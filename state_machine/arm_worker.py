"""The state machine of one arm: task after task from the coordinator, each attempt a behaviour tree (locate the
object and its bin, plan, grasp or back off, carry, release, home, verify the drop).

    worker = threading.Thread(target=arm_worker, args=(arm, robot, coordinator, marks, last_q, args, io,
                                                       handoff_points, camera))

The coordinator (state_machine/coordinator.py) decides which arm does which task and keeps the arms out of each
other's regions; the tree's nodes are in state_machine/behaviour.py; the planning is planning/pick_planner.py.
robot None: plan only (no arm moves). camera None: the objects' known positions are used as they are.
"""
import time
import traceback

import numpy as np

from core import arms as arm_config
from core.arms import HOME_GRIPPER
from hardware.gripper import (GRIP_CONFIRM_READINGS, GRIP_SQUEEZE_UNITS, GRIPPER_OPEN,
                              close_gripper_with_current_feedback)
from planning.pick_planner import HOME_Q, PLAN_LOCK, miss_position, plan_task
from state_machine.coordinator import TO_HANDOFF

CURRENT = {"coordinator": None}  # the running coordinator (for tests and the window)


def short(arm):
    return arm.replace("follower_", "")


def follow(robot, path, gripper, send_joints=None, precise=True):
    """Move along a joint path in one smooth motion (hardware/arm.py follow_path), the gripper going to
    `gripper` on the way; precise=False: its end only roughly (faster). send_joints is not used any more."""
    from hardware.arm import follow_path

    return follow_path(robot, path, gripper, precise=precise)


def go_home(robot, send_joints, read_joints_rad):
    follow(robot, [read_joints_rad(robot), HOME_Q], HOME_GRIPPER, send_joints, precise=False)


def task_tree(task, w):
    """The behaviour tree of one attempt at a task (state_machine/behaviour.py). w: the worker's actions (TaskActions)."""
    from state_machine.behaviour import Action, Fail, Fallback, Sequence

    act = lambda name, fn: Action(name, fn, on_start=w.say)
    to_bin = task.kind != TO_HANDOFF
    return Sequence("attempt", [
        act("locate object", w.locate_object),           # re-detect before every pick (must be seen after a miss)
        act("claim regions", w.claim_regions),            # it may have moved into a region the other arm holds
        act("locate bin", w.locate_bin) if to_bin else None,     # the bin may have been moved
        act("plan", w.plan),
        Fallback("grasp", [
            act("close on it", w.grasp),
            Sequence("missed", [act("back off", w.back_off), act("home", w.home), Fail("missed")]),
        ]),
        act("carry to the bin", w.carry_to_bin) if to_bin else act("place on the drop-off", w.place_on_dropoff),
        act("release", w.release),
        act("leave the drop-off", w.leave_dropoff) if not to_bin else None,
        act("home", w.home),
        act("verify drop", w.verify_drop) if to_bin else None,    # not seen outside the bin afterwards
    ])


class TaskActions:
    """The actions of one arm's behaviour tree; the blackboard bb holds this attempt's data (plans, hold, why)."""

    def __init__(self, arm, robot, coordinator, marks, last_q, args, io, handoff_points, camera):
        self.arm, self.robot, self.coordinator, self.marks = arm, robot, coordinator, marks
        self.last_q, self.args, self.handoff_points, self.camera = last_q, args, handoff_points, camera
        self.send_joints, self.read_joints_rad = io
        self.home_at = time.monotonic()                     # when the arm was last home (out of the camera's view)

    def say(self, text):
        self.coordinator.set_state(self.arm, text)

    @staticmethod
    def _move(shape, xyz, pixel=None):
        for s in {id(shape): shape, id(shape.get("origin", shape)): shape.get("origin", shape)}.values():
            s["xyz"] = np.asarray(xyz, float)
            if pixel is not None:
                s["pixel"] = pixel

    # ---- seeing
    def locate_object(self, bb):
        task = bb["task"]
        if self.camera is None:
            return True
        seen = self.camera.find_object(task.shape, self.home_at)
        if seen is None:
            if task.attempts == 0:                          # first try: a detection gap is no reason to give up
                return True
            bb["why"] = "not seen again"
            return False
        xyz, pixel = seen
        moved = np.linalg.norm(np.asarray(xyz[:2]) - np.asarray(task.shape["xyz"][:2]))
        base = arm_config.base(self.arm)
        if np.hypot(xyz[0] - base["x"], xyz[1] - base["y"]) > self.args.max_reach:
            bb["why"] = f"moved {100 * moved:.0f} cm, out of reach"
            return False
        if moved > 0.01:
            print(f"  {self.arm}: {task.shape.get('label', 'object')} is {100 * moved:.0f} cm from where it was")
        self._move(task.shape, xyz, pixel)
        self.coordinator.relocate(task)
        return True

    def claim_regions(self, bb):
        if self.coordinator.claim(bb["task"]):
            return True
        bb["why"] = "region taken"                         # back to the queue, no attempt counted
        return False

    def locate_bin(self, bb):
        task = bb["task"]
        if self.camera is not None:
            xyz = self.camera.find_bin(task.box, self.home_at)
            if xyz is not None:
                if np.linalg.norm(np.asarray(xyz[:2]) - np.asarray(task.box["xyz"][:2])) > 0.02:
                    print(f"  {self.arm}: the {task.box['colour']} bin has moved")
                task.box["xyz"] = np.asarray(xyz, float)
        return True                                         # not seen: its last known place

    def verify_drop(self, bb):
        """After the release, from home: the object must not be seen outside the bin (objects in a bin are left
        out of the detections); seen near the bin or where it was picked, it fell: try again from there."""
        task = bb["task"]
        if self.camera is None or self.robot is None:
            return True
        stray = self.camera.find_stray(task.shape, self.home_at,
                                       near=[task.box["xyz"], bb["picked_at"]], marks=self.marks)
        if stray is None:
            return True
        print(f"  {self.arm}: it is not in the {task.box['colour']} bin (seen at x {stray[0][0]:.2f} "
              f"y {stray[0][1]:.2f})")
        self._move(task.shape, *stray)
        self.coordinator.relocate(task)
        bb["why"] = "drop missed"
        return False

    # ---- planning
    def plan(self, bb):
        task, args = bb["task"], self.args
        bb["picked_at"] = np.asarray(task.shape["xyz"], float)
        if task.kind == TO_HANDOFF:
            hx, hy = self.handoff_points[task.slot]
            bb["handoff_xyz"] = np.array([hx, hy, task.shape["xyz"][2]])   # same top height: it stands on the table
        obstacles = {"shapes": [m for m in self.marks if m["kind"] == "shape"
                                and m["state"] in ("todo", "handoff", "failed")],
                     "boxes": [m for m in self.marks if m["kind"] == "box"], "arms": dict(self.last_q)}
        start_q = self.last_q[self.arm] if self.robot is None else self.read_joints_rad(self.robot)
        try:
            with PLAN_LOCK:
                bb["plans"], z = plan_task(task, start_q, obstacles, args, bb.get("handoff_xyz"))
        except RuntimeError as error:
            bb["why"] = f"no plan: {error}"
            return False
        print(f"  {self.arm}: grasp at {100 * z:.1f} cm above the table")
        return True

    # ---- moving (each one is a no-op without a robot: plan only)
    PRECISE_STEPS = ("grasp", "over box", "place")     # where the gripper is put somewhere; the rest only roughly

    def _follow(self, bb, step, gripper):
        if self.robot is not None:
            follow(self.robot, dict(bb["plans"])[step], gripper, self.send_joints,
                   precise=step in self.PRECISE_STEPS)

    def grasp(self, bb):
        task = bb["task"]
        bb["opening"] = task.shape.get("open_cmd", GRIPPER_OPEN)     # as wide as this shape needs
        if self.robot is None:
            bb["hold"] = bb["opening"]
            return True
        self._follow(bb, "above shape", bb["opening"])
        self._follow(bb, "grasp", bb["opening"])
        grasped, stop, hold, current_ma = close_gripper_with_current_feedback(
            self.robot, 0.0, self.args.grip_current_ma, GRIP_CONFIRM_READINGS, GRIP_SQUEEZE_UNITS,
            miss_below=miss_position(task.shape, self.args))
        if not grasped:
            print(f"  {self.arm}: missed it (gripper closed to {stop:.0f}/100, {current_ma:.0f} mA)")
            return False
        print(f"  {self.arm}: holding it ({current_ma:.0f} mA)")
        bb["hold"] = hold
        return True

    def back_off(self, bb):
        self._follow(bb, "lift", bb["opening"])
        return True

    def carry_to_bin(self, bb):
        self._follow(bb, "lift", bb["hold"])
        self._follow(bb, "over box", bb["hold"])
        bb["release_at"] = "over box"
        return True

    def place_on_dropoff(self, bb):
        self._follow(bb, "lift", bb["hold"])
        self._follow(bb, "over handoff", bb["hold"])
        self._follow(bb, "place", bb["hold"])
        bb["release_at"] = "place"
        return True

    def release(self, bb):
        if self.robot is not None:
            self.send_joints(self.robot, dict(bb["plans"])[bb["release_at"]][-1], gripper=GRIPPER_OPEN)
        return True

    def leave_dropoff(self, bb):
        self._follow(bb, "leave handoff", GRIPPER_OPEN)
        return True

    def home(self, bb):
        if self.robot is not None:
            go_home(self.robot, self.send_joints, self.read_joints_rad)
            self.last_q[self.arm] = self.read_joints_rad(self.robot)
        self.home_at = time.monotonic()
        return True


def arm_worker(arm, robot, coordinator, marks, last_q, args, io, handoff_points, camera=None):
    """One arm, task after task: the coordinator gives it the next task it may do, and each attempt at it is one run
    of its behaviour tree (task_tree). camera: what the camera sees now (games/pick_and_place.py; find_object, find_bin,
    find_stray), or None (the known positions are used as they are)."""
    w = TaskActions(arm, robot, coordinator, marks, last_q, args, io, handoff_points, camera)
    task = None
    try:
        while True:
            task = coordinator.next_task(arm)
            if task is None:
                break
            print(f"\n{task.label()}  (regions: {', '.join(sorted(task.regions))}, attempt {task.attempts + 1})")
            origin = task.shape.get("origin", task.shape)
            bb = {"task": task}
            if task_tree(task, w).tick(bb):
                if task.kind == TO_HANDOFF:
                    origin["state"], origin["handoff_xyz"] = "handoff", bb["handoff_xyz"]
                    print(f"  {arm}: left it on drop-off point {task.slot} for {task.then_arm}")
                else:
                    origin["state"] = "placed"
                    print(f"  {arm}: dropped it in the {task.box['colour']} bin")
                coordinator.finish(task, "done", bb.get("handoff_xyz"))
                continue
            why = bb.get("why", "failed")
            if why == "region taken":
                print(f"  {arm}: it is now where the other arm works: later")
                coordinator.requeue(task, count=False)
            elif why in ("missed", "drop missed") and task.attempts + 1 < args.attempts:
                print(f"  {arm}: {why}; trying again ({task.attempts + 2}/{args.attempts})")
                coordinator.requeue(task)
            else:
                print(f"  {arm}: {why}: left")
                origin["state"] = "failed"
                coordinator.finish(task, "failed")
    except Exception:
        traceback.print_exc()
        print(f"{arm}: stopped after an error; the other arm carries on with the tasks that do not need it")
        coordinator.abort(arm, locals().get("task"))
        w.say("ERROR (see the terminal)")
        return
    w.say("done")
