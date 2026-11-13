"""The state machine of one arm: task after task from the coordinator, each attempt a behaviour tree (locate the
object and its bin, plan, grasp or back off, carry, release, home, verify the drop).

    worker = threading.Thread(target=arm_worker, args=(arm, robot, coordinator, marks, last_q, args, io,
                                                       handoff_points, camera))

The coordinator (state_machine/coordinator.py) decides which arm does which task and keeps the arms out of each
other's regions; the tree's nodes are in state_machine/behaviour.py; the planning is planning/pick_planner.py.
robot None: plan only (no arm moves). camera None: the objects' known positions are used as they are.
"""
import csv
import math
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
NEAR = 0.12              # another object closer than this to one with no plan may be what blocks it [m]


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

    def __init__(self, arm, robot, coordinator, marks, last_q, args, io, handoff_points, camera, planner=None):
        self.arm, self.robot, self.coordinator, self.marks = arm, robot, coordinator, marks
        self.planner = planner                              # planning/planner_service.py ArmPlanner, or None
        self.prefetch = None                                # the next task, planned ahead while this one moves
        self.last_q, self.args, self.handoff_points, self.camera = last_q, args, handoff_points, camera
        self.send_joints, self.read_joints_rad = io
        self.home_at = time.monotonic()                     # when the arm was last home (out of the camera's view)

    def say(self, text):
        self.coordinator.set_state(self.arm, text)

    @staticmethod
    def _move(shape, xyz, pixel=None, yaw=None):
        for s in {id(shape): shape, id(shape.get("origin", shape)): shape.get("origin", shape)}.values():
            s["xyz"] = np.asarray(xyz, float)
            if pixel is not None:
                s["pixel"] = pixel
            if yaw is not None:
                s["yaw"] = yaw

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
        xyz, pixel, yaw = (*seen, None)[:3]
        moved = np.linalg.norm(np.asarray(xyz[:2]) - np.asarray(task.shape["xyz"][:2]))
        base = arm_config.base(self.arm)
        if np.hypot(xyz[0] - base["x"], xyz[1] - base["y"]) > self.args.max_reach:
            bb["why"] = f"moved {100 * moved:.0f} cm, out of reach"
            return False
        if moved > 0.01:
            print(f"  {self.arm}: {task.shape.get('label', 'object')} is {100 * moved:.0f} cm from where it was")
        self._move(task.shape, xyz, pixel, yaw)
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
    def _obstacles(self, leave_out=()):
        """What the planner avoids: the objects still on the table (columns, with --neighbour-radius), the bins, and
        the other arm at its last known joint angles. leave_out: marks that will be gone by then."""
        gone = {id(m) for m in leave_out}
        return {"shapes": [m for m in self.marks if m["kind"] == "shape" and id(m) not in gone
                           and m["state"] in ("todo", "handoff", "failed")],
                "boxes": [m for m in self.marks if m["kind"] == "box"], "arms": dict(self.last_q)}

    def _inputs(self, task, start_q, obstacles):
        """Everything a plan depends on, to tell whether a plan made ahead of time is still right."""
        return {"shape": np.array(task.shape["xyz"], float), "box": np.array(task.box["xyz"], float),
                "opening": task.shape.get("open_cmd"), "start": np.array(start_q, float),
                "boxes": [np.array(b["xyz"], float) for b in obstacles["boxes"]],
                "shapes": [np.array(m["xyz"], float) for m in obstacles["shapes"]] if self.args.neighbour_radius != 0
                else [],
                "arms": {a: np.array(q, float) for a, q in obstacles["arms"].items() if a != self.arm}}

    def _still_valid(self, planned, now):
        """Why a plan made ahead of time cannot be used now (None: it can). Positions within 2 mm, the other arm
        within 1 deg, the start pose within --prefetch-tolerance (the arm starts where the plan assumed)."""
        close = lambda a, b: np.linalg.norm(a - b) <= 0.002
        if not close(planned["shape"], now["shape"]):
            return "the object moved"
        if not close(planned["box"], now["box"]) or len(planned["boxes"]) != len(now["boxes"]) or \
                not all(close(a, b) for a, b in zip(planned["boxes"], now["boxes"])):
            return "a bin moved"
        if len(planned["shapes"]) != len(now["shapes"]) or \
                not all(close(a, b) for a, b in zip(planned["shapes"], now["shapes"])):
            return "the other objects changed"
        if planned["opening"] != now["opening"]:
            return "another gripper opening"
        if planned["arms"].keys() != now["arms"].keys() or any(
                np.abs(planned["arms"][a] - now["arms"][a]).max() > math.radians(1.0) for a in now["arms"]):
            return "the other arm is elsewhere"
        off = math.degrees(np.abs(planned["start"] - now["start"]).max())
        if off > getattr(self.args, "prefetch_tolerance", 3.0):
            return f"the arm starts {off:.1f} deg from where the plan assumed"
        return None

    def _prefetch_next(self, bb):
        """While this task's object is carried: plan the next task from the home pose, in the planner's process."""
        if self.planner is None or self.planner.mode != "process" or not getattr(self.args, "prefetch", True):
            return
        task = bb["task"]
        upcoming = self.coordinator.peek(self.arm, exclude=task)
        if upcoming is None:
            return
        now = self._obstacles(leave_out=[task.shape, task.shape.get("origin", task.shape)])
        now["arms"][self.arm] = list(HOME_Q)
        self.prefetch = {"task": upcoming, "inputs": self._inputs(upcoming, HOME_Q, now),
                         "future": self.planner.submit(upcoming, HOME_Q, now, self.args)}

    def _take_prefetch(self, task, start_q, obstacles):
        """The plan made ahead of time for this task, if it is still right: (plans, z, log), else None."""
        ahead, self.prefetch = self.prefetch, None
        if ahead is None or ahead["task"] is not task:
            if ahead is not None:
                ahead["future"].cancel()
            return None
        try:
            plans, z, log = ahead["future"].result()
        except Exception:                                   # noqa: BLE001 - no plan ahead: plan now, as always
            return None
        why = self._still_valid(ahead["inputs"], self._inputs(task, start_q, obstacles))
        if why is not None:
            print(f"  {self.arm}: the plan made ahead is not used: {why}")
            return None
        plans[0][1][0] = list(map(float, start_q))           # the path starts where the arm really is
        return plans, z, log

    def plan(self, bb):
        task, args = bb["task"], self.args
        bb["picked_at"] = np.asarray(task.shape["xyz"], float)
        if task.kind == TO_HANDOFF:
            hx, hy = self.handoff_points[task.slot]
            bb["handoff_xyz"] = np.array([hx, hy, task.shape["xyz"][2]])   # same top height: it stands on the table
        obstacles = self._obstacles()
        start_q = self.last_q[self.arm] if self.robot is None else self.read_joints_rad(self.robot)
        started = time.monotonic()
        try:
            ahead = self._take_prefetch(task, start_q, obstacles)
            if ahead is not None:
                bb["plans"], z, log = ahead
                print(log, end="")
                print(f"  {self.arm}: planned ahead, while the last object was carried")
            elif self.planner is not None:
                bb["plans"], z, log = self.planner.plan(task, start_q, obstacles, args, bb.get("handoff_xyz"))
                print(log, end="")
            else:
                with PLAN_LOCK:
                    bb["plans"], z = plan_task(task, start_q, obstacles, args, bb.get("handoff_xyz"))
        except RuntimeError as error:
            print(getattr(error, "log", ""), end="")       # the planner's printout up to the failure
            bb["why"] = f"no plan: {error}"
            return False
        bb["grasp_z"] = z
        print(f"  {self.arm}: grasp at {100 * z:.1f} cm above the table "
              f"({time.monotonic() - started:.1f} s waiting for the plan)")
        return True

    def _check_grasp(self, bb, reached_q, grasped, current_ma):
        """Where the grasp point really went (forward kinematics of the joint readings) against the plan and the
        object: printed, and a row in outputs/grasp_log.csv. Separates the errors: planned vs object = the planner's
        tolerance; reached vs planned = the motors not reaching the planned joint angles. When both are small but
        grasps still miss, the error is outside the model: the camera's position of the object, or the arm's
        calibration (the model's joint angles are not the real ones), or bending the encoders do not see."""
        try:
            from core.arms import arm_from_table
            from core.paths import OUTPUTS
            from planning.kinematics import grasp_pose, reach_frame

            task = bb["task"]
            planned_q = dict(bb["plans"])["grasp"][-1]
            target = (arm_from_table(self.arm) @ np.append(task.shape["xyz"], 1.0))[:3]
            target[2] = bb.get("grasp_z", target[2])
            planned, tilt = grasp_pose(planned_q)
            reached, _ = grasp_pose(reached_q)
            along, across = reach_frame(target)
            parts = lambda v: (1000 * float((v - target)[:2] @ along), 1000 * float((v - target)[:2] @ across),
                               1000 * float((v - target)[2]))
            p_off, r_off = parts(planned), parts(reached)
            joints = np.degrees(np.asarray(reached_q, float) - np.asarray(planned_q, float))
            posture = "top-down" if tilt < 45 else "level"
            text = lambda o: f"{o[0]:+.0f} along, {o[1]:+.0f} across, {o[2]:+.0f} z"
            print(f"  {self.arm}: grasp check ({posture}, {100 * np.linalg.norm(target[:2]):.0f} cm out), mm from the "
                  f"object (along: + beyond, - short; across: + left): planned {text(p_off)} | reached {text(r_off)} | "
                  f"joints off " + " ".join(f"{d:+.1f}" for d in joints) + " deg")
            path = OUTPUTS / "grasp_log.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.is_file()
            with path.open("a", newline="") as handle:
                out = csv.writer(handle)
                if new:
                    out.writerow(["time", "arm", "object", "posture", "reach_cm", "target_x", "target_y", "target_z",
                                  "planned_along_mm", "planned_across_mm", "planned_z_mm", "reached_along_mm",
                                  "reached_across_mm", "reached_z_mm", "pan_deg", "lift_deg", "elbow_deg",
                                  "wrist_flex_deg", "wrist_roll_deg", "grasped", "current_ma"])
                out.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.arm, task.shape.get("label", ""), posture,
                              round(100 * float(np.linalg.norm(target[:2])), 1), *np.round(target, 4),
                              *np.round(p_off, 1), *np.round(r_off, 1), *np.round(joints, 2), int(bool(grasped)),
                              round(float(current_ma), 1)])
        except Exception as error:                          # noqa: BLE001 - a check must never stop the arm
            print(f"  {self.arm}: grasp check failed ({error})")

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
        reached_q = self.read_joints_rad(self.robot)            # where the arm stopped, for the grasp check
        trace, info = [], {}
        from planning.pick_planner import grasp_settings

        squeeze = float(grasp_settings(task.shape.get("name")).get("squeeze", GRIP_SQUEEZE_UNITS))
        grasped, stop, hold, current_ma = close_gripper_with_current_feedback(
            self.robot, 0.0, self.args.grip_current_ma, GRIP_CONFIRM_READINGS, squeeze,
            miss_below=miss_position(task.shape, self.args),
            stall_readings=getattr(self.args, "grip_stall_readings", 5) or None,
            empty_below=getattr(self.args, "grip_empty", 8.0), trace=trace, info=info,
            close_speed=getattr(self.args, "grip_close_speed", 40.0) or None)
        self._check_grasp(bb, reached_q, grasped, current_ma)
        self._log_gripper(task, bb["opening"], trace, info.get("reason", ""), grasped)
        if not grasped:
            print(f"  {self.arm}: missed it (gripper {info.get('reason', '')} at {stop:.0f}/100, {current_ma:.0f} mA, "
                  f"peak {max((c for _, _, c in trace), default=0):.0f} mA)")
            return False
        print(f"  {self.arm}: holding it ({info.get('reason', '')}: stopped at {stop:.0f}/100, {current_ma:.0f} mA)")
        bb["hold"] = hold
        return True

    def _log_gripper(self, task, opening, trace, reason, grasped):
        """Every reading of this close (time, gripper position, current) as rows of outputs/gripper_trace.csv, to see
        how the real gripper behaves on each object (where it stops, how high the current gets)."""
        try:
            from core.paths import OUTPUTS

            path = OUTPUTS / "gripper_trace.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.is_file()
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with path.open("a", newline="") as handle:
                out = csv.writer(handle)
                if new:
                    out.writerow(["time", "arm", "object", "opened_to", "result", "grasped", "t_s", "position",
                                  "current_ma"])
                for t, position, current in trace:
                    out.writerow([stamp, self.arm, task.shape.get("label", ""), round(float(opening), 1), reason,
                                  int(bool(grasped)), t, position, current])
        except Exception as error:                          # noqa: BLE001 - logging must never stop the arm
            print(f"  {self.arm}: gripper trace not saved ({error})")

    def back_off(self, bb):
        self._follow(bb, "lift", bb["opening"])
        return True

    def carry_to_bin(self, bb):
        self._prefetch_next(bb)                             # plan the next object while this one is carried
        self._follow(bb, "lift", bb["hold"])
        self._follow(bb, "over box", bb["hold"])
        bb["release_at"] = "over box"
        return True

    def place_on_dropoff(self, bb):
        self._prefetch_next(bb)
        self._follow(bb, "lift", bb["hold"])
        self._follow(bb, "over handoff", bb["hold"])
        self._follow(bb, "place", bb["hold"])
        bb["release_at"] = "place"
        return True

    def release(self, bb):
        if self.robot is not None:
            # open only as wide as it was picked with: a wide-open jaw swings into the object just put down
            self.send_joints(self.robot, dict(bb["plans"])[bb["release_at"]][-1],
                             gripper=bb.get("opening", GRIPPER_OPEN))
        return True

    def leave_dropoff(self, bb):
        self._follow(bb, "leave handoff", bb.get("opening", GRIPPER_OPEN))
        return True

    def home(self, bb):
        if self.robot is not None:
            go_home(self.robot, self.send_joints, self.read_joints_rad)
            self.last_q[self.arm] = self.read_joints_rad(self.robot)
        self.home_at = time.monotonic()
        return True


def arm_worker(arm, robot, coordinator, marks, last_q, args, io, handoff_points, camera=None, actions=None,
               planner=None):
    """One arm, task after task: the coordinator gives it the next task it may do, and each attempt at it is one run
    of its behaviour tree (task_tree). camera: what the camera sees now (games/pick_and_place.py; find_object, find_bin,
    find_stray), or None (the known positions are used as they are). actions: the TaskActions class to use (the
    simulator passes one that moves the arms in PyBullet instead of the real motors). planner: the arm's
    planning/planner_service.py ArmPlanner (plans in its own process, and the next task ahead of time), or None
    (plan in this thread, one arm at a time)."""
    w = (actions or TaskActions)(arm, robot, coordinator, marks, last_q, args, io, handoff_points, camera, planner)
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
            elif why.startswith("no plan"):
                xyz = np.asarray(task.shape["xyz"], float)[:2]
                near = any(m["kind"] == "shape" and m is not origin and m is not task.shape
                           and m["state"] in ("todo", "handoff", "failed")
                           and np.linalg.norm(np.asarray(m.get("handoff_xyz") if m["state"] == "handoff"
                                                         else m["xyz"], float)[:2] - xyz) < NEAR
                           for m in marks)
                outcome = coordinator.blocked(task, near)
                print(f"  {arm}: {why}: " + {"later": "other objects are close around it, tried again once one of "
                                                      "them is moved",
                                             "other arm": "the other arm tries it from its side",
                                             "failed": "left"}[outcome])
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
