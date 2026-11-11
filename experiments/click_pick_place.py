"""Pick and place with clicked positions instead of YOLO: mark the shapes and the boxes in the camera image, then
the arms pick every shape and drop it into the box of its colour.

    python experiments/click_pick_place.py                       # both arms, one task at a time
    python experiments/click_pick_place.py --parallel            # both arms at the same time where it is safe
    python experiments/click_pick_place.py --arm follower_blue   # one arm only
    python experiments/click_pick_place.py --no-robot --gui      # no arm: plan and show the moves in PyBullet

In the camera window:
    left click        mark a point, then say what it is:
      r g b y           a SHAPE of that colour (click its centre)
      R G B Y / 1-4     the BOX of that colour (1 red, 2 green, 3 blue, 4 yellow; click where to drop)
    u                 undo the last mark          c   clear all marks
    Space / Enter     go: sort every shape into its box
    h                 send the arms home          q / Esc   quit: after the tasks running now, each arm goes back
                                                            to the pose it started in and holds it (torque on)

Who does what (control/coordinator.py): a shape goes to the arm nearest to it if that arm also reaches its box
(DIRECT). If only the other arm reaches the box, the near arm puts the shape down on a handoff point in the middle
(arms.json "handoff") and the other arm takes it from there to the box (HANDOFF). Each mark is labelled with its
plan. The field is split into each arm's side and a middle strip; a task reserves every region it touches before
it starts, and frees them when its arm is home again, so the arms are never in the same region. With --parallel
both arms work at once (e.g. each on its own side); without it, one task runs at a time.

Each arm runs in its own thread; each attempt at a task is one run of a small behaviour tree (task_tree: locate the
object and its bin, plan, grasp or back off, carry, release, home, verify the drop; control/behaviour.py); the planner (sim_plan's World: the table, the other marked shapes, the boxes and the OTHER ARM at its
last known joint angles as obstacles) is used by one arm at a time. The click's 3D point comes from the ZED point
cloud and the ArUco calibration; for a shape it is its top, so the grasp height follows the shape.
"""
import argparse
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from calibration.calibrate import FILE as CALIBRATION_FILE, arm_from_table, arms, camera_to_table, open_zed  # noqa: E402
from control import arm_config  # noqa: E402
from control.coordinator import BOX_SLACK, DIRECT, TO_HANDOFF, Coordinator, Regions, build_tasks, drop_point  # noqa: E402
from experiments.click_to_go import click_to_camera_xyz  # noqa: E402
from experiments.Full_pipeline import (GRIP_CONFIRM_READINGS, GRIP_CURRENT_MA, GRIP_SQUEEZE_UNITS,  # noqa: E402
                                       GRIPPER_OPEN, HOME_Q, close_gripper_with_current_feedback)
from experiments.pick_and_place import SEED_TRIES, follow, go_home, plan_sequence  # noqa: E402
from experiments.sim_plan import JOINTS, TABLE_CLEARANCE, URDF_FILE, World  # noqa: E402

WINDOW = "click pick and place"
SHAPE_KEYS = {ord("r"): "red", ord("g"): "green", ord("b"): "blue", ord("y"): "yellow"}
BOX_KEYS = {ord("R"): "red", ord("G"): "green", ord("B"): "blue", ord("Y"): "yellow",
            ord("1"): "red", ord("2"): "green", ord("3"): "blue", ord("4"): "yellow"}
BGR = {"red": (40, 40, 230), "green": (40, 180, 40), "blue": (230, 90, 30), "yellow": (30, 220, 240)}
SHAPE_RADIUS = 0.03      # other marked shapes are obstacles: a column of this radius up to their top [m]
PLAN_LOCK = threading.Lock()   # PyBullet is used by one thread at a time
CURRENT = {"coordinator": None}  # the running coordinator (for tests and the window)


def short(arm):
    return arm.replace("follower_", "")


# ---------------------------------------------------------------------------------------------- the planning world
def add_obstacle(world, x, y, half_x, half_y, height):
    """A box from the table up to `height` at (x, y) in the arm frame, as an obstacle for the planners."""
    import pybullet as p

    c = world.client
    half = [half_x, half_y, height / 2]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.6, 0.4, 0.2, 0.5], physicsClientId=c)
    body = p.createMultiBody(0, col, vis, basePosition=[x, y, height / 2], physicsClientId=c)
    world.checker.obstacle_body_ids.append(body)
    return body


def add_other_arm(world, to_arm, other, q):
    """The other arm's robot model at its base and joint angles q, as an obstacle (it is not planned)."""
    import pybullet as p

    c = world.client
    b = arm_config.base(other)
    xy = (to_arm @ np.array([b["x"], b["y"], 0.0, 1.0]))[:2]
    yaw = math.atan2(to_arm[1, 0], to_arm[0, 0]) + math.radians(b["yaw_deg"])
    body = p.loadURDF(str(URDF_FILE), basePosition=[*xy, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                      useFixedBase=True, physicsClientId=c)
    index = {p.getJointInfo(body, i, physicsClientId=c)[1].decode(): i
             for i in range(p.getNumJoints(body, physicsClientId=c))}
    for name, value in zip(JOINTS, q):
        p.resetJointState(body, index[name], float(value), physicsClientId=c)
    world.checker.obstacle_body_ids.append(body)
    return body


def handoff_checker(handoff_points, args):
    """can_grasp(arm, point index, top z): can that arm grasp a shape of that height on that handoff point (an IK
    check in an empty world, from home)? Cached, so each combination is solved once. Returns (can_grasp, close)."""
    cache, worlds = {}, {}

    def can_grasp(arm, i, top):
        # keyed by the point's position, not its index: a caller may replace the list's points between rounds
        key = (arm, round(float(handoff_points[i][0]), 3), round(float(handoff_points[i][1]), 3), round(float(top), 2))
        if key not in cache:
            with PLAN_LOCK:
                if arm not in worlds:
                    worlds[arm] = World([], list(HOME_Q), gui=False, clearance=args.clearance, arm=arm,
                                        calibration_path=arm_config.follower_calibration(arm))
                    worlds[arm].level_tolerance_deg = getattr(args, "level_tolerance", 30.0)
                    worlds[arm].stall_iterations = getattr(args, "ik_stall", 30) or None
                world = worlds[arm]
                h = (arm_from_table(arm) @ np.array([*handoff_points[i][:2], 0.0, 1.0]))[:3]
                grasp_z = grasp_height(top, args)
                ok = False
                for posture in postures_for(h, args):
                    for z in grasp_heights(top, posture, args):
                        world.start_q = list(HOME_Q)
                        result = world.find_pose([h[0], h[1], z], args.seed, posture=posture)
                        if result is not None and result.success:
                            ok = True
                            break
                    if ok:
                        break
            cache[key] = ok
            if not ok:
                print(f"  handoff point H{i} {handoff_points[i]}: {arm} cannot grasp a {100 * top:.1f} cm shape there")
        return cache[key]

    def close():
        for world in worlds.values():
            world.close()
    return can_grasp, close


def grasp_height(top, args):
    """Height of the point between the jaws at the grasp: half the object's height (--grasp-at middle, the default)
    or --grasp-depth below its top (--grasp-at top); never below --min-grasp-z (the lowest the jaws reach). If
    there is no pose there, the planner tries 5 mm higher each time."""
    z = top / 2 if getattr(args, "grasp_at", "middle") == "middle" else top - args.grasp_depth
    return max(z, args.min_grasp_z)


# How far the fingers reach below the grasp point when the gripper is level (PyBullet, robot_urdfs/so101: plates from
# 3.3 cm below to 2.9 cm above it). A level grasp point cannot go lower than about 3.5 cm (table clearance).
LEVEL_JAW_BELOW = 0.033


def grasp_heights(top, posture, args):
    """The heights to try for the grasp point, lowest first (5 mm steps); the planner's table clearance check
    decides which are possible.
      level (horizontal): the jaws are at that height, so from half the object's height (--grasp-at) up to
          --level-below-top under its top (higher, they close over it); not below --min-grasp-z. With
          --pick-posture level, up to where the fingers still cover --level-min-contact of the object.
      down: the fingertips hang tip_below_grasp() (from the URDF) below the grasp point, so it never goes lower than
          that + 1.2 cm (the tips 1.2 cm over the table), and at most so high that the tips still reach 1.2 cm
          below the object's top."""
    from control.gripper import tip_below_grasp

    if posture == "down":
        low = max(grasp_height(top, args), tip_below_grasp() + 0.012)
        high = top + tip_below_grasp() - 0.012
    else:
        low, high = grasp_height(top, args), top - args.level_below_top
        # Never below where the level fingers still clear the table (--clearance): a lower target is unreachable, and
        # the IK would then stop at the nearest reachable pose up to PRECISE (12 mm) away, partly sideways, so the
        # jaws ended off-centre on small objects. Starting at the real lowest height keeps the error near zero.
        low = max(low, getattr(args, "clearance", 0.01) + LEVEL_JAW_BELOW + 0.002)
        if getattr(args, "pick_posture", "level") == "level":
            # The level fingers are tall plates reaching LEVEL_JAW_BELOW under the grasp point, so a short object is
            # still between them with the grasp point above its top: allow that while they cover
            # --level-min-contact of it (the table clearance check decides the lowest possible height).
            high = max(high, top + LEVEL_JAW_BELOW - getattr(args, "level_min_contact", 0.015))
    if high < low - 1e-6:
        return np.array([])
    return np.arange(low, max(high, low) + 1e-6, 0.005)


def postures_for(xy_arm, args):
    """The gripper postures to try for a pick at this point (arm frame).
    --pick-posture level (default): only level (fingers parallel to the table, within --level-tolerance); an object
        too close to the base for a level gripper (about 20 cm) then has no plan instead of a top-down grasp.
    --pick-posture auto: level (straight in) first, unless it is within --close-radius of the base (a level gripper
        cannot get that close): top-down first there. The other one when the first has no pose."""
    if getattr(args, "pick_posture", "level") == "level":
        return ["horizontal"]
    close = np.hypot(xy_arm[0], xy_arm[1]) < args.close_radius
    return ["down", "horizontal"] if close else ["horizontal", "down"]


def plan_task(task, start_q, obstacles, args, handoff_xyz=None):
    """Plan the whole task in the arm's frame. Returns ([(label, path)], grasp z) or raises RuntimeError."""
    arm = task.arm
    to_arm = arm_from_table(arm)
    o = (to_arm @ np.append(task.shape["xyz"], 1.0))[:3]
    base = arm_config.base(arm)
    drop = drop_point(task.box["xyz"], [base["x"], base["y"]])      # inside the box, on the side nearest the arm
    b = (to_arm @ np.array([*drop, 0.0, 1.0]))[:3]
    top = max(o[2], 0.0)
    opening = task.shape.get("open_cmd", GRIPPER_OPEN)        # the jaws open only as wide as this shape needs
    world = World([], start_q, gui=args.gui, clearance=args.clearance, arm=arm,
                  calibration_path=arm_config.follower_calibration(arm), gripper_open=opening / 100)
    world.level_tolerance_deg = getattr(args, "level_tolerance", 30.0)
    world.stall_iterations = getattr(args, "ik_stall", 30) or None
    try:
        for other in obstacles["shapes"]:
            if other is task.shape or other is task.shape.get("origin"):
                continue
            xyz = other.get("handoff_xyz") if other["state"] == "handoff" else other["xyz"]
            xy = (to_arm @ np.append(xyz, 1.0))[:2]
            add_obstacle(world, *xy, SHAPE_RADIUS, SHAPE_RADIUS, max(xyz[2], 0.01))
        for box in obstacles["boxes"]:
            xy = (to_arm @ np.append(box["xyz"], 1.0))[:2]
            add_obstacle(world, *xy, args.box_size / 2, args.box_size / 2, args.box_height + args.box_margin)
        for other, q in obstacles["arms"].items():
            if other != arm:
                add_other_arm(world, to_arm, other, q)

        if task.kind == TO_HANDOFF:
            h = (to_arm @ np.array([*handoff_xyz[:2], 0.0, 1.0]))[:3]
            rest = lambda z: [("over handoff", [h[0], h[1], top + args.lift]), ("place", [h[0], h[1], z]),
                              ("leave handoff", [h[0], h[1], top + args.approach])]
            place_postures = postures_for(h, args)
        else:
            drop_z = max(args.box_height + args.drop_above, top + args.lift)
            rest = lambda z: [("over box", [b[0], b[1], drop_z])]
            place_postures = ["horizontal", "down"]
        error = None
        # one posture for the whole pick (above, down, lift: a straight descent); the other one if that fails.
        # A pose that exists is found in ~10 ms, but each failed search takes ~1.5 s, and a grasp height that is out
        # of reach fails for every random start. So scan every height with ONE start first and only then give the
        # same heights the other starts (before: 3 starts per height, ~20 s for an object whose lowest heights are
        # out of reach).
        def attempt(posture, z, seed, tries):
            return plan_sequence(world, start_q, [
                ("above shape", [o[0], o[1], top + args.approach]),
                ("grasp", [o[0], o[1], z]),
                ("lift", [o[0], o[1], top + args.lift]),
                *rest(z),
            ], seed, postures={"above shape": [posture], "grasp": [posture], "lift": [posture],
                               "over handoff": place_postures, "place": place_postures,
                               "leave handoff": place_postures, "over box": ["horizontal", "down"]},
                seed_tries=tries)

        def done(posture, plans, z):
            print(f"  {arm}: {posture} grasp ({100 * np.hypot(*o[:2]):.0f} cm from its base)")
            if args.gui:
                for _, path in plans:
                    world.start_q = list(path[0])
                    world.play(path)
            return plans, z

        retry = (args.seed + 1, SEED_TRIES - 1)
        passes = [(args.seed, 1), retry] if SEED_TRIES > 1 else [(args.seed, 1)]
        for seed, tries in passes:
            for posture in postures_for(o, args):
                heights = grasp_heights(top, posture, args)
                if not len(heights) and seed == args.seed:
                    print(f"  {arm}: a {100 * top:.1f} cm object is too short for a {posture} grasp")
                for i, z in enumerate(heights):
                    try:
                        plans = attempt(posture, z, seed, tries)
                    except RuntimeError as exc:
                        error = exc
                        if "grasp" not in str(exc) and "place" not in str(exc):
                            break
                        continue
                    if seed == args.seed and i > 0 and SEED_TRIES > 1:
                        # Found, but not at the lowest height: a lower one may only have missed its single quick
                        # try (near the edge of what the arm reaches level). Give those the other starts first, so
                        # the grasp is as low as possible (costs ~1-3 s, only for these picks).
                        for lower in heights[:i]:
                            try:
                                return done(posture, attempt(posture, lower, *retry), lower)
                            except RuntimeError:
                                pass
                    return done(posture, plans, z)
        raise RuntimeError(str(error))
    finally:
        world.close()


# ---------------------------------------------------------------------------------------------- an arm's worker
def task_tree(task, w):
    """The behaviour tree of one attempt at a task (control/behaviour.py). w: the worker's actions (TaskActions)."""
    from control.behaviour import Action, Fail, Fallback, Sequence

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
    def _follow(self, bb, step, gripper):
        if self.robot is not None:
            follow(self.robot, dict(bb["plans"])[step], gripper, self.send_joints)

    def grasp(self, bb):
        task = bb["task"]
        bb["opening"] = task.shape.get("open_cmd", GRIPPER_OPEN)     # as wide as this shape needs
        if self.robot is None:
            bb["hold"] = bb["opening"]
            return True
        self._follow(bb, "above shape", bb["opening"])
        self._follow(bb, "grasp", bb["opening"])
        grasped, stop, hold, current_ma = close_gripper_with_current_feedback(
            self.robot, 0.0, self.args.grip_current_ma, GRIP_CONFIRM_READINGS, GRIP_SQUEEZE_UNITS)
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
    of its behaviour tree (task_tree). camera: what the camera sees now (pick_and_place.py; find_object, find_bin,
    find_stray), or None (click_pick_place: the clicked positions are used as they are)."""
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


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="both", help="both (default), follower_pink or follower_blue")
    ap.add_argument("--parallel", action="store_true",
                    help="both arms at the same time when their tasks are in different regions")
    ap.add_argument("--max-reach", type=float, default=0.45,
                    help="an arm only takes shapes, boxes and handoff points within this distance of its base [m]")
    ap.add_argument("--no-robot", action="store_true", help="no arm connected: plan only")
    ap.add_argument("--gui", action="store_true", help="show each planned move in PyBullet (one task at a time)")
    ap.add_argument("--box-height", type=float, default=0.08, help="height of the boxes [m]")
    ap.add_argument("--box-size", type=float, default=0.15, help="width of the boxes, for the planner [m]")
    ap.add_argument("--box-margin", type=float, default=0.03,
                    help="the arm keeps this much above the boxes' rim (room for the shape it holds) [m]")
    ap.add_argument("--drop-above", type=float, default=0.08, help="jaws this high over the box's rim at release [m]")
    ap.add_argument("--drop-inset", type=float, default=0.04,
                    help="the shape is released this far inside the box's rim nearest the arm (so a box whose centre "
                         "is a little out of reach still works) [m]")
    ap.add_argument("--approach", type=float, default=0.06, help="height above the shape's top to come down from [m]")
    ap.add_argument("--grasp-at", choices=["middle", "top"], default="middle",
                    help="grasp at half the shape's height, or --grasp-depth below its top")
    ap.add_argument("--level-below-top", type=float, default=0.01,
                    help="a level (horizontal) grasp goes at least this far below the shape's top [m]; higher, the "
                         "jaws close over it, so a shorter shape is grasped top-down")
    ap.add_argument("--grasp-depth", type=float, default=0.015, help="--grasp-at top: this far below the top [m]")
    ap.add_argument("--min-grasp-z", type=float, default=0.02,
                    help="lowest grasp point for a level grasp [m] (the table clearance check still applies)")
    ap.add_argument("--lift", type=float, default=0.10, help="lift height above the shape's top [m]")
    ap.add_argument("--close-radius", type=float, default=0.15,
                    help="--pick-posture auto: closer than this to an arm's base, the gripper picks top-down first [m]")
    ap.add_argument("--pick-posture", choices=["level", "auto"], default="level",
                    help="level (default): the fingers always parallel to the table when picking (never top-down; "
                         "objects too close to the base for that are left); auto: top-down near the base or as a fallback")
    ap.add_argument("--level-tolerance", type=float, default=15.0,
                    help="how far a level gripper may tilt from level [deg] (was 30)")
    ap.add_argument("--ik-stall", type=int, default=30,
                    help="the IK search gives up on a pose after this many iterations without progress (0 = run all "
                         "160); unreachable grasp heights then cost ~0.6 s instead of ~1.5 s")
    ap.add_argument("--level-min-contact", type=float, default=0.015,
                    help="--pick-posture level: a short object may be grasped with the grasp point above its top while "
                         "the fingers cover at least this much of it [m]")
    ap.add_argument("--attempts", type=int, default=2, help="grasp attempts per task")
    ap.add_argument("--clearance", type=float, default=0.005,
                    help="the gripper keeps this far above the table in the plans [m] (was 0.01; a level grasp point "
                         "then goes down to 4.0 cm instead of 4.5 cm; raise it if the fingers touch the table)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grip-current-ma", type=float, default=GRIP_CURRENT_MA)
    ap.add_argument("--serial", type=int, help="the ZED's serial number (default: the first camera)")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    from control.IK_control import add_motion_args, apply_motion_args

    add_motion_args(ap)
    args = ap.parse_args()
    apply_motion_args(args)
    BOX_SLACK["value"] = max(0.0, args.box_size / 2 - args.drop_inset)
    if args.parallel and args.gui:
        raise SystemExit("--gui shows one task at a time: use it without --parallel")

    T_table = camera_to_table()
    if T_table is None:
        raise SystemExit(f"No camera_to_table in {CALIBRATION_FILE}: run `python calibration/calibrate.py` first")
    all_arms = arms()
    names = list(all_arms) if args.arm == "both" else [args.arm]
    if any(name not in all_arms for name in names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: both, {', '.join(all_arms)}")
    handoff_points = arm_config.handoff_points()

    import pyzed.sl as sl
    from control.IK_control import connect_robot, on_every_arm, park_and_disconnect, read_joints_rad, send_joints

    robots = {}
    for name in names:
        if args.no_robot:
            robots[name] = None
            continue
        try:
            robots[name] = connect_robot(name)
        except (ConnectionError, OSError, RuntimeError) as error:
            if len(names) == 1:
                raise
            print(f"{name}: could not connect ({str(error).strip().splitlines()[0]}); the other arm does the whole "
                  f"field. Check `python ports/setup_usb_ports.py --check` and the arm's power.")
    if not robots:
        raise SystemExit("no arm connected")
    bases = {name: np.array([all_arms[name]["x"], all_arms[name]["y"]]) for name in robots}
    regions = Regions(bases, arm_config.middle_half_width())
    missing = [name for name in names if name not in robots]
    warning = (f"{', '.join(missing)} NOT CONNECTED: no handoffs, {next(iter(robots))} does what it reaches alone"
               if missing else "")
    if warning:
        print("\n*** " + warning + " ***\n")
    print("Arms: " + ", ".join(f"{name} at y {bases[name][1]:+.2f}" for name in bases)
          + f" | handoff points: {handoff_points} | {'parallel' if args.parallel else 'one task at a time'}")
    last_q = {name: list(HOME_Q) for name in robots}
    workers = []
    can_grasp, close_checker = handoff_checker(handoff_points, args)

    def disconnect_all():
        """Let running tasks finish, then park every arm (back to its start pose, torque kept) and disconnect."""
        if CURRENT["coordinator"] is not None:
            CURRENT["coordinator"].stop.set()
        for worker in workers:
            worker.join()
        on_every_arm(robots, lambda name, robot: park_and_disconnect(robot))    # both arms at the same time

    def home_all():
        def home(name, robot):
            go_home(robot, send_joints, read_joints_rad)
            last_q[name] = read_joints_rad(robot)
        on_every_arm(robots, home)                     # both arms at the same time

    def running():
        return any(w.is_alive() for w in workers)

    marks = []                     # {"kind": "shape"/"box", "colour", "xyz" (table frame), "pixel", "state"}
    state = {"pending": None, "status": "click a shape's centre or a box", "plan": {}}
    frame = cloud = None
    try:
        home_all()                           # start at the home pose (control/arm_config.py), out of the camera's view
        zed = open_zed(args.serial)
    except BaseException:
        disconnect_all()
        raise
    image, cloud_mat, runtime = sl.Mat(), sl.Mat(), sl.RuntimeParameters()
    camera_from_table = np.linalg.inv(T_table)
    try:
        from vision.detection import camera_intrinsics

        intrinsics = camera_intrinsics(zed)
    except Exception:
        intrinsics = None              # handoff points are then not drawn

    def to_pixel(xyz):
        """TABLE-frame point -> image pixel (None without the camera's intrinsics)."""
        if intrinsics is None:
            return None
        fx, fy, cx, cy = intrinsics
        x, y, z = camera_from_table[:3, :3] @ np.asarray(xyz, float) + camera_from_table[:3, 3]
        return (int(round(fx * x / z + cx)), int(round(fy * y / z + cy))) if z > 0 else None

    def plan_labels():
        """{id(mark): text} of what will happen to each todo shape."""
        shapes = [m for m in marks if m["kind"] == "shape" and m["state"] == "todo"]
        tasks, skipped = build_tasks(shapes, [m for m in marks if m["kind"] == "box"], bases, handoff_points,
                                     args.max_reach, regions, can_grasp)
        labels = {id(t.shape): (f"-> {short(t.arm)}" if t.kind == DIRECT else
                                f"-> {short(t.arm)} -> handoff -> {short(t.then_arm)}") for t in tasks}
        labels.update({id(s): "-> no arm" for s, _ in skipped})
        return labels

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or cloud is None or running():
            return
        scale = frame.shape[1] / args.width if args.width < frame.shape[1] else 1.0
        u, v = int(round(x * scale)), int(round(y * scale))
        xyz_cam = click_to_camera_xyz(cloud, u, v)
        if xyz_cam is None:
            state.update(pending=None, status="no depth at that pixel: click again")
            return
        xyz = T_table[:3, :3] @ xyz_cam + T_table[:3, 3]
        state.update(pending={"pixel": (u, v), "xyz": xyz},
                     status="shape: r g b y | box: R G B Y (or 1 2 3 4)")

    def draw(view):
        for i, (hx, hy) in enumerate(handoff_points):
            pixel = to_pixel([hx, hy, 0.0])
            if pixel:
                cv2.drawMarker(view, pixel, (255, 255, 255), cv2.MARKER_DIAMOND, 22, 2)
                cv2.putText(view, f"H{i}", (pixel[0] + 12, pixel[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
        for m in marks:
            colour = BGR[m["colour"]]
            u, v = m["pixel"]
            if m["kind"] == "box":
                cv2.rectangle(view, (u - 22, v - 22), (u + 22, v + 22), colour, 3)
                text = f"{m['colour']} box"
            else:
                cv2.circle(view, (u, v), 12, colour, -1 if m["state"] == "todo" else 2)
                if m["state"] == "placed":
                    cv2.line(view, (u - 12, v), (u + 12, v), colour, 2)
                elif m["state"] == "failed":
                    cv2.drawMarker(view, (u, v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 3)
                elif m["state"] == "handoff" and m.get("handoff_xyz") is not None:
                    pixel = to_pixel(m["handoff_xyz"])
                    if pixel:
                        cv2.circle(view, pixel, 12, colour, -1)
                        cv2.arrowedLine(view, (u, v), pixel, colour, 2, tipLength=0.08)
                text = f"{m['colour']} {m['state']} {state['plan'].get(id(m), '')}"
            cv2.putText(view, text, (u + 16, v - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(view, text, (u + 16, v - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        if state["pending"] is not None:
            cv2.drawMarker(view, state["pending"]["pixel"], (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
        coordinator = CURRENT["coordinator"]
        lines = [state["status"], "u undo | c clear | Space go | h home | q quit"] + ([warning] if warning else [])
        if coordinator is not None:
            lines += [f"{short(a)}: {s}" for a, s in sorted(coordinator.arm_state.items())]
        for i, text in enumerate(lines):
            cv2.putText(view, text, (15, 35 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
            cv2.putText(view, text, (15, 35 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def show():
        view = frame.copy()
        draw(view)
        if args.width < view.shape[1]:
            view = cv2.resize(view, (args.width, round(view.shape[0] * args.width / view.shape[1])))
        cv2.imshow(WINDOW, view)
        return view

    def go():
        shapes = [m for m in marks if m["kind"] == "shape" and m["state"] == "todo"]
        boxes = [m for m in marks if m["kind"] == "box"]
        if not shapes:
            state["status"] = "no shapes left to pick: mark some first"
            return
        tasks, skipped = build_tasks(shapes, boxes, bases, handoff_points, args.max_reach, regions, can_grasp)
        for shape, why in skipped:
            print(f"{shape['colour']} shape at x {shape['xyz'][0]:.2f} y {shape['xyz'][1]:.2f}: skipped, {why}")
        if not tasks:
            state["status"] = "nothing the arms can do: see the terminal"
            if warning:
                print(f"\n*** {warning} ***")
            return
        print("\nPlan:")
        for task in tasks:
            then = f", then {task.then_arm} to the {task.box['colour']} box" if task.kind == TO_HANDOFF else ""
            print(f"  {task.label()}{then}")
        home_all()                           # every arm starts at home (its pose is an obstacle for the other)
        coordinator = Coordinator(tasks, regions, handoff_points, args.parallel, arm_config.handoff_min_spacing())
        CURRENT["coordinator"] = coordinator
        workers.clear()
        for name, robot in robots.items():
            worker = threading.Thread(target=arm_worker, daemon=True, name=name,
                                      args=(name, robot, coordinator, marks, last_q, args,
                                            (send_joints, read_joints_rad), handoff_points))
            workers.append(worker)
            worker.start()
        state["status"] = "working ... (q stops after the tasks running now)"

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(WINDOW, on_mouse)
    sized, was_running, last_report = False, False, time.monotonic()
    print(__doc__.split("In the camera window:")[1].split("Who does what")[0])
    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
                frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                cloud = np.array(cloud_mat.get_data()[:, :, :3], np.float32)
            if frame is None:
                cv2.waitKey(1)
                continue
            busy = running()
            if busy and CURRENT["coordinator"] is not None and time.monotonic() - last_report > 15:
                last_report = time.monotonic()
                print("  status: " + " | ".join(f"{short(a)}: {s}"
                                                 for a, s in sorted(CURRENT["coordinator"].arm_state.items())))
            if was_running and not busy:
                done = sum(m["state"] == "placed" for m in marks if m["kind"] == "shape")
                state["status"] = f"done: {done} placed; mark more or q to quit"
            was_running = busy
            if not busy:
                state["plan"] = plan_labels()
            view = show()
            if not sized:
                cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                sized = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                if busy:
                    print("Stopping after the tasks running now ...")
                    CURRENT["coordinator"].stop.set()
                    for worker in workers:
                        worker.join()
                break
            if busy:
                continue                    # no marking while the arms work
            pending = state["pending"]
            if pending is not None and (key in SHAPE_KEYS or key in BOX_KEYS):
                kind, colour = ("shape", SHAPE_KEYS[key]) if key in SHAPE_KEYS else ("box", BOX_KEYS[key])
                if kind == "box":       # one box per colour: a new one replaces the old
                    marks[:] = [m for m in marks if not (m["kind"] == "box" and m["colour"] == colour)]
                marks.append({"kind": kind, "colour": colour, "state": "todo", **pending})
                x, y, z = pending["xyz"]
                print(f"{colour} {kind} at x {x:.3f} y {y:.3f} z {z:.3f} m (table frame)")
                state.update(pending=None, status=f"{colour} {kind} marked; click the next one or Space to go")
            elif key == ord("u") and marks:
                removed = marks.pop()
                state["status"] = f"removed the {removed['colour']} {removed['kind']}"
            elif key == ord("c"):
                marks.clear()
                state.update(pending=None, status="all marks cleared")
            elif key in (32, 13, 10):           # Space or Enter
                state["pending"] = None
                go()
            elif key == ord("h"):
                home_all()
                state["status"] = "home"
    finally:
        zed.close()
        cv2.destroyAllWindows()
        close_checker()
        disconnect_all()


if __name__ == "__main__":
    main()
