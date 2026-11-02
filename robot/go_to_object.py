"""Detect the objects with YOLO + ZED, then move the SO-101's gripper to one of them. Nothing else.

    1. detect      ZED + YOLO for a few frames -> each object's centre in the robot base frame (robot/detect.py)
    2. pick        the --target class (or the most confident object the arm can reach)
    3. plan        so101ik's inverse kinematics: a straight vertical descent from HOVER above the object down
                   to the grasp height, at the steepest gripper pitch that the arm can hold all the way down
                   (top-down where possible), every pose checked against the joint ranges, the table and
                   the arm's own parts
    4. move        unfold -> above the object -> jaws open -> straight down, stopping early if the tip touches
                   something -> close the jaws on the object -> lift it 12 cm (stops if it slips) -> Enter -> put it
                   back where it was, or at the place clicked in the picture (or --place X Y) -> let go ->
                   straight up -> fold home.  --no-grasp: only go to it

In the picture window (YOLO's masks drawn on it): click an object to pick it, click the table to place it there,
click a bin to drop it in (the jaws' middle hovers over the bin's centre at DROP_HEIGHT, 15 cm, higher if the
object would not clear the rim, and the jaws open); g = go, c = clear the place, q = quit. --drop-in N: bin N.

The inverse kinematics and the servo driver are so101ik (github.com/LukeGrif/lerobot-so101-inverse-kinematics,
pip install below). It reads LeRobot's calibration from ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
and finds the port itself (or set SO101_PORT / SO101_CALIBRATION).

Usage:
    python robot/go_to_object.py                        # camera, then the real arm (asks before moving)
    python robot/go_to_object.py --target cube          # go to the most confident cube
    python robot/go_to_object.py --dry                  # camera + plan only, the arm does not move
    python robot/go_to_object.py --case point_cloud_matching/cases/case_... --sim   # no camera, no arm

Safety: start with the arm folded in its rest pose. Ctrl-C stops the move and folds the arm home; a second
Ctrl-C switches the torque off (hold the arm). A stall or overload stops the move by itself.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

for _fonts in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype", "/usr/share/fonts"):
    if os.path.isdir(_fonts) and not os.path.isdir(os.environ.get("QT_QPA_FONTDIR", "")):
        os.environ["QT_QPA_FONTDIR"] = _fonts      # OpenCV's Qt has no fonts of its own: stops its warnings

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect  # noqa: E402
import gripper  # noqa: E402

# ---- change these --------------------------------------------------------------------------------------
HOVER = 0.12           # m above the grasp point the arm stops first and opens the jaws (an opening jaw swings
                       # several cm down when the gripper is tilted), before going straight down
GRASP_DEPTH = 0.015    # m below the object's top the middle of the jaws goes
MIN_GRASP_Z = 0.015    # m: the middle of the jaws never lower than this above the table
TIP_MIN = 0.012        # m: the fixed finger's tip never lower than this above the table: the object sits less deep
                       # in the jaws when the gripper is steep (near the base), so the tip does not hit the table
PRESS = 0.008          # m: on the way down, stop when the tip is held this far above where it should be
JAW_DEPTH = 0.025      # m: the object sits this far in from the finger ends when the gripper is not steep
JAW_MARGIN = 0.008     # m of room each side between the object and the jaws as they come down
OPEN_MIN, OPEN_MAX = 15.0, 60.0   # % the jaws open to: just wider than the object across the jaws, within this
DEFAULT_WIDTH = {"cube": 0.057, "cylinder": 0.040, "dodecahedron": 0.055, "hexagonal": 0.047}  # m, without a fit
GRIP_RATE = 40.0       # percent/s the jaws close at, until they stop on the object
GRIP_SQUEEZE = 2.5     # percent the jaws are held past where they stopped: a firm hold, not a stall
GRIP_EMPTY = 4.0       # jaws closing to under this percent: nothing in them
GRIP_SLIP = 6.0        # percent the jaws may move while carrying before it counts as the object slipping
DROP_HEIGHT = 0.15     # m: over a bin, the jaws' middle hovers this high above the table, then the jaws open
DROP_CLEAR = 0.015     # m the carried object's bottom must clear the bin's rim by (else it hovers higher)
LIFT = 0.12            # m the object is lifted (straight back up the way the jaws came down)
ROLL = -90.0           # wrist roll, degrees, for every pose: the jaws open sideways (the camera is on one side)
# Where the object goes without a plan's own (check_calibration, defaults): so101ik's tip frame (the tip = the fixed
# jaw's grasp point; x towards the moving jaw is negative, z along the fingers): the middle of the jaws at 50 %.
JAW_CENTRE = np.array([-0.030, 0.0, -0.025])
OPEN = 50.0
SPEED = 20.0           # deg/s peak, the moves in joint space (unfold, to the hover point)
DOWN_SPEED = 0.03      # m/s peak, the vertical descent and the rise
PITCHES = list(range(-90, -19, 5))   # gripper pitches tried, steepest first (-90 = pointing straight down)
# --------------------------------------------------------------------------------------------------------

# The follower's LeRobot calibration (there are several files in that folder). SO101_CALIBRATION or
# --calibration override it.
CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json"

# Fine joint offsets from fine_joint_calibration_gui.py (degrees; true URDF angle = LeRobot angle + offset,
# motor target = wanted angle - offset). so101ik uses the same convention for its offsets.
FINE_OFFSETS = ROOT / "calibration" / "robots" / "so_follower" / "fine_joint_offsets.json"

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def fine_offsets(path):
    """The offsets by joint, read with fine_joint_offsets.py as the rest of the repo reads them (0 without the file)."""
    sys.path.insert(0, str(ROOT))
    from fine_joint_offsets import load_fine_joint_offsets

    return dict(load_fine_joint_offsets(path))


def grasp_point(d):
    """Where the middle of the jaws goes: the object's centre, GRASP_DEPTH below its top, at least MIN_GRASP_Z above
    the table. With a shape fit (robot/shape_fit.py) the centre and the top are the fitted STL's (true height,
    on the table under it, or z = 0 if that is higher); without one the camera's (dark objects read too flat)."""
    if d.fit is not None:
        table = max(d.fit.table_z, 0.0)
        x, y = d.fit.centre[:2]
        top = table + d.fit.height
    else:
        table = 0.0
        x, y, top = d.base_xyz
    return np.array([x, y, max(top - GRASP_DEPTH, table + MIN_GRASP_Z)])


def width_across(d):
    """How wide the object is across the jaws (m): the jaws close across the direction the arm reaches in (roll
    -90), so across the line from the base to the object. From the fitted outline, else the class's largest."""
    x, y = (d.fit.centre if d.fit is not None else d.base_xyz)[:2]
    across = np.array([-y, x]) / max(np.hypot(x, y), 1e-6)
    if d.fit is not None:
        return d.fit.width_across(across)
    return DEFAULT_WIDTH.get(d.name, 0.055)


class Poses(list):
    """A planned chain of so101ik Poses, with the gripper opening (%) and the jaw centre (tip frame) it uses."""
    grip = OPEN
    centre = JAW_CENTRE


def centre_of(model, pose, centre=None):
    """Where the middle of the open jaws is (m, base frame) for a pose (centre: tip frame, default JAW_CENTRE)."""
    T = model.frames(pose.deg)["gripper_frame_link"]
    return T[:3, :3] @ (JAW_CENTRE if centre is None else centre) + T[:3, 3]


def ik_centre(model, c, pitch, near=None, grip=OPEN, centre=None):
    """The pose (roll ROLL, gripper grip) that puts the middle of the jaws (centre, tip frame) at c: the tip aimed at
    c minus the offset, three rounds since the offset turns with the pose. pitch None: the least-loaded pose, or
    with near the next one along a path (its roll and gripper kept)."""
    c = np.asarray(c, float)
    tip = c.copy()
    for _ in range(3):
        if pitch is not None:
            pose = model.ik(*tip, pitch=pitch, roll=ROLL, gripper=grip)
        elif near is not None:
            pose = model.ik(*tip, near=near)
        else:
            pose = model.ik(*tip, roll=ROLL, gripper=grip)
        tip = tip + (c - centre_of(model, pose, centre))
    return pose


def depth_for(z, pitch):
    """How far in from the finger ends the object can sit (m) with the jaws' middle at height z and the gripper at
    pitch, so the fixed finger's tip stays TIP_MIN above the table: JAW_DEPTH, less when the gripper is steep."""
    s = np.sin(np.radians(abs(pitch if pitch is not None else 45.0)))
    return float(np.clip((z - TIP_MIN) / max(s, 0.2), 0.005, JAW_DEPTH))


def plan(model, goal, hover=HOVER, step=0.004, width=None, like=None):
    """(pitch, Poses) from goal + hover straight down to goal, one pose every `step` m, all at one pitch so the
    gripper keeps its angle on the way down; the middle of the jaws on the goal. Steepest pitch first; None (the
    least-loaded pose at every point) last. The opening and the jaw centre: like's (a carried object stays where
    it is in the jaws), else from the object's width across the jaws (just wider, at the depth this pitch allows).
    Raises so101ik.Unreachable when nothing works."""
    from so101ik import Unreachable

    x, y, zg = goal
    zs = list(np.arange(zg + hover, zg, -step)) + [zg]
    for pitch in PITCHES + [None]:
        if like is not None:
            grip, centre = like.grip, like.centre
        else:
            depth = depth_for(zg, pitch)
            grip, _ = gripper.opening_for(width if width is not None else 0.055, depth, JAW_MARGIN, OPEN_MIN, OPEN_MAX)
            centre = gripper.centre(grip, depth)
        try:
            poses, prev = Poses(), None
            for z in zs:
                prev = ik_centre(model, (x, y, z), pitch, near=prev if pitch is None else None, grip=grip, centre=centre)
                poses.append(prev)
        except Unreachable:
            continue
        if np.linalg.norm(centre_of(model, poses[-1], centre) - goal) > 0.002:
            continue
        if poses[-1].tip[2] < TIP_MIN - 0.002:
            continue
        # follow() wants neighbours within 3 degrees of each other: a jump means the IK flipped branch.
        jumps = [max(abs(a.deg[j] - b.deg[j]) for j in ARM) for a, b in zip(poses, poses[1:])]
        if max(jumps, default=0) < 2.5:
            poses.grip, poses.centre = grip, centre
            return pitch, poses
    raise Unreachable(f"no pose reaches {100 * x:.1f} {100 * y:.1f} {100 * zg:.1f} cm from {100 * hover:.0f} cm above")


def arm_only(pose):
    return {j: pose.deg[j] for j in ARM}


def choose(dets, model, target=None, index=None):
    """Mark every detection (d.status: "skipped class", "not --target", "out of reach" or "reachable") and return
    the one to go to with its plan: --index, else the most confident reachable --target (or any object)."""
    from so101ik import Unreachable

    pick = None
    for i, d in enumerate(dets):
        d.plan = None
        if index is not None and i != index:
            d.status = "not --index"
        elif d.name == "bin":
            d.status = "bin: click to drop in"
        elif index is None and d.name in detect.SKIP_CLASSES:
            d.status = "skipped class"
        elif index is None and target is not None and d.name != target:
            d.status = "not --target"
        else:
            d.plan = None
            plan_pick(d, model)
        if pick is None and d.plan is not None:
            pick = d
    return pick


def plan_pick(d, model):
    """d.plan for one detection (also one choose() did not plan): (goal, pitch, poses), or None if out of reach."""
    from so101ik import Unreachable

    if d.plan is None:
        d.width = width_across(d)
        try:
            d.plan = (grasp_point(d), *plan(model, grasp_point(d), width=d.width))
            d.status = "reachable"
        except Unreachable:
            d.status = "out of reach"
    return d.plan


@dataclass
class Place:
    """Where the carried object goes. kind "table": lowered until it touches the table at (x, y); "bin": the jaws'
    middle hovers at goal over the bin's centre and the jaws open (a drop)."""
    kind: str
    goal: np.ndarray
    pitch: float
    poses: list
    bin: object = None          # the bin's Detection
    note: str = ""

    def __str__(self):
        x, y, z = self.goal
        what = f"drop into the bin at x {x:+.3f} y {y:+.3f}, jaws' middle {100 * z:.1f} cm up" if self.kind == "bin" \
            else f"put down on the table at x {x:+.3f} y {y:+.3f}"
        return what + (f" ({self.note})" if self.note else "")


def plan_place(model, xy, grasp_z, like=None):
    """Put down on the table: the jaws' middle at (x, y) and the height it was picked at, so the object's bottom
    comes back to the table. A Place (from HOVER above), or None if out of reach."""
    from so101ik import Unreachable

    goal = np.array([xy[0], xy[1], grasp_z])
    try:
        pitch, poses = plan(model, goal, like=like)
    except Unreachable:
        return None
    return Place("table", goal, pitch, poses)


def plan_drop(model, b, grasp_z, like=None):
    """Drop into bin b: the jaws' middle over its centre at DROP_HEIGHT, or higher if the carried object's bottom
    (grasp_z below the jaws' middle) would not clear the bin's rim (its measured top) by DROP_CLEAR."""
    from so101ik import Unreachable

    x, y, rim = b.base_xyz
    need = rim + grasp_z + DROP_CLEAR
    z = max(DROP_HEIGHT, need)
    note = f"raised from {100 * DROP_HEIGHT:.0f} cm to clear the rim at {100 * rim:.1f} cm" if need > DROP_HEIGHT else ""
    try:
        pitch, poses = plan(model, np.array([x, y, z]), hover=0.0, like=like)
    except Unreachable:
        return None
    return Place("bin", np.array([x, y, z]), pitch, poses, bin=b, note=note)


def replan(model, place, grasp_z, like=None):
    """The same Place again with another model (the arm's own) or another object (its grasp height and its plan,
    like, for the opening and where it sits in the jaws)."""
    if place is None:
        return None
    like = like if like is not None else place.poses
    if place.kind == "bin":
        return plan_drop(model, place.bin, grasp_z, like)
    return plan_place(model, place.goal[:2], grasp_z, like)


def carry_chain(arm, target, target_centre=None):
    """Joint-space steps (under 2 degrees) from where the arm was last sent to `target`, every one checked: no
    problem (ranges, table, the arm itself) and the jaws' middle at least 5 cm up. None if a step fails."""
    start = {j: arm.goal[j] for j in ARM}
    end = arm_only(target)
    n = max(1, int(max(abs(end[j] - start[j]) for j in ARM) / 2.0) + 1)
    steps = [{j: start[j] + (end[j] - start[j]) * k / n for j in ARM} for k in range(n + 1)]
    for q in steps:
        pose = arm.model.pose(dict(q, gripper=arm.goal["gripper"]))
        if arm.model.problems(pose) or centre_of(arm.model, pose, target_centre)[2] < 0.05:
            return None
    return steps


def settle_to(arm, pose):
    """From where the joints were last sent to `pose`, in joint steps under 2 degrees, with each joint led ahead by
    what it lags (gravity sag, gearbox friction), held until all stand within 0.5 degrees."""
    start = {j: arm.goal[j] for j in ARM}
    end = arm_only(pose)
    n = max(1, int(max(abs(end[j] - start[j]) for j in ARM) / 2.0) + 1)
    chain = [{j: start[j] + (end[j] - start[j]) * k / n for j in ARM} for k in range(n + 1)]
    arm.follow(chain, duration=max(0.6, 0.15 * n), lead=True, within=0.5, settle=1.0, step="settle")


def go(arm, goal, pitch, poses, hold, sim_floor=None, grasp=True, place=None):
    """Drive it: to the hover pose, jaws open, down (stopping on contact), close on the object, lift it, then
    put it down at `place` (a Place) or, without one, back where it was; let go, up. Leaving the caller's
    `with Arm()` block folds the arm home. Returns "placed", "empty" (the jaws closed on nothing), "slipped",
    "touched" (stopped on something on the way down) or "no grasp"."""
    from so101ik import Fault, GripLost

    def measured():
        deg = arm.read()[0]
        pose = arm.model.pose(deg)
        return pose.tip, centre_of(arm.model, pose, poses.centre)

    c0 = centre_of(arm.model, poses[0], poses.centre)
    print(f"\n>> to the hover point: jaws' middle {100 * c0[2]:.1f} cm above the table (tip {100 * poses[0].tip[2]:.1f}),"
          f" pitch {poses[0].pitch:.0f}, roll {ROLL:.0f}")
    arm.move_joints(arm_only(poses[0]), speed=SPEED)              # unfolds from the rest pose first
    settle_to(arm, poses[0])                                      # the joints onto it, not a few degrees short
    tip, c = measured()
    print(f">> at the hover point (from the servos: jaws' middle {100 * c[2]:.1f} cm, tip {100 * tip[2]:.1f} cm up);"
          f" opening the jaws to {poses.grip:.0f} % ({100 * gripper.gap(poses.grip, -poses.centre[2]):.1f} cm), "
          f"the object {100 * -poses.centre[2]:.1f} cm in from the finger ends")
    arm.move_joints({"gripper": poses.grip}, speed=60, check=False)
    print(f">> going down: jaws' middle to {100 * goal[2]:.1f} cm")
    if sim_floor is not None:
        arm.sim(floor=sim_floor)
    try:
        touched = arm.follow([arm_only(p) for p in poses], speed=DOWN_SPEED, press=PRESS, lead=True, step="down")
    except Fault as e:                 # a joint stalled before the press saw it: something is in the way
        print(f"  stopped on the way down: {e}")
        touched = True
        arm.pose = None
    if not touched:                    # the press ends with the goals where the joints lag: close that last bit
        settle_to(arm, poses[-1])      # (no press: it would end where the joints lag again)
    tip, c = measured()
    err = 1000 * np.linalg.norm(c - goal)
    print(f"\nJAWS' MIDDLE at {c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f} m (tip {100 * tip[2]:.1f} cm up); object point "
          f"{np.round(goal, 3)}; {err:.1f} mm off" + ("  (stopped on contact)" if touched else ""))
    def wait(text):
        if hold is None:
            input(text)
        else:
            time.sleep(hold)

    def up_and_out(z):
        if sim_floor is not None:
            arm.sim(floor=-1.0)        # the test floor away again, so the arm can fold onto the "table"
        if arm.pose is None:           # after a stop: the pose where the joints stand
            arm.pose = arm.model.pose(dict(arm.goal))
        arm.rise(float(z), speed=DOWN_SPEED)

    def chain(seq):                    # from where the joints were last sent, then the planned poses
        return [{j: arm.goal[j] for j in ARM}] + [arm_only(p) for p in seq]

    if touched or not grasp:
        print(">> stopped on something before the grasp point: the jaws stay open" if touched else ">> --no-grasp")
        wait("Enter to go back up and fold home ... ")
        up_and_out(goal[2] + HOVER)
        return "touched" if touched else "no grasp"

    # close on it
    if arm_is_sim(arm):
        arm.sim(block={"gripper": (max(2.0, poses.grip - SIM_OBJECT), None)})   # a simulated object stops the jaws
    at = arm.grasp(rate=GRIP_RATE, squeeze=GRIP_SQUEEZE, empty=GRIP_EMPTY)
    if at < GRIP_EMPTY:
        print(f">> the jaws closed on nothing (stopped at {at:.0f} %): opening, back up and home")
        arm.move_joints({"gripper": poses.grip}, speed=60, check=False)
        up_and_out(goal[2] + HOVER)
        return "empty"
    guard = (max(0.0, at - GRIP_SLIP), min(100.0, at + GRIP_SLIP))
    print(f">> holding it: the jaws stopped at {at:.0f} % (it counts as slipping outside {guard[0]:.0f}-{guard[1]:.0f} %)")

    # lift it straight back up the way the jaws came down, watching the grip
    lift = [p for p in reversed(poses) if centre_of(arm.model, p, poses.centre)[2] <= goal[2] + LIFT + 1e-6]
    try:
        arm.follow(chain(lift), speed=DOWN_SPEED, lead=True, grip_range=guard, step="lift")
        print(f">> lifted {100 * LIFT:.0f} cm")
        if place is None:
            wait("Holding it up. Enter to put it back down where it was ... ")
            arm.follow(chain(reversed(lift)), speed=DOWN_SPEED, lead=True, press=PRESS, grip_range=guard,
                       step="put down")
        else:
            print(f">> carrying it: {place}")
            steps = carry_chain(arm, place.poses[0], place.poses.centre)
            if steps is not None:
                big = max(abs(steps[-1][j] - steps[0][j]) for j in ARM)
                arm.follow(steps, duration=max(1.5, 1.875 * big / SPEED), lead=True, grip_range=guard, step="carry")
            else:                      # the straight joint path is not clear: the checked move, without the guard
                arm.move_joints(arm_only(place.poses[0]), speed=SPEED)
            arm.pose = place.poses[0]
            settle_to(arm, place.poses[0])
            if place.kind == "table":
                arm.follow(chain(place.poses), speed=DOWN_SPEED, lead=True, press=PRESS, grip_range=guard,
                           step="place down")
            tip, c = measured()
            print(f">> {'over the bin' if place.kind == 'bin' else 'put down'}: jaws' middle at "
                  f"{c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f} m, {1000 * np.linalg.norm(c - place.goal):.1f} mm from the place")
            goal = place.goal - [0, 0, HOVER - 0.05] if place.kind == "bin" else place.goal
        status = "placed"
    except GripLost as e:
        print(f">> it slipped ({e}): opening the jaws where the arm stands")
        arm.pose = arm.model.pose(dict(arm.goal))
        status = "slipped"
    if arm_is_sim(arm):
        arm.sim(block={})
    arm.move_joints({"gripper": poses.grip}, speed=60, check=False)    # let go
    print(">> let go: back up and home")
    up_and_out(goal[2] + HOVER)
    return status


def arm_is_sim(arm):
    return getattr(arm, "_sim", False)


SIM_OBJECT = 12.0      # with --sim: the simulated object stops the closing jaws this many % below the opening


def reach_outline(scene, model, z=0.04, step=12):
    """The outline (OpenCV contours, image pixels) of the table area the jaws can reach at height z, wrist roll
    ROLL: every step-th pixel's table point (point cloud -> base frame) tried with the IK. Roughly where
    objects can be picked and bins reached; the exact answer is each object's own plan."""
    import cv2
    from so101ik import Unreachable

    h, w = scene.cloud.shape[:2]
    vs, us = np.arange(step // 2, h, step), np.arange(step // 2, w, step)
    pts = np.asarray(scene.cloud[np.ix_(vs, us)], float)[..., :3]
    ok = np.isfinite(pts).all(axis=-1)
    T = scene.camera_to_table
    base = detect.table_to_base_many(pts @ T[:3, :3].T + T[:3, 3], scene.transform)
    reach = np.zeros(ok.shape, np.uint8)
    for i, j in zip(*np.nonzero(ok)):
        x, y, _ = base[i, j]
        if not 0.08 < np.hypot(x, y) < 0.6:
            continue
        try:
            model.ik(x, y, z, roll=ROLL)
            reach[i, j] = 255
        except Unreachable:
            pass
    big = cv2.resize(reach, (len(us) * step, len(vs) * step), interpolation=cv2.INTER_LINEAR)[:h, :w]
    big = cv2.GaussianBlur(big, (0, 0), step)
    big = cv2.morphologyEx((big > 127).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3 * step, 3 * step), np.uint8))
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 50 * step * step]


def draw_reach(img, contours):
    """The reach outline: a white line with a dark edge and its name."""
    import cv2

    cv2.drawContours(img, contours, -1, (0, 0, 0), 6)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 2)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y = c[c[:, 0, 1].argmin(), 0]
        cv2.putText(img, "arm reach", (int(x) + 8, int(y) + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(img, "arm reach", (int(x) + 8, int(y) + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def show(scene, dets, chosen, model, save_dir, place=None, wait=True):
    """The camera picture with every detection (list number, class, confidence, base-frame position; the one to
    pick in red with a ring, other reachable ones green, out of reach orange, skipped grey) and the place to put
    it (a cross). Click an object to pick it; click the table to put it there. Saved in save_dir.
    Click a bin to drop it in. Returns (chosen, place) on g/Enter, None on q/Esc; place is a Place, or None (put
    it back where it was)."""
    import cv2

    state = {"chosen": chosen, "place": place, "place_px": None, "place_txt": "", "msg": ""}
    reach = reach_outline(scene, model)
    colours = {"reachable": (0, 200, 0), "out of reach": (0, 140, 255), "bin: click to drop in": (255, 160, 0)}

    def colour(d):
        return (0, 0, 255) if d is state["chosen"] else colours.get(d.status, (160, 160, 160))

    def label(img, text, org, c, scale=0.6):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4 if scale < 1 else 6)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, c, 2)

    def draw():
        img = scene.frame.copy()
        ch, pl = state["chosen"], state["place"]
        shade = img.copy()             # YOLO's predictions: each mask shaded, outlined
        for d in dets:
            if d.polygon is not None and len(d.polygon) >= 3:
                cv2.fillPoly(shade, [np.rint(d.polygon).astype(np.int32)], colour(d))
        img = cv2.addWeighted(shade, 0.30, img, 0.70, 0)
        draw_reach(img, reach)
        for d in dets:
            if d.polygon is not None and len(d.polygon) >= 3:
                thick = 4 if (d is ch or (pl is not None and pl.bin is d)) else 2
                cv2.polylines(img, [np.rint(d.polygon).astype(np.int32)], True, colour(d), thick)
        for i, d in enumerate(dets):
            if d.pixel is None:
                continue
            c = colour(d)
            px = tuple(map(int, d.pixel))
            cv2.circle(img, px, 7, c, -1)
            if d is ch:
                cv2.circle(img, px, 28, c, 4)
            x, y, z = d.base_xyz
            size = f"{d.fit.size} " if d.fit is not None else ""
            label(img, f"{i}. {size}{d.name} {d.conf:.2f}" + ("  <- PICK" if d is ch else f"  ({d.status})"),
                  (px[0] + 14, px[1] - 6), c)
            label(img, f"x {x:+.3f} y {y:+.3f} z {z:+.3f}", (px[0] + 14, px[1] + 18), c)
        if state["place_px"] is not None and pl is not None:
            u, v = state["place_px"]
            c = (255, 0, 255)
            cv2.drawMarker(img, (u, v), c, cv2.MARKER_TILTED_CROSS, 34, 4)
            if ch is not None and ch.pixel is not None:
                cv2.arrowedLine(img, tuple(map(int, ch.pixel)), (u, v), c, 3, tipLength=0.04)
            label(img, ("DROP IN BIN " if pl.kind == "bin" else "PLACE ") + state["place_txt"],
                  (u + 20, v + (56 if pl.kind == "bin" else 8)), c)
        if ch is None:
            head = "no object to pick: click a reachable one, q = quit"
        else:
            head = (f"pick {dets.index(ch)}. {ch.name} -> " + ("put back where it is" if pl is None else
                    f"DROP into bin {dets.index(pl.bin)}" if pl.kind == "bin" else "PLACE at the cross"))
        label(img, head, (20, 45), (255, 255, 255), 1.1)
        label(img, "click an object = pick it   click the table = place it there   click a bin = drop it in   "
                   "g / Enter = go   c = clear the place   q / Esc = quit", (20, 85), (255, 255, 255), 0.75)
        if state["msg"]:
            label(img, state["msg"], (20, 120), (0, 200, 255), 0.75)
        return img

    def inside(d, u, v):
        if d.polygon is not None and len(d.polygon) >= 3:
            if cv2.pointPolygonTest(np.asarray(d.polygon, np.float32), (float(u), float(v)), False) >= 0:
                return True
        return d.pixel is not None and np.hypot(d.pixel[0] - u, d.pixel[1] - v) < 45

    def set_place(new, px, txt, fail):
        if new is None:                # keep the place there was
            state["msg"] = fail
        else:
            state.update(place=new, place_px=px, place_txt=txt)

    def click(event, u, v, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state["msg"] = ""
        hit = [d for d in dets if inside(d, u, v)]
        objects = [d for d in hit if d.name not in detect.SKIP_CLASSES]
        bins = [d for d in hit if d.name == "bin"]
        if objects:                    # an object: pick it
            d = min(objects, key=lambda d: np.hypot(d.pixel[0] - u, d.pixel[1] - v) if d.pixel else 1e9)
            if plan_pick(d, model) is None:
                state["msg"] = f"{d.name} is out of reach"
            else:
                state["chosen"] = d
                if state["place"] is not None:          # the same place, at this object's grasp height
                    state["place"] = replan(model, state["place"], d.plan[0][2], like=d.plan[2])
                    if state["place"] is None:
                        state.update(place_px=None, msg="the place is out of reach for this object: click again")
        elif state["chosen"] is None:
            state["msg"] = "click an object to pick first"
        elif bins:                     # a bin: drop it in
            b = bins[0]
            x, y, _ = b.base_xyz
            set_place(plan_drop(model, b, state["chosen"].plan[0][2], state["chosen"].plan[2]), tuple(map(int, b.pixel)),
                      f"x {x:+.3f} y {y:+.3f}", f"the arm cannot reach over bin {dets.index(b)}")
        elif hit:
            state["msg"] = f"that is the {hit[0].name}: click an object, a bin or the table"
        else:                          # the table: put it down there
            xyz = scene.pixel_to_base(u, v)
            if xyz is None:
                state["msg"] = "no depth there: click again"
            elif xyz[2] > 0.025:
                state["msg"] = f"that is not the table ({100 * xyz[2]:.1f} cm up): click an empty spot"
            else:
                set_place(plan_place(model, xyz[:2], state["chosen"].plan[0][2], state["chosen"].plan[2]), (u, v),
                          f"x {xyz[0]:+.3f} y {xyz[1]:+.3f}",
                          f"the arm cannot put it down at x {xyz[0]:+.3f} y {xyz[1]:+.3f}: click closer")
        print(f"  {state['msg']}" if state["msg"] else f"  pick: {state['chosen'].name if state['chosen'] else '-'}"
              + (f", then {state['place']}" if state["place"] is not None else ""))
        cv2.imshow(win, draw())

    win = "go_to_object"
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(save_dir / f"detections_{stamp}.jpg"), draw())
    print(f"Picture: {save_dir / f'detections_{stamp}.jpg'}")
    if not wait:
        return (state["chosen"], state["place"]) if state["chosen"] is not None else None
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, draw())
        cv2.setMouseCallback(win, click)
    except cv2.error:                  # no display: ask in the terminal instead
        ok = state["chosen"] is not None and input("Move the arm? [y/N] ").strip().lower() == "y"
        return (state["chosen"], state["place"]) if ok else None
    print("Picture window: click an object to pick it, click the table to place it there, click a bin to drop it "
          "in; g or Enter = go, c = clear the place, q or Esc = quit")
    try:
        while True:
            k = cv2.waitKey(100)       # short waits, so Ctrl-C gets through
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if k == -1:
                continue
            k &= 0xFF
            if k in (ord("q"), 27):
                return None
            if k == ord("c"):
                state.update(place=None, place_px=None, msg="")
                cv2.imshow(win, draw())
            if k in (ord("g"), 13, 10) and state["chosen"] is not None:
                cv2.imwrite(str(save_dir / f"detections_{stamp}.jpg"), draw())
                return state["chosen"], state["place"]
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="class name to go to (cube, cylinder, dodecahedron, hexagonal, ...)")
    ap.add_argument("--index", type=int, help="go to this entry of the printed list instead")
    ap.add_argument("--case", help="use a saved point-cloud case instead of the camera")
    ap.add_argument("--frames", type=int, default=15, help="camera frames averaged for the positions")
    ap.add_argument("--weights", help="YOLO weights; default the newest runs/segment/*/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--serial", type=int)
    ap.add_argument("--depth-mode", default="NEURAL", choices=["QUALITY", "ULTRA", "NEURAL"])
    ap.add_argument("--base-offset", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW"),
                    help="TABLE -> robot base frame: m, m, m, degrees (default detect.TABLE_TO_BASE)")
    ap.add_argument("--calibration", default=os.environ.get("SO101_CALIBRATION") or str(CALIBRATION),
                    help=f"the arm's LeRobot calibration file (default {CALIBRATION.name})")
    ap.add_argument("--offsets", default=str(FINE_OFFSETS),
                    help="fine joint offsets JSON (fine_joint_calibration_gui.py); 'none' for no offsets")
    ap.add_argument("--dry", action="store_true", help="detect and plan only; the arm does not move")
    ap.add_argument("--sim", action="store_true", help="drive so101ik's simulated servos instead of the arm")
    ap.add_argument("--sim-floor", type=float, default=None,
                    help="with --sim: a surface at this height (m) the tip stops on, to try the contact stop")
    ap.add_argument("--place", type=float, nargs=2, metavar=("X", "Y"),
                    help="put the object down here (robot base frame, m) instead of back where it was; in the "
                         "picture window, clicking the table does the same")
    ap.add_argument("--drop-in", type=int, metavar="N",
                    help="drop the object into bin N of the printed list (clicking a bin does the same)")
    ap.add_argument("--no-grasp", action="store_true", help="only go to the object; do not close the jaws")
    ap.add_argument("--hold", type=float, default=None, help="seconds at the object; default wait for Enter")
    ap.add_argument("--yes", action="store_true", help="do not ask before moving")
    args = ap.parse_args()

    from so101ik import Arm, Model     # here, so a missing install shows before the camera opens

    transform = None
    if args.base_offset:
        transform = dict(zip(("x", "y", "z", "yaw_deg"), args.base_offset))

    # 1. detect -- the position estimate, once, before the arm moves (the arm would hide the objects).
    if args.case:
        dets, scene = detect.detect_from_case(args.case, transform)
    else:
        dets, scene = detect.detect_live(args.frames, args.weights, args.conf, args.serial, depth_mode=args.depth_mode,
                                         transform=transform)
    print(f"\n{len(dets)} objects (robot base frame):")
    for i, d in enumerate(dets):
        print(f"  {i:2d}. {d}")
    frame = scene.frame if scene is not None else None

    # 2-3. pick and plan, with the arm's own calibration when there is one (offline: the one compiled in).
    cal = Path(args.calibration).expanduser()
    if not (args.sim or args.dry) and not cal.is_file():
        raise SystemExit(f"Calibration file not found: {cal}; pass --calibration")
    offsets = {j: 0.0 for j in ARM} if args.offsets == "none" else fine_offsets(args.offsets)
    print("Fine joint offsets (deg): " + ", ".join(f"{j} {v:+.2f}" for j, v in offsets.items())
          + ("" if args.offsets == "none" or Path(args.offsets).is_file() else f"  ({args.offsets} not found)"))
    model = Model(calibration_file=cal if cal.is_file() else None, offsets=offsets)
    d = choose(dets, model, args.target, args.index)
    print()
    for i, x in enumerate(dets):
        print(f"  {i:2d}. {x.name:<14} {x.status}")
    place = None
    if args.place and d is not None:
        place = plan_place(model, args.place, d.plan[0][2], d.plan[2])
        if place is None:
            raise SystemExit(f"The arm cannot put it down at {args.place}")
    if args.drop_in is not None and d is not None:
        b = dets[args.drop_in]
        if b.name != "bin":
            raise SystemExit(f"--drop-in {args.drop_in} is a {b.name}, not a bin")
        place = plan_drop(model, b, d.plan[0][2], d.plan[2])
        if place is None:
            raise SystemExit(f"The arm cannot reach over bin {args.drop_in}")
    go_ok = True
    if frame is not None:
        picked = show(scene, dets, d, model, ROOT / "outputs" / "go_to_object", place=place,
                      wait=not (args.yes or args.dry))
        go_ok = picked is not None
        if picked is not None:
            d, place = picked
    if d is None:
        raise SystemExit("No reachable object" + (f" of class {args.target!r}" if args.target else "")
                         + ". The arm reaches about 15 to 45 cm from its base at table height.")
    goal, pitch, poses = d.plan
    print(f"\nGoing to {d.name}: jaws' middle to {np.round(goal, 3)} m from {100 * HOVER:.0f} cm above, "
          f"pitch {'least-loaded' if pitch is None else f'{pitch} deg'}, {len(poses)} poses down")
    print(f"  hover pose: {poses[0]}\n  goal pose:  {poses[-1]}")
    if place is not None:
        print(f"Then: {place}, pitch {'least-loaded' if place.pitch is None else f'{place.pitch} deg'}")
    if args.dry:
        return

    # 4. move
    if not args.yes:
        if frame is not None:
            if not go_ok:
                return
        elif input("Move the arm? [y/N] ").strip().lower() != "y":
            return
    print(f"Calibration: {cal}")
    with Arm(port="sim" if args.sim else None, calibration_file=cal if cal.is_file() else None, offsets=offsets,
             name="go_to_object") as arm:
        arm._sim = args.sim
        if args.sim:
            arm.raw.sag = 0.0      # the simulated servos have no gravity, so no sag to make up for
        pitch, poses = plan(arm.model, goal, width=d.width)    # again with exactly the arm's calibration and offsets
        if place is not None:
            place = replan(arm.model, place, goal[2], like=poses)
        go(arm, goal, pitch, poses, 0.5 if args.sim and args.hold is None else args.hold, args.sim_floor,
           grasp=not args.no_grasp, place=place)
    print("Done: folded home, torque off.")


if __name__ == "__main__":
    main()
