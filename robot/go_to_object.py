"""Detect the objects with YOLO + ZED, then move the SO-101's gripper to one of them. Nothing else.

    1. detect      ZED + YOLO for a few frames -> each object's centre in the robot base frame (robot/detect.py)
    2. pick        the --target class (or the most confident object the arm can reach)
    3. plan        so101ik's inverse kinematics: a straight vertical descent from HOVER above the object down
                   to the grasp height, at the steepest gripper pitch that the arm can hold all the way down
                   (top-down where possible), every pose checked against the joint ranges, the table and
                   the arm's own parts
    4. move        unfold -> above the object -> jaws open -> straight down, stopping early if the tip touches
                   something -> hold -> straight up -> fold home

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
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")   # as the other scripts: quiets Qt

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect  # noqa: E402

# ---- change these --------------------------------------------------------------------------------------
HOVER = 0.07           # m above the grasp point the arm stops first, before going straight down
GRASP_DEPTH = 0.015    # m below the object's measured top surface the tip goes (the jaws straddle it)
MIN_TIP_Z = 0.012      # m: never aim the tip lower than this above the table (z = 0)
PRESS = 0.008          # m: on the way down, stop when the tip is held this far above where it should be
OPEN = 70.0            # gripper opening (0 closed .. 100 open) before going down
SPEED = 40.0           # deg/s peak, the moves in joint space (unfold, to the hover point)
DOWN_SPEED = 0.03      # m/s peak, the vertical descent and the rise
PITCHES = list(range(-90, -19, 5))   # gripper pitches tried, steepest first (-90 = pointing straight down)
# --------------------------------------------------------------------------------------------------------

# The follower's LeRobot calibration (there are several files in that folder). SO101_CALIBRATION or
# --calibration override it.
CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json"

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def grasp_point(d):
    """Where the tip goes: GRASP_DEPTH below the measured top (the mask's mean height), not below MIN_TIP_Z.
    The ZED reads dark objects too flat (see point_cloud_matching/README.md), so the floor matters."""
    x, y, z_top = d.base_xyz
    return np.array([x, y, max(z_top - GRASP_DEPTH, MIN_TIP_Z)])


def plan(model, goal, hover=HOVER, step=0.004):
    """(pitch, [Pose]) from goal + hover straight down to goal, one pose every `step` m, all at one pitch so the
    gripper keeps its angle on the way down. Steepest pitch first; None (the least-loaded pose at every point)
    last. Raises so101ik.Unreachable when nothing works."""
    from so101ik import Unreachable

    x, y, zg = goal
    zs = list(np.arange(zg + hover, zg, -step)) + [zg]
    for pitch in PITCHES + [None]:
        try:
            poses, prev = [], None
            for z in zs:
                if pitch is None:
                    prev = model.ik(x, y, z, near=prev) if prev else model.ik(x, y, z)
                else:
                    prev = model.ik(x, y, z, pitch=pitch)
                poses.append(prev)
        except Unreachable:
            continue
        # follow() wants neighbours within 3 degrees of each other: a jump means the IK flipped branch.
        jumps = [max(abs(a.deg[j] - b.deg[j]) for j in ARM) for a, b in zip(poses, poses[1:])]
        if max(jumps, default=0) < 2.5:
            return pitch, poses
    raise Unreachable(f"no pose reaches {100 * x:.1f} {100 * y:.1f} {100 * zg:.1f} cm from {100 * hover:.0f} cm above")


def arm_only(pose):
    return {j: pose.deg[j] for j in ARM}


def choose(dets, model, target=None, index=None):
    """The object to go to and its plan: --index, else the most confident --target (or any) the arm can reach."""
    from so101ik import Unreachable

    if index is not None:
        cands = [dets[index]]
    else:
        cands = [d for d in dets if d.name not in detect.SKIP_CLASSES and (target is None or d.name == target)]
    for d in cands:
        try:
            return d, grasp_point(d), *plan(model, grasp_point(d))
        except Unreachable as e:
            print(f"  skip {d.name} at {np.round(d.base_xyz, 3)}: {e}")
    return None


def settle_to(arm, pose):
    """From where the joints were last sent to `pose`, in joint steps under 2 degrees, with each joint led ahead by
    what it lags (gravity sag, gearbox friction), held until all stand within 0.5 degrees."""
    start = {j: arm.goal[j] for j in ARM}
    end = arm_only(pose)
    n = max(1, int(max(abs(end[j] - start[j]) for j in ARM) / 2.0) + 1)
    chain = [{j: start[j] + (end[j] - start[j]) * k / n for j in ARM} for k in range(n + 1)]
    arm.follow(chain, duration=max(0.6, 0.15 * n), lead=True, within=0.5, settle=1.0, step="settle")


def go(arm, goal, pitch, poses, hold, sim_floor=None):
    """Drive it: to the hover pose, jaws open, down (stopping on contact), hold, up. Leaving the caller's
    `with Arm()` block folds the arm home."""
    from so101ik import Fault

    arm.move_joints(arm_only(poses[0]), speed=SPEED)              # unfolds from the rest pose first
    settle_to(arm, poses[0])                                      # the joints onto it, not a few degrees short
    arm.move_joints({"gripper": OPEN}, speed=60, check=False)
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
    x, y, z, p = arm.where()
    err = 1000 * np.linalg.norm(np.array([x, y, z]) - goal)
    print(f"\nTIP at {x:+.3f} {y:+.3f} {z:+.3f} m, pitch {p:.0f}; goal {np.round(goal, 3)}; {err:.1f} mm off"
          + ("  (stopped on contact)" if touched else ""))
    if hold is None:
        input("At the object. Enter to go back up and fold home ... ")
    else:
        time.sleep(hold)
    if sim_floor is not None:
        arm.sim(floor=-1.0)            # the test floor away again, so the arm can fold onto the "table"
    if arm.pose is None:               # after a stall: the pose where the joints stand
        arm.pose = arm.model.pose(dict(arm.goal))
    arm.rise(float(goal[2] + HOVER), speed=DOWN_SPEED)
    return touched, err


def show(frame, dets, chosen):
    """The detections on the last camera frame; returns True on g/Enter (go), False on q/Esc."""
    import cv2

    img = frame.copy()
    for d in dets:
        if d.pixel is None:
            continue
        colour = (0, 0, 255) if d is chosen else (0, 255, 0)
        cv2.circle(img, tuple(map(int, d.pixel)), 10 if d is chosen else 5, colour, -1)
        x, y, z = d.base_xyz
        cv2.putText(img, f"{d.name} {x:+.3f} {y:+.3f} {z:+.3f}", (int(d.pixel[0]) + 12, int(d.pixel[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.putText(img, "g / Enter: go     q / Esc: quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.namedWindow("go_to_object", cv2.WINDOW_NORMAL)
    cv2.imshow("go_to_object", img)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("g"), 13, 10):
            cv2.destroyAllWindows()
            return True
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            return False


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
    ap.add_argument("--dry", action="store_true", help="detect and plan only; the arm does not move")
    ap.add_argument("--sim", action="store_true", help="drive so101ik's simulated servos instead of the arm")
    ap.add_argument("--sim-floor", type=float, default=None,
                    help="with --sim: a surface at this height (m) the tip stops on, to try the contact stop")
    ap.add_argument("--hold", type=float, default=None, help="seconds at the object; default wait for Enter")
    ap.add_argument("--yes", action="store_true", help="do not ask before moving")
    args = ap.parse_args()

    from so101ik import Arm, Model     # here, so a missing install shows before the camera opens

    transform = None
    if args.base_offset:
        transform = dict(zip(("x", "y", "z", "yaw_deg"), args.base_offset))

    # 1. detect -- the position estimate, once, before the arm moves (the arm would hide the objects).
    if args.case:
        dets, frame = detect.detect_from_case(args.case, transform)
    else:
        dets, frame = detect.detect_live(args.frames, args.weights, args.conf, args.serial, depth_mode=args.depth_mode,
                                         transform=transform)
    print(f"\n{len(dets)} objects (robot base frame):")
    for i, d in enumerate(dets):
        print(f"  {i:2d}. {d}")

    # 2-3. pick and plan, with the arm's own calibration when there is one (offline: the one compiled in).
    cal = Path(args.calibration).expanduser()
    if not (args.sim or args.dry) and not cal.is_file():
        raise SystemExit(f"Calibration file not found: {cal}; pass --calibration")
    model = Model(calibration_file=cal) if cal.is_file() else Model()
    pick = choose(dets, model, args.target, args.index)
    if pick is None:
        raise SystemExit("No reachable object" + (f" of class {args.target!r}" if args.target else "")
                         + ". The arm reaches about 15 to 40 cm from its base at table height.")
    d, goal, pitch, poses = pick
    print(f"\nGoing to {d.name}: tip to {np.round(goal, 3)} m from {100 * HOVER:.0f} cm above, "
          f"pitch {'least-loaded' if pitch is None else f'{pitch} deg'}, {len(poses)} poses down")
    print(f"  hover pose: {poses[0]}\n  goal pose:  {poses[-1]}")
    if args.dry:
        return

    # 4. move
    if not args.yes:
        if frame is not None:
            if not show(frame, dets, d):
                return
        elif input("Move the arm? [y/N] ").strip().lower() != "y":
            return
    print(f"Calibration: {cal}")
    with Arm(port="sim" if args.sim else None, calibration_file=cal if cal.is_file() else None,
             name="go_to_object") as arm:
        if args.sim:
            arm.raw.sag = 0.0      # the simulated servos have no gravity, so no sag to make up for
        pitch, poses = plan(arm.model, goal)      # again with exactly the arm's calibration and offsets
        go(arm, goal, pitch, poses, 0.5 if args.sim and args.hold is None else args.hold, args.sim_floor)
    print("Done: folded home, torque off.")


if __name__ == "__main__":
    main()
