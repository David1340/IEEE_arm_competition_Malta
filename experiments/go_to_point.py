"""Send an arm's grasp point to an x y z, with the SAME planner pick and place uses.

The point is the grasp point between the jaws (grasp_center_link, the red ball in object_3d_live.py), not the
gripper frame IK_control.py aims at. The pose comes from M-FRPSO with the gripper level (or top-down / free), the
path from RRT-Connect around the table, and the arm follows it smoothly (control/IK_control.py follow_path).

    python experiments/go_to_point.py --arm follower_pink 0.30 0.00 0.08          # arm's base frame [m]
    python experiments/go_to_point.py --arm follower_pink 0.30 0.00 0.08 --gripper 60
    python experiments/go_to_point.py --arm follower_blue --frame table 0.10 -0.05 0.08   # shared TABLE frame
    python experiments/go_to_point.py --arm follower_pink --interactive            # type points at a prompt
    python experiments/go_to_point.py --arm follower_pink --no-robot --gui 0.3 0 0.08   # plan and show only

Base frame: x forward from the arm, y to its left, z up from the table. TABLE frame: calibration/calibrate.py.

From your own code (the arm already connected with control.IK_control.connect_robot):

    from experiments.go_to_point import go_to
    reached, q = go_to(robot, "follower_pink", [0.30, 0.0, 0.08], gripper=60)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control import arm_config  # noqa: E402
from control.arm_config import HOME_DEG  # noqa: E402

HOME_Q = [math.radians(v) for v in HOME_DEG]


def plan_to(arm, xyz, start_q, *, posture="horizontal", tolerance=0.012, level_tolerance=15.0, clearance=0.005,
            seed=0, tries=3, gui=False):
    """Plan start_q -> a pose with the grasp point at xyz (arm's base frame, m). Returns (path, IK result).
    Raises RuntimeError when there is no pose or no collision-free path."""
    from experiments.sim_plan import World

    world = World([], start_q, gui=gui, clearance=clearance, arm=arm,
                  calibration_path=arm_config.follower_calibration(arm))
    world.level_tolerance_deg = level_tolerance
    try:
        q = list(start_q)
        q[4] = world.start_q[4]                     # wrist roll stays where it is (as in pick and place)
        world.start_q = q
        result = None
        for attempt in range(tries):
            result = world.find_pose(list(xyz), seed + attempt, tolerance=tolerance, posture=posture)
            if result is not None and result.success:
                break
        if result is None or not result.success:
            off = "" if result is None else f" (best {1000 * result.position_error_m:.0f} mm off)"
            raise RuntimeError(f"no {posture} pose for {np.round(xyz, 3)}{off}")
        route = world.find_path(result.joint_positions, seed)
        if route is None:
            raise RuntimeError(f"no collision-free path to {np.round(xyz, 3)}")
        if gui:
            world.play(route.path)
        return route.path, result
    finally:
        if not gui:
            world.close()


def go_to(robot, arm, xyz, gripper=None, *, frame="arm", **plan_options):
    """Move a connected arm's grasp point to xyz ("arm": its base frame, "table": the shared TABLE frame) [m].
    robot None: plan only. Returns (reached, final joint angles). plan_options: see plan_to."""
    from control.IK_control import follow_path, read_joints_rad

    xyz = np.asarray(xyz, float)
    if frame == "table":
        from calibration.calibrate import arm_from_table

        xyz = (arm_from_table(arm) @ np.append(xyz, 1.0))[:3]
    start = read_joints_rad(robot) if robot is not None else plan_options.pop("start_q", HOME_Q)
    path, result = plan_to(arm, xyz, start, **plan_options)
    print(f"{arm}: pose {1000 * result.position_error_m:.0f} mm from {np.round(xyz, 3)} (base frame), "
          f"path of {len(path)} points")
    if robot is None:
        return True, list(path[-1])
    reached = follow_path(robot, path, gripper)
    return reached, read_joints_rad(robot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xyz", type=float, nargs="*", help="x y z [m]")
    ap.add_argument("--arm", required=True, help="follower_pink or follower_blue (arms.json)")
    ap.add_argument("--frame", choices=["arm", "table"], default="arm", help="frame of x y z (default: the arm's)")
    ap.add_argument("--gripper", type=float, default=None, help="gripper 0-100 at the end (default: unchanged)")
    ap.add_argument("--posture", choices=["horizontal", "down", "free"], default="horizontal",
                    help="gripper level (default, as pick and place), pointing down, or any orientation")
    ap.add_argument("--level-tolerance", type=float, default=15.0, help="tilt allowed from the posture [deg]")
    ap.add_argument("--clearance", type=float, default=0.005, help="the gripper stays this far above the table [m]")
    ap.add_argument("--tolerance", type=float, default=0.012,
                    help="how close the planned grasp point must be to x y z [m] (default 0.012, as pick and place's "
                         "grasp; smaller is more exact but fails more often)")
    ap.add_argument("--interactive", action="store_true", help="type points at a prompt: x y z [gripper], q quits")
    ap.add_argument("--no-robot", action="store_true", help="plan only (no arm connected)")
    ap.add_argument("--gui", action="store_true", help="show the plan in PyBullet")
    ap.add_argument("--home", action="store_true", help="go to the home pose first")
    from control.IK_control import add_motion_args, apply_motion_args

    add_motion_args(ap)
    args = ap.parse_args()
    apply_motion_args(args)
    if not args.interactive and len(args.xyz) != 3:
        ap.error("give x y z, or --interactive")

    robot = None
    if not args.no_robot:
        from control.IK_control import connect_robot, go_home

        robot = connect_robot(args.arm)
        if args.home:
            go_home(robot)
    options = dict(frame=args.frame, posture=args.posture, level_tolerance=args.level_tolerance, tolerance=args.tolerance,
                   clearance=args.clearance, gui=args.gui)

    def run(xyz, gripper):
        try:
            reached, _ = go_to(robot, args.arm, xyz, gripper, **options)
            print("REACHED" if reached else "NOT REACHED (the arm stopped short; see the joints)")
        except RuntimeError as error:
            print(f"Not moving: {error}")

    try:
        if args.interactive:
            print(f"{args.arm}, {args.frame} frame. Type: x y z [gripper 0-100]   (q = quit)")
            while True:
                words = input("point> ").replace(",", " ").split()
                if not words:
                    continue
                if words[0] in ("q", "quit"):
                    break
                try:
                    values = [float(w) for w in words]
                    assert len(values) in (3, 4)
                except (ValueError, AssertionError):
                    print("x y z [gripper], in metres (e.g. 0.30 0 0.08 60)")
                    continue
                run(values[:3], values[3] if len(values) == 4 else None)
        else:
            run(args.xyz, args.gripper)
            if robot is not None:
                input("At the point. Enter = park and quit (--no-park leaves it there) ")
    finally:
        if robot is not None:
            from control.IK_control import park_and_disconnect

            park_and_disconnect(robot)


if __name__ == "__main__":
    main()
