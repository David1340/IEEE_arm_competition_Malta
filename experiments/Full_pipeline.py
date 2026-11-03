"""Camera -> M-FRPSO pose -> RRT-Connect path -> real arm. Simple version.

1. Vision (same code as object_3d_live.py): one YOLO/ZED frame, top-face centre of every object.
2. World (same code as sim_plan.py): table + detected objects + the arm at its measured pose.
3. M-FRPSO finds the pose that puts the point between the jaws on the chosen object.
4. RRT-Connect finds the path to that pose; it is played in PyBullet.
5. With --execute (and a typed 'yes') the arm follows the path waypoint by waypoint.

    python experiments/Full_pipeline.py                       # plan + PyBullet preview, arm does not move
    python experiments/Full_pipeline.py --no-robot            # no arm connected, start from sim_plan's START_Q
    python experiments/Full_pipeline.py --object 2 --height 0.04
    python experiments/Full_pipeline.py --execute
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sim_plan import HEIGHT, START_Q, TABLE_CLEARANCE, World  # noqa: E402
from vision.detection import IGNORED_OBJECT_CLASSES, OBJECT_STL_FILES  # noqa: E402
from vision.live import Vision, add_vision_args  # noqa: E402

GRIPPER_OPEN = 80.0      # LeRobot 0-100 scale
WAYPOINT_STEP_DEG = 4.0  # the motors lag on long moves, so send short steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_vision_args(ap)
    ap.add_argument("--object", type=int, default=1, help="1-based index in the printed object list")
    ap.add_argument("--height", type=float, default=HEIGHT, help="z of the point between the jaws [m]")
    ap.add_argument("--clearance", type=float, default=TABLE_CLEARANCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-robot", action="store_true", help="do not connect to the arm; start from START_Q")
    ap.add_argument("--no-gui", action="store_true", help="no PyBullet window")
    ap.add_argument("--execute", action="store_true", help="move the real arm after a typed confirmation")
    args = ap.parse_args()
    if args.execute and args.no_robot:
        raise SystemExit("--execute needs the arm: drop --no-robot")

    # 1. vision
    vision = Vision(args)
    try:
        canvas, objects = vision.detect(*vision.grab())
    finally:
        vision.close()
    cv2.imshow("YOLO + ZED (any key = continue, q/Esc = cancel)", canvas)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key in (ord("q"), 27):
        raise SystemExit("cancelled")
    objects = [o for o in objects
               if o[2] is not None and o[0].casefold() not in IGNORED_OBJECT_CLASSES
               and o[0].casefold() in OBJECT_STL_FILES]
    for i, (name, conf, center, _) in enumerate(objects, 1):
        print(f"  {i}. {name} {conf:.2f}  X={center[0]:+.3f} Y={center[1]:+.3f} Z={center[2]:+.3f} m")
    if not 1 <= args.object <= len(objects):
        raise SystemExit(f"--object {args.object}: {len(objects)} usable object(s) detected")
    target = np.array([*objects[args.object - 1][2][:2], args.height])

    # 2. world, starting from the real arm pose
    robot = None
    start_q = START_Q
    if not args.no_robot:
        from control.IK_control import connect_robot, read_joints_rad, send_joints

        robot = connect_robot()
        start_q = read_joints_rad(robot)
    try:
        world = World([(OBJECT_STL_FILES[name.casefold()], c[0], c[1], 0.0) for name, _, c, _ in objects],
                      start_q, gui=not args.no_gui, clearance=args.clearance)
        print(f"start joints [deg]: {np.round(np.degrees(start_q), 1)}")

        # 3. M-FRPSO
        result = world.find_pose(target, args.seed)
        if result is None:
            raise SystemExit("M-FRPSO found no collision-free pose (try --seed, --height or another --object)")
        goal_q = result.joint_positions
        world.show(goal_q)
        print(f"M-FRPSO: goal [deg] {np.round(np.degrees(goal_q), 1)}, error {result.position_error_m * 1000:.1f} mm, "
              f"jaw point {np.round(world.tcp(), 3)} m (target {np.round(target, 3)})")

        # 4. RRT-Connect
        route = world.find_path(goal_q, args.seed)
        if route is None:
            raise SystemExit("RRT-Connect found no collision-free path")
        print(f"RRT-Connect: {len(route.path)} configurations")
        world.play(route.path, realtime=not args.no_gui)

        # 5. execute
        if not args.execute:
            print("Plan only (add --execute to move the arm).")
            return
        if input("Type 'yes' to move the real arm along this path: ").strip().lower() != "yes":
            print("Not executed.")
            return
        for a, b in zip(route.path[:-1], route.path[1:]):
            a, b = np.asarray(a), np.asarray(b)
            n = max(1, int(np.ceil(np.degrees(np.abs(b - a).max()) / WAYPOINT_STEP_DEG)))
            for t in np.linspace(0, 1, n + 1)[1:]:
                send_joints(robot, (1 - t) * a + t * b, gripper=GRIPPER_OPEN)
        print("Path finished.")
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
