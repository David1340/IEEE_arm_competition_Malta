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
    python experiments/Full_pipeline.py --execute --close-gripper   # then close until the motor current shows contact
"""
import argparse
import sys
import time
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
HOME_Q = np.radians([0.0, 0.0, 0.0, 0.0, -90.0])  # shoulder_pan, lift, elbow, wrist_flex, wrist_roll
GRIP_CURRENT_MA = 80.0   # gripper motor current that means the object is held
GRIP_CONFIRM_READINGS = 3
GRIP_SQUEEZE_UNITS = 2.0  # units held past the contact position
GRIPPER_MA_PER_UNIT = 6.5


def read_gripper_current_ma(robot):
    """STS3215 Present_Current in mA; bit 15 holds the direction."""
    raw = int(robot.bus.read("Present_Current", "gripper", normalize=False))
    return (raw & 0x7FFF) * GRIPPER_MA_PER_UNIT


def close_gripper_with_current_feedback(robot, closed_target, current_threshold_ma, confirm_readings,
                                        squeeze_units, *, timeout=10.0):
    """Close the gripper until the motor current shows contact, then hold a light squeeze.

    Returns (grasped, stop_position, hold_position, current_ma). grasped is False if the gripper
    reached closed_target without a current rise (nothing between the jaws).
    """
    deadline = time.monotonic() + timeout
    position = float(robot.get_observation()["gripper.pos"])
    above = 0
    current_ma = 0.0
    while time.monotonic() < deadline:
        observation = robot.get_observation()
        hold = {key: value for key, value in observation.items()
                if key.endswith(".pos") and key != "gripper.pos"}
        robot.send_action({**hold, "gripper.pos": float(closed_target)})
        time.sleep(0.04)
        position = float(robot.get_observation()["gripper.pos"])
        current_ma = read_gripper_current_ma(robot)
        above = above + 1 if current_ma >= current_threshold_ma else 0
        if above >= confirm_readings:
            hold_position = max(position - squeeze_units, closed_target)
            robot.send_action({**hold, "gripper.pos": hold_position})
            return True, position, hold_position, current_ma
        if abs(position - closed_target) <= 1.0:
            break
    return False, position, position, current_ma


def move_along(robot, path, send_joints):
    """Send the joint path in short steps, gripper open."""
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        n = max(1, int(np.ceil(np.degrees(np.abs(b - a).max()) / WAYPOINT_STEP_DEG)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            send_joints(robot, (1 - t) * a + t * b, gripper=GRIPPER_OPEN)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_vision_args(ap)
    ap.add_argument("--object", type=int, default=1, help="1-based index in the printed object list")
    ap.add_argument("--height", type=float, default=HEIGHT, help="z of the point between the jaws [m]")
    ap.add_argument("--clearance", type=float, default=TABLE_CLEARANCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-robot", action="store_true", help="do not connect to the arm; start from START_Q")
    ap.add_argument("--no-gui", action="store_true", help="no PyBullet window")
    ap.add_argument("--close-gripper", action="store_true",
                    help="after the path, ask for CLOSE and close the gripper until the motor current shows contact")
    ap.add_argument("--grip-current-ma", type=float, default=GRIP_CURRENT_MA)
    ap.add_argument("--grip-confirm", type=int, default=GRIP_CONFIRM_READINGS,
                    help="consecutive readings above the current threshold")
    ap.add_argument("--grip-squeeze", type=float, default=GRIP_SQUEEZE_UNITS)
    ap.add_argument("--execute", action="store_true", help="move the real arm after a typed confirmation")
    args = ap.parse_args()
    if args.execute and args.no_robot:
        raise SystemExit("--execute needs the arm: drop --no-robot")

    # 0. robot: go to the home pose before anything else (--execute)
    robot = None
    if not args.no_robot:
        from control.IK_control import connect_robot, read_joints_rad, send_joints

        robot = connect_robot()
        if args.execute:
            print(f"Home pose [deg]: {np.round(np.degrees(HOME_Q), 1)}; now at {np.round(np.degrees(read_joints_rad(robot)), 1)}")
            if input("Type 'yes' to move the real arm to the home pose: ").strip().lower() != "yes":
                robot.disconnect()
                raise SystemExit("Not executed.")
            move_along(robot, [read_joints_rad(robot), HOME_Q], send_joints)

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
    start_q = START_Q if robot is None else read_joints_rad(robot)
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
            input("Final pose shown. Press Enter to end... ")
            return
        if input("Type 'yes' to move the real arm along this path: ").strip().lower() != "yes":
            print("Not executed.")
            return
        move_along(robot, route.path, send_joints)
        print("Path finished.")
        if args.close_gripper and input("Type CLOSE to close the gripper on the object: ").strip().upper() == "CLOSE":
            grasped, stop, hold, current_ma = close_gripper_with_current_feedback(
                robot, 0.0, args.grip_current_ma, args.grip_confirm, args.grip_squeeze)
            if grasped:
                print(f"Object grasped: contact at {stop:.1f}/100 with {current_ma:.0f} mA; holding at {hold:.1f}/100")
            else:
                print(f"No contact detected (gripper at {stop:.1f}/100, {current_ma:.0f} mA)")
        input("Final pose held. Press Enter to end... ")
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
