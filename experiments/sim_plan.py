"""Simulation only: table + objects in PyBullet, then M-FRPSO IK and RRT-Connect, one step at a time.

No camera, no robot. Edit OBJECTS / START_Q / TARGET below, or pass --target X Y.

    python experiments/sim_plan.py                  # steps 1-3, press Enter between steps
    python experiments/sim_plan.py --target 0.35 0.05 --height 0.04
    python experiments/sim_plan.py --no-pause       # run everything without waiting
    python experiments/sim_plan.py --direct         # no window (headless check)

Steps:
  1. scene: table (top at z=0), robot at START_Q, objects
  2. M-FRPSO: finds ONE joint pose that puts the point between the jaws at TARGET
  3. RRT-Connect: collision-free path from START_Q to that pose, played back
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.modified_frpso_ik import ModifiedFRPSOIK, load_robot_for_collision  # noqa: E402
from control.rrt_connect import RRTConnectPlanner  # noqa: E402

URDF_FILE = ROOT / "robot_urdfs" / "so101" / "robot.urdf"
STL_DIR = ROOT / "stl"

# ---- edit here -------------------------------------------------------------
# (stl name in stl/, x [m], y [m], yaw [deg]) in the robot frame; the object stands on the table (z=0).
OBJECTS = [
    ("cylinder_large", 0.37, -0.04, 0.0),
    ("cube_small", 0.25, 0.20, 30.0),
]
TARGET = (0.37, -0.04)         # x, y of the point between the jaws (default: first object)
HEIGHT = 0.04                  # z of the point between the jaws [m]
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll  [rad]
START_Q = [0.0553, -0.7104, 0.1461, -1.1745, -math.pi / 2]
TABLE_CLEARANCE = 0.015        # gripper links stay this far above the table [m]
OBJECT_COLOR = [(0.9, 0.3, 0.2), (0.2, 0.5, 0.9), (0.2, 0.7, 0.3), (0.8, 0.7, 0.1)]
# ---------------------------------------------------------------------------

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRASP_REFERENCE_DEG = (57.0, -33.0, -25.0)   # lift/elbow/wrist_flex of a horizontal grasp attitude
GRIPPER_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")


def wait(args, text):
    print(f"\n== {text}")
    if not args.no_pause and not args.direct:
        input("   Enter to continue... ")


def add_table(client):
    half = [0.75, 0.9, 0.02]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.57, 0.6, 1], physicsClientId=client)
    return p.createMultiBody(0, col, vis, basePosition=[0, 0.25, -half[2]], physicsClientId=client)


def add_object(client, stl, x, y, yaw_deg, color):
    path = str(STL_DIR / f"{stl}.stl")
    scale = [0.001] * 3
    col = p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=scale, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=scale,
                              rgbaColor=[*color, 1], physicsClientId=client)
    orn = p.getQuaternionFromEuler([0, 0, math.radians(yaw_deg)])
    return p.createMultiBody(0, col, vis, basePosition=[x, y, 0], baseOrientation=orn, physicsClientId=client)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"), default=None)
    ap.add_argument("--height", type=float, default=HEIGHT)
    ap.add_argument("--clearance", type=float, default=TABLE_CLEARANCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--direct", action="store_true", help="headless, no pauses or playback")
    args = ap.parse_args()
    target_xy = args.target or (TARGET if TARGET else OBJECTS[0][1:3])

    client = p.connect(p.DIRECT if args.direct else p.GUI)
    p.setGravity(0, 0, 0, physicsClientId=client)

    # ---- 1. scene ----------------------------------------------------------
    table = add_table(client)
    bodies = [add_object(client, s, x, y, yaw, OBJECT_COLOR[i % len(OBJECT_COLOR)])
              for i, (s, x, y, yaw) in enumerate(OBJECTS)]
    robot = load_robot_for_collision(str(URDF_FILE), physics_client_id=client)
    joint = {p.getJointInfo(robot, i, physicsClientId=client)[1].decode(): i
             for i in range(p.getNumJoints(robot, physicsClientId=client))}
    link = {p.getJointInfo(robot, i, physicsClientId=client)[12].decode(): i
            for i in range(p.getNumJoints(robot, physicsClientId=client))}
    arm = [joint[n] for n in JOINTS]
    for i, q in zip(arm, START_Q):
        p.resetJointState(robot, i, q, physicsClientId=client)
    low, high = p.getJointInfo(robot, joint["gripper"], physicsClientId=client)[8:10]
    p.resetJointState(robot, joint["gripper"], low + 0.8 * (high - low), physicsClientId=client)  # 80% open

    checker = ModifiedFRPSOIK(
        robot, link["grasp_center_link"], arm,
        obstacle_body_ids=[table, *bodies],
        physics_client_id=client,
        check_self_collision=False,
        fixed_joint_values={joint["wrist_roll"]: START_Q[4]},
        allowed_robot_obstacle_link_pairs=[(table, -1, -1)],   # the base sits on the table
        minimum_link_heights={link[n]: args.clearance for n in GRIPPER_LINKS},
    )
    rrt = RRTConnectPlanner(checker, max_joint_step_rad=math.radians(3.0))
    target = np.array([target_xy[0], target_xy[1], args.height])
    # Visual-only balls (no collision shape, not in the obstacle list): blue = target,
    # red = the point the IK uses (grasp_center_link), which follows the robot.
    def ball(color, position):
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=color, physicsClientId=client)
        return p.createMultiBody(0, -1, shape, basePosition=list(position), physicsClientId=client)

    ball([0.1, 0.3, 1, 0.6], target)
    tcp_ball = ball([1, 0, 0, 1], target)

    def show(q):
        checker._set_joint_state(q)
        position = p.getLinkState(robot, checker.ee_link_index, computeForwardKinematics=True,
                                  physicsClientId=client)[4]
        p.resetBasePositionAndOrientation(tcp_ball, position, [0, 0, 0, 1], physicsClientId=client)

    show(START_Q)
    if not args.direct:
        p.resetDebugVisualizerCamera(1.2, 45, -40, [0.3, 0.1, 0.1], physicsClientId=client)
    wait(args, f"1/3 scene: {len(OBJECTS)} object(s); blue ball = target, red ball = IK point (grasp_center_link) {np.round(target, 3)} m")

    # ---- 2. M-FRPSO --------------------------------------------------------
    # Horizontal attitude of the gripper, panned toward the target, used as the orientation goal.
    ref = np.array(START_Q, dtype=float)
    ref[1:4] = np.deg2rad(GRASP_REFERENCE_DEG)
    for _ in range(4):
        checker._set_joint_state(ref)
        tcp = p.getLinkState(robot, checker.ee_link_index, computeForwardKinematics=True,
                             physicsClientId=client)[4]
        ref[0] -= math.atan2(target[1], target[0]) - math.atan2(tcp[1], tcp[0])
    checker._set_joint_state(ref)
    orientation = p.getLinkState(robot, checker.ee_link_index, computeForwardKinematics=True,
                                 physicsClientId=client)[5]

    result = checker.solve(
        target, target_orientation=orientation, start_q=START_Q,
        num_particles=160, elite_count=5, max_iterations=160, orientation_weight=0.05,
        position_tolerance_m=0.012, orientation_tolerance_rad=math.radians(30.0), seed=args.seed,
    )
    if result is None:
        raise SystemExit("M-FRPSO found no collision-free pose (try another --seed, --height or --target)")
    goal_q = result.joint_positions
    show(goal_q)
    tcp = np.array(p.getLinkState(robot, checker.ee_link_index, computeForwardKinematics=True,
                                  physicsClientId=client)[4])
    print(f"   goal joints [deg]: {np.round(np.degrees(goal_q), 1)}")
    print(f"   position error {result.position_error_m * 1000:.1f} mm, "
          f"between-jaws point {np.round(tcp, 3)} m, {result.iterations} iterations")
    wait(args, "2/3 M-FRPSO pose (robot shown at the found pose)")

    # ---- 3. RRT-Connect ----------------------------------------------------
    route = rrt.plan(START_Q, goal_q, seed=args.seed)
    if route is None:
        raise SystemExit("RRT-Connect found no collision-free path")
    path = route.path
    print(f"   path: {len(path)} configurations, {route.iterations} iterations")
    wait(args, "3/3 RRT-Connect path (the robot returns to the start pose and moves along the path)")
    while True:
        show(START_Q)
        if not args.direct:
            time.sleep(0.7)
        for a, b in zip(path[:-1], path[1:]):
            a, b = np.asarray(a), np.asarray(b)
            steps = max(2, int(np.ceil(np.degrees(np.abs(b - a).max()) / 0.5)))   # 0.5 deg per frame
            for t in np.linspace(0, 1, steps):
                show((1 - t) * a + t * b)
                if not args.direct:
                    time.sleep(0.02 / args.speed)
        print("   path finished")
        if args.direct or input("   Enter = replay, q = quit: ").strip().lower() == "q":
            break
    p.disconnect(client)


if __name__ == "__main__":
    main()
