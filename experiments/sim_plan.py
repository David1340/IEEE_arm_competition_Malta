"""Simulation only: table + objects in PyBullet, then M-FRPSO IK and RRT-Connect, one step at a time.

No camera, no robot. Edit OBJECTS / START_Q / TARGET below, or pass --target X Y.
The same World class is used by experiments/Full_pipeline.py.

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
if str(ROOT) not in sys.path:
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
HEIGHT = 0.02                  # z of the point between the jaws [m]
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll  [rad]
START_Q = [0.0553, -0.7104, 0.1461, -1.1745, -math.pi / 2]
TABLE_CLEARANCE = 0.015        # gripper links stay this far above the table [m]
OBJECT_COLOR = [(0.9, 0.3, 0.2), (0.2, 0.5, 0.9), (0.2, 0.7, 0.3), (0.8, 0.7, 0.1)]
# ---------------------------------------------------------------------------

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRASP_REFERENCE_DEG = (57.0, -33.0, -25.0)   # lift/elbow/wrist_flex of a horizontal grasp attitude
GRIPPER_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")
GRIPPER_OPEN = 0.8                           # fraction of the gripper range


def add_table(client):
    half = [0.75, 0.9, 0.02]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.57, 0.6, 1], physicsClientId=client)
    return p.createMultiBody(0, col, vis, basePosition=[0, 0.25, -half[2]], physicsClientId=client)


def add_object(client, stl_path, x, y, yaw_deg, color):
    path = str(stl_path)
    scale = [0.001] * 3
    col = p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=scale, physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=scale,
                              rgbaColor=[*color, 1], physicsClientId=client)
    orn = p.getQuaternionFromEuler([0, 0, math.radians(yaw_deg)])
    return p.createMultiBody(0, col, vis, basePosition=[x, y, 0], baseOrientation=orn, physicsClientId=client)


def tilt_down(orientation, target):
    """The level grasp orientation turned 90 deg about the horizontal axis across the reach, so the gripper
    points straight down at the target (the jaws still open sideways)."""
    R = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    radial = np.array([target[0], target[1], 0.0])
    radial /= max(np.linalg.norm(radial), 1e-9)
    k = int(np.argmax(np.abs(R.T @ radial)))           # the link axis that points along the reach
    approach = R[:, k] * np.sign(R[:, k] @ radial)
    axis = np.cross(approach, [0.0, 0.0, -1.0])
    angle = math.atan2(np.linalg.norm(axis), approach @ [0.0, 0.0, -1.0])
    axis /= max(np.linalg.norm(axis), 1e-9)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    turn = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    return matrix_to_quaternion(turn @ R)


def matrix_to_quaternion(R):
    """Rotation matrix -> PyBullet quaternion (x, y, z, w), Shepperd's method (stable for every rotation)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        r = math.sqrt(1 + t) * 2
        q = [(R[2, 1] - R[1, 2]) / r, (R[0, 2] - R[2, 0]) / r, (R[1, 0] - R[0, 1]) / r, r / 4]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        r = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [r / 4, (R[0, 1] + R[1, 0]) / r, (R[0, 2] + R[2, 0]) / r, (R[2, 1] - R[1, 2]) / r]
    elif R[1, 1] > R[2, 2]:
        r = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / r, r / 4, (R[1, 2] + R[2, 1]) / r, (R[0, 2] - R[2, 0]) / r]
    else:
        r = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / r, (R[1, 2] + R[2, 1]) / r, r / 4, (R[1, 0] - R[0, 1]) / r]
    return q


class World:
    """Table, objects and robot in PyBullet, plus the M-FRPSO solver and the RRT-Connect planner."""

    def __init__(self, objects, start_q, *, gui=True, clearance=TABLE_CLEARANCE, arm=None, calibration_path=None,
                 gripper_open=None):
        """objects: [(stl_path, x, y, yaw_deg), ...]; arm: which arm's joint limits (calibration_path, its LeRobot
        calibration file) and fine offsets to use (default: the older single-arm files); gripper_open: how far the
        jaws are open in the collision model, as a fraction of the gripper range (default GRIPPER_OPEN), i.e. the
        LeRobot 0-100 command / 100 (control/gripper.py)."""
        self.start_q = list(start_q)
        self.client = c = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=c)
        self.table = add_table(c)
        self.bodies = [add_object(c, stl, x, y, yaw, OBJECT_COLOR[i % len(OBJECT_COLOR)])
                       for i, (stl, x, y, yaw) in enumerate(objects)]
        self.robot = load_robot_for_collision(str(URDF_FILE), physics_client_id=c)
        infos = [p.getJointInfo(self.robot, i, physicsClientId=c) for i in range(p.getNumJoints(self.robot, physicsClientId=c))]
        joint = {info[1].decode(): info[0] for info in infos}
        link = {info[12].decode(): info[0] for info in infos}
        arm = [joint[n] for n in JOINTS]
        low, high = infos[joint["gripper"]][8:10]
        opening = GRIPPER_OPEN if gripper_open is None else gripper_open
        p.resetJointState(self.robot, joint["gripper"], low + opening * (high - low), physicsClientId=c)

        self.checker = ModifiedFRPSOIK(
            self.robot, link["grasp_center_link"], arm,
            obstacle_body_ids=[self.table, *self.bodies],
            physics_client_id=c,
            check_self_collision=False,
            fixed_joint_values={joint["wrist_roll"]: self.start_q[4]},
            allowed_robot_obstacle_link_pairs=[(self.table, -1, -1)],   # the base sits on the table
            minimum_link_heights={link[n]: clearance for n in GRIPPER_LINKS},
            **({} if calibration_path is None else {"calibration_path": calibration_path}),
            arm=arm,
        )
        # Artificial joint limits that also contain the current joint reading, so a start pose
        # slightly outside the URDF limits is valid for both M-FRPSO and RRT-Connect.
        self.checker.lower = np.minimum(self.checker.lower, self.start_q)
        self.checker.upper = np.maximum(self.checker.upper, self.start_q)
        self.rrt = RRTConnectPlanner(self.checker, max_joint_step_rad=math.radians(3.0))
        # Visual-only balls (no collision shape, not in the obstacle list): blue = target,
        # red = the point the IK uses (grasp_center_link), which follows the robot.
        self.target_ball = self._ball([0.1, 0.3, 1, 0.6])
        self.tcp_ball = self._ball([1, 0, 0, 1])
        self.show(self.start_q)
        if gui:
            p.resetDebugVisualizerCamera(1.2, 45, -40, [0.3, 0.1, 0.1], physicsClientId=c)

    def _ball(self, color):
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=color, physicsClientId=self.client)
        return p.createMultiBody(0, -1, shape, basePosition=[0, 0, 0], physicsClientId=self.client)

    def tcp(self):
        return np.array(p.getLinkState(self.robot, self.checker.ee_link_index, computeForwardKinematics=True,
                                       physicsClientId=self.client)[4])

    def show(self, q):
        self.checker._set_joint_state(q)
        p.resetBasePositionAndOrientation(self.tcp_ball, self.tcp().tolist(), [0, 0, 0, 1], physicsClientId=self.client)

    def find_pose(self, target, seed=0, tolerance=0.012, posture="horizontal"):
        """M-FRPSO: one collision-free joint pose with the point between the jaws at target (x, y, z), within
        `tolerance` metres. posture: "horizontal" (the gripper roughly level, pointing away from the base, within
        30 deg), "down" (pointing straight down, within 30 deg: for objects close to the base, where a level
        gripper would need the wrist folded back) or "free" (any orientation)."""
        target = np.asarray(target, dtype=float)
        p.resetBasePositionAndOrientation(self.target_ball, target.tolist(), [0, 0, 0, 1], physicsClientId=self.client)
        # Horizontal attitude of the gripper, panned toward the target, used as the orientation goal.
        ref = np.array(self.start_q, dtype=float)
        ref[1:4] = np.deg2rad(GRASP_REFERENCE_DEG)
        for _ in range(4):
            self.checker._set_joint_state(ref)
            tcp = self.tcp()
            ref[0] -= math.atan2(target[1], target[0]) - math.atan2(tcp[1], tcp[0])
        self.checker._set_joint_state(ref)
        orientation = p.getLinkState(self.robot, self.checker.ee_link_index, computeForwardKinematics=True,
                                     physicsClientId=self.client)[5]
        if posture == "down":
            orientation = tilt_down(orientation, target)
        elif posture == "free":
            orientation = None
        return self.checker.solve(
            target, target_orientation=orientation, start_q=self.start_q,
            num_particles=160, elite_count=5, max_iterations=160, orientation_weight=0.05,
            position_tolerance_m=tolerance, orientation_tolerance_rad=math.radians(30.0), seed=seed,
        )

    def find_path(self, goal_q, seed=0):
        """RRT-Connect: collision-free path from the start pose to goal_q."""
        return self.rrt.plan(self.start_q, goal_q, seed=seed)

    def play(self, path, speed=1.0, realtime=True):
        self.show(self.start_q)
        if realtime:
            time.sleep(0.7)
        for a, b in zip(path[:-1], path[1:]):
            a, b = np.asarray(a), np.asarray(b)
            steps = max(2, int(np.ceil(np.degrees(np.abs(b - a).max()) / 0.5)))   # 0.5 deg per frame
            for t in np.linspace(0, 1, steps):
                self.show((1 - t) * a + t * b)
                if realtime:
                    time.sleep(0.02 / speed)

    def close(self):
        p.disconnect(self.client)


def wait(args, text):
    print(f"\n== {text}")
    if not args.no_pause and not args.direct:
        input("   Enter to continue... ")


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
    target = np.array([target_xy[0], target_xy[1], args.height])

    # 1. scene
    world = World([(STL_DIR / f"{s}.stl", x, y, yaw) for s, x, y, yaw in OBJECTS], START_Q,
                  gui=not args.direct, clearance=args.clearance)
    world.show(START_Q)
    p.resetBasePositionAndOrientation(world.target_ball, target.tolist(), [0, 0, 0, 1], physicsClientId=world.client)
    wait(args, f"1/3 scene: {len(OBJECTS)} object(s); blue ball = target, red ball = IK point "
               f"(grasp_center_link) {np.round(target, 3)} m")

    # 2. M-FRPSO
    result = world.find_pose(target, args.seed)
    if result is None:
        raise SystemExit("M-FRPSO found no collision-free pose (try another --seed, --height or --target)")
    goal_q = result.joint_positions
    world.show(goal_q)
    print(f"   goal joints [deg]: {np.round(np.degrees(goal_q), 1)}")
    print(f"   position error {result.position_error_m * 1000:.1f} mm, "
          f"between-jaws point {np.round(world.tcp(), 3)} m, {result.iterations} iterations")
    wait(args, "2/3 M-FRPSO pose (robot shown at the found pose)")

    # 3. RRT-Connect
    route = world.find_path(goal_q, args.seed)
    if route is None:
        raise SystemExit("RRT-Connect found no collision-free path")
    print(f"   path: {len(route.path)} configurations, {route.iterations} iterations")
    wait(args, "3/3 RRT-Connect path (the robot returns to the start pose and moves along the path)")
    while True:
        world.play(route.path, args.speed, realtime=not args.direct)
        print("   path finished")
        if args.direct or input("   Enter = replay, q = quit: ").strip().lower() == "q":
            break
    world.close()


if __name__ == "__main__":
    main()
