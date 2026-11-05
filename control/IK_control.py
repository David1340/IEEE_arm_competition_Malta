"""End-effector control for the SO-101: PyBullet IK -> LeRobot SOFollower.

Assumes the URDF zero pose matches the calibration midpoint (so101_new_calib),
so LeRobot degrees (use_degrees=True) map directly to URDF joint angles.

Usage:
    python -m control.IK_control --sim-only            # PyBullet GUI only, no hardware
    python -m control.IK_control                         # drives the real arm
    python -m control.IK_control --arm follower_blue     # one of the two arms: port from ports/<arm>.txt, LeRobot
                                                         # calibration <arm>.json, fine offsets of that arm
Commands in the prompt (meters, base frame):
    x y z [gripper 0-100]    absolute target
    d dx dy dz               relative move
    o ox oy oz               set offset added to every target (meters, default 0 0 0)
    q                        quit
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import logging

import pybullet as p
from control.modified_frpso_ik import IKResult, ModifiedFRPSOIK
from control.rrt_connect import RRTConnectPlanner
from control.fine_joint_offsets import correct_joint_position_deg, load_fine_joint_offsets, motor_position_deg

logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

ROOT = Path("/home/cris/.cache/huggingface/lerobot")
URDF = ROOT / "lerobot" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
CALIB = ROOT / "calibration" / "robots" / "so_follower" / "my_follower_arm.json"
PORT_FILE = ROOT / "ports" / "follower_port.txt"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_LINK = "gripper_frame_link"
REPO = Path(__file__).resolve().parents[1]


def arm_files(arm: str | None) -> tuple[Path, Path]:
    """(port file, LeRobot calibration file) of an arm by its LeRobot id (follower_pink, follower_blue); None is
    the single arm of the older setup (PORT_FILE, CALIB = my_follower_arm.json)."""
    if arm is None:
        return PORT_FILE, CALIB
    return REPO / "ports" / f"{arm}.txt", CALIB.parent / f"{arm}.json"


def robot_arm(robot) -> str | None:
    """The arm id connect_robot gave this robot (None for the older single-arm setup)."""
    return getattr(robot, "arm_name", None)


def calib_limits_deg(calib: dict, arm: str | None = None) -> dict[str, tuple[float, float]]:
    """Joint limits in degrees relative to the calibrated midpoint (0 deg)."""
    offsets = load_fine_joint_offsets(arm=arm)
    out = {}
    for name in ARM_JOINTS:
        c = calib[name]
        mid = (c["range_min"] + c["range_max"]) / 2
        out[name] = (
            (c["range_min"] - mid) * 360 / 4095 + offsets[name],
            (c["range_max"] - mid) * 360 / 4095 + offsets[name],
        )
    return out


class EEController:
    def __init__(self, gui: bool = False, lock_wrist_roll: bool = True, arm: str | None = None):
        self.lock_wrist_roll = lock_wrist_roll
        self.arm = arm
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(str(URDF.parent), physicsClientId=self.client)
        collision_flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        self.robot = p.loadURDF(
            str(URDF),
            useFixedBase=True,
            flags=collision_flags,
            physicsClientId=self.client,
        )

        self.idx, self.ee_idx = {}, None
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.client)
            name = info[1].decode()
            if info[2] != p.JOINT_FIXED:
                self.idx[name] = i
            if info[12].decode() == EE_LINK:
                self.ee_idx = i
        assert self.ee_idx is not None, f"{EE_LINK} not found in URDF"

        self.arm_joint_indices = [self.idx[name] for name in ARM_JOINTS]
        self.planner = ModifiedFRPSOIK(
            self.robot,
            self.ee_idx,
            self.arm_joint_indices,
            physics_client_id=self.client,
            check_self_collision=True,
            calibration_path=arm_files(arm)[1],
            arm=arm,
            fixed_joint_values={
                self.idx["wrist_roll"]: math.radians(-90.0),
            } if self.lock_wrist_roll else None,
        )
        self.rrt = RRTConnectPlanner(self.planner)
        self.lower = self.planner.lower.tolist()
        self.upper = self.planner.upper.tolist()
        g = p.getJointInfo(self.robot, self.idx["gripper"], physicsClientId=self.client)
        self.g_lower, self.g_upper = g[8], g[9]
        self.ranges = [u - l for l, u in zip(self.lower, self.upper)]
        self.q = [0.0] * len(ARM_JOINTS)
        if self.lock_wrist_roll:
            self.q[ARM_JOINTS.index("wrist_roll")] = math.radians(-90.0)
        self.last_ik_result: IKResult | None = None
        self._apply_sim(self.q)

    def _apply_sim(self, q):
        for n, v in zip(ARM_JOINTS, q):
            p.resetJointState(self.robot, self.idx[n], v, physicsClientId=self.client)

    def set_sim_state(self, q):
        self.q = list(q)
        self._apply_sim(self.q)

    def ee_position(self) -> np.ndarray:
        return np.array(p.getLinkState(
            self.robot,
            self.ee_idx,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )[4])

    def solve(self, target, orientation=None, tol=5e-3):
        """Return the best M-FRPSO joint solution and its predicted position error."""
        self._apply_sim(self.q)
        result = self.planner.solve(
            target,
            target_orientation=orientation,
            start_q=self.q,
            num_particles=120,
            elite_count=5,
            max_iterations=120,
            position_tolerance_m=tol,
        )
        self.last_ik_result = result
        if result is None:
            return self.q.copy(), float("inf")
        q = result.joint_positions.tolist()
        self._apply_sim(q)
        return q, result.position_error_m


def connect_robot(arm: str | None = None):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    port_file, calib = arm_files(arm)
    for f in (port_file, calib):
        if not f.is_file():
            raise SystemExit(f"{f} not found" + ("" if arm is None else f" (needed for --arm {arm})"))
    cfg = SOFollowerRobotConfig(
        port=port_file.read_text().strip(),
        id=calib.stem,
        calibration_dir=calib.parent,
        use_degrees=True,
        max_relative_target=15.0,  # deg per command, safety
    )
    robot = SOFollower(cfg)
    robot.connect(calibrate=False)
    robot.arm_name = arm
    return robot


def read_joints_rad(robot) -> list[float]:
    obs = robot.get_observation()
    arm = robot_arm(robot)
    return [math.radians(correct_joint_position_deg(name, obs[f"{name}.pos"], arm)) for name in ARM_JOINTS]


def send_joints(
    robot,
    q,
    gripper=None,
    steps=20,
    dt=0.03,
    tol_deg=3.0,
    gripper_tolerance=5.0,
    timeout=None,
    feedback_callback=None,
):
    """Interpolates from the current pose to q so the arm moves smoothly."""
    start = np.array(read_joints_rad(robot))
    goal = np.array(q)
    if timeout is None:
        # Pan measured at ~9 deg/s; use 6 deg/s to leave margin.
        timeout = 3.0 + float(np.max(np.abs(np.degrees(goal - start)))) / 6.0
    obs = robot.get_observation()
    g0 = obs["gripper.pos"]
    g1 = g0 if gripper is None else gripper
    for s in np.linspace(0, 1, steps)[1:]:
        cur = start + (goal - start) * s
        action = {
            f"{name}.pos": motor_position_deg(name, math.degrees(value), robot_arm(robot))
            for name, value in zip(ARM_JOINTS, cur)
        }
        action["gripper.pos"] = g0 + (g1 - g0) * s
        robot.send_action(action)
        time.sleep(dt)
        if feedback_callback is not None:
            feedback_callback(robot.get_observation())

    # Keep commanding the final goal until the joints get there (gripper may stall on an object).
    final = {
        f"{name}.pos": motor_position_deg(name, math.degrees(value), robot_arm(robot))
        for name, value in zip(ARM_JOINTS, goal)
    }
    final["gripper.pos"] = g1
    t0 = time.time()
    while time.time() - t0 < timeout:
        robot.send_action(final)
        time.sleep(dt)
        obs = robot.get_observation()
        if feedback_callback is not None:
            feedback_callback(obs)
        arm_errors = {
            name: abs(obs[f"{name}.pos"] - final[f"{name}.pos"])
            for name in ARM_JOINTS
        }
        arm_err = max(arm_errors.values())
        grip_err = abs(obs["gripper.pos"] - g1)
        if arm_err <= tol_deg and (gripper is None or grip_err <= gripper_tolerance):
            return True
    error_summary = ", ".join(f"{name}={error:.1f}deg" for name, error in arm_errors.items())
    if gripper is not None:
        error_summary += f", gripper={grip_err:.1f}units"
    print(f"Joint motion timeout after {timeout:.1f}s; remaining errors: {error_summary}")
    return False


# ---------------------------------------------------------------------------------------------- smooth motion
# How follow_path() moves the arm; the scripts' --speed / --accel / --profile set these (add_motion_args).
MOTION = {"speed": 20.0, "accel": 40.0, "profile": "trapezoid", "rate": 50.0, "max_lag": 8.0}


def add_motion_args(ap):
    ap.add_argument("--speed", type=float, default=MOTION["speed"],
                    help="top speed of the fastest joint along a path [deg/s]")
    ap.add_argument("--accel", type=float, default=MOTION["accel"], help="speed-up / slow-down [deg/s^2]")
    ap.add_argument("--profile", choices=["trapezoid", "minjerk"], default=MOTION["profile"],
                    help="trapezoid: constant acceleration; minjerk: smoother start and stop, a little slower")
    ap.add_argument("--max-lag", type=float, default=MOTION["max_lag"],
                    help="the path waits while the arm is this far behind the command [deg]")


def apply_motion_args(args):
    MOTION.update(speed=args.speed, accel=args.accel, profile=args.profile, max_lag=args.max_lag)


def motion_profile(length, speed, accel, profile):
    """(duration T, s(t)) moving `length` degrees: s goes 0 -> length with zero speed at both ends."""
    if length <= 0:
        return 0.0, lambda t: 0.0
    if profile == "minjerk":
        # s = L (10 r^3 - 15 r^4 + 6 r^5): peak speed 1.875 L / T, peak acceleration 5.77 L / T^2
        T = max(1.875 * length / speed, math.sqrt(5.7735 * length / accel))
        return T, lambda t: length * (lambda r: 10 * r ** 3 - 15 * r ** 4 + 6 * r ** 5)(min(max(t / T, 0.0), 1.0))
    ta = speed / accel
    if length < speed * ta:                       # too short to reach the top speed: a triangle
        ta = math.sqrt(length / accel)
        speed = accel * ta
    T = 2 * ta + (length - speed * ta) / speed

    def s(t):
        t = min(max(t, 0.0), T)
        if t < ta:
            return 0.5 * accel * t * t
        if t > T - ta:
            return length - 0.5 * accel * (T - t) ** 2
        return 0.5 * accel * ta * ta + speed * (t - ta)
    return T, s


def follow_path(robot, path, gripper=None, *, tol_deg=3.0, gripper_tolerance=5.0, feedback_callback=None,
                **motion) -> bool:
    """Move along a joint path (radians, corrected angles) in one smooth motion: targets are streamed at
    MOTION["rate"] Hz along the whole path with a speed profile (gentle start, top speed, gentle stop), and the
    arm only has to arrive at the end. When the arm falls more than max_lag degrees behind, the path waits for
    it. gripper (0-100) is reached gradually over the move; None keeps it where it is. Returns True when the
    arm arrived (within tol_deg)."""
    settings = {**MOTION, **motion}
    points = [np.asarray(q, dtype=float) for q in path]
    if not points:
        return True
    points[0] = np.array(read_joints_rad(robot))           # start from where the arm really is
    lengths = [float(np.degrees(np.max(np.abs(b - a)))) for a, b in zip(points[:-1], points[1:])]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    g0 = float(robot.get_observation()["gripper.pos"])
    g1 = g0 if gripper is None else float(gripper)
    arm = robot_arm(robot)

    def point_at(distance):
        i = int(np.clip(np.searchsorted(cumulative, distance, side="right") - 1, 0, max(len(lengths) - 1, 0)))
        if not lengths:
            return points[-1]
        f = 0.0 if lengths[i] <= 0 else min(max((distance - cumulative[i]) / lengths[i], 0.0), 1.0)
        return points[i] + f * (points[i + 1] - points[i])

    duration, s_of_t = motion_profile(total, settings["speed"], settings["accel"], settings["profile"])
    dt = 1.0 / settings["rate"]
    t, next_tick = 0.0, time.monotonic()
    give_up = next_tick + 3 * duration + 10.0
    while t < duration and time.monotonic() < give_up:
        distance = s_of_t(t)
        q = point_at(distance)
        action = {f"{name}.pos": motor_position_deg(name, math.degrees(value), arm)
                  for name, value in zip(ARM_JOINTS, q)}
        action["gripper.pos"] = g0 + (g1 - g0) * (distance / total if total > 0 else 1.0)
        robot.send_action(action)
        obs = robot.get_observation()
        if feedback_callback is not None:
            feedback_callback(obs)
        measured = np.array([math.radians(correct_joint_position_deg(n, obs[f"{n}.pos"], arm)) for n in ARM_JOINTS])
        if float(np.degrees(np.max(np.abs(measured - q)))) <= settings["max_lag"]:
            t += dt                                          # on track: the path moves on
        next_tick += dt
        time.sleep(max(0.0, next_tick - time.monotonic()))
    # the end: keep commanding the goal until the joints (and the gripper, if asked) are there
    return send_joints(robot, points[-1], gripper, steps=2, tol_deg=tol_deg, gripper_tolerance=gripper_tolerance,
                       timeout=3.0, feedback_callback=feedback_callback)


LOCKED_WRIST_ROLL = math.radians(-90.0)
WRIST_ROLL_SNAP = math.radians(5.0)  # measured roll within this of -90 deg counts as -90 deg


def planning_start(robot, ctrl) -> list[float]:
    """The measured joints as a start for the planners.

    With wrist roll locked, the planners only accept exactly -90 deg, which a real arm never reads. A roll
    within 5 deg is treated as -90 deg (the first waypoint then corrects it); further away, the wrist is
    turned to -90 deg in place first, with the other joints held still."""
    q = read_joints_rad(robot)
    if not ctrl.lock_wrist_roll:
        return q
    roll = ARM_JOINTS.index("wrist_roll")
    if abs(q[roll] - LOCKED_WRIST_ROLL) > WRIST_ROLL_SNAP:
        print(f"Wrist roll is at {math.degrees(q[roll]):.1f} deg; turning it to -90 deg first "
              "(use --free-wrist-roll to plan with any roll)")
        target = list(q)
        target[roll] = LOCKED_WRIST_ROLL
        follow_path(robot, [q, target])
        q = read_joints_rad(robot)
        if abs(q[roll] - LOCKED_WRIST_ROLL) > WRIST_ROLL_SNAP:
            raise RuntimeError(f"wrist roll stayed at {math.degrees(q[roll]):.1f} deg; could not reach -90 deg")
    q[roll] = LOCKED_WRIST_ROLL
    return q


def report_start(ctrl, q) -> None:
    """Say when the measured start pose is outside the planner's joint limits or in collision in the model
    (the planners still start from it, but this is the usual reason a move fails)."""
    planner = ctrl.planner
    for name, value, lower, upper in zip(ARM_JOINTS, q, planner.lower, planner.upper):
        if not lower <= value <= upper:
            print(f"Note: {name} is at {math.degrees(value):.1f} deg, outside the planner limits "
                  f"[{math.degrees(lower):.1f}, {math.degrees(upper):.1f}] deg")
    if not planner.state_is_collision_free(q, check_joint_limits=False):
        print("Note: the start pose is in collision in the robot model; the path first leaves it")
    ctrl.set_sim_state(q)


def move_to(ctrl, robot, aim, gripper=None, orient=None, tol=5e-3):
    """Move the IK point (gripper_frame_link) to `aim` (base frame, m): M-FRPSO pose, RRT-Connect path, then on
    the real arm up to 3 corrections for sag/backlash. Without a robot the path is played in the simulation.
    Returns (reached, final IK-point position), or (False, None) when no pose or path was found."""
    aim = np.asarray(aim, dtype=float)
    try:
        start_q = planning_start(robot, ctrl) if robot else ctrl.q.copy()
    except RuntimeError as exc:
        print(exc)
        return False, None
    ctrl.set_sim_state(start_q)
    report_start(ctrl, start_q)
    print("M-FRPSO: searching for a pose ...", flush=True)
    t0 = time.time()
    q, err = ctrl.solve(aim, orient)
    print(f"M-FRPSO: done in {time.time() - t0:.1f} s", flush=True)
    if ctrl.last_ik_result is None or not ctrl.last_ik_result.success:
        if ctrl.last_ik_result is None:
            print("M-FRPSO não encontrou uma configuração livre de colisão")
        else:
            print(f"Alvo não confirmado: erro previsto {err * 1000:.1f} mm")
        ctrl.set_sim_state(start_q)
        return False, None
    print("RRT-Connect: planning the path ...", flush=True)
    t0 = time.time()
    try:
        route = ctrl.rrt.plan(start_q, q)
    except ValueError as exc:
        print(f"Configuração inicial inválida para o RRT-Connect: {exc}")
        ctrl.set_sim_state(start_q)
        return False, None
    if route is None:
        print("RRT-Connect não encontrou um caminho livre de colisão; alvo ignorado")
        ctrl.set_sim_state(start_q)
        return False, None
    print(f"RRT-Connect: {len(route.path)} configurações, {route.iterations} iterações, {time.time() - t0:.1f} s")

    if robot is None:
        for waypoint in route.path[1:]:
            ctrl.set_sim_state(waypoint)
        final = ctrl.ee_position()
        return float(np.linalg.norm(final - aim)) <= tol, final

    motion_settled = follow_path(robot, route.path if len(route.path) > 1 else [q, q], gripper)
    ctrl.set_sim_state(read_joints_rad(robot))
    final = ctrl.ee_position()
    # Sag/backlash leave a static offset; shift the IK aim point by the measured error.
    corrected = aim.copy()
    for _ in range(3):
        if not motion_settled or np.linalg.norm(final - aim) < 4e-3:
            break
        corrected = corrected + (aim - final)
        q, _ = ctrl.solve(corrected, orient)
        if ctrl.last_ik_result is None or not ctrl.last_ik_result.success:
            break
        current_q = planning_start(robot, ctrl)
        try:
            correction_route = ctrl.rrt.plan(current_q, q)
        except ValueError as exc:
            print(f"Configuração de correção inválida para o RRT-Connect: {exc}")
            break
        if correction_route is None:
            print("RRT-Connect não encontrou caminho para corrigir o erro de posição")
            break
        motion_settled = follow_path(robot, correction_route.path)
        if not motion_settled:
            break
        ctrl.set_sim_state(read_joints_rad(robot))
        final = ctrl.ee_position()
    print(f"Joint motion settled: {motion_settled}")
    return motion_settled and float(np.linalg.norm(final - aim)) <= tol, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--arm", default=None, help="which arm, by LeRobot id (follower_pink, follower_blue); "
                    "default: the older single-arm files (my_follower_arm.json)")
    gui_options = ap.add_mutually_exclusive_group()
    gui_options.add_argument("--gui", dest="gui", action="store_true", help="open the PyBullet visualizer")
    gui_options.add_argument("--no-gui", dest="gui", action="store_false", help="run PyBullet headless (default)")
    ap.set_defaults(gui=False)
    ap.add_argument("--free-wrist-roll", action="store_true", help="allow M-FRPSO to vary wrist roll")
    ap.add_argument("--orient-down", action="store_true", help="constrain gripper to point down")
    ap.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, -0.04], metavar=("X", "Y", "Z"),
                    help="meters added to every target, base frame")
    add_motion_args(ap)
    args = ap.parse_args()
    apply_motion_args(args)

    ctrl = EEController(gui=args.gui, lock_wrist_roll=not args.free_wrist_roll, arm=args.arm)
    robot = None if args.sim_only else connect_robot(args.arm)
    if robot:
        ctrl.set_sim_state(read_joints_rad(robot))

    orient = p.getQuaternionFromEuler([math.pi, 0, 0]) if args.orient_down else None
    offset = np.array(args.offset)
    target = ctrl.ee_position() - offset
    print(f"EE atual: {np.round(target, 3)} m")

    try:
        while True:
            line = input("alvo> ").replace("[", " ").replace("]", " ").replace(",", " ").split()
            if not line:
                continue
            if line[0] == "q":
                break
            if line[0] == "o":
                try:
                    offset = np.array(line[1:4], dtype=float)
                    assert offset.shape == (3,)
                except (ValueError, AssertionError):
                    print("Formato inválido")
                    continue
                print(f"offset: {offset} m")
                continue
            gripper = None
            try:
                if line[0] == "d":
                    new = target + np.array(line[1:4], dtype=float)
                else:
                    new = np.array(line[:3], dtype=float)
                    if len(line) > 3:
                        gripper = float(line[3])
            except ValueError:
                print("Formato inválido")
                continue

            reached, final = move_to(ctrl, robot, new + offset, gripper=gripper, orient=orient)
            if final is None:
                continue
            target = new
            final_error = float(np.linalg.norm(final - (new + offset)))
            print(f"EE final: {np.round(final - offset, 3)} m | target error: {final_error * 1000:.1f} mm")
            print(f"Position check: {'REACHED' if reached else 'NOT REACHED'}"
                  f"{'' if robot else ' (simulation)'}")
    finally:
        if robot:
            robot.disconnect()
        p.disconnect()


if __name__ == "__main__":
    main()
