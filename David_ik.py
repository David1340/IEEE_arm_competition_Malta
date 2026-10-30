"""End-effector control for the SO-101: PyBullet IK -> LeRobot SOFollower.

Assumes the URDF zero pose matches the calibration midpoint (so101_new_calib),
so LeRobot degrees (use_degrees=True) map directly to URDF joint angles.

Usage:
    python David_ik.py --sim-only            # PyBullet GUI only, no hardware
    python David_ik.py                       # drives the real arm
Commands in the prompt (meters, base frame):
    x y z [gripper 0-100]    absolute target
    d dx dy dz               relative move
    o ox oy oz               set offset added to every target (meters, default 0 0 0)
    q                        quit
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import logging

import pybullet as p

logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

ROOT = Path("/home/cris/.cache/huggingface/lerobot")
URDF = ROOT / "lerobot" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
CALIB = ROOT / "calibration" / "robots" / "so_follower" / "my_follower_arm.json"
PORT_FILE = ROOT / "ports" / "follower_port.txt"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_LINK = "gripper_frame_link"


def calib_limits_deg(calib: dict) -> dict[str, tuple[float, float]]:
    """Joint limits in degrees relative to the calibrated midpoint (0 deg)."""
    out = {}
    for name in ARM_JOINTS:
        c = calib[name]
        mid = (c["range_min"] + c["range_max"]) / 2
        out[name] = (
            (c["range_min"] - mid) * 360 / 4095,
            (c["range_max"] - mid) * 360 / 4095,
        )
    return out


class EEController:
    def __init__(self, gui: bool = True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(str(URDF.parent))
        self.robot = p.loadURDF(str(URDF), useFixedBase=True)

        self.idx, self.ee_idx = {}, None
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            name = info[1].decode()
            if info[2] != p.JOINT_FIXED:
                self.idx[name] = i
            if info[12].decode() == EE_LINK:
                self.ee_idx = i
        assert self.ee_idx is not None, f"{EE_LINK} not found in URDF"

        # Intersect URDF limits with calibration limits.
        lim = calib_limits_deg(json.loads(CALIB.read_text()))
        self.lower, self.upper = [], []
        for n in ARM_JOINTS:
            info = p.getJointInfo(self.robot, self.idx[n])
            self.lower.append(max(info[8], math.radians(lim[n][0])))
            self.upper.append(min(info[9], math.radians(lim[n][1])))
        g = p.getJointInfo(self.robot, self.idx["gripper"])
        self.g_lower, self.g_upper = g[8], g[9]
        self.ranges = [u - l for l, u in zip(self.lower, self.upper)]
        self.q = [0.0] * len(ARM_JOINTS)
        self._apply_sim(self.q)

    def _apply_sim(self, q):
        for n, v in zip(ARM_JOINTS, q):
            p.resetJointState(self.robot, self.idx[n], v)

    def set_sim_state(self, q):
        self.q = list(q)
        self._apply_sim(self.q)

    def ee_position(self) -> np.ndarray:
        return np.array(p.getLinkState(self.robot, self.ee_idx, computeForwardKinematics=True)[4])

    def solve(self, target, orientation=None, tol=2e-3):
        """Returns (joint angles in rad, error in m). Warm-started from the current state."""
        self._apply_sim(self.q)
        # calculateInverseKinematics returns all movable joints in URDF order; arm joints come first.
        kwargs = dict(
            lowerLimits=self.lower + [self.g_lower],
            upperLimits=self.upper + [self.g_upper],
            jointRanges=self.ranges + [self.g_upper - self.g_lower],
            restPoses=list(self.q) + [0.0],
            maxNumIterations=200,
            residualThreshold=1e-5,
        )
        if orientation is not None:
            kwargs["targetOrientation"] = orientation
        sol = None
        for _ in range(5):
            sol = p.calculateInverseKinematics(self.robot, self.ee_idx, list(target), **kwargs)
            self._apply_sim(sol[: len(ARM_JOINTS)])
            err = np.linalg.norm(self.ee_position() - target)
            kwargs["restPoses"] = list(sol[: len(ARM_JOINTS)]) + [0.0]
            if err < tol:
                break
        q = [float(np.clip(v, l, u)) for v, l, u in zip(sol[: len(ARM_JOINTS)], self.lower, self.upper)]
        self._apply_sim(q)
        return q, float(np.linalg.norm(self.ee_position() - target))


def connect_robot():
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    cfg = SOFollowerRobotConfig(
        port=PORT_FILE.read_text().strip(),
        id=CALIB.stem,
        calibration_dir=CALIB.parent,
        use_degrees=True,
        max_relative_target=15.0,  # deg per command, safety
    )
    robot = SOFollower(cfg)
    robot.connect(calibrate=False)
    return robot


def read_joints_rad(robot) -> list[float]:
    obs = robot.get_observation()
    return [math.radians(obs[f"{n}.pos"]) for n in ARM_JOINTS]


def send_joints(robot, q, gripper=None, steps=20, dt=0.03, tol_deg=3.0, timeout=None):
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
        action = {f"{n}.pos": math.degrees(v) for n, v in zip(ARM_JOINTS, cur)}
        action["gripper.pos"] = g0 + (g1 - g0) * s
        robot.send_action(action)
        time.sleep(dt)

    # Keep commanding the final goal until the joints get there (gripper may stall on an object).
    final = {f"{n}.pos": math.degrees(v) for n, v in zip(ARM_JOINTS, goal)}
    final["gripper.pos"] = g1
    t0 = time.time()
    while time.time() - t0 < timeout:
        robot.send_action(final)
        time.sleep(dt)
        obs = robot.get_observation()
        arm_err = max(abs(obs[k] - final[k]) for k in final if k != "gripper.pos")
        grip_err = abs(obs["gripper.pos"] - g1)
        if arm_err < tol_deg and (gripper is None or grip_err < tol_deg):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--orient-down", action="store_true", help="constrain gripper to point down")
    ap.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, -0.04], metavar=("X", "Y", "Z"),
                    help="meters added to every target, base frame")
    args = ap.parse_args()

    ctrl = EEController(gui=not args.no_gui)
    robot = None if args.sim_only else connect_robot()
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

            q, err = ctrl.solve(new + offset, orient)
            if err > 5e-3:
                print("Alvo fora do alcance, ignorado")
                ctrl.set_sim_state(read_joints_rad(robot) if robot else ctrl.q)
                continue
            ctrl.set_sim_state(q)
            target = new
            new = new + offset
            if robot:
                send_joints(robot, q, gripper)
                ctrl.set_sim_state(read_joints_rad(robot))
                final = ctrl.ee_position()
                # Sag/backlash leave a static offset; shift the IK aim point by the measured error.
                aim = new.copy()
                for _ in range(3):
                    if np.linalg.norm(final - new) < 4e-3:
                        break
                    aim = aim + (new - final)
                    q, _ = ctrl.solve(aim, orient)
                    send_joints(robot, q)
                    ctrl.set_sim_state(read_joints_rad(robot))
                    final = ctrl.ee_position()
                print(f"EE final: {np.round(final - offset, 3)} m | erro: {np.linalg.norm(final - new) * 1000:.1f} mm")
            else:
                print(f"EE final: {np.round(ctrl.ee_position() - offset, 3)} m | erro: {err * 1000:.1f} mm")
    finally:
        if robot:
            robot.disconnect()
        p.disconnect()


if __name__ == "__main__":
    main()
