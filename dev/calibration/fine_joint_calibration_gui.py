"""Fine-tune SO-101 software joint offsets while hand-guiding the arm.

Torque is disabled after connection. PyBullet shows measured joint angles plus
the live slider offsets. Press s to save; q/Escape discards unsaved changes.

Usage:
    python dev/calibration/fine_joint_calibration_gui.py
    python dev/calibration/fine_joint_calibration_gui.py --arm follower_blue     # port from arms.json,
                                                                             # saves that arm's own offsets
    python dev/calibration/fine_joint_calibration_gui.py --port /dev/ttyACM2     # any port (with or without --arm)
    python dev/calibration/fine_joint_calibration_gui.py --no-robot
    python dev/calibration/fine_joint_calibration_gui.py --reset-offsets
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.joint_offsets import (  # noqa: E402
    ARM_JOINTS,
    load_fine_joint_offsets,
    own_offsets_file,
    save_fine_joint_offsets,
)
from hardware.arm import URDF, connect_robot  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-robot", action="store_true", help="adjust offsets against a zero-pose preview only")
    parser.add_argument("--arm", default=None, help="which arm (follower_pink, follower_blue): its port file "
                        "port and LeRobot calibration from arms.json, and its own offsets file")
    parser.add_argument("--port", default=None, help="serial port, e.g. /dev/ttyACM2 (default: the arm's port file)")
    parser.add_argument("--offset-range-deg", type=float, default=10.0,
                        help="default +/- range for each PyBullet fine-offset slider")
    parser.add_argument("--reset-offsets", action="store_true", help="reset all saved offsets to zero")
    args = parser.parse_args(argv)
    if not math.isfinite(args.offset_range_deg) or args.offset_range_deg <= 0:
        parser.error("--offset-range-deg must be finite and positive")

    offsets_file = own_offsets_file(args.arm)
    offsets = load_fine_joint_offsets(arm=args.arm)
    print(f"Fine offsets of {args.arm or 'the arm'}: saved to {offsets_file}")
    if args.reset_offsets:
        if input("Type RESET to set every fine offset to 0 degrees: ").strip().upper() != "RESET":
            print("Cancelled; saved fine offsets were not changed.")
            return 1
        zero_offsets = {name: 0.0 for name in ARM_JOINTS}
        save_fine_joint_offsets(zero_offsets, offsets_file)
        print("All saved fine offsets reset to 0 degrees.")
        return 0

    robot = None
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not open the PyBullet calibration GUI")
    try:
        if not args.no_robot:
            robot = connect_robot(args.arm, port=args.port)
            robot.bus.disable_torque()
            print("SO-101 torque disabled. Move the arm by hand; no motor commands will be sent.")
        else:
            print("Offline preview: using zero arm pose; robot was not connected.")

        p.setAdditionalSearchPath(str(URDF.parent), physicsClientId=client)
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setRealTimeSimulation(0, physicsClientId=client)
        robot_id = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=client)

        slider_ids = {}
        for joint_name in ARM_JOINTS:
            slider_range = max(args.offset_range_deg, abs(offsets[joint_name]) + 0.5)
            slider_ids[joint_name] = p.addUserDebugParameter(
                f"{joint_name} fine offset (deg)",
                -slider_range,
                slider_range,
                offsets[joint_name],
                physicsClientId=client,
            )

        table_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.75, 0.9, 0.02],
            rgbaColor=[0.55, 0.57, 0.60, 1],
            physicsClientId=client,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=table_shape,
            basePosition=[0, 0.25, -0.02],
            physicsClientId=client,
        )
        p.resetDebugVisualizerCamera(1.65, 45, -45, [0.15, 0.25, 0.20], physicsClientId=client)

        joint_indices = {
            p.getJointInfo(robot_id, index, physicsClientId=client)[1].decode(): index
            for index in range(p.getNumJoints(robot_id, physicsClientId=client))
        }
        gripper_info = p.getJointInfo(robot_id, joint_indices["gripper"], physicsClientId=client)
        status_id = -1
        saved_offsets = offsets.copy()
        print("Adjust sliders: s saves, q/Esc cancels unsaved changes and exits.")

        while p.isConnected(client):
            if robot is None:
                raw_positions = {name: 0.0 for name in ARM_JOINTS}
                gripper_position = 50.0
            else:
                observation = robot.get_observation()
                raw_positions = {name: float(observation[f"{name}.pos"]) for name in ARM_JOINTS}
                gripper_position = float(observation["gripper.pos"])

            offsets = {
                name: float(p.readUserDebugParameter(slider_ids[name], physicsClientId=client))
                for name in ARM_JOINTS
            }
            for name in ARM_JOINTS:
                joint_index = joint_indices[name]
                corrected_radians = math.radians(raw_positions[name] + offsets[name])
                p.resetJointState(robot_id, joint_index, corrected_radians, physicsClientId=client)

            gripper_angle = gripper_info[8] + np.clip(gripper_position / 100.0, 0.0, 1.0) * (
                gripper_info[9] - gripper_info[8]
            )
            p.resetJointState(robot_id, joint_indices["gripper"], float(gripper_angle), physicsClientId=client)
            end_effector = p.getLinkState(
                robot_id,
                next(index for index in range(p.getNumJoints(robot_id, physicsClientId=client))
                     if p.getJointInfo(robot_id, index, physicsClientId=client)[12].decode() == "gripper_frame_link"),
                computeForwardKinematics=True,
                physicsClientId=client,
            )[4]
            status = (
                "Fine offsets (deg): " + ", ".join(f"{name}={offsets[name]:+.2f}" for name in ARM_JOINTS)
                + f" | EE={np.round(end_effector, 3)} m | "
                + ("UNSAVED" if not np.allclose(
                    [offsets[name] for name in ARM_JOINTS],
                    [saved_offsets[name] for name in ARM_JOINTS],
                    atol=0.01,
                ) else "SAVED")
                + f" | {offsets_file.name}"
            )
            status_id = p.addUserDebugText(
                status,
                [0.0, 0.0, 0.75],
                textColorRGB=[0.05, 0.05, 0.05],
                textSize=1.1,
                lifeTime=0,
                replaceItemUniqueId=status_id,
                physicsClientId=client,
            )
            p.stepSimulation(physicsClientId=client)

            keys = p.getKeyboardEvents(physicsClientId=client)
            if ord("s") in keys and keys[ord("s")] & p.KEY_WAS_TRIGGERED:
                save_fine_joint_offsets(offsets, offsets_file)
                saved_offsets = offsets.copy()
                print("Saved fine offsets (deg):", {name: round(offsets[name], 2) for name in ARM_JOINTS})
            if (ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED) or (
                27 in keys and keys[27] & p.KEY_WAS_TRIGGERED
            ):
                if not np.allclose(
                    [offsets[name] for name in ARM_JOINTS],
                    [saved_offsets[name] for name in ARM_JOINTS],
                    atol=0.01,
                ):
                    print("Cancelled; unsaved slider changes were discarded.")
                break
            time.sleep(0.02)
    finally:
        if robot is not None:
            robot.disconnect()
        if p.isConnected(client):
            p.disconnect(physicsClientId=client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())