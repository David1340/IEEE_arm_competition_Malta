"""Read and optionally set selected SO-101 arm joints in degrees.

Unspecified joints remain at their measured positions. Without --execute this
script connects read-only, displays the proposed pose, and sends no commands.

Examples:
    python set_so101_joint_positions.py --shoulder-pan 5
    python set_so101_joint_positions.py --shoulder-pan 5 --wrist-roll -90 --execute
    python set_so101_joint_positions.py --degrees 0 0 0 0 0
"""

from __future__ import annotations

import argparse
import json
import math


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--degrees", type=float, nargs=5, metavar=("PAN", "LIFT", "ELBOW", "WRIST", "ROLL"),
                              help="five target angles in degrees, in ARM_JOINTS order")
    target_group.add_argument("--shoulder-pan", type=float, default=None, metavar="DEG")
    target_group.add_argument("--shoulder-lift", type=float, default=None, metavar="DEG")
    target_group.add_argument("--elbow-flex", type=float, default=None, metavar="DEG")
    target_group.add_argument("--wrist-flex", type=float, default=None, metavar="DEG")
    target_group.add_argument("--wrist-roll", type=float, default=None, metavar="DEG")
    parser.add_argument("--steps", type=int, default=20, help="interpolation steps for the move")
    parser.add_argument("--tolerance-deg", type=float, default=3.0, help="joint arrival tolerance")
    parser.add_argument("--execute", action="store_true", help="send the move after typing MOVE")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    requested = dict(zip(joint_names, args.degrees)) if args.degrees is not None else {
        "shoulder_pan": args.shoulder_pan,
        "shoulder_lift": args.shoulder_lift,
        "elbow_flex": args.elbow_flex,
        "wrist_flex": args.wrist_flex,
        "wrist_roll": args.wrist_roll,
    }
    if all(value is None for value in requested.values()):
        parser.error("set at least one joint target")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.tolerance_deg <= 0 or not math.isfinite(args.tolerance_deg):
        parser.error("--tolerance-deg must be finite and positive")
    if any(value is not None and not math.isfinite(value) for value in requested.values()):
        parser.error("joint targets must be finite")

    from IK_control import CALIB, ARM_JOINTS, calib_limits_deg, connect_robot, read_joints_rad, send_joints

    robot = connect_robot()
    try:
        current_rad = read_joints_rad(robot)
        target_degrees = {name: math.degrees(value) for name, value in zip(ARM_JOINTS, current_rad)}
        target_degrees.update({name: value for name, value in requested.items() if value is not None})
        limits = calib_limits_deg(json.loads(CALIB.read_text()))

        print("Joint targets in degrees (unrequested joints hold their current position):")
        for name in ARM_JOINTS:
            target = target_degrees[name]
            lower, upper = limits[name]
            if not lower <= target <= upper:
                raise ValueError(f"{name} target {target:.2f} deg is outside calibrated limits "
                                 f"[{lower:.2f}, {upper:.2f}] deg")
            print(f"  {name:14s} current={math.degrees(current_rad[ARM_JOINTS.index(name)]):7.2f} "
                  f"target={target:7.2f} limits=[{lower:.2f}, {upper:.2f}]")

        target_rad = [math.radians(target_degrees[name]) for name in ARM_JOINTS]
        if not args.execute:
            print("Dry run: no motor commands sent. Add --execute to move the arm.")
            return 0
        if input("Type MOVE to send this joint target: ").strip().upper() != "MOVE":
            print("Cancelled; no motor commands sent.")
            return 1

        reached = send_joints(
            robot,
            target_rad,
            steps=args.steps,
            tol_deg=args.tolerance_deg,
        )
        measured_rad = read_joints_rad(robot)
        print("Measured result:")
        for name, target, measured in zip(ARM_JOINTS, target_rad, measured_rad):
            error_deg = math.degrees(measured - target)
            print(f"  {name:14s} measured={math.degrees(measured):7.2f} deg "
                  f"error={error_deg:+6.2f} deg")
        print("All targets reached." if reached else "Move did not reach tolerance before timeout.")
        return 0 if reached else 2
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())