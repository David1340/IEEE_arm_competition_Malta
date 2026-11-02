"""Does the arm's model match the real arm? Torque OFF: move the arm by hand and compare.

Prints, 5 times a second, each joint's LeRobot angle, the angle the model uses (+ fine offset) and where the
model thinks the gripper tip is (robot base frame, cm). Nothing moves; the torque is switched off, so HOLD THE
ARM if it is not resting on something.

Checks to do (tip = the end of the fixed jaw):
    1. Tip touching the table 25 cm straight in front of the base   -> should read  x 25  y 0   z 0
    2. Tip touching the table 35 cm in front                         -> x 35  y 0   z 0
    3. Tip touching the table 30 cm in front, 15 cm to the LEFT      -> x 30  y 15  z 0
    4. Upper arm straight up, forearm horizontal pointing forward     -> lift and elbow near 0 (URDF zero pose)
If z reads several cm above 0 while the tip is on the table, the model thinks the arm is higher than it is:
that is what drives the arm into the table. Send the printouts.

Usage:
    python robot/check_calibration.py
    python robot/check_calibration.py --offsets none        # without the fine joint offsets
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go_to_object import ARM, CALIBRATION, FINE_OFFSETS, fine_offsets  # noqa: E402


SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "wrist",
         "wrist_roll": "roll"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", default=str(CALIBRATION))
    ap.add_argument("--offsets", default=str(FINE_OFFSETS), help="fine joint offsets JSON, or 'none'")
    ap.add_argument("--port", default=None, help="default: found by so101ik (or SO101_PORT); 'sim' for no arm")
    args = ap.parse_args()

    from so101ik import Arm

    offsets = {j: 0.0 for j in ARM} if args.offsets == "none" else fine_offsets(args.offsets)
    cal = Path(args.calibration).expanduser()
    arm = Arm(port=args.port, calibration_file=cal if cal.is_file() else None, offsets=offsets,
              name="check_calibration")
    try:
        arm.check()
        arm.torque_off()
        print("Torque OFF: move the arm by hand. Ctrl-C to stop.")
        print("fine offsets: " + "  ".join(f"{j} {offsets[j]:+.2f}" for j in ARM))
        print("joints: LeRobot deg -> model deg (LeRobot + offset)")
        while True:
            deg = arm.read()[0]
            x, y, z, pitch = arm.where()
            joints = "  ".join(f"{SHORT[j]} {deg[j]:+6.1f}>{deg[j] + offsets[j]:+6.1f}" for j in ARM)
            print(f"TIP x {100 * x:5.1f} y {100 * y:5.1f} z {100 * z:5.1f} cm  pitch {pitch:+5.0f} | {joints}", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()


if __name__ == "__main__":
    main()
