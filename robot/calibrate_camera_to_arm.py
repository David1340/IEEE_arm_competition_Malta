"""Make the camera and the arm agree on where things are: TABLE frame (camera) -> robot base frame (arm).

For each of 3 to 5 positions:
    1. you put ONE block (a cube is best) somewhere new in the arm's reach, hands away, Enter;
    2. the camera finds it (the most confident object, top-face centre);
    3. the arm's torque is OFF: you put the TIP OF THE FIXED (long) FINGER on the centre of the block's top
       face by hand, hold it there, Enter.
Then the rotation about z and the shift (x, y, z) that best map the camera's points onto the arm's are fitted
and saved to robot/table_to_base.json, which robot/detect.py (and so go_to_object.py) uses from then on.
Spread the positions out: left, right, near, far.

Usage:
    python robot/calibrate_camera_to_arm.py            # 4 positions
    python robot/calibrate_camera_to_arm.py --points 5
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect  # noqa: E402
from go_to_object import ARM, CALIBRATION, FINE_OFFSETS, fine_offsets  # noqa: E402

IDENTITY = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}


def fit(cam, arm):
    """Rotation about z and shift that map the camera's points onto the arm's (least squares, 2D for x, y;
    the mean difference for z). Returns the transform dict and the residuals (m)."""
    cam, arm = np.asarray(cam, float), np.asarray(arm, float)
    if len(cam) >= 2:
        a, b = cam[:, :2] - cam[:, :2].mean(0), arm[:, :2] - arm[:, :2].mean(0)
        yaw = math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    else:
        yaw = 0.0
    t = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": math.degrees(yaw)}
    rotated = np.array([detect.table_to_base(p, t) for p in cam])
    shift = (arm - rotated).mean(0)
    t.update(x=float(shift[0]), y=float(shift[1]), z=float(shift[2]))
    residual = arm - np.array([detect.table_to_base(p, t) for p in cam])
    return t, residual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--calibration", default=str(CALIBRATION))
    ap.add_argument("--offsets", default=str(FINE_OFFSETS))
    ap.add_argument("--no-z", action="store_true", help="fit x, y and the rotation only; leave z at 0")
    args = ap.parse_args()

    from so101ik import Arm

    offsets = {j: 0.0 for j in ARM} if args.offsets == "none" else fine_offsets(args.offsets)
    cal = Path(args.calibration).expanduser()
    arm = Arm(calibration_file=cal if cal.is_file() else None, offsets=offsets, name="calibrate_camera_to_arm")
    cam_pts, arm_pts = [], []
    try:
        arm.check()
        arm.torque_off()
        print("Torque OFF (hold the arm when you move it).")
        while len(cam_pts) < args.points:
            n = len(cam_pts) + 1
            input(f"\n[{n}/{args.points}] Put ONE block somewhere new in reach (only it on the table near the arm), "
                  "move the arm and your hands out of the camera's view, then Enter ... ")
            dets, _ = detect.detect_live(args.frames, transform=IDENTITY)
            dets = [d for d in dets if d.name not in detect.SKIP_CLASSES]
            if not dets:
                print("  the camera found no block; try again")
                continue
            d = dets[0]
            print(f"  camera: {d.name} conf {d.conf:.2f} at x {100 * d.table_xyz[0]:.1f} y {100 * d.table_xyz[1]:.1f} "
                  f"top z {100 * d.table_xyz[2]:.1f} cm")
            input("  Now put the TIP OF THE FIXED FINGER on the centre of its top face, hold it still, Enter ... ")
            x, y, z, _ = arm.where()
            print(f"  arm:    tip at x {100 * x:.1f} y {100 * y:.1f} z {100 * z:.1f} cm")
            cam_pts.append(d.table_xyz)
            arm_pts.append([x, y, z])
    except KeyboardInterrupt:
        print()
    finally:
        arm.close()

    if not cam_pts:
        return
    t, residual = fit(cam_pts, arm_pts)
    if args.no_z:
        t["z"] = 0.0
    print("\npoint   camera (cm)             arm (cm)                after the fit, off by (mm)")
    for c, a, r in zip(cam_pts, arm_pts, residual):
        print(f"  {np.round(np.array(c) * 100, 1)}   {np.round(np.array(a) * 100, 1)}   "
              f"xy {1000 * np.hypot(*r[:2]):.0f}  z {1000 * r[2]:+.0f}")
    print(f"\nfit: rotate {t['yaw_deg']:+.2f} deg, then shift x {100 * t['x']:+.1f} y {100 * t['y']:+.1f} "
          f"z {100 * t['z']:+.1f} cm")
    if len(cam_pts) < 2:
        print("Only one point: shift only, no rotation. Use 3 or more for a proper fit.")
    if input(f"Save to {detect.TRANSFORM_FILE}? [Y/n] ").strip().lower() in ("", "y", "yes"):
        detect.TRANSFORM_FILE.write_text(json.dumps(
            {**t, "points": [{"camera": list(map(float, c)), "arm": list(map(float, a))}
                             for c, a in zip(cam_pts, arm_pts)]}, indent=2) + "\n")
        print("Saved: detect.py and go_to_object.py use it from now on.")


if __name__ == "__main__":
    main()
