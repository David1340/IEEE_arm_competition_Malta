#!/usr/bin/env python3
"""Step 2: calibrate the arms and the camera. Every result goes to config/ (arms.yaml, calibration/).

    python 02_calibrate.py status                       # what is calibrated, per arm and for the camera

  Motors (LeRobot: each joint's range and middle; once per arm, again only after re-assembling a joint)
    python 02_calibrate.py motors --arm follower_pink   # runs lerobot-calibrate for that arm (any of the four, or
                                                        # all), then copies the result into the repo (sync pull)
    python 02_calibrate.py sync status|pull|push        # LeRobot's own calibration files <-> config/calibration/
                                                        # (push on a new computer after git pull)

  Fine joint offsets (small software corrections on top of the motor calibration; followers only)
    python 02_calibrate.py offsets --arm follower_pink  # PyBullet sliders over the live arm; s saves
    python 02_calibrate.py offsets --arm both --objects # both arms in one window, with the camera's objects

  Camera and arm placement (the TABLE frame; see calibration/calibrate.py)
    python 02_calibrate.py camera tape                  # the camera from the black tape (no markers needed)
    python 02_calibrate.py camera aruco                 # the camera from the ArUco markers
    python 02_calibrate.py camera ids                   # live view: which marker has which ID
    python 02_calibrate.py camera touch --arm both      # each arm's base x, y, yaw from its fingertip on the tape
    python 02_calibrate.py camera show                  # print the camera calibration

  Moving from the old layout (calibration.json at the repo root)
    python 02_calibrate.py import-old [path/to/calibration.json]

Order on a new setup: 01_setup_usb.py, then motors (every arm), then offsets (followers), then camera tape (or
aruco), then camera touch. Then commit config/.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def motors(argv):
    from core import arms as arm_config

    ap = argparse.ArgumentParser(prog="02_calibrate.py motors",
                                 description="LeRobot motor calibration of one arm (or all), then sync pull.")
    ap.add_argument("--arm", required=True, help=f"one of {', '.join(arm_config.names())}, or all")
    ap.add_argument("--print", action="store_true", help="only print the lerobot-calibrate command(s)")
    args = ap.parse_args(argv)
    names = arm_config.names() if args.arm == "all" else [args.arm]
    for name in names:
        arm_config.entry(name)                                  # unknown name: a clear error
        kind = "robot" if arm_config.role(name) == "follower" else "teleop"
        command = ["lerobot-calibrate", f"--{kind}.type={arm_config.lerobot_type(name)}",
                   f"--{kind}.port={arm_config.port(name)}", f"--{kind}.id={name}"]
        print(f"\n# {name}\n" + " ".join(command))
        if args.print:
            continue
        if not shutil.which("lerobot-calibrate"):
            raise SystemExit("lerobot-calibrate not found: activate the LeRobot environment (conda activate lerobot)")
        if not Path(arm_config.port(name)).exists():
            raise SystemExit(f"{arm_config.port(name)} not found: check the USB (python 01_setup_usb.py --check)")
        if subprocess.run(command).returncode != 0:
            raise SystemExit(f"lerobot-calibrate failed for {name}")
        from calibration.lerobot_sync import pull

        pull([name])


def offsets(argv):
    from calibration.calibrate import main

    main(["offsets", *argv])


def camera(argv):
    from calibration.calibrate import main

    if not argv or argv[0].startswith("-"):
        raise SystemExit("python 02_calibrate.py camera <tape|aruco|ids|touch|check|arm|show> [options]  "
                         "(--help after the mode for its options)")
    main(argv)


def sync(argv):
    from calibration.lerobot_sync import main

    main(argv)


def status(argv):
    from calibration import calibrate, lerobot_sync
    from core import arms as arm_config
    from hardware.joint_offsets import load_fine_joint_offsets, own_offsets_file

    print("Motor calibration (LeRobot) and its copy in the repo:")
    lerobot_sync.status(arm_config.names())
    print("\nFine joint offsets (deg):")
    for name in arm_config.followers():
        values = load_fine_joint_offsets(arm=name)
        source = own_offsets_file(name)
        print(f"  {name:14s} " + ("  ".join(f"{j} {v:+.2f}" for j, v in values.items())
                                  if source.is_file() else "(none: python 02_calibrate.py offsets --arm " + name + ")"))
    print("\nBases in the TABLE frame (config/arms.yaml):")
    for name, b in arm_config.bases().items():
        print(f"  {name:14s} x {b['x']:+.4f}  y {b['y']:+.4f}  yaw {b['yaw_deg']:+.2f} deg")
    data = calibrate.load()
    camera_info = data.get("camera", {})
    print("\nCamera: " + ("calibrated" + (f" ({camera_info.get('method', '?')}, {camera_info.get('time', '?')})"
                                         if camera_info else "") if data.get("camera_to_table") is not None
                         else "NOT calibrated (python 02_calibrate.py camera tape)"))


def import_old(argv):
    """The old layout's calibration.json (repo root) -> config/calibration/camera.yaml."""
    import json

    from calibration import calibrate

    source = Path(argv[0]) if argv else ROOT / "calibration.json"
    if not source.is_file():
        raise SystemExit(f"{source} not found (give its path: python 02_calibrate.py import-old path/to/calibration.json)")
    data = json.loads(source.read_text())
    if calibrate.FILE.is_file() and input(f"{calibrate.FILE} exists; replace it? [y/N] ").strip().lower() != "y":
        return
    calibrate.save(data)
    print(f"{source} -> {calibrate.FILE} (and {calibrate.LEGACY_CAMERA.name}). The fine joint offsets of the old "
          "layout are already in config/calibration/<arm>/joint_offsets.yaml. You can delete the old file.")


COMMANDS = {"status": status, "motors": motors, "sync": sync, "offsets": offsets, "camera": camera,
            "import-old": import_old}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in COMMANDS:
        print(__doc__)
        raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else
                         f"unknown command {sys.argv[1]!r}; one of: {', '.join(COMMANDS)}")
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
