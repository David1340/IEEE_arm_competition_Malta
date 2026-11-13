"""Copy each arm's LeRobot motor calibration between LeRobot's own folder and the repo.

LeRobot reads and writes an arm's motor calibration (each motor's id, homing offset and range) in its own folder,
e.g. ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_pink.json (config/arms.yaml "lerobot").
That folder is per computer and not versioned. The repo keeps a copy of every arm's calibration in
config/calibration/<arm>/motors.yaml, so it is versioned with the rest of the setup and moves to a new computer
with `git pull`.

    python 02_calibrate.py sync status            # each arm: LeRobot's file, the repo copy, and whether they match
    python 02_calibrate.py sync pull              # LeRobot -> repo   (after calibrating; then commit config/)
    python 02_calibrate.py sync push              # repo -> LeRobot   (on a new computer, after git pull)
    python 02_calibrate.py sync pull --arm follower_pink leader_blue     # only these arms
    python 02_calibrate.py sync push --force      # overwrite LeRobot's files even when they differ

push keeps a backup of every LeRobot file it replaces (<name>.json.bak).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import arms as arm_config  # noqa: E402
from core import config  # noqa: E402

COMMENT = """LeRobot motor calibration of {arm} (each motor's id, drive mode, homing offset and range).
A copy of LeRobot's own file {lerobot}; LeRobot reads that one, not this.
python 02_calibrate.py sync pull   copies LeRobot's file here (after `python 02_calibrate.py motors`)
python 02_calibrate.py sync push   copies this back to LeRobot's folder (e.g. on a new computer)"""


def lerobot_file(arm: str) -> Path:
    """LeRobot's calibration file for the arm. When it is not where config/arms.yaml says, LeRobot's whole
    calibration folder is searched for <arm>.json (its folder names changed between LeRobot versions)."""
    path = arm_config.lerobot_calibration(arm)
    if path.is_file():
        return path
    root = arm_config.lerobot_calibration_dir(arm).parents[1]       # .../lerobot/calibration
    found = sorted(root.glob(f"**/{arm}.json")) if root.is_dir() else []
    return found[0] if len(found) == 1 else path


def read_lerobot(path: Path) -> dict:
    return json.loads(path.read_text())


def read_copy(arm: str) -> dict | None:
    return config.load_copy(arm_config.calibration_file(arm), None, required=False)


def status(arms: list[str]) -> None:
    print(f"{'arm':14s} {'LeRobot file':62s} repo copy")
    for arm in arms:
        source = lerobot_file(arm)
        copy = read_copy(arm)
        here = source.is_file()
        if here and copy is not None:
            state = "same" if read_lerobot(source) == copy else "DIFFERENT"
        else:
            state = "missing" if copy is None else "only in repo (push it)"
            if here and copy is None:
                state = "only in LeRobot (pull it)"
        print(f"{arm:14s} {str(source) if here else '(none) ' + str(source):62s} {state}")


def pull(arms: list[str]) -> None:
    """LeRobot -> repo."""
    for arm in arms:
        source = lerobot_file(arm)
        if not source.is_file():
            print(f"{arm}: no LeRobot calibration at {source}; calibrate it first (python 02_calibrate.py motors)")
            continue
        data = read_lerobot(source)
        target = arm_config.calibration_file(arm)
        old = read_copy(arm)
        if old == data:
            print(f"{arm}: repo copy already the same")
            continue
        config.save(target, data, comment=COMMENT.format(arm=arm, lerobot=source))
        print(f"{arm}: {source} -> {target.relative_to(ROOT)}" + ("" if old is None else " (updated)"))
    print("Commit config/calibration/ to keep it.")


def push(arms: list[str], force: bool = False) -> None:
    """repo -> LeRobot."""
    for arm in arms:
        data = read_copy(arm)
        if data is None:
            print(f"{arm}: no repo copy ({arm_config.calibration_file(arm).relative_to(ROOT)}); pull it on the "
                  "computer where the arm was calibrated")
            continue
        target = arm_config.lerobot_calibration(arm)
        if target.is_file():
            current = read_lerobot(target)
            if current == data:
                print(f"{arm}: LeRobot's file already the same")
                continue
            if not force:
                print(f"{arm}: {target} differs from the repo copy; left as it is (--force replaces it, keeping a "
                      ".bak; or pull to keep LeRobot's)")
                continue
            shutil.copy2(target, target.with_name(target.name + ".bak"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=4) + "\n")
        print(f"{arm}: {arm_config.calibration_file(arm).relative_to(ROOT)} -> {target}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="status", choices=["status", "pull", "push"])
    ap.add_argument("--arm", nargs="+", default=None, help="only these arms (default: all four)")
    ap.add_argument("--force", action="store_true", help="push: replace LeRobot files that differ (a .bak is kept)")
    args = ap.parse_args(argv)
    arms = args.arm or arm_config.names()
    unknown = [a for a in arms if a not in arm_config.names()]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {', '.join(arm_config.names())}")
    if args.action == "status":
        status(arms)
    elif args.action == "pull":
        pull(arms)
    else:
        push(arms, args.force)


if __name__ == "__main__":
    main()
