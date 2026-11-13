"""Game 2, two-arm coordination: both arms together on multi-step tasks (rules section 5.8, GAME 2).

    python run_game.py two_arm_coordination --round pouring         # Round 1: rice from a container into a bowl
    python run_game.py two_arm_coordination --round folding         # Round 2: fold three cloths, place them
    python run_game.py two_arm_coordination --round wrap_and_pack   # Round 3: wrap the mascot, box it, close the box

NOT IMPLEMENTED YET: this is the skeleton the game is to be built in, like games/pick_and_place.py:
    config/games/two_arm_coordination.yaml   the rounds and their settings
    perception/        find the objects (the pouring YOLO dataset tools are in dev/dataset_tools/yolo_pouring/)
    planning/          IK and collision-free paths for each arm (planning/pick_planner.py, planning/sim.py)
    state_machine/     the steps of each round as behaviour trees (state_machine/behaviour.py); both arms
                       coordinated (state_machine/coordinator.py)
    hardware/          connect the arms (hardware/arm.py), the gripper (hardware/gripper.py)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games import rules  # noqa: E402

GAME = "two_arm_coordination"


def main(argv=None):
    settings = rules.load(GAME)
    rounds = settings.get("rounds", {})
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", choices=list(rounds), default=settings.get("round", next(iter(rounds), None)))
    ap.add_argument("--no-robot", action="store_true", help="no arms connected")
    args = ap.parse_args(argv)
    round_ = rounds.get(args.round, {})
    print(f"Game 2, round {args.round}: {round_.get('description', '')}")
    raise SystemExit(f"{GAME} is not implemented yet (see games/two_arm_coordination.py for where it goes)")


if __name__ == "__main__":
    main()
