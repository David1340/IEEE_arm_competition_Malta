#!/usr/bin/env python3
"""Step 4: play a game. Each game is one script in games/, built from the shared parts.

    python run_game.py                                   # list the games
    python run_game.py pick_and_place                    # Game 1 (its options: python run_game.py pick_and_place -h)
    python run_game.py pick_and_place --round shape --arm follower_pink
    python run_game.py two_arm_coordination --round pouring      # Game 2 (not implemented yet)
"""
import importlib
import sys

from games import GAMES


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Games: " + ", ".join(GAMES))
        return
    game = sys.argv[1]
    if game not in GAMES:
        raise SystemExit(f"unknown game {game!r}; one of: {', '.join(GAMES)}")
    sys.argv = [f"{sys.argv[0]} {game}", *sys.argv[2:]]
    importlib.import_module(GAMES[game]).main(sys.argv[1:])


if __name__ == "__main__":
    main()
