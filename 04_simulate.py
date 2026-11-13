#!/usr/bin/env python3
"""Simulate a game round in PyBullet: no arms, no camera. Place objects and bins, then watch both arms plan and
move with the same code as the real game (planner, collision avoidance, two-arm coordination).

    python 04_simulate.py edit example              # place objects and bins from above (simulation/scenes/example.yaml)
    python 04_simulate.py edit my_test --random 8   # a new scene with 8 random objects; Enter saves and runs it
    python 04_simulate.py run example               # play the round in the PyBullet window
    python 04_simulate.py run example --round shape --speed 2
    python 04_simulate.py run example --headless    # no window: plan, move and check only
    python 04_simulate.py list                      # the saved scenes

`run` takes every game option (python 04_simulate.py run -h). Details: simulation/run.py, simulation/editor.py.
"""
import argparse
import sys


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if command == "run":
        from simulation.run import main as run

        run(rest)
    elif command == "edit":
        from games.pick_and_place import add_game_args
        from simulation.editor import __doc__ as editor_doc, edit
        from simulation.run import main as run

        ap = argparse.ArgumentParser(prog="04_simulate.py edit", description=editor_doc,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("scene", nargs="?", default="new")
        ap.add_argument("--random", type=int, default=None, metavar="N", help="start from N random objects")
        add_game_args(ap)                                    # --max-reach, --box-size, ... drawn as the game uses them
        args, run_options = ap.parse_known_args(rest)
        run_after, path = edit(args.scene, args)
        if run_after:
            run([str(path), *run_options])
    elif command == "list":
        from simulation.scene import SCENES, available

        print("\n".join(f"{name}  ({SCENES / name}.yaml)" for name in available()) or "no scenes yet")
    else:
        print(__doc__)
        if command not in ("", "-h", "--help"):
            raise SystemExit(f"unknown command {command!r}: edit, run or list")


if __name__ == "__main__":
    main()
