#!/usr/bin/env python3
"""Step 3: teleoperate each follower with its leader (the pairs and ports come from config/arms.yaml).

    python 03_teleop.py                       # every pair, each in its own terminal window (lerobot-teleoperate)
    python 03_teleop.py --here                # every pair in this terminal (Ctrl+C stops all)
    python 03_teleop.py --pair follower_blue  # only that follower (or name its leader)
    python 03_teleop.py --gui                 # live telemetry: every joint's angle, current, load, temperature
    python 03_teleop.py --print               # only print the lerobot-teleoperate commands
    python 03_teleop.py net receive           # over the network: on the computer with the FOLLOWERS (first)
    python 03_teleop.py net send              # over the network: on the computer with the LEADERS
                                              # (addresses in config/teleop.yaml)

Details: teleop/teleop.py, teleop/teleop_gui.py, teleop/teleop_net.py.
"""
import sys


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["net"]:
        from teleop.teleop_net import main as net_main

        sys.argv = [sys.argv[0] + " net", *argv[1:]]
        net_main(argv[1:])
    else:
        from teleop.teleop import main as teleop_main

        teleop_main(argv)


if __name__ == "__main__":
    main()
