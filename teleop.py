#!/usr/bin/env python3
"""Teleoperate the followers with their leader arms; the pairs and ports come from arms.json.

    python teleop.py                          # every pair, each in its own terminal window
    python teleop.py --here                   # every pair in this terminal (Ctrl+C stops all)
    python teleop.py --pair follower_blue     # only that follower (or name its leader)
    python teleop.py --print                  # only print the lerobot-teleoperate commands
    python teleop.py -- --display_data=true   # anything after -- is passed on to lerobot-teleoperate

With arms.json as it is: follower_pink <- leader_blue, follower_blue <- leader_green.
"""
import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from control import arm_config  # noqa: E402


def command(follower, leader, extra):
    return ["lerobot-teleoperate",
            f"--robot.type={arm_config.lerobot_type(follower)}", f"--robot.port={arm_config.port(follower)}",
            f"--robot.id={follower}",
            f"--teleop.type={arm_config.lerobot_type(leader)}", f"--teleop.port={arm_config.port(leader)}",
            f"--teleop.id={leader}", *extra]


def open_terminal(title, cmd, conda_env):
    """Run cmd in a new terminal window that stays open afterwards, so errors can be read."""
    line = shlex.join(cmd)
    inner = (f"conda activate {shlex.quote(conda_env)} 2>/dev/null || true; echo {shlex.quote(title)}; "
             f"echo {shlex.quote(line)}; {line}; echo; read -rp 'Finished. Press Enter to close...'")
    for terminal, args in (("gnome-terminal", [f"--title={title}", "--", "bash", "-ic", inner]),
                           ("konsole", ["-p", f"tabtitle={title}", "-e", "bash", "-ic", inner]),
                           ("xterm", ["-T", title, "-e", "bash", "-ic", inner]),
                           ("x-terminal-emulator", ["-e", "bash", "-ic", inner])):
        if shutil.which(terminal):
            subprocess.Popen([terminal, *args])
            return True
    return False


def run_here(jobs):
    """Run every pair in this terminal, each line prefixed with its pair; Ctrl+C stops them all."""
    procs = []
    for title, cmd in jobs:
        print(f"[{title}] {shlex.join(cmd)}")
        procs.append((title, subprocess.Popen(cmd)))
    try:
        while any(p.poll() is None for _, p in procs):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping ...")
        for _, p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for _, p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    for title, p in procs:
        print(f"[{title}] exited with {p.returncode}")


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", metavar="ARM", help="only these followers (or their leaders)")
    ap.add_argument("--here", action="store_true", help="run in this terminal instead of new windows")
    ap.add_argument("--print", action="store_true", help="only print the commands")
    ap.add_argument("--conda-env", default=os.environ.get("CONDA_ENV", "lerobot"),
                    help="conda env activated in the new windows (default: lerobot)")
    args = ap.parse_args(argv)

    pairs = arm_config.teleop_pairs(args.pair)
    if not pairs:
        raise SystemExit(f"No teleop pair matches {args.pair}; pairs in arms.json: {arm_config.teleop_pairs()}")
    jobs = [(f"{follower} <- {leader}", command(follower, leader, extra)) for follower, leader in pairs]
    if args.print:
        for title, cmd in jobs:
            print(f"# {title}\n{shlex.join(cmd)}\n")
        return

    missing = [(name, arm_config.port(name)) for pair in pairs for name in pair
               if not os.path.exists(arm_config.port(name))]
    if missing:
        for name, port in missing:
            print(f"{name}: {port} not found (USB unplugged or arm not set up?)")
        raise SystemExit("Check with:  python ports/setup_usb_ports.py --check")

    if args.here:
        run_here(jobs)
        return
    for title, cmd in jobs:
        if not open_terminal(title, cmd, args.conda_env):
            print("No terminal program found (gnome-terminal, konsole, xterm): running here instead.")
            run_here(jobs)
            return
        print(f"Started {title} in a new window")
        time.sleep(1)


if __name__ == "__main__":
    main()
