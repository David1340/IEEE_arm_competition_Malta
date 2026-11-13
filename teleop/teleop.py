#!/usr/bin/env python3
"""Teleoperate the followers with their leader arms; the pairs and ports come from config/arms.yaml.

    python 03_teleop.py                          # every pair, each in its own terminal window
    python 03_teleop.py --here                   # every pair in this terminal (Ctrl+C stops all)
    python 03_teleop.py --pair follower_blue     # only that follower (or name its leader)
    python 03_teleop.py --print                  # only print the lerobot-teleoperate commands
    python 03_teleop.py --gui                    # live telemetry: every joint's angle, current, load, temperature
                                              # (teleop_gui.py drives the arms itself; --gui --log saves a CSV)
    python 03_teleop.py -- --display_data=true   # anything after -- is passed on to lerobot-teleoperate

With config/arms.yaml as it is: follower_pink <- leader_blue, follower_blue <- leader_green.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import arms as arm_config  # noqa: E402


def command(follower, leader, extra):
    return ["lerobot-teleoperate",
            f"--robot.type={arm_config.lerobot_type(follower)}", f"--robot.port={arm_config.port(follower)}",
            f"--robot.id={follower}",
            f"--teleop.type={arm_config.lerobot_type(leader)}", f"--teleop.port={arm_config.port(leader)}",
            f"--teleop.id={leader}", *extra]


# Variables a snap app (e.g. VS Code installed as a snap) sets in its terminal; inherited, they make gnome-terminal
# load the snap's libraries and crash ("symbol lookup error ... /snap/core20/.../libpthread.so.0").
SNAP_VARIABLES = ("GTK_PATH", "GTK_EXE_PREFIX", "GTK_IM_MODULE_FILE", "GIO_MODULE_DIR", "GSETTINGS_SCHEMA_DIR",
                  "GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR", "LOCPATH", "LD_LIBRARY_PATH", "LD_PRELOAD")


def clean_environment():
    """This process's environment without what a snap app put in it."""
    env = dict(os.environ)
    for name in list(env):
        if name in SNAP_VARIABLES or (name.startswith("SNAP") and name != "SNAP_NAME"):
            env.pop(name)
        elif name != "PATH" and "/snap/" in env[name]:
            env.pop(name)
    if "XDG_DATA_DIRS" in os.environ:   # keep the system entries, drop the snap's
        dirs = [d for d in os.environ["XDG_DATA_DIRS"].split(":") if d and "/snap/" not in d]
        env["XDG_DATA_DIRS"] = ":".join(dirs) or "/usr/local/share:/usr/share"
    return env


def open_terminal(title, cmd, conda_env):
    """Run cmd in a new terminal window that stays open afterwards, so errors can be read. Returns True only when
    the terminal program really started (it is watched for a moment: a crash on start is caught)."""
    line = shlex.join(cmd)
    inner = (f"conda activate {shlex.quote(conda_env)} 2>/dev/null || true; echo {shlex.quote(title)}; "
             f"echo {shlex.quote(line)}; {line}; echo; read -rp 'Finished. Press Enter to close...'")
    env = clean_environment()
    for terminal, args in (("gnome-terminal", [f"--title={title}", "--", "bash", "-ic", inner]),
                           ("konsole", ["-p", f"tabtitle={title}", "-e", "bash", "-ic", inner]),
                           ("xterm", ["-T", title, "-e", "bash", "-ic", inner]),
                           ("x-terminal-emulator", ["-e", "bash", "-ic", inner])):
        if not shutil.which(terminal):
            continue
        proc = subprocess.Popen([terminal, *args], env=env, stderr=subprocess.PIPE, text=True)
        try:
            proc.wait(timeout=2.0)      # gnome-terminal hands the window to its server and exits 0 quickly
        except subprocess.TimeoutExpired:
            return True                 # still running (konsole, xterm): the window is open
        if proc.returncode == 0:
            return True
        error = (proc.stderr.read() or "").strip().splitlines()
        print(f"  {terminal} failed to start: {error[-1] if error else f'exit code {proc.returncode}'}")
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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", metavar="ARM", help="only these followers (or their leaders)")
    ap.add_argument("--here", action="store_true", help="run in this terminal instead of new windows")
    ap.add_argument("--print", action="store_true", help="only print the commands")
    ap.add_argument("--gui", action="store_true",
                    help="teleoperate with a live telemetry window (teleop_gui.py) instead of lerobot-teleoperate")
    from teleop import teleop_gui
    teleop_gui.add_args(ap)
    ap.add_argument("--conda-env", default=os.environ.get("CONDA_ENV", "lerobot"),
                    help="conda env activated in the new windows (default: lerobot)")
    args = ap.parse_args(argv)

    pairs = arm_config.teleop_pairs(args.pair)
    if not pairs:
        raise SystemExit(f"No teleop pair matches {args.pair}; pairs in config/arms.yaml: {arm_config.teleop_pairs()}")
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
        raise SystemExit("Check with:  python 01_setup_usb.py --check   (names missing but the boards "
                         "found: python 01_setup_usb.py --install-saved)")
    if args.gui:
        teleop_gui.run(pairs, args)
        return

    if args.here:
        run_here(jobs)
        return
    for index, (title, cmd) in enumerate(jobs):
        if not open_terminal(title, cmd, args.conda_env):
            rest = jobs[index:]
            print("Could not open a terminal window: running " + ", ".join(t for t, _ in rest) + " here instead "
                  "(Ctrl+C stops it).")
            run_here(rest)
            return
        print(f"Started {title} in a new window")
        time.sleep(1)


if __name__ == "__main__":
    main()
