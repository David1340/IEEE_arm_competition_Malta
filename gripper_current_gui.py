"""Live graph of the SO-101 gripper's current, load and position.

Standalone: only needs LeRobot, matplotlib and the arm's calibration file. Use it to see whether
closing on a block looks different from closing on nothing. Only the gripper moves; the other
joints hold the pose they had when the script started.

Controls:
    Open button  / o key   open the gripper
    Close button / c key   close the gripper; stops squeezing once a block is detected (see below)
    Stop button  / s key   stop commanding the gripper (it stays where it is)
    Clear button / x key   clear the graph
    close the window       quit (the arm's motors are switched off: support the arm first)

Grip protection (change the values below, or override them on the command line):
    While closing, once the current stays above GRIP_CURRENT_MA for GRIP_CONFIRM readings in a row,
    a block is in the gripper: it stops closing and holds the fingers GRIP_SQUEEZE units past where
    they stopped, so the servo keeps a light grip instead of stalling at full force.

Usage:
    python gripper_current_gui.py --port /dev/ttyACM0
    python gripper_current_gui.py --port /dev/ttyACM0 --grip-current 150 --grip-squeeze 3
    python gripper_current_gui.py --port /dev/ttyACM0 --calib path/to/my_follower_arm.json
"""

import argparse
import collections
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"  # LeRobot's default data folder
CALIB = ROOT / "calibration" / "robots" / "so_follower" / "my_follower_arm.json"
PORT_FILE = ROOT / "ports" / "follower_port.txt"

MA_PER_UNIT = 6.5  # STS3215 Present_Current: ~6.5 mA per unit (Feetech docs)

# ---- Grip protection: change these ----
GRIP_CURRENT_MA = 80.0  # block detected above this (measured: empty ~0 mA, block 600 mA peak then ~100 mA)
GRIP_CONFIRM = 3        # readings in a row above the threshold (~40 ms apart) so a single spike is ignored
GRIP_SQUEEZE = 2.0      # gripper units held past where it stopped: 0 = just hold, larger = firmer grip


def connect(port, calib):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    calib = Path(calib)
    if not calib.exists():
        raise SystemExit(f"calibration file not found: {calib}\npass it with --calib path/to/<arm>.json")
    if port is None:
        if not PORT_FILE.exists():
            raise SystemExit(f"no --port given and {PORT_FILE} not found\npass it with e.g. --port /dev/ttyACM0")
        port = PORT_FILE.read_text().strip()
    cfg = SOFollowerRobotConfig(
        port=port,
        id=calib.stem,
        calibration_dir=calib.parent,
        use_degrees=True,
        max_relative_target=15.0,  # per command, so the gripper moves in small steps and squeezes gently
    )
    robot = SOFollower(cfg)
    robot.connect(calibrate=False)
    return robot


def signed(raw: int, sign_bit: int) -> int:
    """Feetech registers keep the direction in one bit instead of two's complement."""
    return -(raw & ((1 << sign_bit) - 1)) if raw & (1 << sign_bit) else raw


def read_gripper(robot) -> dict:
    """Position (LeRobot 0-100 scale), current (mA) and load (% of max torque) of the gripper."""
    pos = robot.get_observation()["gripper.pos"]
    cur = signed(int(robot.bus.read("Present_Current", "gripper", normalize=False)), 15)
    load = signed(int(robot.bus.read("Present_Load", "gripper", normalize=False)), 10)
    return {"pos": float(pos), "current_ma": abs(cur) * MA_PER_UNIT, "load_pct": abs(load) / 10.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help=f"serial port, e.g. /dev/ttyACM0 (default: read from {PORT_FILE})")
    ap.add_argument("--calib", default=str(CALIB), help="LeRobot calibration json of the arm")
    ap.add_argument("--open-target", type=float, default=60.0, help="gripper position for Open (0-100)")
    ap.add_argument("--closed-target", type=float, default=0.0, help="gripper position for Close (0-100)")
    ap.add_argument("--window", type=float, default=20.0, help="seconds of history shown")
    ap.add_argument("--grip-current", type=float, default=GRIP_CURRENT_MA, help="mA that means a block is held")
    ap.add_argument("--grip-confirm", type=int, default=GRIP_CONFIRM, help="readings in a row above it")
    ap.add_argument("--grip-squeeze", type=float, default=GRIP_SQUEEZE, help="units held past the stop point")
    ap.add_argument("--no-grip-protection", action="store_true", help="keep squeezing (old behaviour)")
    args = ap.parse_args()

    robot = connect(args.port, args.calib)
    obs = robot.get_observation()
    hold = {k: v for k, v in obs.items() if k.endswith(".pos") and k != "gripper.pos"}
    state = {"target": None, "mode": "idle", "above": 0, "t0": time.time()}
    closing_dir = -1.0 if args.closed_target < args.open_target else 1.0
    keep = int(args.window * 60)
    t, cur, load, pos = (collections.deque(maxlen=keep) for _ in range(4))

    fig, (ax_c, ax_l, ax_p) = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    fig.subplots_adjust(bottom=0.14, top=0.9, hspace=0.15)
    title = fig.suptitle("", fontsize=15, family="monospace")
    line_c, = ax_c.plot([], [], color="tab:red", lw=2)
    line_l, = ax_l.plot([], [], color="tab:orange", lw=2)
    line_p, = ax_p.plot([], [], color="tab:blue", lw=2)
    ax_c.set_ylabel("current (mA)")
    ax_l.set_ylabel("load (%)")
    ax_p.set_ylabel("position (0-100)")
    ax_p.set_xlabel("time (s)")
    for ax in (ax_c, ax_l, ax_p):
        ax.grid(alpha=0.3)

    def mark(name):
        x = time.time() - state["t0"]
        for ax in (ax_c, ax_l, ax_p):
            ax.axvline(x, color="grey", ls="--", lw=1)
        ax_c.text(x, 1.0, f" {name}", transform=ax_c.get_xaxis_transform(), va="top", fontsize=10)

    def command(name, target):
        state.update(target=target, mode=name, above=0)
        mark(name)

    def clear(_=None):
        for d in (t, cur, load, pos):
            d.clear()
        for ax in (ax_c, ax_l, ax_p):
            for artist in list(ax.lines)[1:] + list(ax.texts):
                artist.remove()
        state["t0"] = time.time()

    actions = {
        "o": lambda _=None: command("open", args.open_target),
        "c": lambda _=None: command("close", args.closed_target),
        "s": lambda _=None: command("stop", None),
        "x": clear,
    }
    buttons = []
    for i, (label, key) in enumerate([("Open (o)", "o"), ("Close (c)", "c"), ("Stop (s)", "s"), ("Clear (x)", "x")]):
        b = Button(fig.add_axes([0.13 + i * 0.2, 0.02, 0.16, 0.06]), label)
        b.on_clicked(actions[key])
        buttons.append(b)
    fig.canvas.mpl_connect("key_press_event", lambda e: actions[e.key]() if e.key in actions else None)

    def update(_):
        if state["target"] is not None:
            robot.send_action({**hold, "gripper.pos": state["target"]})
        r = read_gripper(robot)
        if state["mode"] == "close" and not args.no_grip_protection:
            state["above"] = state["above"] + 1 if r["current_ma"] >= args.grip_current else 0
            if state["above"] >= args.grip_confirm:
                # Block in the gripper: stop closing, hold just past where the fingers stopped.
                state.update(target=r["pos"] + closing_dir * args.grip_squeeze, mode="GRIPPED")
                mark("gripped")
        now = time.time() - state["t0"]
        t.append(now)
        cur.append(r["current_ma"])
        load.append(r["load_pct"])
        pos.append(r["pos"])
        line_c.set_data(t, cur)
        line_l.set_data(t, load)
        line_p.set_data(t, pos)
        ax_p.set_xlim(max(0.0, now - args.window), max(args.window, now))
        for ax, data in ((ax_c, cur), (ax_l, load), (ax_p, pos)):
            ax.set_ylim(0, max(10.0, max(data) * 1.15))
        last_s = [c for tt, c in zip(t, cur) if tt >= now - 1.0]
        title.set_text(f"current {r['current_ma']:5.0f} mA  (1 s avg {sum(last_s) / len(last_s):5.0f})   "
                       f"load {r['load_pct']:4.1f}%   pos {r['pos']:5.1f}   [{state['mode']}]")
        title.set_color("tab:green" if state["mode"] == "GRIPPED" else "black")
        return line_c, line_l, line_p, title

    _anim = FuncAnimation(fig, update, interval=40, cache_frame_data=False)
    try:
        plt.show()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
