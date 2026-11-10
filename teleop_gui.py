#!/usr/bin/env python3
"""Teleoperate with live telemetry: every follower joint's angle, current, load and temperature, plotted while you
drive the arms with their leaders. The pairs and ports come from arms.json, as for teleop.py.

    python teleop_gui.py                          # every pair (as arms.json pairs them)
    python teleop_gui.py --pair follower_blue     # only that follower (or name its leader)
    python teleop_gui.py --log                    # also write every reading to outputs/telemetry/<time>.csv
    ./teleoperate.sh --gui                        # the same through teleop.py

Why not alongside lerobot-teleoperate: a serial port can be open in one program only, so the telemetry has to come
from the program that drives the arms. This one does what lerobot-teleoperate does (the same robot / teleop types and
ids, so the same LeRobot calibration files): the leader's joints are read and sent to the follower, --fps times a
second, in a background thread; between those commands the follower's registers are read --telemetry-hz times a
second (Present_Position, Present_Current, Present_Load, Present_Temperature of every motor).

The window has one column per follower:
  table     per joint: angle (follower), target (leader), error between them, current (mA), load (%), temperature;
            a joint whose current is above --current-warn or temperature above --temp-warn ends in "<-- !"
  angle     every joint's angle over the last --window seconds (degrees; the gripper 0-100)
  current   every joint's motor current (mA; STS3215: ~6.5 mA per unit)
  load      every joint's load (% of the motor's maximum torque)
Close the window or Ctrl+C to stop; the arms are then disconnected as lerobot-teleoperate does (torque off: hold
the followers, they go limp).
"""
import argparse
import collections
import csv
import importlib
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from control import arm_config  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "wrist",
         "wrist_roll": "roll", "gripper": "grip"}
MA_PER_UNIT = 6.5          # STS3215 Present_Current: ~6.5 mA per unit (Feetech docs)
COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def signed(raw, sign_bit):
    """Feetech registers keep the direction in one bit instead of two's complement."""
    raw = int(raw)
    return -(raw & ((1 << sign_bit) - 1)) if raw & (1 << sign_bit) else raw


# ---------------------------------------------------------------------------------------------- the arms
def lerobot_device(kind, type_name, port, device_id, degrees):
    """A LeRobot robot (kind 'robot') or teleoperator ('teleop') of that type, built like lerobot-teleoperate does
    (its config registry by type name, the default calibration folder, id = the arm's name); degrees: use_degrees
    (raises TypeError when this LeRobot's config has no such option). Not connected yet."""
    package = "lerobot.robots" if kind == "robot" else "lerobot.teleoperators"
    modules = (["so_follower", "so101_follower", "so100_follower"] if kind == "robot"
               else ["so_leader", "so101_leader", "so100_leader"])
    for name in modules:                     # importing them registers their config types
        try:
            importlib.import_module(f"{package}.{name}")
        except ImportError:
            pass
    base = importlib.import_module(package)
    config_base = getattr(base, "RobotConfig" if kind == "robot" else "TeleoperatorConfig")
    make_name = "make_robot_from_config" if kind == "robot" else "make_teleoperator_from_config"
    make = getattr(base, make_name, None) or getattr(importlib.import_module(f"{package}.utils"), make_name)
    config_class = config_base.get_choice_class(type_name)
    config = (config_class(port=port, id=device_id, use_degrees=True) if degrees
              else config_class(port=port, id=device_id))
    return make(config)


class Pair:
    """One follower driven by its leader, with the follower's telemetry history."""

    def __init__(self, follower, leader, keep):
        self.follower_name, self.leader_name = follower, leader
        self.follower = self.leader = None
        self.degrees = False
        self.lock = threading.Lock()
        self.t = collections.deque(maxlen=keep)
        self.series = {key: {j: collections.deque(maxlen=keep) for j in JOINTS}
                       for key in ("angle", "current", "load")}
        self.latest = {}                     # joint -> {angle, target, current, load, temp}
        self.error = ""

    def connect(self):
        """Leader and follower in the SAME units: both in degrees when both configs allow it, else both in LeRobot's
        default scale (the leader's values are sent to the follower as they are, so they must match)."""
        def make(degrees):
            return (lerobot_device("robot", arm_config.lerobot_type(self.follower_name),
                                   arm_config.port(self.follower_name), self.follower_name, degrees),
                    lerobot_device("teleop", arm_config.lerobot_type(self.leader_name),
                                   arm_config.port(self.leader_name), self.leader_name, degrees))
        try:
            (self.follower, self.leader), self.degrees = make(True), True
        except TypeError:                    # a LeRobot whose configs have no use_degrees
            (self.follower, self.leader), self.degrees = make(False), False
        self.follower.connect()
        self.leader.connect()

    def step(self):
        """Leader -> follower; returns the target (the leader's joints)."""
        action = self.leader.get_action()
        self.follower.send_action(action)
        return action

    def read(self, target, t0):
        """The follower's registers -> history and latest values."""
        bus = self.follower.bus
        angle = bus.sync_read("Present_Position")
        current = bus.sync_read("Present_Current", normalize=False)
        load = bus.sync_read("Present_Load", normalize=False)
        temp = bus.sync_read("Present_Temperature", normalize=False)
        now = time.monotonic() - t0
        row = {}
        with self.lock:
            self.t.append(now)
            for j in JOINTS:
                if j not in angle:
                    continue
                values = {"angle": float(angle[j]), "target": float(target.get(f"{j}.pos", float("nan"))),
                          "current": abs(signed(current[j], 15)) * MA_PER_UNIT,
                          "load": abs(signed(load[j], 10)) / 10.0, "temp": float(temp[j])}
                for key in ("angle", "current", "load"):
                    self.series[key][j].append(values[key])
                self.latest[j] = values
                row[j] = values
        return now, row

    def disconnect(self):
        for device in (self.leader, self.follower):
            if device is not None:
                try:
                    device.disconnect()
                except Exception as error:          # noqa: BLE001 - keep closing the others
                    print(f"disconnect: {error}")


def drive(pairs, args, stop, log_writer):
    """The teleop loop (background thread): every pair leader -> follower at --fps, telemetry at --telemetry-hz."""
    period = 1.0 / args.fps
    every = max(1, round(args.fps / args.telemetry_hz))
    t0, cycle = time.monotonic(), 0
    while not stop.is_set():
        start = time.monotonic()
        for pair in pairs:
            try:
                target = pair.step()
                if cycle % every == 0:
                    now, row = pair.read(target, t0)
                    if log_writer is not None:
                        for j, v in row.items():
                            log_writer.writerow([f"{now:.3f}", pair.follower_name, j, f"{v['angle']:.2f}",
                                                 f"{v['target']:.2f}", f"{v['current']:.0f}", f"{v['load']:.1f}",
                                                 f"{v['temp']:.0f}"])
                pair.error = ""
            except Exception as error:              # noqa: BLE001 - shown in the window, the loop goes on
                pair.error = str(error).splitlines()[0][:80]
        cycle += 1
        time.sleep(max(0.0, period - (time.monotonic() - start)))


# ---------------------------------------------------------------------------------------------- the window
def build_window(pairs, args):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, len(pairs), figsize=(7.5 * len(pairs), 10), squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1, 1, 1]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.05, hspace=0.35, wspace=0.18)
    views = []
    for col, pair in enumerate(pairs):
        table_ax, angle_ax, current_ax, load_ax = axes[:, col]
        table_ax.axis("off")
        table_ax.set_title(f"{pair.follower_name}  <-  {pair.leader_name}", fontsize=13, loc="left")
        lines = {}
        unit = "deg" if pair.degrees else "%"
        for ax, key, label in ((angle_ax, "angle", f"angle ({unit}; gripper 0-100)"),
                               (current_ax, "current", "current (mA)"), (load_ax, "load", "load (%)")):
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            lines[key] = {j: ax.plot([], [], color=c, lw=1.6, label=SHORT[j])[0] for j, c in zip(JOINTS, COLOURS)}
        angle_ax.legend(loc="upper left", ncol=6, fontsize=8)
        if args.current_warn:
            current_ax.axhline(args.current_warn, color="red", ls="--", lw=1)
        load_ax.set_xlabel("time (s)")
        text = table_ax.text(0.0, 0.95, "", family="monospace", fontsize=10, va="top", transform=table_ax.transAxes)
        views.append({"pair": pair, "lines": lines, "axes": (angle_ax, current_ax, load_ax), "text": text})
    return fig, views


def refresh(views, args):
    for view in views:
        pair = view["pair"]
        with pair.lock:
            t = list(pair.t)
            data = {key: {j: list(d) for j, d in series.items()} for key, series in pair.series.items()}
            latest = dict(pair.latest)
        rows = [f"{'joint':6s} {'angle':>7s} {'target':>7s} {'error':>6s} {'mA':>6s} {'load%':>6s} {'temp':>5s}"]
        warn = False
        for j in JOINTS:
            v = latest.get(j)
            if v is None:
                continue
            hot = v["current"] > args.current_warn or v["temp"] > args.temp_warn
            warn |= hot
            rows.append(f"{SHORT[j]:6s} {v['angle']:7.1f} {v['target']:7.1f} {v['angle'] - v['target']:6.1f} "
                        f"{v['current']:6.0f} {v['load']:6.1f} {v['temp']:4.0f}C" + ("  <-- !" if hot else ""))
        if pair.error:
            rows.append(f"ERROR: {pair.error}")
        view["text"].set_text("\n".join(rows) if latest or pair.error else "connecting ...")
        view["text"].set_color("red" if pair.error else "black")         # a hot joint: its row ends in "<-- !"
        if not t:
            continue
        for key, ax in zip(("angle", "current", "load"), view["axes"]):
            for j, line in view["lines"][key].items():
                y = data[key][j]
                line.set_data(t[-len(y):], y)
            ax.set_xlim(max(0.0, t[-1] - args.window), max(args.window, t[-1]))
            ax.relim()
            ax.autoscale_view(scalex=False)


def run(pair_names, args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    keep = int(args.window * args.telemetry_hz) + 10
    pairs = [Pair(f, l, keep) for f, l in pair_names]
    for pair in pairs:
        print(f"Connecting {pair.follower_name} <- {pair.leader_name} ...")
        pair.connect()
    log_file = log_writer = None
    if args.log:
        path = ROOT / "outputs" / "telemetry" / f"teleop_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["t_s", "follower", "joint", "angle", "target", "current_ma", "load_pct", "temp_c"])
        print(f"Logging to {path}")
    stop = threading.Event()
    worker = threading.Thread(target=drive, args=(pairs, args, stop, log_writer), daemon=True, name="teleop")
    worker.start()
    print("Teleoperating: move the leaders. Close the window or Ctrl+C to stop.")
    try:
        fig, views = build_window(pairs, args)
        animation = FuncAnimation(fig, lambda _: refresh(views, args), interval=150, cache_frame_data=False)
        plt.show()
        del animation
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=2.0)
        for pair in pairs:
            pair.disconnect()
        if log_file is not None:
            log_file.close()
        print("Stopped.")


def add_args(ap):
    ap.add_argument("--fps", type=float, default=60.0, help="leader -> follower commands per second")
    ap.add_argument("--telemetry-hz", type=float, default=20.0, help="telemetry readings per second")
    ap.add_argument("--window", type=float, default=20.0, help="seconds of history shown")
    ap.add_argument("--current-warn", type=float, default=600.0, help="mark currents above this (mA)")
    ap.add_argument("--temp-warn", type=float, default=55.0, help="mark temperatures above this (C)")
    ap.add_argument("--log", action="store_true", help="write every reading to outputs/telemetry/<time>.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs="+", metavar="ARM", help="only these followers (or their leaders)")
    add_args(ap)
    args = ap.parse_args()
    pairs = arm_config.teleop_pairs(args.pair)
    if not pairs:
        raise SystemExit(f"No teleop pair matches {args.pair}; pairs in arms.json: {arm_config.teleop_pairs()}")
    missing = [(name, arm_config.port(name)) for pair in pairs for name in pair
               if not Path(arm_config.port(name)).exists()]
    if missing:
        for name, port in missing:
            print(f"{name}: {port} not found (USB unplugged, no power, or the port names not set up on this PC?)")
        raise SystemExit("Check with:  python ports/setup_usb_ports.py --check   (names missing but the boards "
                         "found: python ports/setup_usb_ports.py --install-saved)")
    run(pairs, args)


if __name__ == "__main__":
    main()
