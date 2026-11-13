#!/usr/bin/env python3
"""Teleop over the network: the leaders on one computer (send), the followers on another (receive).

    python 03_teleop.py net receive                 # on the computer with the FOLLOWERS (start this first)
    python 03_teleop.py net send                    # on the computer with the LEADERS
    python 03_teleop.py net send --pair follower_blue          # only that pair (or name its leader)
    python 03_teleop.py net send --to 192.168.1.60             # override the address in config/teleop.yaml

What it does: the same as lerobot-teleoperate (the follower goes where its leader is), split in two. "send" reads
the leaders' joints --fps times a second and sends them in one small UDP message to the receiver; "receive" opens
UDP port <port> and sends each follower its leader's joints. The pairs come from config/arms.yaml (follower <- leader), and
each computer only needs its own arms: ports registered (01_setup_usb.py) and LeRobot calibration files.

Settings: config/teleop.yaml "network" (each computer's own config/teleop.yaml):
    "network": {"receiver_ip": "192.168.1.50", "sender_ip": "192.168.1.20", "port": 5555, "key": "some-word"}
  receiver_ip  the follower computer's address (hostname -I on it); send sends there
  sender_ip    the leader computer's address; receive only accepts messages from it ("" = from anywhere)
  port         the UDP port the receiver opens
  key          a word both sides must have the same; messages without it are ignored
--to / --from / --port / --key override them.

Safety: a follower only follows messages that are newer than the last one (late ones are dropped), from the right
address with the right key, and in the same units (both sides degrees, or both LeRobot's default scale); each
joint moves at most --max-step per message, so a burst of delayed messages cannot make it lunge; when nothing has
arrived for --timeout s (Wi-Fi drop) the followers hold where they are until messages come again. Ctrl+C on the
receiver disconnects the followers as lerobot-teleoperate does (torque off: hold them, they go limp).

Check the connection first: ping <receiver_ip> from the sender. Guest / university Wi-Fi often blocks devices
from reaching each other. A firewall on the receiver (rarely on by default): sudo ufw allow 5555/udp.
"""
import argparse
import json
import random
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import arms as arm_config  # noqa: E402

DEFAULTS = {"receiver_ip": "", "sender_ip": "", "port": 5555, "key": ""}


def settings(args):
    """config/teleop.yaml "network", overridden by the command line."""
    from core import config
    from core.paths import CONFIG

    net = {**DEFAULTS, **(config.load(CONFIG / "teleop.yaml", {}, required=False).get("network") or {})}
    for name, value in (("receiver_ip", args.to), ("sender_ip", getattr(args, "from_ip", None)),
                        ("port", args.port), ("key", args.key)):
        if value is not None:
            net[name] = value
    net["port"] = int(net["port"])
    return net


def connect(kind, names):
    """Connect these arms (kind 'robot' = followers, 'teleop' = leaders), all in the same units: degrees when this
    LeRobot allows it, else its default scale. Returns ({name: device}, degrees)."""
    from teleop.teleop_gui import lerobot_device

    missing = [(n, arm_config.port(n)) for n in names if not Path(arm_config.port(n)).exists()]
    if missing:
        for name, port in missing:
            print(f"{name}: {port} not found (USB unplugged, no power, or the port names not set up here?)")
        raise SystemExit("Check with:  python 01_setup_usb.py --check")
    for degrees in (True, False):
        try:
            devices = {n: lerobot_device(kind, arm_config.lerobot_type(n), arm_config.port(n), n, degrees)
                       for n in names}
            break
        except TypeError:                     # a LeRobot whose configs have no use_degrees
            continue
    for name, device in devices.items():
        print(f"Connecting {name} ...")
        device.connect()
    return devices, degrees


def disconnect(devices):
    for name, device in devices.items():
        try:
            device.disconnect()
        except Exception as error:            # noqa: BLE001 - keep closing the others
            print(f"{name}: {error}")


# ---------------------------------------------------------------------------------------------- send (leaders)
def send(args, pairs):
    net = settings(args)
    if not net["receiver_ip"]:
        raise SystemExit('No receiver address: set "receiver_ip" in config/teleop.yaml "network" or give --to <ip>')
    leaders, degrees = connect("teleop", [leader for _, leader in pairs])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    session, seq, period = random.getrandbits(31), 0, 1.0 / args.fps
    target = (net["receiver_ip"], net["port"])
    print(f"Sending {', '.join(f'{l} -> {f}' for f, l in pairs)} to {target[0]}:{target[1]} at {args.fps:.0f} Hz "
          f"({'degrees' if degrees else 'LeRobot default scale'}). Ctrl+C stops.")
    last_report, sent, errors = time.monotonic(), 0, 0
    try:
        while True:
            start = time.monotonic()
            arms = {}
            for follower, leader in pairs:
                try:
                    arms[follower] = {k: float(v) for k, v in leaders[leader].get_action().items()}
                except Exception as error:      # noqa: BLE001 - one bad read: skip it, keep going
                    errors += 1
                    if errors % 50 == 1:
                        print(f"{leader}: read failed ({str(error).splitlines()[0][:80]})")
            if arms:
                seq += 1
                message = {"key": net["key"], "session": session, "seq": seq, "degrees": degrees, "arms": arms}
                try:
                    sock.sendto(json.dumps(message).encode(), target)
                    sent += 1
                except OSError as error:        # e.g. Wi-Fi down: keep trying
                    errors += 1
                    if errors % 50 == 1:
                        print(f"send failed: {error}")
            if time.monotonic() - last_report >= 5.0:
                print(f"  sent {sent / (time.monotonic() - last_report):.0f} messages/s")
                last_report, sent = time.monotonic(), 0
            time.sleep(max(0.0, period - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        disconnect(leaders)
        print("Stopped.")


# ---------------------------------------------------------------------------------------------- receive (followers)
def limit_step(target, last, max_step):
    """target, each joint moved at most max_step from last (the previous command)."""
    out = {}
    for name, value in target.items():
        before = last.get(name)
        out[name] = value if before is None else before + max(-max_step, min(max_step, value - before))
    return out


class Receiver:
    """Checks each message and gives the followers their targets; the logic without the network or the arms, so
    it can be tested on its own."""

    def __init__(self, net, degrees, followers, max_step):
        self.net, self.degrees, self.followers, self.max_step = net, degrees, followers, max_step
        self.session = self.seq = None
        self.last_command = {}               # follower -> its last command
        self.units_warned = False

    def handle(self, data, sender_ip):
        """One UDP message -> {follower: action to send}, or ({}, why it was ignored)."""
        if self.net["sender_ip"] and sender_ip != self.net["sender_ip"]:
            return {}, f"from {sender_ip}, not {self.net['sender_ip']}"
        try:
            message = json.loads(data.decode())
        except (ValueError, UnicodeDecodeError):
            return {}, "not a teleop message"
        if message.get("key") != self.net["key"]:
            return {}, "wrong key"
        if message.get("degrees") != self.degrees:
            return {}, (f"units differ: the sender uses {'degrees' if message.get('degrees') else 'the default scale'}"
                        f", this side {'degrees' if self.degrees else 'the default scale'} (same LeRobot on both?)")
        if message.get("session") != self.session:          # the sender (re)started: follow its messages
            self.session, self.seq = message.get("session"), None
        seq = message.get("seq", 0)
        if self.seq is not None and seq <= self.seq:
            return {}, "late"
        self.seq = seq
        actions = {}
        for follower, target in message.get("arms", {}).items():
            if follower in self.followers:
                action = limit_step(target, self.last_command.get(follower, {}), self.max_step)
                self.last_command[follower] = action
                actions[follower] = action
        return actions, ""


def receive(args, pairs):
    net = settings(args)
    if not net["key"]:
        print('Note: no "key" set in config/teleop.yaml "network": any message to this port is accepted (only from '
              f'{net["sender_ip"] or "anywhere"})')
    followers, degrees = connect("robot", [follower for follower, _ in pairs])
    receiver = Receiver(net, degrees, followers, args.max_step)
    for name, robot in followers.items():           # start from where each follower is (no jump)
        try:
            receiver.last_command[name] = {k: float(v) for k, v in robot.get_observation().items()
                                           if k.endswith(".pos")}
        except Exception:                           # noqa: BLE001 - then the first message sets it
            pass
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", net["port"]))
    sock.settimeout(0.05)
    print(f"Listening on UDP port {net['port']} for {', '.join(followers)} "
          f"({'degrees' if degrees else 'LeRobot default scale'}); "
          f"sender {net['sender_ip'] or 'any address'}. Start the sender now. Ctrl+C stops.")
    last_ok, holding, ignored, last_report, applied = time.monotonic(), True, {}, time.monotonic(), 0
    try:
        while True:
            try:
                data, (ip, _) = sock.recvfrom(65536)
            except socket.timeout:
                data = None
            if data is not None:
                actions, why = receiver.handle(data, ip)
                if why:
                    if why not in ignored:
                        print(f"ignored a message: {why}")
                    ignored[why] = ignored.get(why, 0) + 1
                for name, action in actions.items():
                    try:
                        followers[name].send_action(action)
                    except Exception as error:      # noqa: BLE001 - shown, the loop goes on
                        print(f"{name}: send failed ({str(error).splitlines()[0][:80]})")
                if actions:
                    applied += 1
                    last_ok = time.monotonic()
                    if holding:
                        print("Following the leaders.")
                        holding = False
            if not holding and time.monotonic() - last_ok > args.timeout:
                print(f"No messages for {args.timeout:.1f} s: holding the followers where they are.")
                holding = True
            if time.monotonic() - last_report >= 5.0:
                if applied:
                    print(f"  following at {applied / (time.monotonic() - last_report):.0f} messages/s")
                last_report, applied = time.monotonic(), 0
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        disconnect(followers)
        print("Stopped (the followers' torque is off: they go limp).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["send", "receive"], help="send: this computer has the leaders; receive: the "
                                                               "followers")
    ap.add_argument("--pair", nargs="+", metavar="ARM", help="only these followers (or their leaders)")
    ap.add_argument("--to", default=None, help="send: the receiver's address (default: config/teleop.yaml receiver_ip)")
    ap.add_argument("--from", dest="from_ip", default=None,
                    help="receive: accept messages only from this address (default: config/teleop.yaml sender_ip)")
    ap.add_argument("--port", type=int, default=None, help="UDP port (default: config/teleop.yaml port, else 5555)")
    ap.add_argument("--key", default=None, help="shared word (default: config/teleop.yaml key)")
    ap.add_argument("--fps", type=float, default=50.0, help="send: messages per second")
    ap.add_argument("--max-step", type=float, default=6.0,
                    help="receive: largest change of a joint per message (degrees, or the default scale's units)")
    ap.add_argument("--timeout", type=float, default=0.3, help="receive: hold after this long without messages [s]")
    args = ap.parse_args(argv)
    pairs = arm_config.teleop_pairs(args.pair)
    if not pairs:
        raise SystemExit(f"No teleop pair matches {args.pair}; pairs in config/teleop.yaml: {arm_config.teleop_pairs()}")
    (send if args.mode == "send" else receive)(args, pairs)


if __name__ == "__main__":
    main()
