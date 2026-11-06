#!/usr/bin/env python3
"""Give every arm a fixed USB name, so a replug (which can change /dev/ttyACM0, 1, 2 ...) never breaks anything.

USB serial boards have no MAC address; their unique hardware ID is the USB serial number (the 5AE6083782 in
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083782-if00). This script finds which serial number belongs to
which arm (you unplug each arm's USB when asked), then installs a udev rule so Linux always creates

    /dev/so101_follower_pink   /dev/so101_follower_blue   /dev/so101_leader_blue   /dev/so101_leader_green

pointing at the right board, whatever ttyACM number it gets, and writes those names into ports/<arm>.txt, which
teleoperate.sh, click_pick_place.py, click_to_go.py, IK_control.py, calibrate.py ... all read.

    python ports/setup_usb_ports.py                 # find each arm (unplug / replug when asked), install the rule
    python ports/setup_usb_ports.py --check         # which arm is plugged in, and where its fixed name points
    python ports/setup_usb_ports.py --install-saved # (re)install the rule from ports/usb_ids.json, e.g. on a new PC
    python ports/setup_usb_ports.py --no-udev       # no sudo: write /dev/serial/by-id/... paths instead (also fixed)
    python ports/setup_usb_ports.py --names follower_pink follower_blue     # only some arms

Installing the rule needs sudo once (it copies ports/99-so101-arms.rules to /etc/udev/rules.d/). The rule also
lets every user open the boards (MODE 0666), so no dialout group is needed. A board without a serial number, or
two boards with the same one, is matched by the physical USB socket it is plugged into instead (keep it there).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ["follower_pink", "follower_blue", "leader_blue", "leader_green"]
IDS_FILE = HERE / "usb_ids.json"
RULES_FILE = HERE / "99-so101-arms.rules"
RULES_TARGET = Path("/etc/udev/rules.d/99-so101-arms.rules")
LINK_PREFIX = "so101_"


# ---------------------------------------------------------------------------------------------- USB devices
def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def serial_devices():
    """{tty path: {serial, vendor, product, usb_path, by_id}} of every USB serial board plugged in now."""
    by_id = {}
    for link in glob.glob("/dev/serial/by-id/*"):
        by_id[os.path.realpath(link)] = link
    devices = {}
    for tty in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        node = os.path.realpath(f"/sys/class/tty/{os.path.basename(tty)}/device")
        # walk up from the tty's interface to the USB device, which holds idVendor / serial
        while node not in ("/", "") and not os.path.exists(os.path.join(node, "idVendor")):
            node = os.path.dirname(node)
        if node in ("/", ""):
            continue
        devices[tty] = {
            "serial": _read(os.path.join(node, "serial")),
            "vendor": _read(os.path.join(node, "idVendor")),
            "product": _read(os.path.join(node, "idProduct")),
            "usb_path": os.path.basename(node),          # the physical socket, e.g. 1-2.3
            "by_id": by_id.get(tty),
        }
    return devices


def key(device):
    return device["serial"] or f"socket:{device['usb_path']}"


def wait_for(condition, seconds=15.0):
    end = time.time() + seconds
    while time.time() < end:
        result = condition()
        if result:
            return result
        time.sleep(0.3)
    return None


# ---------------------------------------------------------------------------------------------- finding the arms
def identify(names):
    """Ask for each arm's USB to be unplugged and plugged back; returns {name: device}."""
    found = {}
    print("Plug in every arm's USB now (and power the arms).")
    input("Press Enter when they are all plugged in ... ")
    present = serial_devices()
    print(f"{len(present)} USB serial board(s): " + ", ".join(f"{os.path.basename(t)} ({key(d)})" for t, d in present.items()))
    for name in names:
        while True:
            answer = input(f"\nUNPLUG the USB of {name}, then press Enter (s = skip this arm): ").strip().lower()
            if answer == "s":
                break
            before = {key(d): d for d in present.values()}
            now = {key(d): d for d in serial_devices().values()}
            gone = [k for k in before if k not in now]
            if len(gone) != 1:
                print(f"  {len(gone)} boards disappeared (expected 1). Plug everything back and try again.")
                present = serial_devices()
                continue
            device = before[gone[0]]
            print(f"  {name} = serial {device['serial'] or '(none)'}, USB {device['vendor']}:{device['product']}, "
                  f"socket {device['usb_path']}")
            input(f"  Plug {name} back in, then press Enter ... ")
            back = wait_for(lambda: {key(d): d for d in serial_devices().values()}.get(gone[0]), 10.0)
            if back is None:
                print("  It did not come back (unplugged? another socket without a serial number?). Try this arm again.")
                present = serial_devices()
                continue
            found[name] = back
            present = serial_devices()
            break
    return found


# ---------------------------------------------------------------------------------------------- the udev rule
def rule_lines(ids):
    """One udev line per arm: match the board by vendor, product and serial (or by socket), add the fixed name."""
    serials = [d["serial"] for d in ids.values() if d["serial"]]
    lines = ["# Fixed names for the SO-101 arms' USB boards; made by ports/setup_usb_ports.py",
             "# /dev/so101_<arm> always points at that arm's board, whatever ttyACM number it gets."]
    for name, d in ids.items():
        match = f'SUBSYSTEM=="tty", ATTRS{{idVendor}}=="{d["vendor"]}", ATTRS{{idProduct}}=="{d["product"]}"'
        if d["serial"] and serials.count(d["serial"]) == 1:
            match += f', ATTRS{{serial}}=="{d["serial"]}"'
        else:
            print(f"  {name}: no unique serial number; matched by its USB socket {d['usb_path']} (keep it plugged there)")
            match += f', KERNELS=="{d["usb_path"]}"'
        lines.append(f'{match}, SYMLINK+="{LINK_PREFIX}{name}", MODE="0666"')
    return "\n".join(lines) + "\n"


def install_rule(ids):
    RULES_FILE.write_text(rule_lines(ids))
    print(f"\nWrote {RULES_FILE}:")
    print(RULES_FILE.read_text())
    commands = [["sudo", "cp", str(RULES_FILE), str(RULES_TARGET)],
                ["sudo", "udevadm", "control", "--reload-rules"],
                ["sudo", "udevadm", "trigger", "--subsystem-match=tty", "--action=add"]]
    print("Installing it (sudo asks for your password):")
    for command in commands:
        print("  " + " ".join(command))
        if subprocess.run(command).returncode != 0:
            raise SystemExit("Could not install the rule; run the commands above yourself, then --check.")
    # udev makes the links within a moment for the boards that are plugged in
    wait_for(lambda: all(os.path.exists(f"/dev/{LINK_PREFIX}{n}") for n in ids), 3.0)
    for name in ids:
        link = f"/dev/{LINK_PREFIX}{name}"
        print(f"  {link} -> {os.path.realpath(link) if os.path.exists(link) else 'not there yet (replug it)'}")


def write_port_files(ids, use_udev):
    for name, d in ids.items():
        path = f"/dev/{LINK_PREFIX}{name}" if use_udev else d["by_id"]
        if not path:
            print(f"  {name}: no /dev/serial/by-id name (no serial number); use the udev rule instead")
            continue
        (HERE / f"{name}.txt").write_text(path + "\n")
        print(f"  ports/{name}.txt = {path}")


# ---------------------------------------------------------------------------------------------- check
def check(names):
    ids = json.loads(IDS_FILE.read_text()) if IDS_FILE.is_file() else {}
    devices = serial_devices()
    by_key = {key(d): tty for tty, d in devices.items()}
    print(f"{'arm':15s} {'ports file':34s} {'points at':14s} saved serial")
    for name in names:
        port_file = HERE / f"{name}.txt"
        port = port_file.read_text().strip() if port_file.is_file() else "(no ports file)"
        target = os.path.realpath(port) if os.path.exists(port) else "NOT PLUGGED IN"
        saved = ids.get(name)
        saved_text = "(not set up)" if not saved else key(saved)
        if saved and key(saved) not in by_key:
            saved_text += "  <- board not connected"
        print(f"{name:15s} {port:34s} {target:14s} {saved_text}")
    unknown = [tty for tty, d in devices.items() if key(d) not in {key(v) for v in ids.values()}]
    if unknown:
        print("Boards not assigned to any arm: " + ", ".join(f"{t} ({key(devices[t])})" for t in unknown))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", default=NAMES, help="which arms (default: all four)")
    ap.add_argument("--check", action="store_true", help="show where each arm's name points now")
    ap.add_argument("--install-saved", action="store_true", help="install the rule from ports/usb_ids.json")
    ap.add_argument("--no-udev", action="store_true",
                    help="no sudo / udev: write the /dev/serial/by-id/... names into ports/<arm>.txt instead")
    args = ap.parse_args()
    if not sys.platform.startswith("linux"):
        raise SystemExit("This is for Linux (udev).")

    if args.check:
        check(args.names)
        return
    if args.install_saved:
        if not IDS_FILE.is_file():
            raise SystemExit(f"No {IDS_FILE.name} yet: run without --install-saved first")
        ids = {n: d for n, d in json.loads(IDS_FILE.read_text()).items() if n in args.names}
    else:
        ids = identify(args.names)
        if not ids:
            raise SystemExit("No arm identified.")
        saved = json.loads(IDS_FILE.read_text()) if IDS_FILE.is_file() else {}
        saved.update(ids)
        IDS_FILE.write_text(json.dumps(saved, indent=2) + "\n")
        print(f"\nSaved {IDS_FILE}")
        ids = {n: d for n, d in saved.items()}          # the rule covers every arm set up so far

    if not args.no_udev:
        install_rule(ids)
    print("\nPort files:")
    write_port_files({n: d for n, d in ids.items() if n in args.names or args.install_saved}, not args.no_udev)
    print("\nDone. Check any time with:  python ports/setup_usb_ports.py --check")


if __name__ == "__main__":
    main()
