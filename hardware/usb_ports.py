#!/usr/bin/env python3
"""Step 1: find the four arms' USB boards and give each a fixed name, so a replug (which can change /dev/ttyACM0,
1, 2 ...) never breaks anything. Run it through `python 01_setup_usb.py`.

USB serial boards have no MAC address; their unique hardware ID is the USB serial number (the 5AE6083782 in
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083782-if00). This finds which serial number belongs to which arm
(you unplug each arm's USB when asked), then installs a udev rule so Linux always creates

    /dev/so101_follower_pink   /dev/so101_follower_blue   /dev/so101_leader_blue   /dev/so101_leader_green

pointing at the right board, whatever ttyACM number it gets, and saves the serial numbers and those names in
config/arms.yaml, which every script reads. The arm names are the ones in config/arms.yaml.

    python 01_setup_usb.py                   # find each arm (unplug / replug when asked), install the rule
    python 01_setup_usb.py --list            # every USB serial board plugged in now, with its IDs
    python 01_setup_usb.py --check           # which arm is plugged in, and where its fixed name points
    python 01_setup_usb.py --install-saved   # (re)install the rule from the serials in config/arms.yaml (new PC)
    python 01_setup_usb.py --no-udev         # no sudo: put /dev/serial/by-id/... paths in config/arms.yaml
    python 01_setup_usb.py --names follower_pink follower_blue     # only some arms

Installing the rule needs sudo once (it copies hardware/99-so101-arms.rules to /etc/udev/rules.d/). The rule also
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
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from core import arms as arm_config  # noqa: E402

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
    lines = ["# Fixed names for the SO-101 arms' USB boards; made by 01_setup_usb.py",
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


def saved_ids():
    """{arm: device} from the 'usb' entries in config/arms.yaml."""
    out = {}
    for name in arm_config.names():
        usb = arm_config.entry(name).get("usb")
        if usb and usb.get("vendor"):
            text = lambda v: None if v is None else str(v)     # a hand-edited serial may have become a number
            out[name] = {"serial": text(usb.get("serial")), "vendor": text(usb["vendor"]),
                         "product": text(usb.get("product")), "usb_path": text(usb.get("socket")),
                         "by_id": usb.get("by_id")}
    return out


def save_to_config(ids, use_udev):
    """Write each arm's USB serial and its fixed port name into config/arms.yaml."""
    data = arm_config.load_copy()
    for name, d in ids.items():
        group = "followers" if name in data.get("followers", {}) else "leaders"
        arm = data.setdefault(group, {}).setdefault(name, {})
        arm["usb"] = {"serial": d["serial"], "vendor": d["vendor"], "product": d["product"], "socket": d["usb_path"]}
        arm["usb"] = {k: (None if v is None else str(v)) for k, v in arm["usb"].items()}   # IDs stay text
        if d.get("by_id"):
            arm["usb"]["by_id"] = d["by_id"]
        path = f"/dev/{LINK_PREFIX}{name}" if use_udev else d.get("by_id")
        if path:
            arm["port"] = path
            print(f"  {name}: port {path}")
        else:
            print(f"  {name}: no /dev/serial/by-id name (no serial number); use the udev rule instead")
    arm_config.save(data)
    print(f"Saved {arm_config.FILE}")


# ---------------------------------------------------------------------------------------------- list / check
def list_devices():
    """Every USB serial board plugged in now, with its IDs, and which arm it is (if it is set up)."""
    devices = serial_devices()
    owners = {key(d): name for name, d in saved_ids().items()}
    if not devices:
        print("No USB serial boards found (/dev/ttyACM*, /dev/ttyUSB*). Plugged in and powered?")
        return
    print(f"{'device':14s} {'serial':14s} {'vendor:product':15s} {'socket':10s} arm")
    for tty, d in devices.items():
        print(f"{tty:14s} {d['serial'] or '(none)':14s} {d['vendor']}:{d['product']:10s} {d['usb_path']:10s} "
              f"{owners.get(key(d), '(not set up)')}")
    print(f"{len(devices)} board(s); {len(arm_config.names())} arms in {arm_config.FILE.name}: "
          + ", ".join(arm_config.names()))


def check(names):
    ids = saved_ids()
    devices = serial_devices()
    by_key = {key(d): tty for tty, d in devices.items()}
    print(f"{'arm':15s} {'port (config/arms.yaml)':34s} {'points at':14s} saved serial")
    for name in names:
        port = arm_config.entry(name).get("port", "(no port)")
        target = os.path.realpath(port) if os.path.exists(port) else "NOT PLUGGED IN"
        saved = ids.get(name)
        saved_text = "(not set up)" if not saved else key(saved)
        if saved and key(saved) not in by_key:
            saved_text += "  <- board not connected"
        print(f"{name:15s} {port:34s} {target:14s} {saved_text}")
    unknown = [tty for tty, d in devices.items() if key(d) not in {key(v) for v in ids.values()}]
    if unknown:
        print("Boards not assigned to any arm: " + ", ".join(f"{t} ({key(devices[t])})" for t in unknown))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", default=arm_config.names(), help="which arms (default: all in config/arms.yaml)")
    ap.add_argument("--list", action="store_true", help="list every USB serial board plugged in now, with its IDs")
    ap.add_argument("--check", action="store_true", help="show where each arm's name points now")
    ap.add_argument("--install-saved", action="store_true", help="install the rule from the serials in config/arms.yaml")
    ap.add_argument("--no-udev", action="store_true",
                    help="no sudo / udev: put the /dev/serial/by-id/... names in config/arms.yaml instead")
    args = ap.parse_args(argv)
    if not sys.platform.startswith("linux"):
        raise SystemExit("This is for Linux (udev).")

    if args.list:
        list_devices()
        return
    if args.check:
        check(args.names)
        return
    if args.install_saved:
        ids = {n: d for n, d in saved_ids().items() if n in args.names}
        if not ids:
            raise SystemExit("No USB serials in config/arms.yaml yet: run without --install-saved first")
        new = ids
    else:
        new = identify(args.names)
        if not new:
            raise SystemExit("No arm identified.")
        ids = {**saved_ids(), **new}                     # the rule covers every arm set up so far

    if not args.no_udev:
        install_rule(ids)
    print("\nconfig/arms.yaml:")
    save_to_config(new, not args.no_udev)
    print("\nDone. Check any time with:  python 01_setup_usb.py --check")


if __name__ == "__main__":
    main()
