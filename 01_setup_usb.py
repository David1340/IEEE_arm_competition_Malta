#!/usr/bin/env python3
"""Step 1: detect the four arms' USB boards, record their IDs, and give each a fixed name with a udev rule.

    python 01_setup_usb.py --list            # every USB serial board plugged in now, with its IDs
    python 01_setup_usb.py                   # identify each arm (unplug / replug when asked) and install the rule
    python 01_setup_usb.py --check           # which arm is plugged in, and where its fixed name points
    python 01_setup_usb.py --install-saved   # a new computer: install the rule from the IDs in config/arms.yaml

Result: /dev/so101_follower_pink, /dev/so101_follower_blue, /dev/so101_leader_blue, /dev/so101_leader_green,
and each board's serial number in config/arms.yaml. Details: hardware/usb_ports.py.
"""
from hardware.usb_ports import main

if __name__ == "__main__":
    main()
