"""The arms' setup from arms.json (repo root): ports, USB serials, teleop pairs, where each follower stands.

    from control.arm_config import followers, port, base, teleop_pairs, follower_calibration
    port("follower_pink")            # "/dev/so101_follower_pink"
    base("follower_blue")            # {"x": 0.0, "y": -0.25, "yaw_deg": 0.0}
    teleop_pairs()                   # [("follower_pink", "leader_blue"), ("follower_blue", "leader_green")]

Edit arms.json to change a port, a pairing or a base position; ports/setup_usb_ports.py fills in the ports and
USB serials. Every script reads it from here, so there is one place for these values.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "arms.json"


@lru_cache(maxsize=4)
def _read(path: str, mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text())


def load() -> dict:
    """arms.json as a dict (re-read when the file changes)."""
    if not FILE.is_file():
        raise SystemExit(f"{FILE} not found: it holds the arms' ports and positions (see control/arm_config.py)")
    return _read(str(FILE), FILE.stat().st_mtime_ns)


def save(data: dict) -> None:
    tmp = FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(FILE)


def followers() -> dict:
    return load().get("followers", {})


def leaders() -> dict:
    return load().get("leaders", {})


def names(role: str | None = None) -> list[str]:
    """Arm names: followers, leaders, or both (role None)."""
    if role == "follower":
        return list(followers())
    if role == "leader":
        return list(leaders())
    return list(followers()) + list(leaders())


def entry(name: str) -> dict:
    arms = {**followers(), **leaders()}
    if name not in arms:
        raise SystemExit(f"unknown arm {name!r} in {FILE.name}; known: {', '.join(arms)}")
    return arms[name]


def role(name: str) -> str:
    return "follower" if name in followers() else "leader"


def port(name: str) -> str:
    return entry(name)["port"]


def base(name: str) -> dict:
    """A follower's base in the TABLE frame: {x, y, yaw_deg}."""
    b = entry(name).get("base", {})
    return {"x": float(b.get("x", 0.0)), "y": float(b.get("y", 0.0)), "yaw_deg": float(b.get("yaw_deg", 0.0))}


def default_arm() -> str:
    return load().get("default_arm") or next(iter(followers()))


def lerobot_type(name: str) -> str:
    settings = load().get("lerobot", {})
    return settings.get("follower_type" if role(name) == "follower" else "leader_type",
                        "so101_follower" if role(name) == "follower" else "so101_leader")


def follower_calibration(name: str) -> Path:
    """The follower's LeRobot calibration file (<dir>/<name>.json, or the arm's own 'calibration' entry)."""
    own = entry(name).get("calibration")
    if own:
        return Path(own).expanduser()
    folder = load().get("lerobot", {}).get("follower_calibration_dir",
                                           "~/.cache/huggingface/lerobot/calibration/robots/so_follower")
    return Path(folder).expanduser() / f"{name}.json"


def teleop_pairs(only: list[str] | None = None) -> list[tuple[str, str]]:
    """(follower, leader) for every follower with a leader (only those named, if given)."""
    pairs = [(f, e["leader"]) for f, e in followers().items() if e.get("leader")]
    if only:
        pairs = [pair for pair in pairs if pair[0] in only or pair[1] in only]
    return pairs
