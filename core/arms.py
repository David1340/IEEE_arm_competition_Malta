"""The arms' setup from config/arms.yaml: ports, USB serials, teleop pairs, where each follower stands, home pose.

    from core import arms
    arms.port("follower_pink")          # "/dev/so101_follower_pink"
    arms.base("follower_blue")          # {"x": 0.0017, "y": -0.2336, "yaw_deg": -0.41}
    arms.teleop_pairs()                 # [("follower_pink", "leader_blue"), ("follower_blue", "leader_green")]
    arms.HOME_DEG, arms.HOME_GRIPPER    # the home pose (config/arms.yaml "home")

Edit config/arms.yaml to change a port, a pairing, a base position or the home pose; `python 01_setup_usb.py`
fills in the ports and USB serials. Every script reads it from here, so there is one place for these values.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core import config
from core.paths import ARMS_FILE, arm_calibration_dir

FILE = ARMS_FILE
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def load() -> dict:
    """config/arms.yaml as a dict (re-read when the file changes). Do not change it in place: use load_copy()."""
    data = config.load(FILE, {})
    if not data.get("followers"):
        raise config.ConfigError(f"{FILE}: no followers")
    return data


def load_copy() -> dict:
    """A copy of config/arms.yaml that is safe to change and save()."""
    return config.load_copy(FILE, {})


def save(data: dict) -> None:
    config.save(FILE, data)


def _home() -> tuple[list[float], float]:
    home = load().get("home", {})
    joints = [float(v) for v in home.get("joints_deg", [0.0, -100.0, 90.0, -75.0, -90.0])]
    if len(joints) != len(ARM_JOINTS):
        raise config.ConfigError(f"{FILE}: home.joints_deg needs {len(ARM_JOINTS)} angles ({', '.join(ARM_JOINTS)})")
    return joints, float(home.get("gripper", 60.0))


# The home pose: where every script starts (the arms go there first) and finishes (park_and_disconnect), and where
# the arms wait between picks, out of the camera's view. Set in config/arms.yaml "home".
HOME_DEG, HOME_GRIPPER = _home()


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


def lerobot_calibration_dir(name: str) -> Path:
    """The folder LeRobot keeps this arm's motor calibration in (config/arms.yaml "lerobot")."""
    settings = load().get("lerobot", {})
    if role(name) == "follower":
        folder = settings.get("follower_calibration_dir", "~/.cache/huggingface/lerobot/calibration/robots/so_follower")
    else:
        folder = settings.get("leader_calibration_dir",
                              "~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader")
    return Path(folder).expanduser()


def lerobot_calibration(name: str) -> Path:
    """The arm's LeRobot calibration file, the one LeRobot reads (<dir>/<name>.json, or the arm's own
    'calibration' entry). Its versioned copy is calibration_file(name)."""
    own = entry(name).get("calibration")
    if own:
        return Path(own).expanduser()
    return lerobot_calibration_dir(name) / f"{name}.json"


def follower_calibration(name: str) -> Path:
    """The follower's LeRobot calibration file (kept for older callers; same as lerobot_calibration)."""
    return lerobot_calibration(name)


def calibration_file(name: str) -> Path:
    """config/calibration/<arm>/motors.yaml: the versioned copy of the arm's LeRobot calibration."""
    return arm_calibration_dir(name) / "motors.yaml"


def handoff_points() -> list[list[float]]:
    """TABLE-frame xy of the points in the middle where one arm leaves a shape for the other (config/arms.yaml
    handoff.points); the first free one is used."""
    return [list(map(float, p)) for p in load().get("handoff", {}).get("points", [[0.30, 0.0]])]


def handoff_min_spacing() -> float:
    """A handoff point closer than this to an occupied one is not used [m] (config/arms.yaml "handoff")."""
    return float(load().get("handoff", {}).get("min_spacing", 0.10))


def middle_half_width() -> float:
    """Half the width of the shared middle strip between the arms' sides [m] (config/arms.yaml "handoff")."""
    return float(load().get("handoff", {}).get("middle_half_width", 0.10))


def teleop_pairs(only: list[str] | None = None) -> list[tuple[str, str]]:
    """(follower, leader) for every follower with a leader (only those named, if given)."""
    pairs = [(f, e["leader"]) for f, e in followers().items() if e.get("leader")]
    if only:
        pairs = [pair for pair in pairs if pair[0] in only or pair[1] in only]
    return pairs


def bases() -> dict:
    """{follower: {x, y, yaw_deg}}: every follower's base in the TABLE frame."""
    return {name: base(name) for name in followers()}


def arm_from_table(name: str) -> np.ndarray:
    """4 x 4: TABLE frame -> that follower's base frame (from its config/arms.yaml base)."""
    all_bases = bases()
    if name not in all_bases:
        raise KeyError(f"unknown arm {name!r}; known: {', '.join(all_bases)}")
    b = all_bases[name]
    yaw = math.radians(b["yaw_deg"])
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.array([b["x"], b["y"], 0.0])
    return T
