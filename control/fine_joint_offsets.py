"""Shared software fine offsets for the SO-101 arm joints."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FINE_OFFSETS_FILE = ROOT / "calibration" / "robots" / "so_follower" / "fine_joint_offsets.json"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@lru_cache(maxsize=8)
def _read_offsets(path: str, modified_ns: int, file_size: int) -> dict[str, float]:
	file_path = Path(path)
	if not file_path.exists():
		return {name: 0.0 for name in ARM_JOINTS}
	data = json.loads(file_path.read_text())
	values = data.get("joint_offsets_deg", {})
	offsets = {name: float(values.get(name, 0.0)) for name in ARM_JOINTS}
	invalid = [name for name, value in offsets.items() if not math.isfinite(value)]
	if invalid:
		raise ValueError(f"Fine joint offsets must be finite degrees; invalid: {invalid}")
	return offsets


def offsets_file(arm: str | None = None) -> Path:
	"""The fine offsets file of one arm (by its LeRobot id, e.g. follower_blue): fine_joint_offsets_<arm>.json
	when it exists (or arm is None: the shared fine_joint_offsets.json). An arm without its own file uses the
	shared one until `python calibration/calibrate.py offsets --arm <arm>` saves its own."""
	if arm is None:
		return FINE_OFFSETS_FILE
	own = FINE_OFFSETS_FILE.with_name(f"fine_joint_offsets_{arm}.json")
	return own if own.exists() else FINE_OFFSETS_FILE


def own_offsets_file(arm: str | None) -> Path:
	"""Where `arm`'s own offsets are saved (the shared file when arm is None)."""
	return FINE_OFFSETS_FILE if arm is None else FINE_OFFSETS_FILE.with_name(f"fine_joint_offsets_{arm}.json")


def load_fine_joint_offsets(path: str | Path | None = FINE_OFFSETS_FILE, *, arm: str | None = None) -> dict[str, float]:
	"""Load software corrections in degrees, refreshing the cache when JSON changes. With arm, that arm's file."""
	file_path = offsets_file(arm) if arm is not None or path is None else Path(path)
	if not file_path.exists():
		return {name: 0.0 for name in ARM_JOINTS}
	stat = file_path.stat()
	return _read_offsets(str(file_path.resolve()), stat.st_mtime_ns, stat.st_size)


def save_fine_joint_offsets(offsets: Mapping[str, float], path: str | Path = FINE_OFFSETS_FILE) -> None:
	"""Persist one finite offset in degrees for each arm joint."""
	values = {name: float(offsets.get(name, 0.0)) for name in ARM_JOINTS}
	invalid = [name for name, value in values.items() if not math.isfinite(value)]
	if invalid:
		raise ValueError(f"Fine joint offsets must be finite degrees; invalid: {invalid}")
	file_path = Path(path)
	file_path.parent.mkdir(parents=True, exist_ok=True)
	data = {
		"version": 1,
		"unit": "degrees",
		"description": "Software offsets added to live arm joint readings and subtracted from motor targets.",
		"joint_offsets_deg": values,
	}
	temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
	temporary_path.write_text(json.dumps(data, indent=2) + "\n")
	temporary_path.replace(file_path)
	_read_offsets.cache_clear()


def correct_joint_position_deg(joint_name: str, raw_position_deg: float, arm: str | None = None) -> float:
	"""Map a LeRobot joint reading to the corrected logical angle."""
	return float(raw_position_deg) + load_fine_joint_offsets(arm=arm).get(joint_name, 0.0)


def motor_position_deg(joint_name: str, corrected_position_deg: float, arm: str | None = None) -> float:
	"""Map a corrected logical target back to the LeRobot motor coordinate."""
	return float(corrected_position_deg) - load_fine_joint_offsets(arm=arm).get(joint_name, 0.0)


def corrected_observation(observation: Mapping[str, object]) -> dict[str, object]:
	"""Return an observation copy with fine offsets applied to arm `.pos` fields."""
	corrected = dict(observation)
	for joint_name in ARM_JOINTS:
		key = f"{joint_name}.pos"
		if key in corrected:
			corrected[key] = correct_joint_position_deg(joint_name, float(corrected[key]))
	return corrected


def motor_action_from_corrected(action: Mapping[str, float]) -> dict[str, float]:
	"""Convert corrected arm targets back to motor coordinates; leave other keys unchanged."""
	converted = dict(action)
	for joint_name in ARM_JOINTS:
		key = f"{joint_name}.pos"
		if key in converted:
			converted[key] = motor_position_deg(joint_name, float(converted[key]))
	return converted


def corrected_to_motor_radians(joint_names: Sequence[str], positions_rad: Sequence[float]):
	"""Translate corrected logical radians to raw motor-coordinate radians."""
	return [
		float(position) - math.radians(load_fine_joint_offsets().get(name, 0.0))
		for name, position in zip(joint_names, positions_rad)
	]


def offsets_radians(joint_names: Sequence[str]):
	"""Return fine offsets in radians, in the requested joint order."""
	return [math.radians(load_fine_joint_offsets().get(name, 0.0)) for name in joint_names]