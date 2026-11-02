"""Shared software fine offsets for the SO-101 arm joints."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parent
FINE_OFFSETS_FILE = ROOT / "calibration" / "robots" / "so_follower" / "fine_joint_offsets.json"  # the older file
CALIBRATION_FILE = ROOT / "calibration.json"   # every calibration in one file (robot/calibration.py): arm.joint_offsets_deg
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@lru_cache(maxsize=8)
def _read_offsets(path: str, modified_ns: int, file_size: int) -> dict[str, float]:
	file_path = Path(path)
	if not file_path.exists():
		return {name: 0.0 for name in ARM_JOINTS}
	data = json.loads(file_path.read_text())
	values = data.get("arm", {}).get("joint_offsets_deg") if "arm" in data else data.get("joint_offsets_deg", {})
	values = values or {}
	offsets = {name: float(values.get(name, 0.0)) for name in ARM_JOINTS}
	invalid = [name for name, value in offsets.items() if not math.isfinite(value)]
	if invalid:
		raise ValueError(f"Fine joint offsets must be finite degrees; invalid: {invalid}")
	return offsets


def _default_path() -> Path:
	"""calibration.json when it holds the offsets, else the older fine_joint_offsets.json."""
	if CALIBRATION_FILE.exists() and "joint_offsets_deg" in json.loads(CALIBRATION_FILE.read_text()).get("arm", {}):
		return CALIBRATION_FILE
	return FINE_OFFSETS_FILE


def load_fine_joint_offsets(path: str | Path | None = None) -> dict[str, float]:
	"""Load software corrections in degrees, refreshing the cache when JSON changes."""
	file_path = Path(path) if path is not None else _default_path()
	if not file_path.exists():
		return {name: 0.0 for name in ARM_JOINTS}
	stat = file_path.stat()
	return _read_offsets(str(file_path.resolve()), stat.st_mtime_ns, stat.st_size)


def save_fine_joint_offsets(offsets: Mapping[str, float], path: str | Path | None = None) -> None:
	"""Persist one finite offset in degrees for each arm joint: into calibration.json's arm section (the rest of
	that file kept), or into the given (older format) file."""
	values = {name: float(offsets.get(name, 0.0)) for name in ARM_JOINTS}
	invalid = [name for name, value in values.items() if not math.isfinite(value)]
	if invalid:
		raise ValueError(f"Fine joint offsets must be finite degrees; invalid: {invalid}")
	if path is None:
		data = json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else {}
		data.setdefault("arm", {})["joint_offsets_deg"] = values
		temporary_path = CALIBRATION_FILE.with_suffix(".json.tmp")
		temporary_path.write_text(json.dumps(data, indent=2) + "\n")
		temporary_path.replace(CALIBRATION_FILE)
		_read_offsets.cache_clear()
		return
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


def correct_joint_position_deg(joint_name: str, raw_position_deg: float) -> float:
	"""Map a LeRobot joint reading to the corrected logical angle."""
	return float(raw_position_deg) + load_fine_joint_offsets().get(joint_name, 0.0)


def motor_position_deg(joint_name: str, corrected_position_deg: float) -> float:
	"""Map a corrected logical target back to the LeRobot motor coordinate."""
	return float(corrected_position_deg) - load_fine_joint_offsets().get(joint_name, 0.0)


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