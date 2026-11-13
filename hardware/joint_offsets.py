"""Fine joint offsets of the SO-101 followers: software corrections on top of LeRobot's motor calibration.

Each follower's offsets are in config/calibration/<arm>/joint_offsets.yaml (degrees; true angle = LeRobot reading +
offset). Set them with `python 02_calibrate.py offsets --arm <arm>`. An arm without a file has zero offsets.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from core import config
from core.paths import arm_calibration_dir

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
FILE_NAME = "joint_offsets.yaml"
COMMENT = """Fine joint offsets: software corrections on top of LeRobot's motor calibration, in degrees.
true angle = LeRobot reading + offset; motor target = wanted angle - offset.
Set with `python 02_calibrate.py offsets --arm <arm>` (PyBullet sliders over the live arm)."""


def own_offsets_file(arm: str | None) -> Path | None:
	"""Where `arm`'s offsets are kept: config/calibration/<arm>/joint_offsets.yaml (None for no arm)."""
	return None if arm is None else arm_calibration_dir(arm) / FILE_NAME


def offsets_file(arm: str | None = None) -> Path | None:
	"""The fine offsets file of one arm, by its name (e.g. follower_blue); None for no arm (zero offsets)."""
	return own_offsets_file(arm)


def _zeros() -> dict[str, float]:
	return {name: 0.0 for name in ARM_JOINTS}


def load_fine_joint_offsets(path: str | Path | None = None, *, arm: str | None = None) -> dict[str, float]:
	"""The offsets in degrees by joint, from `arm`'s file (or `path`); zeros when there is no file. Re-read when the
	file changes."""
	file_path = own_offsets_file(arm) if arm is not None else (Path(path) if path is not None else None)
	if file_path is None or not file_path.is_file():
		return _zeros()
	values = (config.load(file_path, {}) or {}).get("joint_offsets_deg", {}) or {}
	offsets = {name: float(values.get(name, 0.0)) for name in ARM_JOINTS}
	invalid = [name for name, value in offsets.items() if not math.isfinite(value)]
	if invalid:
		raise ValueError(f"{file_path}: fine joint offsets must be finite degrees; invalid: {invalid}")
	return offsets


def save_fine_joint_offsets(offsets: Mapping[str, float], path: str | Path) -> None:
	"""Save one finite offset in degrees for each arm joint to `path` (own_offsets_file(arm))."""
	values = {name: round(float(offsets.get(name, 0.0)), 4) for name in ARM_JOINTS}
	invalid = [name for name, value in values.items() if not math.isfinite(value)]
	if invalid:
		raise ValueError(f"Fine joint offsets must be finite degrees; invalid: {invalid}")
	config.save(Path(path), {"unit": "degrees", "joint_offsets_deg": values}, comment=COMMENT)


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