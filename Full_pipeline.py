"""One-frame YOLO/ZED perception and collision-checked SO-101 object-approach pipeline.

The pipeline estimates object point clouds from one frame, reads the arm pose,
builds a PyBullet scene, plans with M-FRPSO and RRT-Connect, and only moves when
--execute is supplied and the operator confirms.

Usage:
	python Full_pipeline.py                 # show one camera frame, then headless PyBullet
	python Full_pipeline.py --gui            # show the snapshot scene
	python Full_pipeline.py --gui --no-robot # inspect with the URDF zero pose
	python Full_pipeline.py --no-frame-preview # skip the camera window
	python Full_pipeline.py --simulate --gui # camera-seeded simulation; no physical robot connection
	python Full_pipeline.py --execute        # plan, then explicitly authorize arm motion
	python Full_pipeline.py --execute --gui   # animate the real arm feedback in PyBullet
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ROI_FILE = ROOT / "yolo_dataset" / "roi.json"
CALIBRATION_FILE = ROOT / "camera_transform.npy"
URDF_FILE = ROOT / "lerobot" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
TABLE_Z = 0.0
MIN_GRIPPER_TABLE_CLEARANCE_M = 0.04
MAX_TABLE_OBJECT_CENTER_HEIGHT_M = 0.20
FIXED_WRIST_ROLL_RAD = -np.pi / 2.0
MAX_START_LIMIT_CORRECTION_DEG =10.0
GRIPPER_OPEN_POSITION = 80.0
OBJECT_COLLISION_PADDING_M = 0.008
OBJECT_CENTER_APPROACH_LIFT_M = 0.05
CARTESIAN_POSITION_TOLERANCE_M = 0.01

POINT_COLORS = [
	(0.10, 0.85, 0.25),
	(1.00, 0.45, 0.10),
	(0.15, 0.45, 1.00),
	(1.00, 0.85, 0.10),
	(0.90, 0.15, 0.75),
	(0.10, 0.85, 0.85),
]
IGNORED_OBJECT_CLASSES = {"gripper", "bin"}
OBJECT_STL_FILES = {
	"cube": ROOT / "stl" / "cube_large.stl",
	"cylinder": ROOT / "stl" / "cylinder_large.stl",
	"dodecahedron": ROOT / "stl" / "dodecahedron_large.stl",
	"hexagonal": ROOT / "stl" / "hexagonal_prism_large.stl",
	"hexagonal_prism": ROOT / "stl" / "hexagonal_prism_large.stl",
}


def is_ignored_object_class(class_name: str) -> bool:
	return class_name.strip().casefold() in IGNORED_OBJECT_CLASSES


def is_table_object_center(center, max_height_m=MAX_TABLE_OBJECT_CENTER_HEIGHT_M) -> bool:
	return float(center[2]) <= TABLE_Z + max_height_m


def object_stl_path(class_name: str) -> Path | None:
	path = OBJECT_STL_FILES.get(class_name.strip().casefold())
	if path is not None and not path.exists():
		raise FileNotFoundError(f"STL asset not found for {class_name!r}: {path}")
	return path


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--serial", type=int, default=None, help="ZED serial number; defaults to camera ID")
	parser.add_argument("--camera-id", type=int, default=0)
	parser.add_argument("--weights", type=Path, default=None, help="default: latest segmentation model")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--resolution", choices=("HD2K", "HD1080", "HD720", "VGA"), default="HD1080")
	parser.add_argument("--fps", type=int, default=15)
	parser.add_argument("--min-depth", type=float, default=0.10)
	parser.add_argument("--max-depth", type=float, default=2.0)
	parser.add_argument("--max-object-center-height", type=float, default=MAX_TABLE_OBJECT_CENTER_HEIGHT_M,
						help="ignore detections whose point-cloud center is higher than this above the table")
	parser.add_argument("--min-points", type=int, default=20)
	parser.add_argument("--max-points-per-object", type=int, default=800)
	parser.add_argument("--no-frame-preview", action="store_true", help="do not show the captured camera frame")
	parser.add_argument("--gui", action="store_true", help="show the PyBullet snapshot scene; default is headless")
	parser.add_argument("--no-robot", action="store_true", help="use the URDF zero pose instead of reading the arm")
	parser.add_argument("--simulate", action="store_true",
						help="run M-FRPSO and animate the RRT path in PyBullet without connecting to the arm")
	parser.add_argument("--object-standoff", type=float, default=0.0,
						help="extra horizontal clearance beyond the PyBullet object surface, in meters")
	parser.add_argument("--object-center-lift", type=float, default=OBJECT_CENTER_APPROACH_LIFT_M,
						help="small vertical clearance above point-cloud center for the near-side approach, in meters")
	parser.add_argument("--min-gripper-table-clearance", type=float, default=MIN_GRIPPER_TABLE_CLEARANCE_M,
						help="minimum z clearance for gripper collision geometry above the table, in meters")
	parser.add_argument("--gripper-open-position", type=float, default=GRIPPER_OPEN_POSITION,
						help="gripper opening before arm motion, on LeRobot's 0-100 scale")
	parser.add_argument("--joint-limit-margin-deg", type=float, default=0.5,
						help="inward margin used when correcting an out-of-calibration start joint")
	parser.add_argument("--execute", action="store_true", help="execute the planned arm path after typed confirmation")
	return parser


def object_cloud_from_mask(
	cloud: np.ndarray,
	polygon: np.ndarray,
	roi: tuple[int, int, int, int],
	camera_to_robot: np.ndarray,
	*,
	min_depth: float = 0.10,
	max_depth: float = 2.0,
	min_points: int = 20,
	z_tolerance_m: float = 0.015,
):
	"""Return robust table-frame mask points, their mean center, and pixel center."""
	import cv2

	x0, y0, width, height = roi
	mask = np.zeros((height, width), dtype=np.uint8)
	cv2.fillPoly(mask, [np.rint(np.asarray(polygon)).astype(np.int32)], 1)
	mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
	ys, xs = np.nonzero(mask)
	if not len(xs):
		return None, None, None

	camera_xyz = cloud[y0:y0 + height, x0:x0 + width, :3][ys, xs]
	valid = np.isfinite(camera_xyz).all(axis=1)
	valid &= (camera_xyz[:, 2] >= min_depth) & (camera_xyz[:, 2] <= max_depth)
	camera_xyz, xs, ys = camera_xyz[valid], xs[valid], ys[valid]
	if len(camera_xyz) < min_points:
		return None, None, None

	z_median = np.median(camera_xyz[:, 2])
	z_mad = np.median(np.abs(camera_xyz[:, 2] - z_median))
	tolerance = max(z_tolerance_m, 3.0 * 1.4826 * z_mad)
	inliers = np.abs(camera_xyz[:, 2] - z_median) <= tolerance
	camera_xyz, xs, ys = camera_xyz[inliers], xs[inliers], ys[inliers]
	if len(camera_xyz) < min_points:
		return None, None, None

	homogeneous = np.column_stack((camera_xyz, np.ones(len(camera_xyz))))
	table_xyz = (camera_to_robot @ homogeneous.T).T[:, :3]
	center = table_xyz.mean(axis=0)
	pixel_center = (int(round(xs.mean() + x0)), int(round(ys.mean() + y0)))
	return table_xyz.astype(np.float32), center.astype(np.float32), pixel_center


def capture_one_frame(args):
	"""Run YOLO once and pair each valid mask with its filtered point cloud."""
	import cv2
	import pyzed.sl as sl
	from ultralytics import YOLO
	from yolo_dataset.object_3d_live import latest_weights, load_camera_transform, roi_for_frame

	if not ROI_FILE.exists():
		raise FileNotFoundError(f"ROI config not found: {ROI_FILE}")
	roi_data = json.loads(ROI_FILE.read_text())
	camera_to_robot = load_camera_transform(CALIBRATION_FILE)
	weights = args.weights or latest_weights()
	model = YOLO(str(weights))
	print(f"YOLO model: {weights}")
	print(f"Camera-to-robot calibration: {CALIBRATION_FILE}")

	init = sl.InitParameters()
	init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
	init.camera_fps = args.fps
	init.depth_mode = sl.DEPTH_MODE.QUALITY
	init.coordinate_units = sl.UNIT.METER
	init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
	if args.serial is not None:
		init.set_from_serial_number(args.serial)
	else:
		init.set_from_camera_id(args.camera_id)

	camera = sl.Camera()
	status = camera.open(init)
	if status != sl.ERROR_CODE.SUCCESS:
		raise RuntimeError(f"Could not open ZED camera: {status}")

	image_mat, cloud_mat = sl.Mat(), sl.Mat()
	try:
		if camera.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
			raise RuntimeError("Could not capture the initial ZED frame")
		camera.retrieve_image(image_mat, sl.VIEW.LEFT)
		camera.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
		frame = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
		cloud = cloud_mat.get_data()
		if cloud.shape[:2] != frame.shape[:2]:
			raise RuntimeError(f"Image/point cloud sizes differ: {frame.shape[:2]} vs {cloud.shape[:2]}")

		roi = roi_for_frame(frame.shape, roi_data)
		x0, y0, roi_width, roi_height = roi
		roi_image = frame[y0:y0 + roi_height, x0:x0 + roi_width]
		prediction = model.predict(
			roi_image,
			imgsz=max(roi_width, roi_height),
			conf=args.conf,
			verbose=False,
		)[0]

		detections = []
		preview = frame.copy()
		if prediction.masks is not None:
			for polygon, class_id, confidence in zip(
				prediction.masks.xy,
				prediction.boxes.cls.int().tolist(),
				prediction.boxes.conf.tolist(),
			):
				class_name = str(model.names[class_id])
				color = tuple(int(value * 255) for value in POINT_COLORS[class_id % len(POINT_COLORS)][::-1])
				full_polygon = np.rint(np.asarray(polygon) + [x0, y0]).astype(np.int32)
				cv2.polylines(preview, [full_polygon], True, color, 2)
				if is_ignored_object_class(class_name):
					label = f"{class_name} {confidence:.2f} | ignored"
					text_position = tuple(full_polygon.min(axis=0))
					cv2.putText(preview, label, (int(text_position[0]), max(18, int(text_position[1]) - 8)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
					continue
				points, center, pixel_center = object_cloud_from_mask(
					cloud,
					polygon,
					roi,
					camera_to_robot,
					min_depth=args.min_depth,
					max_depth=args.max_depth,
					min_points=args.min_points,
				)
				if center is None:
					print(f"{class_name} conf={confidence:.2f}: insufficient valid cloud points")
					label = f"{class_name} {confidence:.2f} | cloud unavailable"
					text_position = tuple(full_polygon.min(axis=0))
				elif not is_table_object_center(center, args.max_object_center_height):
					print(f"{class_name} conf={confidence:.2f}: center Z={center[2]:.3f} m above tabletop range; ignored")
					label = f"{class_name} {confidence:.2f} | above table, ignored"
					text_position = pixel_center
					cv2.circle(preview, pixel_center, 6, (255, 255, 255), -1)
				else:
					cv2.circle(preview, pixel_center, 6, (255, 255, 255), -1)
					label = f"{class_name} {confidence:.2f} | XYZ {np.round(center, 3)} m"
					text_position = pixel_center
				cv2.putText(preview, label, (int(text_position[0]), max(18, int(text_position[1]) - 8)),
							cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
				if center is None or not is_table_object_center(center, args.max_object_center_height):
					continue
				detections.append({
					"class_name": class_name,
					"confidence": float(confidence),
					"points": points,
					"center": center,
					"pixel_center": pixel_center,
				})
		cv2.rectangle(preview, (x0, y0), (x0 + roi_width, y0 + roi_height), (255, 255, 255), 2)
		return detections, preview
	finally:
		camera.close()


def show_camera_frame(frame) -> bool:
	"""Show the captured, annotated frame once; any key continues, q/ESC cancels."""
	import cv2

	window = "Full pipeline: one-frame YOLO/ZED result"
	cv2.namedWindow(window, cv2.WINDOW_NORMAL)
	scale = min(1.0, 1400.0 / frame.shape[1])
	view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
	cv2.imshow(window, view)
	print("Review the single YOLO/ZED frame. Press any key to continue, or q/Esc to cancel.")
	key = cv2.waitKey(0) & 0xFF
	cv2.destroyWindow(window)
	return key not in (ord("q"), 27)


def near_side_target(detection, current_ee_position, standoff_m=0.0, collision_aabb=None):
	"""Target the near side of the PyBullet object geometry at cloud-center height."""
	if collision_aabb is None:
		points = detection["points"]
		lower = points.min(axis=0).astype(float)
		upper = points.max(axis=0).astype(float)
		lower[:2] -= OBJECT_COLLISION_PADDING_M
		upper[:2] += OBJECT_COLLISION_PADDING_M
	else:
		lower = np.asarray(collision_aabb[0], dtype=float)
		upper = np.asarray(collision_aabb[1], dtype=float)
	box_center = (lower[:2] + upper[:2]) / 2.0
	half_extents = (upper[:2] - lower[:2]) / 2.0
	direction = np.asarray(current_ee_position[:2], dtype=float) - box_center
	distance = float(np.linalg.norm(direction))
	if distance < 1e-6:
		raise ValueError("Cannot determine a near-side approach direction from the current end-effector position")
	direction /= distance
	surface_distance = float(np.dot(np.abs(direction), half_extents))
	return np.array([
		*(box_center + direction * (surface_distance + standoff_m)),
		float(detection["center"][2]),
	])


def set_gripper_position(robot, target, *, tolerance=2.0, timeout=30.0, feedback_callback=None):
	"""Move only the gripper while holding all measured arm joints in place."""
	deadline = time.monotonic() + timeout
	position = float(robot.get_observation()["gripper.pos"])
	while time.monotonic() < deadline:
		observation = robot.get_observation()
		hold = {
			key: value for key, value in observation.items()
			if key.endswith(".pos") and key != "gripper.pos"
		}
		robot.send_action({**hold, "gripper.pos": float(target)})
		time.sleep(0.04)
		observation = robot.get_observation()
		if feedback_callback is not None:
			feedback_callback(observation)
		position = float(observation["gripper.pos"])
		if abs(position - target) <= tolerance:
			return True, position
	return False, position


def gripper_position_to_urdf(lower: float, upper: float, position_0_to_100: float) -> float:
	return float(lower + np.clip(position_0_to_100 / 100.0, 0.0, 1.0) * (upper - lower))


def restore_pybullet_robot_pose(robot_id, joint_names, joint_positions, gripper_position, physics_client_id):
	import pybullet as p

	joint_indices = {
		p.getJointInfo(robot_id, index, physicsClientId=physics_client_id)[1].decode(): index
		for index in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id))
	}
	for name, position in zip(joint_names, joint_positions):
		p.resetJointState(robot_id, joint_indices[name], float(position), physicsClientId=physics_client_id)
	gripper_index = joint_indices["gripper"]
	gripper_info = p.getJointInfo(robot_id, gripper_index, physicsClientId=physics_client_id)
	p.resetJointState(
		robot_id,
		gripper_index,
		gripper_position_to_urdf(gripper_info[8], gripper_info[9], gripper_position),
		physicsClientId=physics_client_id,
	)
	p.performCollisionDetection(physicsClientId=physics_client_id)


def sync_pybullet_from_observation(robot_id, joint_names, observation, physics_client_id):
	"""Mirror one measured LeRobot observation into its PyBullet twin."""
	import pybullet as p
	from fine_joint_offsets import corrected_observation

	logical_observation = corrected_observation(observation)
	joint_indices = {
		p.getJointInfo(robot_id, index, physicsClientId=physics_client_id)[1].decode(): index
		for index in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id))
	}
	for name in joint_names:
		p.resetJointState(
			robot_id,
			joint_indices[name],
			math.radians(float(logical_observation[f"{name}.pos"])),
			physicsClientId=physics_client_id,
		)
	gripper_index = joint_indices["gripper"]
	gripper_info = p.getJointInfo(robot_id, gripper_index, physicsClientId=physics_client_id)
	p.resetJointState(
		robot_id,
		gripper_index,
		gripper_position_to_urdf(gripper_info[8], gripper_info[9], float(observation["gripper.pos"])),
		physicsClientId=physics_client_id,
	)
	p.stepSimulation(physicsClientId=physics_client_id)


def animate_simulated_path(robot_id, joint_indices, path, *, physics_client_id, gui, step_rad=np.deg2rad(2.0)):
	"""Animate a collision-checked joint-space path in PyBullet only."""
	import pybullet as p

	previous = np.asarray(path[0], dtype=float)
	for waypoint in path[1:]:
		waypoint = np.asarray(waypoint, dtype=float)
		steps = max(1, int(np.ceil(np.max(np.abs(waypoint - previous)) / step_rad)))
		for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
			configuration = previous + fraction * (waypoint - previous)
			for joint_index, position in zip(joint_indices, configuration):
				p.resetJointState(robot_id, joint_index, float(position), physicsClientId=physics_client_id)
			p.stepSimulation(physicsClientId=physics_client_id)
			if gui:
				time.sleep(1.0 / 120.0)
		previous = waypoint
	return previous


def load_robot_state(no_robot: bool):
	"""Read the SO-101 joints once and keep the connection open for optional execution."""
	from IK_control import ARM_JOINTS, connect_robot, read_joints_rad

	if no_robot:
		print("Using simulated home pose; robot was not connected")
		joint_positions = np.zeros(len(ARM_JOINTS), dtype=float)
		joint_positions[ARM_JOINTS.index("wrist_roll")] = FIXED_WRIST_ROLL_RAD
		return None, ARM_JOINTS, joint_positions

	robot = connect_robot()
	joint_positions = np.asarray(read_joints_rad(robot), dtype=float)
	print("Read current SO-101 joint positions (rad):", np.round(joint_positions, 4))
	return robot, ARM_JOINTS, joint_positions


def build_planning_scene(
	detections,
	joint_names,
	joint_positions,
	*,
	gui,
	max_points_per_object,
	gripper_position=GRIPPER_OPEN_POSITION,
	min_gripper_table_clearance=MIN_GRIPPER_TABLE_CLEARANCE_M,
	collision_padding=OBJECT_COLLISION_PADDING_M,
):
	"""Build the table, measured arm, point-cloud obstacles, M-FRPSO, and RRT-Connect."""
	import pybullet as p
	from IK_control import EE_LINK
	from modified_frpso_ik import ModifiedFRPSOIK, load_robot_for_collision
	from rrt_connect import RRTConnectPlanner

	client = p.connect(p.GUI if gui else p.DIRECT)
	if client < 0:
		raise RuntimeError("Could not connect to PyBullet")
	try:
		p.setAdditionalSearchPath(str(URDF_FILE.parent), physicsClientId=client)
		p.setGravity(0.0, 0.0, 0.0, physicsClientId=client)
		p.setRealTimeSimulation(0, physicsClientId=client)

		table_half_extents = [0.75, 0.9, 0.02]
		table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents,
													physicsClientId=client)
		table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents,
											   rgbaColor=[0.55, 0.57, 0.60, 1.0], physicsClientId=client)
		table_id = p.createMultiBody(
			0, table_collision, table_visual, basePosition=[0.0, 0.25, TABLE_Z - table_half_extents[2]],
			physicsClientId=client,
		)

		robot_id = load_robot_for_collision(str(URDF_FILE), physics_client_id=client)
		index_by_name = {
			p.getJointInfo(robot_id, i, physicsClientId=client)[1].decode(): i
			for i in range(p.getNumJoints(robot_id, physicsClientId=client))
		}
		link_by_name = {
			p.getJointInfo(robot_id, i, physicsClientId=client)[12].decode(): i
			for i in range(p.getNumJoints(robot_id, physicsClientId=client))
		}
		for name, position in zip(joint_names, joint_positions):
			p.resetJointState(robot_id, index_by_name[name], float(position), physicsClientId=client)
		gripper_info = p.getJointInfo(robot_id, index_by_name["gripper"], physicsClientId=client)
		gripper_angle = gripper_position_to_urdf(gripper_info[8], gripper_info[9], gripper_position)
		p.resetJointState(robot_id, index_by_name["gripper"], float(gripper_angle), physicsClientId=client)

		object_body_ids = []
		p.addUserDebugText(
			"SO-101 measured pose | point-cloud obstacles | table top Z=0",
			[0.0, 0.0, 0.8], textColorRGB=[0.1, 0.1, 0.1], textSize=1.2, physicsClientId=client,
		)
		for index, detection in enumerate(detections):
			all_points = detection["points"]
			color = POINT_COLORS[index % len(POINT_COLORS)]
			class_name = detection["class_name"].strip().casefold()
			mesh_path = object_stl_path(class_name)
			if mesh_path is not None:
				collision_shape = p.createCollisionShape(
					p.GEOM_MESH,
					fileName=str(mesh_path),
					meshScale=[0.001, 0.001, 0.001],
					physicsClientId=client,
				)
				visual_shape = p.createVisualShape(
					p.GEOM_MESH,
					fileName=str(mesh_path),
					meshScale=[0.001, 0.001, 0.001],
					rgbaColor=[*color, 0.8],
					physicsClientId=client,
				)
				body_position = [float(detection["center"][0]), float(detection["center"][1]), TABLE_Z]
				print(f"  {class_name} geometry: {mesh_path.name} (fixed large STL for visual and collision)")
				geometry_description = f"collision STL={mesh_path.name}"
			else:
				lower = all_points.min(axis=0).astype(float)
				upper = all_points.max(axis=0).astype(float)
				lower[:2] -= collision_padding
				upper[:2] += collision_padding
				lower[2] = TABLE_Z
				upper[2] = max(upper[2] + collision_padding, TABLE_Z + 0.01)
				box_center = (lower + upper) / 2.0
				half_extents = np.maximum((upper - lower) / 2.0, 0.005)
				collision_shape = p.createCollisionShape(
					p.GEOM_BOX, halfExtents=half_extents.tolist(), physicsClientId=client
				)
				visual_shape = p.createVisualShape(
					p.GEOM_BOX, halfExtents=half_extents.tolist(), rgbaColor=[*color, 0.18], physicsClientId=client
				)
				body_position = box_center.tolist()
				geometry_description = f"collision AABB={half_extents * 2}"
			body_id = p.createMultiBody(
				baseMass=0,
				baseCollisionShapeIndex=collision_shape,
				baseVisualShapeIndex=visual_shape,
				basePosition=body_position,
				physicsClientId=client,
			)
			object_body_ids.append(body_id)

			if mesh_path is None:
				points = all_points
				if len(points) > max_points_per_object:
					sample_indices = np.linspace(0, len(points) - 1, max_points_per_object).astype(int)
					points = points[sample_indices]
				p.addUserDebugPoints(points.tolist(), [color] * len(points), pointSize=3, lifeTime=0,
									 physicsClientId=client)
			center = detection["center"]
			marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.012,
										 rgbaColor=[*color, 1.0], physicsClientId=client)
			p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker,
							  basePosition=center.tolist(), physicsClientId=client)
			p.addUserDebugText(
				f"{detection['class_name']} {detection['confidence']:.2f} center={np.round(center, 3)} m",
				(center + np.array([0.0, 0.0, 0.025])).tolist(), textColorRGB=list(color), textSize=1.0,
				physicsClientId=client,
			)
			print(f"{index + 1}. {detection['class_name']} conf={detection['confidence']:.2f}: "
				  f"center={np.round(center, 4)} m from {len(all_points)} points; {geometry_description}")

		arm_indices = [index_by_name[name] for name in joint_names]
		wrist_index = index_by_name["wrist_roll"]
		minimum_gripper_link_heights = {
			link_by_name[name]: TABLE_Z + min_gripper_table_clearance
			for name in ("gripper_link", "moving_jaw_so101_v1_link")
			if name in link_by_name
		}
		checker = ModifiedFRPSOIK(
			robot_id,
			link_by_name[EE_LINK],
			arm_indices,
			obstacle_body_ids=[table_id, *object_body_ids],
			physics_client_id=client,
			check_self_collision=False,
			fixed_joint_values={wrist_index: FIXED_WRIST_ROLL_RAD},
			allowed_robot_obstacle_link_pairs=[(table_id, -1, -1)],
			minimum_link_heights=minimum_gripper_link_heights,
		)
		transition_checker = ModifiedFRPSOIK(
			robot_id,
			link_by_name[EE_LINK],
			arm_indices,
			obstacle_body_ids=[table_id, *object_body_ids],
			physics_client_id=client,
			check_self_collision=False,
			allowed_robot_obstacle_link_pairs=[(table_id, -1, -1)],
			minimum_link_heights=minimum_gripper_link_heights,
		)
		rrt = RRTConnectPlanner(checker, max_joint_step_rad=np.deg2rad(3.0))
		ee_state = p.getLinkState(robot_id, link_by_name[EE_LINK],
								  computeForwardKinematics=True, physicsClientId=client)
		print("PyBullet current end-effector FK:", np.round(ee_state[4], 4), "m")
		print(f"Planning scene ready: table top Z={TABLE_Z:.3f} m, {len(object_body_ids)} object obstacle(s), "
			  f"{len(rrt.active_offsets)} active joint dimensions, min gripper clearance="
			  f"{min_gripper_table_clearance:.3f} m; self-collision checks disabled")
		if gui:
			p.resetDebugVisualizerCamera(1.65, 45, -45, [0.15, 0.25, 0.20], physicsClientId=client)
		return client, robot_id, checker, rrt, transition_checker, object_body_ids
	except Exception:
		p.disconnect(physicsClientId=client)
		raise


def main(argv=None):
	args = build_parser().parse_args(argv)
	if not 0.0 <= args.conf <= 1.0:
		raise SystemExit("--conf must be between 0 and 1")
	if args.min_depth < 0 or args.max_depth <= args.min_depth:
		raise SystemExit("Require 0 <= --min-depth < --max-depth")
	if not np.isfinite(args.max_object_center_height) or args.max_object_center_height <= 0:
		raise SystemExit("--max-object-center-height must be finite and positive")
	if args.min_points < 1 or args.max_points_per_object < 1:
		raise SystemExit("--min-points and --max-points-per-object must be positive")
	if not np.isfinite(args.object_standoff) or args.object_standoff < 0:
		raise SystemExit("--object-standoff must be finite and non-negative")
	if not np.isfinite(args.object_center_lift) or args.object_center_lift < 0:
		raise SystemExit("--object-center-lift must be finite and non-negative")
	if not np.isfinite(args.min_gripper_table_clearance) or args.min_gripper_table_clearance < 0:
		raise SystemExit("--min-gripper-table-clearance must be finite and non-negative")
	if not 0.0 <= args.gripper_open_position <= 100.0:
		raise SystemExit("--gripper-open-position must be between 0 and 100")
	if not np.isfinite(args.joint_limit_margin_deg) or args.joint_limit_margin_deg < 0:
		raise SystemExit("--joint-limit-margin-deg must be finite and non-negative")
	if args.simulate and args.execute:
		raise SystemExit("--simulate and --execute are mutually exclusive")
	if args.execute and args.no_robot:
		raise SystemExit("--execute requires a connected robot; remove --no-robot")

	detections, frame = capture_one_frame(args)
	if not args.no_frame_preview and not show_camera_frame(frame):
		print("Cancelled after frame review; robot state was not read and PyBullet was not opened.")
		return
	if not detections:
		raise RuntimeError("YOLO produced no objects with enough valid point-cloud data")

	robot = None
	client = None
	import pybullet as p
	from IK_control import read_joints_rad, send_joints
	try:
		robot, joint_names, joint_positions = load_robot_state(args.no_robot or args.simulate)
		measured_gripper_position = (
			float(robot.get_observation()["gripper.pos"]) if robot is not None else args.gripper_open_position
		)
		simulated_gripper_position = args.gripper_open_position if args.execute else measured_gripper_position
		wrist_offset = joint_names.index("wrist_roll")
		client, robot_id, checker, rrt, transition_checker, object_body_ids = build_planning_scene(
			detections,
			joint_names,
			joint_positions,
			gui=args.gui,
			max_points_per_object=args.max_points_per_object,
			gripper_position=simulated_gripper_position,
			min_gripper_table_clearance=args.min_gripper_table_clearance,
		)
		twin_feedback_callback = None
		if args.gui and robot is not None:
			twin_feedback_callback = lambda observation: sync_pybullet_from_observation(
				robot_id, joint_names, observation, client
			)
		limit_margin = np.deg2rad(args.joint_limit_margin_deg)
		if np.any(transition_checker.upper - transition_checker.lower <= 2.0 * limit_margin):
			raise RuntimeError("Joint-limit margin is too large for one or more calibrated joint ranges")
		aligned_start = np.clip(
			joint_positions,
			transition_checker.lower + limit_margin,
			transition_checker.upper - limit_margin,
		)
		aligned_start[wrist_offset] = FIXED_WRIST_ROLL_RAD
		start_correction_required = not np.allclose(joint_positions, aligned_start, atol=1e-6, rtol=0.0)
		if not transition_checker.state_is_collision_free(joint_positions, check_joint_limits=False):
			collision_message = (
				"Measured start pose collides with the table/object scene; no automatic joint correction is safe. "
				"Move the gripper clear of the table, then retry."
			)
			if args.gui:
				restore_pybullet_robot_pose(
					robot_id,
					joint_names,
					joint_positions,
					measured_gripper_position,
					client,
				)
				p.addUserDebugText(
					"START COLLISION: measured pose shown; no motion planned",
					[0.0, 0.0, 0.72],
					textColorRGB=[1.0, 0.0, 0.0],
					textSize=1.3,
					physicsClientId=client,
				)
				print(collision_message)
				input("PyBullet is showing the measured colliding pose. Press Enter to close: ")
				return
			raise RuntimeError(collision_message)
		urdf_limit_violation = np.maximum(
			transition_checker.urdf_lower - joint_positions,
			joint_positions - transition_checker.urdf_upper,
		)
		urdf_limit_violation = np.maximum(urdf_limit_violation, 0.0)
		if np.max(urdf_limit_violation) > np.deg2rad(MAX_START_LIMIT_CORRECTION_DEG):
			bad_joint = int(np.argmax(urdf_limit_violation))
			raise RuntimeError(
				f"{joint_names[bad_joint]} is {math.degrees(urdf_limit_violation[bad_joint]):.2f} deg "
				f"outside its URDF range; automatic start correction is limited to "
				f"{MAX_START_LIMIT_CORRECTION_DEG:.1f} deg"
			)
		if start_correction_required:
			if not transition_checker.edge_is_collision_free(
				joint_positions,
				aligned_start,
				check_joint_limits=False,
			):
				raise RuntimeError("The collision-checked start correction is unsafe; no motion was sent")
			print("A collision-checked start correction is required before the calibrated 4D route:")
			for name, current, target in zip(joint_names, joint_positions, aligned_start):
				if not np.isclose(current, target, atol=1e-6, rtol=0.0):
					print(f"  {name}: {math.degrees(current):.2f} -> {math.degrees(target):.2f} deg")
		if not checker.state_is_collision_free(aligned_start):
			raise RuntimeError("The current robot configuration is in collision; no plan was generated")

		selected_index = max(range(len(detections)), key=lambda index: detections[index]["confidence"])
		selected = detections[selected_index]
		current_ee = p.getLinkState(
			robot_id,
			checker.ee_link_index,
			computeForwardKinematics=True,
			physicsClientId=client,
		)[4]
		selected_aabb = p.getAABB(object_body_ids[selected_index], physicsClientId=client)
		target = near_side_target(selected, current_ee, args.object_standoff, collision_aabb=selected_aabb)
		target[2] += args.object_center_lift
		print(f"Selected object: {selected['class_name']} conf={selected['confidence']:.2f}")
		print(f"Collision-free near-side target: {np.round(target, 4)} m "
			  f"({args.object_standoff * 1000:.0f} mm outside object box; "
			  f"{args.object_center_lift * 1000:.0f} mm above cloud-center height)")

		ik_result = checker.solve(
			target,
			start_q=aligned_start,
			num_particles=160,
			elite_count=5,
			max_iterations=160,
			position_tolerance_m=CARTESIAN_POSITION_TOLERANCE_M,
			seed=0,
		)
		if ik_result is None or not ik_result.success:
			error = float("inf") if ik_result is None else ik_result.position_error_m
			raise RuntimeError(f"M-FRPSO did not find a collision-free pre-grasp solution; position error={error:.4f} m")
		print(f"M-FRPSO goal found: position error={ik_result.position_error_m * 1000:.1f} mm")

		route = rrt.plan(aligned_start, ik_result.joint_positions, seed=0)
		if route is None:
			raise RuntimeError("RRT-Connect could not find a collision-free path to the M-FRPSO goal")
		print(f"RRT-Connect path: {len(route.path)} configurations, {route.iterations} iterations")
		for index, waypoint in enumerate(route.path):
			print(f"  {index}: {np.round(waypoint, 4)} rad")

		if args.simulate:
			print("Simulation mode: camera snapshot and PyBullet scene only; robot was not connected.")
			restore_pybullet_robot_pose(
				robot_id,
				joint_names,
				joint_positions,
				args.gripper_open_position,
				client,
			)
			simulation_path = [joint_positions.copy()]
			if start_correction_required:
				if not transition_checker.edge_is_collision_free(
					joint_positions, aligned_start, check_joint_limits=False
				):
					raise RuntimeError("Simulated start correction is not collision-free")
				simulation_path.append(aligned_start.copy())
			simulation_path.extend(configuration.copy() for configuration in route.path[1:])
			final_configuration = animate_simulated_path(
				robot_id,
				checker.joint_indices,
				simulation_path,
				physics_client_id=client,
				gui=args.gui,
			)
			final_state = p.getLinkState(
				robot_id, checker.ee_link_index, computeForwardKinematics=True, physicsClientId=client
			)
			final_error = float(np.linalg.norm(np.asarray(final_state[4]) - target))
			print(f"PyBullet simulation finished; final Cartesian error={final_error * 1000:.1f} mm")
		elif args.execute:
			if input("Type MOVE to execute this collision-checked path: ").strip().upper() != "MOVE":
				print("Cancelled; no motion command was sent")
				return
			gripper_opened, gripper_position = set_gripper_position(
				robot,
				args.gripper_open_position,
				feedback_callback=twin_feedback_callback,
			)
			if not gripper_opened:
				raise RuntimeError(
					f"Gripper opening timed out at measured position {gripper_position:.1f}/100 "
					f"(target {args.gripper_open_position:.1f}); "
					"arm motion was not started"
				)
			print(f"Gripper opened to {gripper_position:.1f}/100 before arm motion")
			joint_positions = np.asarray(read_joints_rad(robot), dtype=float)
			aligned_start = np.clip(
				joint_positions,
				transition_checker.lower + limit_margin,
				transition_checker.upper - limit_margin,
			)
			aligned_start[wrist_offset] = FIXED_WRIST_ROLL_RAD
			start_correction_required = not np.allclose(joint_positions, aligned_start, atol=1e-6, rtol=0.0)
			if not transition_checker.state_is_collision_free(joint_positions, check_joint_limits=False):
				raise RuntimeError("Measured arm pose after opening the gripper collides with the table/object scene")
			if start_correction_required and not transition_checker.edge_is_collision_free(
				joint_positions,
				aligned_start,
				check_joint_limits=False,
			):
				raise RuntimeError("Updated start correction after opening the gripper is not collision-free")
			if not checker.state_is_collision_free(aligned_start):
				raise RuntimeError("Updated planning start after opening the gripper is invalid")
			route = rrt.plan(aligned_start, ik_result.joint_positions, seed=0)
			if route is None:
				raise RuntimeError("RRT-Connect could not plan from the measured pose after opening the gripper")
			if start_correction_required:
				correction_steps = max(
					20,
					int(np.ceil(np.max(np.abs(np.degrees(aligned_start - joint_positions))) / 3.0)) + 1,
				)
				correction_reached = send_joints(
					robot,
					aligned_start,
					steps=correction_steps,
					tol_deg=0.5,
					timeout=8.0,
					feedback_callback=twin_feedback_callback,
				)
				joint_positions = np.asarray(read_joints_rad(robot), dtype=float)
				if not transition_checker.state_is_collision_free(joint_positions):
					raise RuntimeError(
						"Measured pose after start correction is outside calibrated limits or colliding; "
						"stopping before object route"
					)
				if not correction_reached:
					print("Correction target timed out, but measured pose is safe; replanning from measured feedback")
				route_start = joint_positions.copy()
				route_start[wrist_offset] = FIXED_WRIST_ROLL_RAD
				if not transition_checker.edge_is_collision_free(joint_positions, route_start):
					raise RuntimeError("Measured start-correction residual cannot be safely bridged; stopping")
				route = rrt.plan(route_start, ik_result.joint_positions, seed=0)
				if route is None:
					raise RuntimeError("RRT-Connect could not replan from the measured aligned pose")
			for waypoint_index, waypoint in enumerate(route.path[1:], start=1):
				if not transition_checker.edge_is_collision_free(joint_positions, waypoint):
					raise RuntimeError(f"Measured-to-waypoint edge {waypoint_index} is not collision-free; stopping")
				steps = max(
					20,
					int(np.ceil(np.max(np.abs(np.degrees(waypoint - joint_positions))) / 3.0)) + 1,
				)
				waypoint_reached = send_joints(
					robot,
					waypoint,
					steps=steps,
					tol_deg=3.0,
					timeout=30.0,
					feedback_callback=twin_feedback_callback,
				)
				joint_positions = np.asarray(read_joints_rad(robot), dtype=float)
				if not transition_checker.state_is_collision_free(joint_positions):
					raise RuntimeError(f"Measured configuration after waypoint {waypoint_index} is in collision; stopping")
				if not waypoint_reached:
					remaining = np.degrees(np.abs(waypoint - joint_positions))
					error_summary = ", ".join(
						f"{name}={error:.1f} deg" for name, error in zip(joint_names, remaining)
					)
					raise RuntimeError(
						f"Arm did not reach RRT waypoint {waypoint_index} within 3 degrees; "
						f"remaining errors: {error_summary}"
					)
			for joint_index, position in zip(checker.joint_indices, joint_positions):
				p.resetJointState(robot_id, joint_index, float(position), physicsClientId=client)
			final_state = p.getLinkState(robot_id, checker.ee_link_index,
										 computeForwardKinematics=True, physicsClientId=client)
			final_error = float(np.linalg.norm(np.asarray(final_state[4]) - target))
			if final_error <= CARTESIAN_POSITION_TOLERANCE_M:
				print(f"Reached near-side object approach; measured FK error={final_error * 1000:.1f} mm")
			else:
				print(f"Cartesian target not reached within {CARTESIAN_POSITION_TOLERANCE_M * 1000:.0f} mm; "
					  f"measured FK error={final_error * 1000:.1f} mm")
			print(f"Gripper remains open at {args.gripper_open_position:.1f}/100")
		elif args.no_robot:
			print("Plan-only mode: no robot was connected and no motion was sent.")
		else:
			if start_correction_required:
				print("Plan-only mode: path assumes the displayed start correction; robot was not moved.")
			print("Plan-only mode: robot was read but not moved. Pass --execute and type MOVE to execute.")

		if args.gui and not args.execute:
			restore_pybullet_robot_pose(
				robot_id,
				joint_names,
				joint_positions,
				measured_gripper_position,
				client,
			)
			print("Restored PyBullet visual to the measured arm and gripper pose after planning.")
		if args.gui:
			input("PyBullet planning scene is open. Press Enter to close: ")
	finally:
		if client is not None:
			p.disconnect(physicsClientId=client)
		if robot is not None:
			robot.disconnect()


if __name__ == "__main__":
	main()
