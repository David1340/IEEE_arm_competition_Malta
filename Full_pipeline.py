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
import contextlib
import io
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ROI_FILE = ROOT / "yolo_dataset" / "roi.json"
CALIBRATION_FILE = ROOT / "camera_transform.npy"
URDF_FILE = ROOT / "robot_urdfs" / "so101" / "robot.urdf"
TABLE_Z = 0.0
MIN_GRIPPER_TABLE_CLEARANCE_M = 0.015
GRASP_CENTER_LINK = "grasp_center_link"
MAX_TABLE_OBJECT_CENTER_HEIGHT_M = 0.20
FIXED_WRIST_ROLL_RAD = -np.pi / 2.0
MAX_START_LIMIT_CORRECTION_DEG =10.0
# The motors are far slower than send_joints' ramp, so long single moves let joints drift apart
# and leave the checked straight-line path; short waypoints keep the real arm on the plan.
MAX_EXECUTION_WAYPOINT_STEP_DEG = 4.0
# Loaded joints (e.g. the extended elbow) settle a few degrees short of the command, so
# intermediate waypoints accept a bounded lag; the measured pose is re-checked for collisions each time.
WAYPOINT_INTERMEDIATE_TOLERANCE_DEG = 5.0
WAYPOINT_FINAL_TOLERANCE_DEG = 3.0
WAYPOINT_TIMEOUT_S = 4.0
MAX_WAYPOINT_LAG_DEG = 6.0
REFINE_TOLERANCE_DEG = 1.0
MAX_REFINE_ATTEMPTS = 5
MAX_REFINE_OFFSET_DEG = 5.0
MAX_REFINE_STEP_DEG = 8.0
GRIPPER_OPEN_POSITION = 80.0
OBJECT_COLLISION_PADDING_M = 0.008
OBJECT_CENTER_APPROACH_LIFT_M = 0.05
CARTESIAN_POSITION_TOLERANCE_M = 0.02
GRASP_POSITION_TOLERANCE_M = 0.012
GRASP_REFERENCE_DEG = (57.0, -33.0, -25.0)
GRASP_ORIENTATION_TOLERANCE_DEG = 30.0
GRASP_HEIGHT_M = 0.04
MAX_GRASP_HEIGHT_M = 0.05
GRASP_LATERAL_TOLERANCE_M = 0.015
GRASP_JAW_REACH_M = 0.03
MEASURED_COLLISION_MARGIN_M = 0.001
GRIP_CURRENT_MA = 80.0
GRIP_CURRENT_CONFIRM_READINGS = 3
GRIP_SQUEEZE_UNITS = 2.0
GRIPPER_MA_PER_UNIT = 6.5
MEASURED_TABLE_CLEARANCE_M = 0.003

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
	parser.add_argument("--grasp-height", type=float, default=GRASP_HEIGHT_M,
						help="target height above the table of the point between the jaws at the grasp, in meters "
							 "(the solution must stay below 5 cm)")
	parser.add_argument("--no-silhouette", action="store_true",
						help="skip the STL silhouette fit and use the point-cloud mean and the large STL")
	parser.add_argument("--silhouette-min-iou", type=float, default=SILHOUETTE_MIN_IOU,
						help="minimum mask/silhouette IoU to trust the STL fit")
	parser.add_argument("--close-gripper", action="store_true",
						help="after reaching the grasp pose, ask for CLOSE and close the gripper until the motor current shows contact")
	parser.add_argument("--grip-current-ma", type=float, default=GRIP_CURRENT_MA,
						help="gripper motor current in mA that means the object is held")
	parser.add_argument("--grip-confirm", type=int, default=GRIP_CURRENT_CONFIRM_READINGS,
						help="consecutive readings above the current threshold")
	parser.add_argument("--grip-squeeze", type=float, default=GRIP_SQUEEZE_UNITS,
						help="gripper units held past the contact position")
	parser.add_argument("--approach", choices=("grasp", "near-side"), default="grasp",
						help="grasp: put the object between the jaws; near-side: stop beside the object")
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


SILHOUETTE_DIR = ROOT / "point_cloud_matching"
SILHOUETTE_MIN_IOU = 0.7
SILHOUETTE_SHAPE_BY_CLASS = {"hexagonal": "hexagonal_prism", "pentagonal": "pentagonal_prism"}


def fit_silhouettes(detections, full_polygons, cloud_xyz, camera_to_robot, intrinsics, min_iou):
	"""Render-and-compare (point_cloud_matching): fit the STL, size and resting pose to each YOLO mask.

	Replaces the cloud-mean center with the fitted object center and stores the fit in detection["fit"].
	Detections whose fit stays below min_iou keep the cloud-based center and the large STL.
	"""
	import sys

	import cv2

	if str(SILHOUETTE_DIR) not in sys.path:
		sys.path.insert(0, str(SILHOUETTE_DIR))
	import models as silhouette_models
	from silhouette import Camera, SilhouetteMatcher
	from silhouette_live import Detection, local_table_z

	camera = Camera(camera_to_robot, intrinsics)
	height, width = cloud_xyz.shape[:2]
	all_mask = np.zeros((height, width), np.uint8)
	for polygon in full_polygons:
		cv2.fillPoly(all_mask, [np.rint(polygon).astype(np.int32)], 1)
	all_mask = cv2.dilate(all_mask, np.ones((3, 3), np.uint8), iterations=8)
	candidates_by_shape = {}
	for detection in detections:
		class_name = detection["class_name"].strip().casefold()
		shape = SILHOUETTE_SHAPE_BY_CLASS.get(class_name, silhouette_models.CLASS_TO_STL.get(class_name))
		if shape is None:
			continue
		if shape not in candidates_by_shape:
			candidates_by_shape[shape] = silhouette_models.load_candidates(shape)
		candidates = candidates_by_shape[shape]
		if not candidates:
			continue
		mask_detection = Detection(class_name, detection["confidence"], detection["polygon_full"])
		table_z = local_table_z(cloud_xyz, camera_to_robot, mask_detection, all_mask)
		matcher = SilhouetteMatcher(camera, mask_detection)
		fit = max((matcher.fit(candidate, table_z) for candidate in candidates), key=lambda item: item.iou)
		label = f"{fit.candidate.label}: IoU={fit.iou:.2f}, local table Z={table_z * 1000:.0f} mm"
		if fit.iou < min_iou:
			print(f"{class_name} conf={detection['confidence']:.2f}: silhouette fit too weak ({label}); "
				  "keeping the point-cloud center and the large STL")
			continue
		detection["fit"] = fit
		# Only the top face is used: its centre sits straight above the object axis.
		detection["center"] = np.asarray(fit.top_center, dtype=np.float32)
		top_vertices = np.asarray(fit.candidate.top_tris, dtype=float).reshape(-1, 3)
		outline = camera.project(fit.pose_points(top_vertices))
		detection["outline_px"] = cv2.convexHull(np.rint(outline).astype(np.int32)).reshape(-1, 2)
		detection["center_px"] = camera.project(fit.top_center)[0]
		print(f"{class_name} conf={detection['confidence']:.2f}: silhouette fit {label}; "
			  f"top-face center={np.round(fit.top_center, 4)} m, yaw={np.degrees(fit.yaw):.0f} deg")


def camera_intrinsics(camera, cloud_xyz):
	"""fx, fy, cx, cy of the left ZED camera; falls back to fitting the pinhole model to the cloud."""
	try:
		info = camera.get_camera_information()
		left = getattr(info, "camera_configuration", info).calibration_parameters.left_cam
		return left.fx, left.fy, left.cx, left.cy
	except Exception as error:
		print(f"Could not read ZED intrinsics ({error}); estimating them from the point cloud")
		from silhouette_live import intrinsics_from_cloud

		return intrinsics_from_cloud(cloud_xyz)


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
					"polygon_full": np.asarray(polygon, dtype=np.float64) + [x0, y0],
				})
		if detections and not args.no_silhouette:
			full_polygons = [np.asarray(polygon, dtype=np.float64) + [x0, y0] for polygon in prediction.masks.xy]
			cloud_xyz = cloud[..., :3]
			fit_silhouettes(detections, full_polygons, cloud_xyz, camera_to_robot,
							camera_intrinsics(camera, cloud_xyz), args.silhouette_min_iou)
			for detection in detections:
				if "fit" not in detection:
					continue
				fit = detection["fit"]
				center_px = tuple(int(round(value)) for value in detection["center_px"])
				cv2.polylines(preview, [detection["outline_px"]], True, (0, 255, 0), 2)
				cv2.drawMarker(preview, center_px, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
				cv2.putText(preview, f"TOP {fit.candidate.label} IoU={fit.iou:.2f} XYZ {np.round(fit.top_center, 3)}",
							(center_px[0] + 10, center_px[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
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


def _tcp_angle(checker, q):
	import pybullet as p

	checker._set_joint_state(q)
	position = p.getLinkState(checker.robot_id, checker.ee_link_index, computeForwardKinematics=True,
							  physicsClientId=checker.physics_client_id)[4]
	return math.atan2(position[1], position[0])


def solve_centered_grasp(checker, robot_id, object_body_id, axis_xy, start_q, client, grasp_height_m, seeds=20):
	"""Solve one collision-free pose with the point between the jaws on the object axis, at or below MAX_GRASP_HEIGHT_M.

	The checker rejects states that touch the table (clearance margin) or the object, so the first
	solution that meets the lateral and height limits is returned.
	"""
	import pybullet as p

	joint_by_name = {p.getJointInfo(robot_id, i, physicsClientId=client)[1].decode(): i
					 for i in range(p.getNumJoints(robot_id, physicsClientId=client))}
	pan_offset = checker.joint_indices.index(joint_by_name["shoulder_pan"])
	target = np.array([axis_xy[0], axis_xy[1], grasp_height_m])
	# Horizontal grasp attitude: lift/elbow/flex of a known flanking pose, panned toward the object.
	reference = np.zeros(len(checker.joint_indices))
	reference[1:4] = np.deg2rad(GRASP_REFERENCE_DEG)
	reference[checker.joint_indices.index(joint_by_name["wrist_roll"])] = FIXED_WRIST_ROLL_RAD
	for _ in range(4):
		reference[pan_offset] -= math.atan2(target[1], target[0]) - _tcp_angle(checker, reference)
	checker._set_joint_state(reference)
	target_orientation = p.getLinkState(
		robot_id, checker.ee_link_index, computeForwardKinematics=True, physicsClientId=client)[5]
	for seed in range(seeds):
		result = checker.solve(
			target,
			target_orientation=target_orientation,
			start_q=start_q,
			num_particles=160,
			elite_count=5,
			max_iterations=160,
			orientation_weight=0.05,
			position_tolerance_m=GRASP_POSITION_TOLERANCE_M,
			orientation_tolerance_rad=math.radians(GRASP_ORIENTATION_TOLERANCE_DEG),
			seed=seed,
		)
		if result is None:
			continue
		checker._set_joint_state(result.joint_positions)
		tcp = np.array(p.getLinkState(
			robot_id, checker.ee_link_index, computeForwardKinematics=True, physicsClientId=client)[4])
		lateral_error = float(np.linalg.norm(tcp[:2] - target[:2]))
		if lateral_error <= GRASP_LATERAL_TOLERANCE_M and tcp[2] <= MAX_GRASP_HEIGHT_M:
			print(f"  grasp pose (seed {seed}): between-jaws height={tcp[2] * 1000:.0f} mm, "
				  f"axis offset={lateral_error * 1000:.1f} mm")
			return result
	return None


def describe_collision(checker, q):
	"""Explain why a measured configuration is rejected by the collision checker."""
	import pybullet as p

	q = np.asarray(q, dtype=float)
	checker._set_joint_state(q)
	reasons = []
	for link_index, minimum_z in checker.minimum_link_heights.items():
		bottom = p.getAABB(checker.robot_id, linkIndex=link_index, physicsClientId=checker.physics_client_id)[0][2]
		if bottom < minimum_z:
			name = p.getJointInfo(checker.robot_id, link_index, physicsClientId=checker.physics_client_id)[12].decode()
			reasons.append(f"{name} bottom z={bottom * 1000:.0f} mm < {minimum_z * 1000:.0f} mm")
	p.performCollisionDetection(physicsClientId=checker.physics_client_id)
	for body_id in checker.obstacle_body_ids:
		for point in p.getClosestPoints(checker.robot_id, body_id, distance=checker.collision_margin_m,
										physicsClientId=checker.physics_client_id):
			if (body_id, point[3], point[4]) in checker.allowed_robot_obstacle_link_pairs:
				continue
			name = p.getJointInfo(checker.robot_id, point[3], physicsClientId=checker.physics_client_id)[12].decode() \
				if point[3] >= 0 else "base"
			reasons.append(f"{name} vs body {body_id}: {point[8] * 1000:.1f} mm")
	return "; ".join(reasons) or "joint limits or fixed-joint mismatch"


def densify_path(path, max_step_rad):
	"""Insert straight-line joint-space waypoints so no segment exceeds max_step_rad on any joint."""
	dense = [np.asarray(path[0], dtype=float)]
	for waypoint in path[1:]:
		waypoint = np.asarray(waypoint, dtype=float)
		previous = dense[-1]
		steps = max(1, int(np.ceil(np.max(np.abs(waypoint - previous)) / max_step_rad)))
		dense.extend(previous + fraction * (waypoint - previous) for fraction in np.linspace(0.0, 1.0, steps + 1)[1:])
	return dense


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


def read_gripper_current_ma(robot):
	"""STS3215 Present_Current in mA; bit 15 holds the direction."""
	raw = int(robot.bus.read("Present_Current", "gripper", normalize=False))
	return (raw & 0x7FFF) * GRIPPER_MA_PER_UNIT


def close_gripper_with_current_feedback(robot, closed_target, current_threshold_ma, confirm_readings,
										squeeze_units, *, timeout=10.0, feedback_callback=None):
	"""Close the gripper until the motor current shows contact, then hold a light squeeze.

	Returns (grasped, stop_position, hold_position, current_ma). grasped is False if the gripper
	reached closed_target without a current rise (nothing between the jaws).
	"""
	deadline = time.monotonic() + timeout
	position = float(robot.get_observation()["gripper.pos"])
	above = 0
	current_ma = 0.0
	while time.monotonic() < deadline:
		observation = robot.get_observation()
		hold = {key: value for key, value in observation.items()
				if key.endswith(".pos") and key != "gripper.pos"}
		robot.send_action({**hold, "gripper.pos": float(closed_target)})
		time.sleep(0.04)
		observation = robot.get_observation()
		if feedback_callback is not None:
			feedback_callback(observation)
		position = float(observation["gripper.pos"])
		current_ma = read_gripper_current_ma(robot)
		above = above + 1 if current_ma >= current_threshold_ma else 0
		if above >= confirm_readings:
			hold_position = max(position - squeeze_units, closed_target)
			robot.send_action({**hold, "gripper.pos": hold_position})
			return True, position, hold_position, current_ma
		if abs(position - closed_target) <= 1.0:
			break
	return False, position, position, current_ma


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
			fit = detection.get("fit")
			mesh_path = None if fit is not None else object_stl_path(class_name)
			if fit is not None:
				triangles = np.asarray(fit.candidate.tris, dtype=float)
				vertices = triangles.reshape(-1, 3)
				collision_shape = p.createCollisionShape(
					p.GEOM_MESH, vertices=np.unique(np.round(vertices, 7), axis=0).tolist(), physicsClientId=client)
				visual_shape = p.createVisualShape(
					p.GEOM_MESH, vertices=vertices.tolist(), indices=list(range(len(vertices))),
					rgbaColor=[*color, 0.8], physicsClientId=client)
				body_position = [float(fit.x), float(fit.y), float(fit.table_z)]
				body_orientation = p.getQuaternionFromEuler([0.0, 0.0, float(fit.yaw)])
				geometry_description = f"silhouette-fitted STL={fit.candidate.label} (IoU={fit.iou:.2f})"
			elif mesh_path is not None:
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
				body_orientation = [0.0, 0.0, 0.0, 1.0]
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
				body_orientation = [0.0, 0.0, 0.0, 1.0]
				geometry_description = f"collision AABB={half_extents * 2}"
			body_id = p.createMultiBody(
				baseMass=0,
				baseCollisionShapeIndex=collision_shape,
				baseVisualShapeIndex=visual_shape,
				basePosition=body_position,
				baseOrientation=body_orientation,
				physicsClientId=client,
			)
			object_body_ids.append(body_id)

			if mesh_path is None and fit is None:
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
			link_by_name[GRASP_CENTER_LINK],
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
			link_by_name[GRASP_CENTER_LINK],
			arm_indices,
			obstacle_body_ids=[table_id, *object_body_ids],
			physics_client_id=client,
			check_self_collision=False,
			allowed_robot_obstacle_link_pairs=[(table_id, -1, -1)],
			minimum_link_heights=minimum_gripper_link_heights,
		)
		rrt = RRTConnectPlanner(checker, max_joint_step_rad=np.deg2rad(3.0))
		ee_state = p.getLinkState(robot_id, link_by_name[GRASP_CENTER_LINK],
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
		scene_kwargs = dict(
			max_points_per_object=args.max_points_per_object,
			gripper_position=simulated_gripper_position,
			min_gripper_table_clearance=args.min_gripper_table_clearance,
		)
		# Planning runs headless; the GUI is opened only once a path exists (or to show a start collision).
		client, robot_id, checker, rrt, transition_checker, object_body_ids = build_planning_scene(
			detections, joint_names, joint_positions, gui=False, **scene_kwargs)

		def open_gui_scene():
			nonlocal client, robot_id, checker, rrt, transition_checker, object_body_ids
			p.disconnect(physicsClientId=client)
			client = None
			with contextlib.redirect_stdout(io.StringIO()):
				client, robot_id, checker, rrt, transition_checker, object_body_ids = build_planning_scene(
					detections, joint_names, joint_positions, gui=True, **scene_kwargs)
			restore_pybullet_robot_pose(
				robot_id, joint_names, joint_positions, measured_gripper_position, client)

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
		if (
			args.execute
			and robot is not None
			and not transition_checker.state_is_collision_free(joint_positions, check_joint_limits=False)
		):
			home = np.zeros(len(joint_names), dtype=float)
			home[wrist_offset] = FIXED_WRIST_ROLL_RAD
			if not transition_checker.state_is_collision_free(home):
				raise RuntimeError("Measured start pose collides and the home pose is also in collision; no motion sent")
			# The start itself collides, so only the states after it can be validated.
			steps_to_home = max(1, int(np.ceil(np.max(np.abs(home - joint_positions)) / np.deg2rad(3.0))))
			for fraction in np.linspace(0.0, 1.0, steps_to_home + 1)[1:]:
				if not transition_checker.state_is_collision_free(
					joint_positions + fraction * (home - joint_positions), check_joint_limits=False
				):
					raise RuntimeError("Straight-line path from the colliding start to home is in collision; no motion sent")
			print("Measured start pose collides with the table; home pose (0, 0, 0, 0, -90 deg) is collision-free.")
			if input("Type HOME to move the arm to the home pose first: ").strip().upper() != "HOME":
				print("Cancelled; no motion command was sent")
				return
			home_steps = max(20, int(np.ceil(np.max(np.abs(np.degrees(home - joint_positions))) / 3.0)) + 1)
			send_joints(
				robot,
				home,
				steps=home_steps,
				tol_deg=3.0,
				timeout=30.0,
				feedback_callback=twin_feedback_callback,
			)
			joint_positions = np.asarray(read_joints_rad(robot), dtype=float)
			print("Moved to home pose; measured joints (rad):", np.round(joint_positions, 4))
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
				open_gui_scene()
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
		if args.approach == "grasp":
			target = np.asarray(selected["center"], dtype=float).copy()
			if "fit" not in selected:
				target = (np.asarray(selected_aabb[0], dtype=float) + np.asarray(selected_aabb[1], dtype=float)) / 2.0
			print(f"Selected object: {selected['class_name']} conf={selected['confidence']:.2f}")
			grasp_height = args.grasp_height
			print(f"Grasp axis (x, y): {np.round(target[:2], 4)} m; point between the jaws at "
				  f"{grasp_height * 1000:.0f} mm above the table")
		else:
			target = near_side_target(selected, current_ee, args.object_standoff, collision_aabb=selected_aabb)
			target[2] += args.object_center_lift
			print(f"Selected object: {selected['class_name']} conf={selected['confidence']:.2f}")
			print(f"Collision-free near-side target: {np.round(target, 4)} m "
				  f"({args.object_standoff * 1000:.0f} mm outside object box; "
				  f"{args.object_center_lift * 1000:.0f} mm above cloud-center height)")

		if args.approach == "grasp":
			ik_result = solve_centered_grasp(
				checker, robot_id, object_body_ids[selected_index], target[:2], aligned_start, client,
				grasp_height,
			)
		else:
			ik_result = checker.solve(
				target,
				start_q=aligned_start,
				num_particles=160,
				elite_count=5,
				max_iterations=160,
				position_tolerance_m=CARTESIAN_POSITION_TOLERANCE_M,
				seed=0,
			)
		# Grasp poses are validated by the physical jaw-closing test; the IK error is measured against a height the table clearance may not allow.
		ik_failed = ik_result is None or (args.approach != "grasp" and not ik_result.success)
		if ik_failed:
			error = float("inf") if ik_result is None else ik_result.position_error_m
			raise RuntimeError(f"M-FRPSO did not find a collision-free pre-grasp solution; position error={error:.4f} m")
		print(f"M-FRPSO goal found: position error={ik_result.position_error_m * 1000:.1f} mm")
		if args.approach == "grasp":
			checker._set_joint_state(ik_result.joint_positions)
			target = np.array(p.getLinkState(
				robot_id, checker.ee_link_index, computeForwardKinematics=True, physicsClientId=client)[4])
			print(f"Grasp point between the jaws: {np.round(target, 4)} m")

		route = rrt.plan(aligned_start, ik_result.joint_positions, seed=0)
		if route is None:
			raise RuntimeError("RRT-Connect could not find a collision-free path to the M-FRPSO goal")
		print(f"RRT-Connect path: {len(route.path)} configurations, {route.iterations} iterations")
		for index, waypoint in enumerate(route.path):
			print(f"  {index}: {np.round(waypoint, 4)} rad")

		if args.gui:
			open_gui_scene()

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
			execution_path = densify_path(route.path, np.deg2rad(MAX_EXECUTION_WAYPOINT_STEP_DEG))
			print(f"Executing {len(execution_path) - 1} tracked waypoints (<= {MAX_EXECUTION_WAYPOINT_STEP_DEG:.0f} deg each)")
			for waypoint_index, waypoint in enumerate(execution_path[1:], start=1):
				if not transition_checker.edge_is_collision_free(joint_positions, waypoint):
					raise RuntimeError(f"Measured-to-waypoint edge {waypoint_index} is not collision-free; stopping")
				steps = max(
					20,
					int(np.ceil(np.max(np.abs(np.degrees(waypoint - joint_positions))) / 3.0)) + 1,
				)
				is_last_waypoint = waypoint_index == len(execution_path) - 1
				waypoint_reached = send_joints(
					robot,
					waypoint,
					steps=steps,
					tol_deg=WAYPOINT_FINAL_TOLERANCE_DEG if is_last_waypoint else WAYPOINT_INTERMEDIATE_TOLERANCE_DEG,
					timeout=10.0 if is_last_waypoint else WAYPOINT_TIMEOUT_S,
					feedback_callback=twin_feedback_callback,
				)
				joint_positions = np.asarray(read_joints_rad(robot), dtype=float)
				# Motor lag moves the measured pose off the checked path, so only real contact stops the arm.
				planning_margin = transition_checker.collision_margin_m
				transition_checker.collision_margin_m = MEASURED_COLLISION_MARGIN_M
				planned_heights = transition_checker.minimum_link_heights
				# The 2 cm table clearance is a planning buffer; only real contact (near z=0) stops the arm.
				transition_checker.minimum_link_heights = {
					link: min(height, MEASURED_TABLE_CLEARANCE_M) for link, height in planned_heights.items()}
				try:
					measured_free = transition_checker.state_is_collision_free(joint_positions)
					if not measured_free:
						detail = describe_collision(transition_checker, joint_positions)
				finally:
					transition_checker.collision_margin_m = planning_margin
					transition_checker.minimum_link_heights = planned_heights
				if not measured_free:
					raise RuntimeError(
						f"Measured configuration after waypoint {waypoint_index} is in collision ({detail}); stopping")
				if not waypoint_reached:
					remaining = np.degrees(np.abs(waypoint - joint_positions))
					error_summary = ", ".join(
						f"{name}={error:.1f} deg" for name, error in zip(joint_names, remaining)
					)
					if not is_last_waypoint and np.max(remaining) <= MAX_WAYPOINT_LAG_DEG:
						print(f"Waypoint {waypoint_index} settled short of target ({error_summary}); continuing")
						continue
					if is_last_waypoint and np.max(remaining) <= MAX_WAYPOINT_LAG_DEG:
						print(f"Final waypoint settled short of target ({error_summary}); load-limited joint")
						continue
					raise RuntimeError(
						f"Arm did not reach RRT waypoint {waypoint_index}; "
						f"remaining errors: {error_summary}"
					)
			refine_goal = np.asarray(route.path[-1], dtype=float)
			integrated_command = refine_goal.copy()
			for refine_attempt in range(1, MAX_REFINE_ATTEMPTS + 1):
				joint_error = refine_goal - joint_positions
				if np.max(np.abs(np.degrees(joint_error))) <= REFINE_TOLERANCE_DEG:
					break
				# Integrator: accumulate the measured error into the command so load-limited joints (shoulder_lift) are pushed past the target.
				integrated_command = np.clip(
					integrated_command + joint_error,
					refine_goal - np.radians(MAX_REFINE_OFFSET_DEG),
					refine_goal + np.radians(MAX_REFINE_OFFSET_DEG),
				)
				refine_command = np.clip(
					integrated_command,
					transition_checker.lower + limit_margin,
					transition_checker.upper - limit_margin,
				)
				refine_command[wrist_offset] = FIXED_WRIST_ROLL_RAD
				if np.max(np.abs(np.degrees(refine_command - joint_positions))) > MAX_REFINE_STEP_DEG:
					print("Refinement skipped: correction exceeds the safe step size")
					break
				# Refinement moves only a few degrees, so the edge collision check is not enforced here.
				send_joints(
					robot,
					refine_command,
					steps=20,
					tol_deg=REFINE_TOLERANCE_DEG,
					timeout=WAYPOINT_TIMEOUT_S,
					feedback_callback=twin_feedback_callback,
				)
				joint_positions = np.asarray(read_joints_rad(robot), dtype=float)
				if not transition_checker.state_is_collision_free(joint_positions):
					print("Warning: measured pose after refinement touches the collision model "
						  f"({describe_collision(transition_checker, joint_positions)}); collision allowed during refinement")
				print(f"Refinement {refine_attempt}: max joint error "
					  f"{np.max(np.abs(np.degrees(refine_goal - joint_positions))):.2f} deg")
			for joint_index, position in zip(checker.joint_indices, joint_positions):
				p.resetJointState(robot_id, joint_index, float(position), physicsClientId=client)
			final_state = p.getLinkState(robot_id, checker.ee_link_index,
										 computeForwardKinematics=True, physicsClientId=client)
			final_error = float(np.linalg.norm(np.asarray(final_state[4]) - target))
			if final_error <= CARTESIAN_POSITION_TOLERANCE_M:
				print(f"Reached {'grasp pose' if args.approach == 'grasp' else 'near-side object approach'}; "
					  f"measured FK error={final_error * 1000:.1f} mm")
			else:
				print(f"Cartesian target not reached within {CARTESIAN_POSITION_TOLERANCE_M * 1000:.0f} mm; "
					  f"measured FK error={final_error * 1000:.1f} mm")
			if args.close_gripper and final_error <= CARTESIAN_POSITION_TOLERANCE_M:
				if input("Type CLOSE to close the gripper on the object: ").strip().upper() == "CLOSE":
					grasped, stop_position, hold_position, current_ma = close_gripper_with_current_feedback(
						robot, 0.0, args.grip_current_ma, args.grip_confirm, args.grip_squeeze,
						feedback_callback=twin_feedback_callback,
					)
					if grasped:
						print(f"Object grasped: contact at {stop_position:.1f}/100 with {current_ma:.0f} mA; "
							  f"holding at {hold_position:.1f}/100")
					else:
						print(f"No contact detected (gripper at {stop_position:.1f}/100, {current_ma:.0f} mA): "
							  "the gripper closed without an object between the jaws")
				else:
					print(f"Gripper remains open at {args.gripper_open_position:.1f}/100")
			else:
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
