"""Show YOLO objects with calibrated 3D centers from the YOLO mask and ZED point cloud.

Coordinates are transformed into the robot/table frame defined by the known
ArUco marker positions in qr_calibrate.py. The center is the mean of valid
point-cloud samples across the full segmented object mask, after depth outlier rejection.

Usage:
    python yolo_dataset/object_3d_live.py
    python yolo_dataset/object_3d_live.py --weights runs/segment/train/weights/best.pt
    python yolo_dataset/object_3d_live.py --serial 27500226
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROI_FILE = HERE / "roi.json"
CALIBRATION_FILE = HERE.parent / "camera_transform.npy"
RUNS = HERE.parent / "runs" / "segment"
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
          (128, 0, 255), (0, 200, 120), (200, 120, 0), (120, 120, 255)]
OBJECT_STL_FILES = {
    "cube": HERE.parent / "stl" / "cube_large.stl",
    "cylinder": HERE.parent / "stl" / "cylinder_large.stl",
    "dodecahedron": HERE.parent / "stl" / "dodecahedron_large.stl",
    "hexagonal": HERE.parent / "stl" / "hexagonal_prism_large.stl",
    "hexagonal_prism": HERE.parent / "stl" / "hexagonal_prism_large.stl",
}
IGNORED_OBJECT_CLASSES = {"gripper", "bin"}


def latest_weights() -> Path:
    candidates = sorted(RUNS.glob("*/weights/best.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No best.pt found under {RUNS}; train a segmentation model first or pass --weights")
    return candidates[-1]


def roi_for_frame(frame_shape, roi_data):
    height, width = frame_shape[:2]
    ref_width, ref_height = roi_data.get("ref_size", [width, height])
    sx, sy = width / ref_width, height / ref_height
    x = round(roi_data["x"] * sx)
    y = round(roi_data["y"] * sy)
    w = round(roi_data["w"] * sx)
    h = round(roi_data["h"] * sy)
    if x < 0 or y < 0 or x + w > width or y + h > height or w <= 0 or h <= 0:
        raise ValueError(f"ROI {(x, y, w, h)} does not fit camera frame {(width, height)}")
    return x, y, w, h


def load_camera_transform(path):
    if not path.exists():
        raise SystemExit(
            f"Camera calibration not found: {path}\n"
            "Run qr_calibrate.py --show-ids, then calibrate with the measured marker IDs first."
        )
    transform = np.load(path)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SystemExit(f"Invalid camera calibration matrix in {path}; expected a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5):
        raise SystemExit(f"Invalid homogeneous-transform last row in {path}: {transform[3]}")
    return transform


def transform_point(transform, point):
    return (transform @ np.append(np.asarray(point, dtype=float), 1.0))[:3]


def visible_surface_center(
    cloud,
    polygon,
    roi,
    min_depth,
    max_depth,
    min_points,
    camera_to_table,
    z_tolerance_m=0.015,
):
    """Return a robust full-mask point-cloud center in calibrated table coordinates."""
    x0, y0, width, height = roi
    poly = np.asarray(polygon, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, 0, None, None

    camera_xyz = cloud[y0 : y0 + height, x0 : x0 + width, :3][ys, xs]
    valid = np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] >= min_depth) & (camera_xyz[:, 2] <= max_depth)
    camera_xyz = camera_xyz[valid]
    xs, ys = xs[valid], ys[valid]
    if len(camera_xyz) < min_points:
        return None, 0, None, None

    z_median = np.median(camera_xyz[:, 2])
    z_mad = np.median(np.abs(camera_xyz[:, 2] - z_median))
    depth_tolerance = max(z_tolerance_m, 3.0 * 1.4826 * z_mad)
    inliers = np.abs(camera_xyz[:, 2] - z_median) <= depth_tolerance
    camera_xyz = camera_xyz[inliers]
    xs, ys = xs[inliers], ys[inliers]
    if len(camera_xyz) < min_points:
        return None, len(camera_xyz), None, None

    center_camera = camera_xyz.mean(axis=0)
    center = transform_point(camera_to_table, center_camera)
    pixel_center = (int(round(xs.mean() + x0)), int(round(ys.mean() + y0)))
    return center, len(camera_xyz), pixel_center, mask * 255


class PyBulletTwin:
    def __init__(self, joint_names, *, gui=True):
        import pybullet as p
        import pybullet_data
        from IK_control import EE_LINK, URDF
        from modified_frpso_ik import load_robot_for_collision

        self.p = p
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
            p.setAdditionalSearchPath(str(URDF.parent), physicsClientId=self.client)
            p.setGravity(0, 0, 0, physicsClientId=self.client)
            p.setRealTimeSimulation(0, physicsClientId=self.client)

            table_half_extents = [0.75, 0.9, 0.02]
            table_collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=table_half_extents, physicsClientId=self.client
            )
            table_visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.55, 0.57, 0.60, 1],
                physicsClientId=self.client,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=table_collision,
                baseVisualShapeIndex=table_visual,
                basePosition=[0, 0.25, -table_half_extents[2]],
                physicsClientId=self.client,
            )
            flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            self.robot_id = p.loadURDF(
                str(URDF), useFixedBase=True, flags=flags, physicsClientId=self.client
            )
            self.joint_names = list(joint_names)
            self.joint_indices = {}
            self.link_indices = {}
            for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
                info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client)
                self.joint_indices[info[1].decode()] = index
                self.link_indices[info[12].decode()] = index
            self.ee_link_index = self.link_indices[EE_LINK]
            self.mesh_shapes = {}
            self.object_bodies = {}
            self.object_markers = {}
            self.object_labels = {}
            self.gui = gui
            if gui:
                p.resetDebugVisualizerCamera(
                    1.65, 45, -45, [0.15, 0.25, 0.20], physicsClientId=self.client
                )
        except Exception:
            p.disconnect(physicsClientId=self.client)
            raise

    def _object_shape(self, class_name):
        class_name = class_name.strip().casefold()
        mesh_path = OBJECT_STL_FILES.get(class_name)
        if mesh_path is None:
            return None
        if not mesh_path.exists():
            raise FileNotFoundError(f"STL asset missing for {class_name}: {mesh_path}")
        if class_name not in self.mesh_shapes:
            color = COLORS[len(self.mesh_shapes) % len(COLORS)]
            rgba = [value / 255.0 for value in color[::-1]] + [0.85]
            mesh_scale = [0.001, 0.001, 0.001]
            self.mesh_shapes[class_name] = (
                self.p.createCollisionShape(
                    self.p.GEOM_MESH, fileName=str(mesh_path), meshScale=mesh_scale,
                    physicsClientId=self.client,
                ),
                self.p.createVisualShape(
                    self.p.GEOM_MESH, fileName=str(mesh_path), meshScale=mesh_scale,
                    rgbaColor=rgba, physicsClientId=self.client,
                ),
                rgba[:3],
            )
        return self.mesh_shapes[class_name]

    def update_objects(self, objects):
        active_keys = set()
        class_indices = {}
        for class_name, confidence, center, point_count in objects:
            normalized_class = class_name.strip().casefold()
            if center is None or normalized_class in IGNORED_OBJECT_CLASSES:
                continue
            mesh_shapes = self._object_shape(normalized_class)
            if mesh_shapes is None:
                continue
            occurrence = class_indices.get(normalized_class, 0)
            class_indices[normalized_class] = occurrence + 1
            key = (normalized_class, occurrence)
            active_keys.add(key)
            collision_shape, visual_shape, color = mesh_shapes
            position = [float(center[0]), float(center[1]), 0.0]
            label = f"{class_name} {confidence:.2f} center={np.round(center, 3)} m"
            if key not in self.object_bodies:
                self.object_bodies[key] = self.p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position,
                    physicsClientId=self.client,
                )
                marker_shape = self.p.createVisualShape(
                    self.p.GEOM_SPHERE, radius=0.009, rgbaColor=[*color, 1],
                    physicsClientId=self.client,
                )
                self.object_markers[key] = self.p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=marker_shape,
                    basePosition=center.tolist(),
                    physicsClientId=self.client,
                )
                self.object_labels[key] = self.p.addUserDebugText(
                    label, (np.asarray(center) + [0, 0, 0.035]).tolist(),
                    textColorRGB=color, textSize=1.0, lifeTime=0,
                    physicsClientId=self.client,
                )
            else:
                self.p.resetBasePositionAndOrientation(
                    self.object_bodies[key], position, [0, 0, 0, 1], physicsClientId=self.client
                )
                self.p.resetBasePositionAndOrientation(
                    self.object_markers[key], center.tolist(), [0, 0, 0, 1], physicsClientId=self.client
                )
                self.p.addUserDebugText(
                    label, (np.asarray(center) + [0, 0, 0.035]).tolist(),
                    textColorRGB=color, textSize=1.0, lifeTime=0,
                    replaceItemUniqueId=self.object_labels[key], physicsClientId=self.client,
                )

        for key in set(self.object_bodies) - active_keys:
            self.p.removeBody(self.object_bodies.pop(key), physicsClientId=self.client)
            self.p.removeBody(self.object_markers.pop(key), physicsClientId=self.client)
            self.p.removeUserDebugItem(self.object_labels.pop(key), physicsClientId=self.client)

    def update_robot(self, joint_positions, gripper_position):
        for name, position in zip(self.joint_names, joint_positions):
            self.p.resetJointState(
                self.robot_id, self.joint_indices[name], float(position), physicsClientId=self.client
            )
        gripper_index = self.joint_indices["gripper"]
        info = self.p.getJointInfo(self.robot_id, gripper_index, physicsClientId=self.client)
        normalized = np.clip(float(gripper_position) / 100.0, 0.0, 1.0)
        gripper_angle = info[8] + normalized * (info[9] - info[8])
        self.p.resetJointState(self.robot_id, gripper_index, float(gripper_angle), physicsClientId=self.client)

    def close(self):
        if self.p.isConnected(self.client):
            self.p.disconnect(physicsClientId=self.client)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", type=int, default=None, help="ZED serial number; default uses camera ID")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--weights", default=None, help="default: most recently modified best.pt under runs/segment")
    ap.add_argument("--calibration", type=Path, default=CALIBRATION_FILE,
                    help="camera-to-robot transform saved by qr_calibrate.py")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="default: longest ROI side")
    ap.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD1080")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--depth-mode", choices=["QUALITY", "ULTRA", "NEURAL"], default="QUALITY")
    ap.add_argument("--min-depth", type=float, default=0.10, help="ignore points closer than this (meters)")
    ap.add_argument("--max-depth", type=float, default=2.00, help="ignore points farther than this (meters)")
    ap.add_argument("--min-points", type=int, default=20, help="minimum valid mask points for a center estimate")
    ap.add_argument("--max-width", type=int, default=1400)
    ap.add_argument("--no-pybullet-gui", action="store_true", help="run the twin headless")
    ap.add_argument("--no-robot", action="store_true", help="use the URDF zero pose instead of live joint readings")
    args = ap.parse_args()

    if not ROI_FILE.exists():
        raise SystemExit(f"{ROI_FILE} not found; run 'python yolo_dataset/roi.py select' first")
    roi_data = json.loads(ROI_FILE.read_text())
    roi = tuple(roi_data[key] for key in ("x", "y", "w", "h"))
    imgsz = args.imgsz or max(roi[2:])
    camera_to_robot = load_camera_transform(args.calibration)
    weights = Path(args.weights) if args.weights else latest_weights()
    print(f"Model: {weights} | imgsz={imgsz} | depth={args.depth_mode}")
    print(f"Camera calibration: {args.calibration}")

    model = YOLO(str(weights))
    names = model.names

    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.camera_fps = args.fps
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    if args.serial is not None:
        init.set_from_serial_number(args.serial)
    else:
        init.set_from_camera_id(args.camera_id)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open ZED camera: {status}")

    image_mat, cloud_mat = sl.Mat(), sl.Mat()
    runtime = sl.RuntimeParameters()
    win = "YOLO + ZED 3D centers"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_report = 0.0
    fps, frames, fps_time = 0.0, 0, time.time()
    robot = None
    twin = None

    try:
        from IK_control import ARM_JOINTS, connect_robot, read_joints_rad

        if not args.no_robot:
            robot = connect_robot()
            robot.bus.disable_torque()
            print("SO-101 torque disabled for hand-guiding; joint feedback remains live and no motor commands are sent.")
        twin = PyBulletTwin(ARM_JOINTS, gui=not args.no_pybullet_gui)
        print("Coordinates: TABLE frame from ArUco calibration; meters. "
              "X increases left-to-right, Y bottom-to-top, Z=0 at marker plane. "
              "SO-101 feedback is read-only; press q/ESC to quit.")
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                cv2.waitKey(1)
                continue
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
            frame = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
            cloud = cloud_mat.get_data()
            current_roi = roi_for_frame(frame.shape, roi_data)
            x0, y0, roi_w, roi_h = current_roi
            if cloud.shape[:2] != frame.shape[:2]:
                raise RuntimeError(f"Image/point cloud sizes differ: {frame.shape[:2]} vs {cloud.shape[:2]}")

            roi_image = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]
            result = model.predict(roi_image, imgsz=imgsz, conf=args.conf, verbose=False)[0]
            canvas = frame.copy()
            overlay = canvas.copy()
            objects = []

            if result.masks is not None:
                classes = result.boxes.cls.int().tolist()
                confidences = result.boxes.conf.tolist()
                for index, (polygon, class_id, confidence) in enumerate(zip(result.masks.xy, classes, confidences), 1):
                    if len(polygon) < 3:
                        continue
                    points = np.rint(polygon + [x0, y0]).astype(np.int32)
                    color = COLORS[class_id % len(COLORS)]
                    center, count, pixel_center, top_mask = visible_surface_center(
                        cloud, polygon, current_roi, args.min_depth, args.max_depth, args.min_points,
                        camera_to_robot,
                    )
                    class_name = names[class_id]
                    if center is None:
                        cv2.polylines(canvas, [points], True, (128, 128, 128), 1)
                        text = f"{class_name} {confidence:.2f} XYZ n/a ({count} pts)"
                        objects.append((class_name, confidence, None, count))
                    else:
                        roi_overlay = overlay[y0 : y0 + roi_h, x0 : x0 + roi_w]
                        roi_overlay[top_mask > 0] = color
                        contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(canvas, contours, -1, color, 2, offset=(x0, y0))
                        cx, cy = pixel_center
                        cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)
                        text = f"{class_name} {confidence:.2f} TABLE X{center[0]:+.3f} Y{center[1]:+.3f} Z{center[2]:+.3f}m"
                        objects.append((class_name, confidence, center, count))
                    px, py = points.min(axis=0)
                    cv2.putText(canvas, text, (int(px), max(int(py) - 5, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            if robot is None:
                joint_positions = [0.0] * len(ARM_JOINTS)
                gripper_position = 50.0
            else:
                observation = robot.get_observation()
                joint_positions = read_joints_rad(robot)
                gripper_position = float(observation["gripper.pos"])
            twin.update_robot(joint_positions, gripper_position)
            twin.update_objects(objects)
            twin.p.stepSimulation(physicsClientId=twin.client)

            canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0)
            cv2.rectangle(canvas, (x0, y0), (x0 + roi_w, y0 + roi_h), (255, 255, 255), 2)
            frames += 1
            if time.time() - fps_time >= 1.0:
                elapsed = time.time() - fps_time
                fps, frames, fps_time = frames / elapsed, 0, time.time()
            cv2.putText(canvas, f"{fps:.1f} FPS | conf>={args.conf:.2f} | {len(objects)} objects",
                        (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            scale = min(1.0, args.max_width / canvas.shape[1])
            view = cv2.resize(canvas, None, fx=scale, fy=scale) if scale < 1.0 else canvas
            cv2.imshow(win, view)

            now = time.time()
            if now - last_report >= 1.0:
                print(f"\n{datetime.now():%H:%M:%S} | {len(objects)} objects")
                for index, (class_name, confidence, center, count) in enumerate(objects, 1):
                    if center is None:
                        print(f"  {index}. {class_name} conf={confidence:.2f}: depth unavailable ({count} points)")
                    else:
                        print(f"  {index}. {class_name} conf={confidence:.2f} TABLE: "
                              f"X={center[0]:+.3f} Y={center[1]:+.3f} Z={center[2]:.3f} m ({count} points)")
                last_report = now

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if twin is not None:
            twin.close()
        if robot is not None:
            robot.disconnect()
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()