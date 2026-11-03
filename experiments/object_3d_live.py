"""Show YOLO objects with calibrated 3D centers from the YOLO mask and ZED point cloud.

Coordinates are transformed into the robot/table frame defined by the known
ArUco marker positions in qr_calibrate.py. The center is the mean of valid
point-cloud samples across the full segmented object mask, after depth outlier rejection.

With the silhouette fit on (default) the STL is fitted to each YOLO mask by render-and-compare
(vision/silhouette_live.py) and the reported center is the CENTRE OF THE TOP FACE
of the fitted STL; the green outline is that top face. Objects the fit cannot match (IoU below
--silhouette-min-iou) keep the point-cloud center. --no-silhouette restores the old behaviour.

Usage:
    python experiments/object_3d_live.py
    python experiments/object_3d_live.py --weights runs/segment/train/weights/best.pt
    python experiments/object_3d_live.py --serial 27500226
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.detection import (  # noqa: E402
    CALIBRATION_FILE,
    IGNORED_OBJECT_CLASSES,
    OBJECT_STL_FILES,
    ROI_FILE,
    camera_intrinsics,
    latest_weights,
    load_camera_transform,
    roi_for_frame,
    visible_surface_center,
)

COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
          (128, 0, 255), (0, 200, 120), (200, 120, 0), (120, 120, 255)]


class PyBulletTwin:
    def __init__(self, joint_names, *, gui=True):
        import pybullet as p
        import pybullet_data
        from control.IK_control import EE_LINK, URDF
        from control.modified_frpso_ik import load_robot_for_collision

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
    ap.add_argument("--no-silhouette", action="store_true",
                    help="use the point-cloud mean instead of the fitted STL top-face center")
    ap.add_argument("--silhouette-min-iou", type=float, default=0.7,
                    help="minimum mask/silhouette IoU to trust the top-face center")
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
    fitter = None
    if not args.no_silhouette:
        from vision.silhouette import Camera
        from vision.silhouette_live import Detection, LiveFitter, SHAPE_CLASSES, intrinsics_from_cloud

        try:
            intrinsics = camera_intrinsics(zed)
        except Exception as error:
            print(f"Could not read ZED intrinsics ({error}); estimating them from the point cloud")
            while zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                pass
            zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
            intrinsics = intrinsics_from_cloud(cloud_mat.get_data()[..., :3])
        fitter = LiveFitter(Camera(camera_to_robot, intrinsics), camera_to_robot)
    win = "YOLO + ZED 3D centers"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_report = 0.0
    fps, frames, fps_time = 0.0, 0, time.time()
    robot = None
    twin = None

    try:
        from control.IK_control import ARM_JOINTS, connect_robot, read_joints_rad

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
                valid = [(polygon, class_id, confidence)
                         for polygon, class_id, confidence in zip(result.masks.xy, classes, confidences)
                         if len(polygon) >= 3]
                track_by_detection = {}
                if fitter is not None:
                    shape_items = [item for item in valid if names[item[1]] in SHAPE_CLASSES]
                    tracks, _ = fitter.update(
                        [Detection(names[class_id], confidence, polygon + [x0, y0])
                         for polygon, class_id, confidence in shape_items],
                        cloud[..., :3],
                    )
                    track_by_detection = {id(item[0]): track for item, track in zip(shape_items, tracks)}
                for index, (polygon, class_id, confidence) in enumerate(valid, 1):
                    points = np.rint(polygon + [x0, y0]).astype(np.int32)
                    color = COLORS[class_id % len(COLORS)]
                    center, count, pixel_center, top_mask = visible_surface_center(
                        cloud, polygon, current_roi, args.min_depth, args.max_depth, args.min_points,
                        camera_to_robot,
                    )
                    class_name = names[class_id]
                    track = track_by_detection.get(id(polygon))
                    if track is not None and track.fit is not None and track.fit.iou >= args.silhouette_min_iou:
                        # Top face of the fitted STL: its centre sits above the object axis.
                        top = fitter.cam.project(track.fit.pose_points(
                            np.asarray(track.fit.candidate.top_tris, dtype=float).reshape(-1, 3)))
                        cv2.polylines(canvas, [cv2.convexHull(np.rint(top).astype(np.int32))], True, color, 2)
                        center = np.asarray(track.top_center, dtype=float)
                        count = len(track.fit.candidate.points)
                        cx, cy = (int(round(value)) for value in fitter.cam.project(center)[0])
                        cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                        text = (f"{class_name} {confidence:.2f} TOP X{center[0]:+.3f} Y{center[1]:+.3f} "
                                f"Z{center[2]:+.3f}m IoU{track.fit.iou:.2f}")
                        objects.append((class_name, confidence, center, count))
                    elif center is None:
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