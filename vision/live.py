"""ZED + YOLO + top-face silhouette detection of table objects, shared by the live viewer and the pipeline.

    vision = Vision(args)            # args from add_vision_args()
    frame, cloud = vision.grab()
    canvas, objects = vision.detect(frame, cloud)
    # objects: [(class_name, confidence, center_xyz_in_robot_frame_or_None, point_count), ...]

Centres: objects with an STL get the fitted top-face centre (else the point-cloud mean of the mask). Bins use
no depth at all: the mean pixel of the YOLO mask is turned into x, y by the camera ray through it meeting the
plane --bin-plane-z above the table (camera calibration + lens parameters only). The mask holds the rim and
the walls the camera sees, so its mean pixel lies between the rim and the base: half the bin's height works best
(simulated 15 x 15 x 8 cm bin across the field: at most ~17 mm off at 0.04 m; 47 mm at 0, 56 mm at the rim).
--bin-point-cloud goes back to the point-cloud mean for bins.

An object with neither a silhouette fit nor --min-points depth points (plain or shiny surfaces, glare under the
camera) is not dropped: its x, y come the same way as a bin's (mean mask pixel, camera ray onto the plane at half
its STL height) and its top is its STL height. Labelled "PIXELS (no depth)"; --no-pixel-fallback drops it instead.

Objects inside a bin are left out (already sorted, or out of the gripper's way): an object whose mask lies at least
--in-bin-overlap inside a bin's outline (its convex hull, so the bin's open inside counts) is drawn grey, labelled
"in bin", and not returned. --keep-in-bin returns them as before.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

from vision.detection import (
    OBJECT_STL_FILES,
    CALIBRATION_FILE,
    ROI_FILE,
    camera_intrinsics,
    latest_weights,
    load_camera_transform,
    roi_for_frame,
    visible_surface_center,
)

BIN_CLASS = "bin"
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
          (128, 0, 255), (0, 200, 120), (200, 120, 0), (120, 120, 255)]


def add_vision_args(ap):
    ap.add_argument("--serial", type=int, default=None, help="ZED serial number; default uses camera ID")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--weights", default=None, help="default: most recently modified best.pt under runs/segment")
    ap.add_argument("--calibration", type=Path, default=CALIBRATION_FILE,
                    help="camera-to-robot transform saved by qr_calibrate.py")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="default: longest ROI side")
    ap.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD720")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--depth-mode", choices=["QUALITY", "ULTRA", "NEURAL"], default="QUALITY")
    ap.add_argument("--min-depth", type=float, default=0.10, help="ignore points closer than this (meters)")
    ap.add_argument("--max-depth", type=float, default=2.00, help="ignore points farther than this (meters)")
    ap.add_argument("--min-points", type=int, default=5, help="minimum valid mask points for a center estimate")
    ap.add_argument("--no-pixel-fallback", action="store_true",
                    help="drop an object with no fit and too few depth points (default: place it from its pixels)")
    ap.add_argument("--no-silhouette", action="store_true",
                    help="use the point-cloud mean instead of the fitted STL top-face center")
    ap.add_argument("--silhouette-min-iou", type=float, default=0.7,
                    help="minimum mask/silhouette IoU to trust the top-face center")
    ap.add_argument("--bin-plane-z", type=float, default=0.04,
                    help="bins: height above the table [m] of the plane the mask's mean pixel is cast onto "
                         "(about half the bin's height)")
    ap.add_argument("--bin-point-cloud", action="store_true",
                    help="bins: use the point-cloud mean of the mask instead of the mean pixel")
    ap.add_argument("--in-bin-overlap", type=float, default=0.5,
                    help="an object with at least this share of its mask inside a bin's outline is in the bin and "
                         "left out")
    ap.add_argument("--keep-in-bin", action="store_true", help="do not leave out the objects inside bins")


def inside_bins(valid, names, roi_shape, min_overlap):
    """ids of the masks (non-bin detections) lying at least min_overlap inside a bin's outline. valid: [(polygon
    in ROI pixels, class_id, confidence)]. The bin's convex hull is used, so its open inside counts even when its
    mask is only the rim."""
    bins = [polygon for polygon, class_id, _ in valid if names[class_id].casefold() == BIN_CLASS]
    if not bins:
        return set()
    height, width = roi_shape
    bin_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in bins:
        cv2.fillPoly(bin_mask, [cv2.convexHull(np.rint(polygon).astype(np.int32))], 1)
    inside = set()
    for polygon, class_id, _ in valid:
        if names[class_id].casefold() == BIN_CLASS:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        area = int(mask.sum())
        if area and int((mask & bin_mask).sum()) >= min_overlap * area:
            inside.add(id(polygon))
    return inside


def bin_center_from_pixels(polygon, roi, cam, plane_z):
    """A bin's centre without depth: the mean of the mask's pixels, then the camera ray through that pixel
    meets the plane Z = plane_z (robot frame, the table at 0). Returns (xyz, pixel count, (u, v)) or
    (None, count, None) for a mask with no pixels."""
    x0, y0, width, height = roi
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)], 1)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, 0, None
    u, v = xs.mean() + x0, ys.mean() + y0
    return cam.ray_to_plane(u, v, plane_z), len(xs), (int(round(u)), int(round(v)))


class Vision:
    def __init__(self, args):
        import pyzed.sl as sl
        from ultralytics import YOLO

        self.sl = sl
        self.args = args
        if not ROI_FILE.exists():
            raise SystemExit(f"{ROI_FILE} not found; run 'python yolo_dataset/roi.py select' first")
        self.roi_data = json.loads(ROI_FILE.read_text())
        roi = tuple(self.roi_data[key] for key in ("x", "y", "w", "h"))
        self.imgsz = args.imgsz or max(roi[2:])
        self.camera_to_robot = load_camera_transform(args.calibration)
        weights = Path(args.weights) if args.weights else latest_weights()
        print(f"Model: {weights} | imgsz={self.imgsz} | depth={args.depth_mode}")
        print(f"Camera calibration: {args.calibration}")
        self.model = YOLO(str(weights))
        self.names = self.model.names

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
        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise SystemExit(f"Could not open ZED camera: {status}")
        self.image_mat, self.cloud_mat = sl.Mat(), sl.Mat()
        self.runtime = sl.RuntimeParameters()

        from vision.silhouette import Camera
        from vision.silhouette_live import LiveFitter, intrinsics_from_cloud

        try:
            intrinsics = camera_intrinsics(self.zed)
        except Exception as error:
            print(f"Could not read ZED intrinsics ({error}); estimating them from the point cloud")
            self.grab()
            intrinsics = intrinsics_from_cloud(self.cloud_mat.get_data()[..., :3])
        self.cam = Camera(self.camera_to_robot, intrinsics)   # pixel <-> robot frame, for the bins' rays
        self.fitter = None if args.no_silhouette else LiveFitter(self.cam, self.camera_to_robot)

    def grab(self):
        """Latest left image (BGR) and XYZRGBA point cloud, blocking until a frame arrives."""
        sl = self.sl
        while self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            cv2.waitKey(1)
        self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.cloud_mat, sl.MEASURE.XYZRGBA)
        frame = cv2.cvtColor(self.image_mat.get_data(), cv2.COLOR_BGRA2BGR)
        cloud = self.cloud_mat.get_data()
        if cloud.shape[:2] != frame.shape[:2]:
            raise RuntimeError(f"Image/point cloud sizes differ: {frame.shape[:2]} vs {cloud.shape[:2]}")
        return frame, cloud

    def detect(self, frame, cloud):
        from vision.silhouette_live import Detection, SHAPE_CLASSES

        args, names, fitter = self.args, self.names, self.fitter
        current_roi = roi_for_frame(frame.shape, self.roi_data)
        x0, y0, roi_w, roi_h = current_roi
        roi_image = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]
        result = self.model.predict(roi_image, imgsz=self.imgsz, conf=args.conf, verbose=False)[0]
        canvas = frame.copy()
        overlay = canvas.copy()
        objects = []
        self.last_polygons = []   # full-frame pixel outline of each entry of objects, in the same order

        if result.masks is not None:
            classes = result.boxes.cls.int().tolist()
            confidences = result.boxes.conf.tolist()
            valid = [(polygon, class_id, confidence)
                     for polygon, class_id, confidence in zip(result.masks.xy, classes, confidences)
                     if len(polygon) >= 3]
            in_bin = set() if args.keep_in_bin else inside_bins(valid, names, (roi_h, roi_w), args.in_bin_overlap)
            for polygon, class_id, confidence in valid:
                if id(polygon) in in_bin:
                    points = np.rint(polygon + [x0, y0]).astype(np.int32)
                    cv2.polylines(canvas, [points], True, (150, 150, 150), 1)
                    px, py = points.min(axis=0)
                    cv2.putText(canvas, f"{names[class_id]} in bin", (int(px), max(int(py) - 5, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            valid = [item for item in valid if id(item[0]) not in in_bin]
            track_by_detection = {}
            if fitter is not None:
                shape_items = [item for item in valid if names[item[1]] in SHAPE_CLASSES]
                tracks, _ = fitter.update(
                    [Detection(names[class_id], confidence, polygon + [x0, y0])
                     for polygon, class_id, confidence in shape_items],
                    cloud[..., :3],
                )
                track_by_detection = {id(item[0]): track for item, track in zip(shape_items, tracks)}
            for polygon, class_id, confidence in valid:
                points = np.rint(polygon + [x0, y0]).astype(np.int32)
                color = COLORS[class_id % len(COLORS)]
                if names[class_id].casefold() == BIN_CLASS and not args.bin_point_cloud:
                    center, count, pixel_center = bin_center_from_pixels(polygon, current_roi, self.cam,
                                                                         args.bin_plane_z)
                    cv2.polylines(canvas, [points], True, color, 2)
                    if center is None:
                        text = f"{BIN_CLASS} {confidence:.2f} XY n/a (mask too small)"
                    else:
                        cv2.drawMarker(canvas, pixel_center, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
                        text = f"{BIN_CLASS} {confidence:.2f} PIXELS X{center[0]:+.3f} Y{center[1]:+.3f}m"
                    objects.append((BIN_CLASS, confidence, center, count))
                    self.last_polygons.append(points)
                    px, py = points.min(axis=0)
                    cv2.putText(canvas, text, (int(px), max(int(py) - 5, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                color, 2)
                    continue
                center, count, pixel_center, top_mask = visible_surface_center(
                    cloud, polygon, current_roi, args.min_depth, args.max_depth, args.min_points,
                    self.camera_to_robot,
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
                    self.last_polygons.append(points)
                elif center is None and not args.no_pixel_fallback and class_name.casefold() in OBJECT_STL_FILES:
                    # No fit and no depth: x, y from the mask's mean pixel and the camera ray at half the object's
                    # height (as for the bins); the top at its STL height (standing).
                    from control.gripper import object_height

                    height = object_height(OBJECT_STL_FILES[class_name.casefold()])
                    xy, _, pixel_center = bin_center_from_pixels(polygon, current_roi, self.cam, height / 2)
                    if xy is None:
                        center = None
                    else:
                        center = np.array([xy[0], xy[1], height])
                        cv2.polylines(canvas, [points], True, color, 2)
                        cv2.drawMarker(canvas, pixel_center, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
                        text = (f"{class_name} {confidence:.2f} PIXELS (no depth) X{center[0]:+.3f} "
                                f"Y{center[1]:+.3f} Z{center[2]:+.3f}m")
                    objects.append((class_name, confidence, center, count))
                    self.last_polygons.append(points)
                    if center is None:
                        cv2.polylines(canvas, [points], True, (128, 128, 128), 1)
                        text = f"{class_name} {confidence:.2f} XYZ n/a ({count} pts)"
                elif center is None:
                    cv2.polylines(canvas, [points], True, (128, 128, 128), 1)
                    text = f"{class_name} {confidence:.2f} XYZ n/a ({count} pts)"
                    objects.append((class_name, confidence, None, count))
                    self.last_polygons.append(points)
                else:
                    roi_overlay = overlay[y0 : y0 + roi_h, x0 : x0 + roi_w]
                    roi_overlay[top_mask > 0] = color
                    contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, color, 2, offset=(x0, y0))
                    cx, cy = pixel_center
                    cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)
                    text = f"{class_name} {confidence:.2f} TABLE X{center[0]:+.3f} Y{center[1]:+.3f} Z{center[2]:+.3f}m"
                    objects.append((class_name, confidence, center, count))
                    self.last_polygons.append(points)
                px, py = points.min(axis=0)
                cv2.putText(canvas, text, (int(px), max(int(py) - 5, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0)
        cv2.rectangle(canvas, (x0, y0), (x0 + roi_w, y0 + roi_h), (255, 255, 255), 2)
        return canvas, objects

    def close(self):
        self.zed.close()
