"""Capture one ZED frame, YOLO predictions, and per-object point-cloud samples.

Usage:
    python yolo_dataset/capture_pointcloud_case.py
    python yolo_dataset/capture_pointcloud_case.py --weights runs/segment/train-3/weights/best.pt

Creates a unique case folder under point_cloud_matching/cases/ containing the source image,
full XYZRGBA point cloud, YOLO overlay, metadata, and one NPZ per detected instance.
This script captures data only; it never connects to or moves the robot.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from object_3d_live import (  # noqa: E402
    CALIBRATION_FILE,
    latest_weights,
    load_camera_transform,
    roi_for_frame,
    visible_surface_center,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", type=int, default=None, help="ZED serial; defaults to camera ID")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--weights", default=None, help="default: latest best.pt under runs/segment")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=None, help="default: longest ROI side")
    parser.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD1080")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth-mode", choices=["QUALITY", "ULTRA", "NEURAL"], default="QUALITY")
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=2.00)
    parser.add_argument("--out", type=Path, default=ROOT / "point_cloud_matching" / "cases")
    args = parser.parse_args()

    roi_file = HERE / "roi.json"
    if not roi_file.exists():
        raise SystemExit(f"ROI config not found: {roi_file}")
    roi_data = json.loads(roi_file.read_text())
    camera_to_table = load_camera_transform(CALIBRATION_FILE)
    weights = Path(args.weights) if args.weights else latest_weights()
    model = YOLO(str(weights))
    names = model.names
    imgsz = args.imgsz
    print(f"Model: {weights}; confidence threshold: {args.conf}")
    print(f"Calibration: {CALIBRATION_FILE}")

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

    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open ZED camera: {status}")

    try:
        image_mat, cloud_mat = sl.Mat(), sl.Mat()
        if camera.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED frame grab failed")
        camera.retrieve_image(image_mat, sl.VIEW.LEFT)
        camera.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
        frame = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
        point_cloud = cloud_mat.get_data().copy()
    finally:
        camera.close()

    roi = roi_for_frame(frame.shape, roi_data)
    x0, y0, roi_w, roi_h = roi
    imgsz = imgsz or max(roi_w, roi_h)
    result = model.predict(
        frame[y0 : y0 + roi_h, x0 : x0 + roi_w],
        imgsz=imgsz,
        conf=args.conf,
        verbose=False,
    )[0]

    annotated = frame.copy()
    detections = []
    object_data = []
    if result.masks is not None:
        for index, (polygon, class_id, confidence) in enumerate(
            zip(result.masks.xy, result.boxes.cls.int().tolist(), result.boxes.conf.tolist()), start=1
        ):
            class_name = str(names[class_id])
            polygon = np.asarray(polygon, dtype=np.float32)
            if len(polygon) < 3:
                continue

            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
            ys, xs = np.nonzero(mask)
            camera_xyz = point_cloud[y0 : y0 + roi_h, x0 : x0 + roi_w, :3][ys, xs]
            valid = (
                np.isfinite(camera_xyz).all(axis=1)
                & (camera_xyz[:, 2] >= args.min_depth)
                & (camera_xyz[:, 2] <= args.max_depth)
            )
            pixel_xy = np.column_stack((xs[valid] + x0, ys[valid] + y0)).astype(np.int32)
            camera_xyz = camera_xyz[valid].astype(np.float32)
            camera_xyz_h = np.column_stack((camera_xyz, np.ones(len(camera_xyz), dtype=np.float32)))
            table_xyz = (camera_to_table @ camera_xyz_h.T).T[:, :3].astype(np.float32)

            center_table, center_points, center_pixel, center_mask = visible_surface_center(
                point_cloud,
                polygon,
                roi,
                args.min_depth,
                args.max_depth,
                20,
                camera_to_table,
            )
            object_id = f"object_{index:03d}"
            object_data.append((object_id, mask, {
                "pixel_xy_full": pixel_xy,
                "camera_xyz_m": camera_xyz,
                "table_xyz_m": table_xyz,
                "mask_roi": mask,
                "polygon_roi_xy_px": polygon,
                "center_mask_roi": center_mask if center_mask is not None else np.zeros_like(mask),
            }))

            color = (0, 220, 0) if center_table is not None else (128, 128, 128)
            full_polygon = np.rint(polygon + [x0, y0]).astype(np.int32)
            cv2.polylines(annotated, [full_polygon], True, color, 3)
            if center_table is not None and center_pixel is not None:
                cv2.circle(annotated, center_pixel, 7, (255, 255, 255), -1)
                label = f"{class_name} {confidence:.2f} TABLE {np.round(center_table, 3)}m"
            else:
                label = f"{class_name} {confidence:.2f} center n/a"
            cv2.putText(annotated, label, tuple(full_polygon.min(axis=0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            detections.append({
                "object_id": object_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox_xyxy_roi_px": result.boxes.xyxy[index - 1].cpu().numpy().tolist(),
                "polygon_roi_xy_px": polygon.tolist(),
                "mask_file": f"masks/{object_id}.png",
                "point_cloud_file": f"objects/{object_id}.npz",
                "valid_mask_point_count": int(len(camera_xyz)),
                "center_inlier_point_count": int(center_points),
                "center_table_xyz_m": center_table.tolist() if center_table is not None else None,
                "center_pixel_xy_full": list(center_pixel) if center_pixel is not None else None,
            })

    cv2.putText(annotated, "ENTER/SPACE/S = save case | Q/ESC = discard", (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    scale = min(1.0, 1400 / annotated.shape[1])
    preview = cv2.resize(annotated, None, fx=scale, fy=scale) if scale < 1.0 else annotated
    cv2.namedWindow("Review point-cloud case", cv2.WINDOW_NORMAL)
    cv2.imshow("Review point-cloud case", preview)
    print("Review the annotated image. Press Enter/Space/s to save, or q/Esc to discard.")
    key = cv2.waitKeyEx(0) & 0xFF
    try:
        cv2.destroyWindow("Review point-cloud case")
    except cv2.error:
        cv2.destroyAllWindows()
    if key not in (ord("s"), 10, 13, 32):
        print("Case discarded; no data files were written.")
        return

    case_id = datetime.now(timezone.utc).strftime("case_%Y%m%dT%H%M%S_%fZ")
    case_dir = args.out / case_id
    (case_dir / "objects").mkdir(parents=True, exist_ok=False)
    (case_dir / "masks").mkdir()
    image_path = case_dir / "image.png"
    cloud_path = case_dir / "point_cloud_xyzrgba.npy"
    overlay_path = case_dir / "yolo_predictions.jpg"
    cv2.imwrite(str(image_path), frame)
    np.save(cloud_path, point_cloud)
    (case_dir / "roi.json").write_text(json.dumps(roi_data, indent=2))
    np.save(case_dir / "camera_transform.npy", camera_to_table)
    cv2.imwrite(str(overlay_path), annotated)
    for object_id, mask, arrays in object_data:
        cv2.imwrite(str(case_dir / "masks" / f"{object_id}.png"), mask)
        np.savez_compressed(case_dir / "objects" / f"{object_id}.npz", **arrays)

    metadata = {
        "case_id": case_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_units": "meters",
        "camera_point_cloud_frame": "ZED IMAGE (+X right, +Y down, +Z forward)",
        "table_frame": "camera_transform.npy / ArUco calibration",
        "image_shape_bgr": list(frame.shape),
        "point_cloud_shape_xyzrgba": list(point_cloud.shape),
        "roi_full_image_xywh_px": list(roi),
        "confidence_threshold": args.conf,
        "imgsz": imgsz,
        "depth_mode": args.depth_mode,
        "depth_range_m": [args.min_depth, args.max_depth],
        "model_weights": str(weights.resolve()),
        "calibration_file": str(CALIBRATION_FILE.resolve()),
        "files": {
            "image": image_path.name,
            "point_cloud_xyzrgba": cloud_path.name,
            "prediction_overlay": overlay_path.name,
            "roi": "roi.json",
            "camera_transform": "camera_transform.npy",
        },
        "detections": detections,
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved case to {case_dir}")
    print(f"Captured one frame; saved {len(detections)} detections")
    for detection in detections:
        print(f"  {detection['object_id']} {detection['class_name']} conf={detection['confidence']:.2f} "
              f"points={detection['valid_mask_point_count']} center={detection['center_table_xyz_m']}")


if __name__ == "__main__":
    main()
