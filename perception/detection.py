"""YOLO + ZED helpers shared by every script: ROI, camera calibration, weights, point-cloud
object centres and the top-face silhouette fit.

Coordinates are in the default arm's base frame given by config/calibration/camera_transform.npy (from the camera
calibration, config/calibration/camera.yaml).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402
from core.paths import CAMERA_TRANSFORM_FILE, OBJECTS_DIR, PERCEPTION_FILE, YOLO_DIR, YOLO_RUNS  # noqa: E402

ROI_FILE = PERCEPTION_FILE                     # the ROI is the "roi" section of config/perception.yaml
CALIBRATION_FILE = CAMERA_TRANSFORM_FILE
RUNS = YOLO_RUNS
OBJECT_STL_FILES = {
    "cube": OBJECTS_DIR / "cube_large.stl",
    "cylinder": OBJECTS_DIR / "cylinder_large.stl",
    "dodecahedron": OBJECTS_DIR / "dodecahedron_large.stl",
    "hexagonal": OBJECTS_DIR / "hexagonal_prism_large.stl",
    "hexagonal_prism": OBJECTS_DIR / "hexagonal_prism_large.stl",
}
# Every size of each object (YOLO does not tell them apart): {class: {"small": stl, "large": stl}}
OBJECT_SIZES = {name: {size: path.with_name(path.name.replace("_large", f"_{size}")) for size in ("small", "large")
                       if path.with_name(path.name.replace("_large", f"_{size}")).is_file()}
                for name, path in OBJECT_STL_FILES.items()}
IGNORED_OBJECT_CLASSES = {"gripper", "bin"}


def settings() -> dict:
    """config/perception.yaml as a dict."""
    return config.load(PERCEPTION_FILE, {})


def load_roi(required: bool = True) -> dict | None:
    """The image region YOLO looks at: {x, y, w, h, ref_size} (config/perception.yaml "roi")."""
    roi = settings().get("roi")
    if not roi and required:
        raise SystemExit(f"No 'roi' in {PERCEPTION_FILE}; set it with "
                         "`python dev/dataset_tools/yolo_dataset/roi.py select` (or edit it here)")
    return dict(roi) if roi else None


def latest_weights() -> Path:
    """The YOLO weights to use: config/perception.yaml yolo.weights (a path relative to the repo root), or with
    "latest" the newest runs/segment/*/weights/best.pt, else the newest models/yolo/*.pt."""
    chosen = str(settings().get("yolo", {}).get("weights", "latest"))
    if chosen != "latest":
        path = Path(chosen).expanduser()
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            raise SystemExit(f"YOLO weights {path} not found (config/perception.yaml yolo.weights)")
        return path
    candidates = sorted(RUNS.glob("*/weights/best.pt"), key=lambda path: path.stat().st_mtime)
    candidates = candidates or sorted(YOLO_DIR.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No YOLO weights: none under {RUNS} or {YOLO_DIR}; copy best.pt into models/yolo/, "
                         "set config/perception.yaml yolo.weights, or pass --weights")
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
    """camera -> the default arm's base frame (4 x 4). For the repo's camera_transform.npy it is rebuilt from
    config/calibration/camera.yaml's camera_to_table and the CURRENT config/arms.yaml, so a base edited by hand in config/arms.yaml cannot leave
    the vision in a stale frame (pick and place converts back to the TABLE frame with the current config/arms.yaml); the
    .npy is rewritten when it was stale. Any other path (a saved case) is read as it is."""
    if Path(path).resolve() == CALIBRATION_FILE.resolve():
        try:
            from calibration.calibrate import arm_from_table, camera_to_table, default_arm

            to_table = camera_to_table()
        except Exception as error:                     # noqa: BLE001 - fall back to the .npy
            print(f"Note: could not read config/calibration/camera.yaml ({error}); using {path.name}")
            to_table = None
        if to_table is not None:
            fresh = arm_from_table(default_arm()) @ to_table
            if path.exists():
                old = np.load(path)
                if old.shape == (4, 4) and not np.allclose(old, fresh, atol=1e-6):
                    print(f"Note: {path.name} was made with an older config/arms.yaml base; rebuilt from config/calibration/camera.yaml "
                          f"({1000 * np.linalg.norm(old[:3, 3] - fresh[:3, 3]):.0f} mm apart) and saved")
                    np.save(path, fresh)
            else:
                np.save(path, fresh)
            return fresh
    if not path.exists():
        raise SystemExit(
            f"Camera calibration not found: {path}\n"
            "Calibrate the camera first: python 02_calibrate.py camera tape (or camera aruco)"
        )
    transform = np.load(path)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SystemExit(f"Invalid camera calibration matrix in {path}; expected a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5):
        raise SystemExit(f"Invalid homogeneous-transform last row in {path}: {transform[3]}")
    return transform


def transform_point(transform, point):
    return (transform @ np.append(np.asarray(point, dtype=float), 1.0))[:3]


def camera_intrinsics(zed, cloud_xyz=None):
    """fx, fy, cx, cy of the left ZED camera; falls back to fitting a pinhole model to the cloud."""
    try:
        info = zed.get_camera_information()
        left = getattr(info, "camera_configuration", info).calibration_parameters.left_cam
        return left.fx, left.fy, left.cx, left.cy
    except Exception as error:
        if cloud_xyz is None:
            raise
        print(f"Could not read ZED intrinsics ({error}); estimating them from the point cloud")
        from perception.silhouette_live import intrinsics_from_cloud

        return intrinsics_from_cloud(np.asarray(cloud_xyz)[..., :3])


def top_face_fits(items, cloud, camera_to_table, intrinsics, min_iou=0.7):
    """Shared one-frame top-face silhouette fit, used by every script that detects objects.

    items: [(class_name, confidence, polygon_full_image_xy)]. Returns one entry per item, None when
    the class has no STL or the fit is weaker than min_iou, else a dict with
      center      top-face centre in the table/robot frame (m)
      center_px   its image pixel (x, y)
      outline_px  convex outline of the projected top face, int32 (N, 2)
      iou, label, fit
    Scripts should use `center` instead of the point-cloud mean when it is not None.
    """
    from perception import object_models as silhouette_models
    from perception.silhouette import Camera, SilhouetteMatcher
    from perception.silhouette_live import Detection, local_table_z

    cloud_xyz = np.asarray(cloud)[..., :3]
    camera = Camera(camera_to_table, intrinsics)
    all_mask = np.zeros(cloud_xyz.shape[:2], np.uint8)
    for _, _, polygon in items:
        cv2.fillPoly(all_mask, [np.rint(polygon).astype(np.int32)], 1)
    all_mask = cv2.dilate(all_mask, np.ones((3, 3), np.uint8), iterations=8)
    candidates_by_shape = {}
    results = []
    for class_name, confidence, polygon in items:
        name = str(class_name).strip().casefold()
        shape = silhouette_models.CLASS_TO_STL.get(name)
        if shape is None or len(polygon) < 3:
            results.append(None)
            continue
        if shape not in candidates_by_shape:
            candidates_by_shape[shape] = silhouette_models.load_candidates(shape)
        candidates = candidates_by_shape[shape]
        if not candidates:
            results.append(None)
            continue
        detection = Detection(name, float(confidence), np.asarray(polygon, dtype=float))
        table_z = local_table_z(cloud_xyz, camera_to_table, detection, all_mask)
        matcher = SilhouetteMatcher(camera, detection)
        fit = max((matcher.fit(candidate, table_z) for candidate in candidates), key=lambda item: item.iou)
        if fit.iou < min_iou:
            results.append(None)
            continue
        top_vertices = np.asarray(fit.candidate.top_tris, dtype=float).reshape(-1, 3)
        outline = camera.project(fit.pose_points(top_vertices))
        results.append({
            "center": np.asarray(fit.top_center, dtype=float),
            "center_px": tuple(int(round(v)) for v in camera.project(fit.top_center)[0]),
            "outline_px": cv2.convexHull(np.rint(outline).astype(np.int32)).reshape(-1, 2),
            "iou": float(fit.iou),
            "label": fit.candidate.label,
            "fit": fit,
        })
    return results


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
