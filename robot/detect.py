"""Where the objects are: YOLO masks + ZED point cloud -> object centres in the robot's base frame.

Each object's mask and point cloud are taken as yolo_dataset/object_3d_live.py does it (visible_surface_center,
TABLE frame from camera_transform.npy); its x, y and height then come from its top face (top_face_centre), since
a tall object's side wall pulls the whole-mask mean towards the camera. Here it is averaged over several frames, so one
noisy frame does not decide where the arm goes, and then moved from the TABLE frame to the arm's
base frame with TABLE_TO_BASE.

    detect_live(...)        opens the ZED, runs YOLO on `frames` frames, returns [Detection]
    TABLE_TO_BASE           identity with calibration.json (robot/calibrate.py); else the older
                            correction from robot/table_to_base.json
    detect_from_case(path)  the same from a case saved by yolo_dataset/capture_pointcloud_case.py
                            (no camera, no YOLO: for trying the arm side on its own)

The robot's base frame is the URDF's (so101ik's): metres, x forward, y to the arm's left, z up; the origin is
base_link's, at the bottom of the base (table level), 3.9 cm behind the shoulder pan axis (x 0.039, y 0).
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

for _fonts in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype", "/usr/share/fonts"):
    if os.path.isdir(_fonts) and not os.path.isdir(os.environ.get("QT_QPA_FONTDIR", "")):
        os.environ["QT_QPA_FONTDIR"] = _fonts      # OpenCV's Qt has no fonts of its own: stops its warnings

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolo_dataset"

# The camera's pose: calibration.json's camera_to_base (robot/calibrate.py) puts the point cloud straight into the
# robot's base frame, and TABLE_TO_BASE stays identity. Without it, the older camera_transform.npy (the old qr_calibrate.py)
# gives a TABLE frame, and TABLE_TO_BASE (from robot/table_to_base.json if there) moves it to the base frame:
# p_base = Rz(yaw) @ p_table + (x, y, z).
import calibrate as _cal  # noqa: E402

IDENTITY_T = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}
TABLE_TO_BASE = dict(IDENTITY_T)
TRANSFORM_FILE = Path(__file__).resolve().parent / "table_to_base.json"
CAMERA_TO_BASE = _cal.camera_to_base()
if CAMERA_TO_BASE is None and TRANSFORM_FILE.is_file():
    _saved = json.loads(TRANSFORM_FILE.read_text())
    TABLE_TO_BASE = {k: float(_saved[k]) for k in ("x", "y", "z", "yaw_deg")}

# Classes that are not objects to go to.
SKIP_CLASSES = {"gripper", "bin"}


@dataclass
class Detection:
    name: str
    conf: float
    table_xyz: np.ndarray          # centre in the TABLE frame (m), as object_3d_live.py prints it
    base_xyz: np.ndarray = None    # the same in the robot base frame (m)
    seen: int = 1                  # frames it was seen in
    spread_mm: float = 0.0         # how much the per-frame centres disagreed (std, XY)
    pixel: tuple = None            # pixel centre in the full image, for drawing
    polygon: np.ndarray = None     # YOLO's mask outline in the full image (N x 2 pixels), for drawing and clicks
    colour: str = "unknown"        # red / yellow / green / blue, from yolo_dataset/block_color.py (HSV vote)
    colour_conf: float = 0.0       # the winning colour's share of the vote
    fit: object = None             # shape_fit.Fit: size, true height, outline (the STL fitted to the mask), or None
    plan: tuple = None             # go_to_object: (goal, pitch, Poses) to pick it, once planned
    width: float = None            # go_to_object: how wide it is across the jaws (m)
    status: str = ""               # go_to_object: reachable / out of reach / ...
    samples: list = field(default_factory=list, repr=False)

    def __str__(self):
        x, y, z = self.base_xyz
        size = f"{self.fit.size} " if self.fit is not None else ""
        return (f"{self.colour:>7} {size + self.name:<19} conf {self.conf:.2f}  base x {x:+.3f} y {y:+.3f} z {z:+.3f} m"
                f"  (seen {self.seen}x, spread {self.spread_mm:.1f} mm)")


TOP_BAND = 0.008       # m: the top face is the mask's points within this of its highest (95th percentile) point


def top_face_centre(cloud, mask, roi, camera_to_table, min_depth, max_depth, min_points=15):
    """x, y of the object's top face and its height (m, TABLE frame), from the mask's point-cloud samples. A tall
    object seen at an angle shows its side wall too, which pulls the whole-mask mean towards the camera; the top
    face does not. None if too few points."""
    x0, y0, w, h = roi
    ys, xs = np.nonzero(mask)
    xyz = cloud[y0 + ys, x0 + xs, :3]
    ok = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= min_depth) & (xyz[:, 2] <= max_depth)
    xyz = xyz[ok] @ camera_to_table[:3, :3].T + camera_to_table[:3, 3]
    if len(xyz) < min_points:
        return None
    z_top = np.percentile(xyz[:, 2], 95)
    top = xyz[xyz[:, 2] >= z_top - TOP_BAND]
    if len(top) < min_points:
        return None
    return np.array([*top[:, :2].mean(axis=0), z_top])


@dataclass
class Scene:
    """The last camera frame and its point cloud, to turn a clicked pixel into a place on the table."""
    frame: np.ndarray                  # BGR image (None if there is none)
    cloud: np.ndarray                  # H x W x 3, ZED camera frame (m)
    camera_to_table: np.ndarray        # 4 x 4
    transform: dict = None             # TABLE -> base (None: TABLE_TO_BASE)

    def pixel_to_base(self, u, v, r=4):
        """The table point (robot base frame, m) under pixel (u, v): the median of the valid point-cloud samples
        within r pixels. None where the camera has no depth."""
        h, w = self.cloud.shape[:2]
        u, v = int(round(u)), int(round(v))
        if not (0 <= u < w and 0 <= v < h):
            return None
        win = np.asarray(self.cloud[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1, :3], float).reshape(-1, 3)
        win = win[np.isfinite(win).all(axis=1)]
        if len(win) < 5:
            return None
        c = np.median(win, axis=0)
        return table_to_base(self.camera_to_table[:3, :3] @ c + self.camera_to_table[:3, 3], self.transform)


def add_colours(frame, dets):
    """Each detection's colour from its mask, with yolo_dataset/block_color.py: the eroded mask's well-saturated
    pixels vote by hue (colors.json there, when calibrated, sets the hue centres). Smaller detections lying on a
    bigger one (blocks in a bin) are left out of the bigger one's vote."""
    if frame is None:
        return dets
    import cv2

    sys.path.insert(0, str(YOLO_DIR))
    import block_color as bc

    colors = bc.load_colors()
    hsv = bc.to_hsv(frame)
    h, w = frame.shape[:2]
    objs = []
    for d in dets:
        if d.polygon is None or len(d.polygon) < 3:
            objs.append(None)
            continue
        poly = np.rint(d.polygon).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(poly)
        box = (max(0, x), max(0, y), min(w, x + bw), min(h, y + bh))
        objs.append((0, poly, box) if box[2] > box[0] and box[3] > box[1] else None)
    valid = [o for o in objs if o is not None]
    k = -1
    for d, o in zip(dets, objs):
        if o is None:
            continue
        k += 1
        exclude = bc.overlap_mask(valid, k)
        hue, sat = bc.object_pixels(hsv, o[1], o[2], 0.35, 0.15, 1.01, exclude)
        if len(hue) < 30:              # badly lit or small: relaxed thresholds, as block_color.py does
            hue, sat = bc.object_pixels(hsv, o[1], o[2], 0.175, 0.075, 1.01, exclude)
        if len(hue):
            d.colour, d.colour_conf, _ = bc.classify(hue, sat, colors)
    return dets


def add_fits(scene, dets):
    """The STL silhouette fit for each object (robot/shape_fit.py); skipped with a note if it cannot run."""
    try:
        import shape_fit
        shape_fit.fit_all(scene, dets, table_to_base_many)
    except Exception as e:          # e.g. scipy missing: the rest works without the fits
        print(f"(shape fit skipped: {type(e).__name__}: {e})")
    return dets


def table_to_base_many(xyz, t=None):
    """table_to_base for an array of points (..., 3)."""
    t = TABLE_TO_BASE if t is None else t
    a = math.radians(t["yaw_deg"])
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return np.asarray(xyz, float) @ rz.T + np.array([t["x"], t["y"], t["z"]])


def table_to_base(xyz, t=None):
    t = TABLE_TO_BASE if t is None else t
    a = math.radians(t["yaw_deg"])
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return rz @ np.asarray(xyz, float) + np.array([t["x"], t["y"], t["z"]])


def merge(per_frame, min_seen, radius=0.03):
    """Per-frame detections -> one Detection per object: same class, centres within radius (m) in XY.
    Kept when seen in at least min_seen frames; the centre is the median of the frames'."""
    groups = []
    for frame in per_frame:
        for d in frame:
            for g in groups:
                if g[0].name == d.name and np.linalg.norm(np.median([s.table_xyz for s in g], 0)[:2] - d.table_xyz[:2]) < radius:
                    g.append(d)
                    break
            else:
                groups.append([d])
    out = []
    for g in groups:
        if len(g) < min_seen:
            continue
        xyz = np.array([s.table_xyz for s in g])
        out.append(Detection(name=g[0].name, conf=float(np.mean([s.conf for s in g])), table_xyz=np.median(xyz, 0),
                             seen=len(g), spread_mm=float(1000 * xyz[:, :2].std(0).max()), pixel=g[-1].pixel,
                             polygon=g[-1].polygon))
    return out


def finish(dets, transform=None):
    for d in dets:
        d.base_xyz = table_to_base(d.table_xyz, transform)
    return sorted(dets, key=lambda d: -d.conf)


def detect_from_case(case_dir, transform=None):
    """Detections from a saved case: its metadata.json already holds each object's centre (TABLE frame)."""
    meta = json.loads((Path(case_dir) / "metadata.json").read_text())
    rx, ry = meta.get("roi_full_image_xywh_px", [0, 0])[:2]
    dets = [Detection(name=d["class_name"], conf=d["confidence"], table_xyz=np.array(d["center_table_xyz_m"], float),
                      pixel=tuple(d["center_pixel_xy_full"]) if d.get("center_pixel_xy_full") else None,
                      polygon=(np.array(d["polygon_roi_xy_px"], float) + [rx, ry]) if d.get("polygon_roi_xy_px") else None)
            for d in meta["detections"] if d.get("center_table_xyz_m") is not None]
    files = meta.get("files", {})
    image = Path(case_dir) / files.get("image", "image.png")
    frame = None
    if image.is_file():
        import cv2
        frame = cv2.imread(str(image))
    cloud = np.load(Path(case_dir) / files.get("point_cloud_xyzrgba", "point_cloud_xyzrgba.npy"), mmap_mode="r")
    T = np.load(Path(case_dir) / files.get("camera_transform", "camera_transform.npy"))
    scene = Scene(frame, cloud, T, transform)
    return add_fits(scene, add_colours(frame, finish(dets, transform))), scene


def detect_live(frames=15, weights=None, conf=0.25, serial=None, camera_id=0, resolution="HD1080",
                depth_mode="NEURAL", min_depth=0.10, max_depth=2.0, min_points=20, calibration=None, transform=None,
                camera_to_table=None, zed=None):
    """Open the ZED, run YOLO on `frames` frames, return ([Detection], Scene of the last frame). The camera is
    closed again before returning, so nothing else holds it while the arm moves. camera_to_table: the camera's
    pose to use (default calibration.json's, else camera_transform.npy); zed: an open camera to use (left open)."""
    import cv2
    import pyzed.sl as sl
    from ultralytics import YOLO

    sys.path.insert(0, str(YOLO_DIR))
    from object_3d_live import CALIBRATION_FILE, latest_weights, load_camera_transform, roi_for_frame, visible_surface_center

    roi_data = json.loads((YOLO_DIR / "roi.json").read_text())
    if camera_to_table is None:
        camera_to_table = (CAMERA_TO_BASE if CAMERA_TO_BASE is not None and calibration is None
                           else load_camera_transform(Path(calibration) if calibration else CALIBRATION_FILE))
    weights = Path(weights) if weights else latest_weights()
    model = YOLO(str(weights))
    print(f"YOLO weights: {weights}")

    own = zed is None
    if own:
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        if serial is not None:
            init.set_from_serial_number(serial)
        else:
            init.set_from_camera_id(camera_id)
        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise SystemExit(f"Could not open ZED camera: {status}")

    image_mat, cloud_mat, runtime = sl.Mat(), sl.Mat(), sl.RuntimeParameters()
    per_frame, frame = [], None
    try:
        for _ in range(5):                      # let exposure and depth settle
            zed.grab(runtime)
        while len(per_frame) < frames:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
            frame = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
            cloud = cloud_mat.get_data()
            roi = roi_for_frame(frame.shape, roi_data)
            x0, y0, w, h = roi
            result = model.predict(frame[y0:y0 + h, x0:x0 + w], imgsz=max(w, h), conf=conf, verbose=False)[0]
            found = []
            if result.masks is not None:
                for poly, cls, c in zip(result.masks.xy, result.boxes.cls.int().tolist(), result.boxes.conf.tolist()):
                    if len(poly) < 3:
                        continue
                    centre, _, pixel, mask = visible_surface_center(cloud, poly, roi, min_depth, max_depth,
                                                                    min_points, camera_to_table)
                    if centre is not None:
                        top = top_face_centre(cloud, mask, roi, camera_to_table, min_depth, max_depth)
                        if top is not None:
                            centre = top
                        found.append(Detection(name=model.names[cls], conf=float(c), table_xyz=np.asarray(centre),
                                               pixel=pixel, polygon=np.asarray(poly, float) + [x0, y0]))
            per_frame.append(found)
        scene = Scene(frame, np.array(cloud[:, :, :3], dtype=np.float32), camera_to_table, transform)
    finally:
        if own:
            zed.close()
    dets = add_colours(scene.frame, finish(merge(per_frame, min_seen=max(1, frames // 2)), transform))
    return add_fits(scene, dets), scene


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print the objects' positions in the robot base frame; nothing moves.")
    ap.add_argument("--case", help="a saved case folder instead of the camera")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--weights")
    a = ap.parse_args()
    dets, _ = detect_from_case(a.case) if a.case else detect_live(a.frames, a.weights)
    for i, d in enumerate(dets):
        print(f"{i:2d}. {d}")
