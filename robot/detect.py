"""Where the objects are: YOLO masks + ZED point cloud -> object centres in the robot's base frame.

The 3D centre of each object is computed exactly as yolo_dataset/object_3d_live.py does it
(visible_surface_center: the mean of the mask's point-cloud samples after depth outlier rejection,
moved into the TABLE frame by camera_transform.npy). Here it is averaged over several frames, so one
noisy frame does not decide where the arm goes, and then moved from the TABLE frame to the arm's
base frame with TABLE_TO_BASE.

    detect_live(...)        opens the ZED, runs YOLO on `frames` frames, returns [Detection]
    detect_from_case(path)  the same from a case saved by yolo_dataset/capture_pointcloud_case.py
                            (no camera, no YOLO: for trying the arm side on its own)

The robot's base frame is the URDF's (so101ik's): metres, x forward, y to the arm's left, z up,
origin under the shoulder pan axis.
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

# TABLE frame (ArUco calibration) -> robot base frame: p_base = Rz(yaw) @ p_table + (x, y, z).
# All zero means the ArUco markers were measured in the robot's own base frame. If the robot's base
# (under the pan axis) sits at table coordinates (bx, by), unrotated, use x=-bx, y=-by.
# Check it once: switch the torque off, run so101ik's examples/where.py, put the gripper tip on a block
# that detect.py has located, and compare the two printouts.
TABLE_TO_BASE = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}

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
    samples: list = field(default_factory=list, repr=False)

    def __str__(self):
        x, y, z = self.base_xyz
        return (f"{self.name:<14} conf {self.conf:.2f}  base x {x:+.3f} y {y:+.3f} z {z:+.3f} m"
                f"  (seen {self.seen}x, spread {self.spread_mm:.1f} mm)")


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
                             seen=len(g), spread_mm=float(1000 * xyz[:, :2].std(0).max()), pixel=g[-1].pixel))
    return out


def finish(dets, transform=None):
    for d in dets:
        d.base_xyz = table_to_base(d.table_xyz, transform)
    return sorted(dets, key=lambda d: -d.conf)


def detect_from_case(case_dir, transform=None):
    """Detections from a saved case: its metadata.json already holds each object's centre (TABLE frame)."""
    meta = json.loads((Path(case_dir) / "metadata.json").read_text())
    dets = [Detection(name=d["class_name"], conf=d["confidence"], table_xyz=np.array(d["center_table_xyz_m"], float),
                      pixel=tuple(d["center_pixel_xy_full"]) if d.get("center_pixel_xy_full") else None)
            for d in meta["detections"] if d.get("center_table_xyz_m") is not None]
    image = Path(case_dir) / meta.get("files", {}).get("image", "image.png")
    frame = None
    if image.is_file():
        import cv2
        frame = cv2.imread(str(image))
    return finish(dets, transform), frame


def detect_live(frames=15, weights=None, conf=0.25, serial=None, camera_id=0, resolution="HD1080",
                depth_mode="NEURAL", min_depth=0.10, max_depth=2.0, min_points=20, calibration=None, transform=None):
    """Open the ZED, run YOLO on `frames` frames, return ([Detection], last BGR frame). The camera is closed
    again before returning, so nothing else holds it while the arm moves."""
    import cv2
    import pyzed.sl as sl
    from ultralytics import YOLO

    sys.path.insert(0, str(YOLO_DIR))
    from object_3d_live import CALIBRATION_FILE, latest_weights, load_camera_transform, roi_for_frame, visible_surface_center

    roi_data = json.loads((YOLO_DIR / "roi.json").read_text())
    camera_to_table = load_camera_transform(Path(calibration) if calibration else CALIBRATION_FILE)
    weights = Path(weights) if weights else latest_weights()
    model = YOLO(str(weights))
    print(f"YOLO weights: {weights}")

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
                    centre, _, pixel, _ = visible_surface_center(cloud, poly, roi, min_depth, max_depth, min_points,
                                                                 camera_to_table)
                    if centre is not None:
                        found.append(Detection(name=model.names[cls], conf=float(c), table_xyz=np.asarray(centre), pixel=pixel))
            per_frame.append(found)
    finally:
        zed.close()
    return finish(merge(per_frame, min_seen=max(1, frames // 2)), transform), frame


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
