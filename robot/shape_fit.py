"""Each object's official STL fitted to its YOLO mask: shape, size (small / large), resting pose, yaw, true height.

This is point_cloud_matching/silhouette.py's render-and-compare fit: the STL is projected through the calibrated
camera and moved (x, y, yaw, size, pose) until its outline matches the YOLO mask best (IoU). It does not use the
depth on the object (the ZED reads dark objects too flat), only the table height around it. From the fit:
    - the object's true height (for how high the jaws go),
    - its centre (straight above the STL's centre),
    - its outline (convex hull) on the table, so how wide it is across the jaws (for how far they open).
A fit under MIN_IOU is not trusted and left out.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "point_cloud_matching"))

MIN_IOU = 0.75
_CANDIDATES = {}


@dataclass
class Fit:
    shape: str                 # STL name: cube, cylinder, dodecahedron, hexagonal_prism
    size: str                  # small / large
    pose: str                  # upright, on side, resting, ...
    iou: float
    centre: np.ndarray         # (3,) base frame (m): x, y of the object's centre, z of its top (on the local table)
    height: float              # m
    hull: np.ndarray           # (n, 2) base frame (m): the object's outline seen from above
    table_z: float             # base frame (m): the table under it (from the point cloud around it)
    yaw_deg: float             # TABLE frame

    @property
    def label(self):
        return f"{self.size} {self.shape} {self.pose} (fit {self.iou:.2f})"

    def width_across(self, axis):
        """How wide the object is along a horizontal unit vector (m)."""
        p = self.hull @ np.asarray(axis, float)[:2]
        return float(p.max() - p.min())


def intrinsics(cloud, step=8):
    """fx, fy, cx, cy of the left camera, from the point cloud itself (as top_face.estimate_intrinsics)."""
    h, w = cloud.shape[:2]
    v, u = np.mgrid[0:h:step, 0:w:step]
    xyz = np.asarray(cloud[::step, ::step, :3], dtype=np.float64)
    ok = np.isfinite(xyz).all(axis=2) & (xyz[..., 2] > 0.1)
    X, Y, Z = xyz[ok, 0], xyz[ok, 1], xyz[ok, 2]
    fx, cx = np.linalg.lstsq(np.column_stack((X / Z, np.ones_like(Z))), u[ok], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.column_stack((Y / Z, np.ones_like(Z))), v[ok], rcond=None)[0]
    return fx, fy, cx, cy


def local_table_z(scene, polygon, ring=(8, 30)):
    """The table's height (TABLE frame) in a ring of pixels around the mask: the median of the point cloud there."""
    import cv2

    h, w = scene.cloud.shape[:2]
    poly = np.rint(polygon).astype(np.int32)
    x0, y0 = np.maximum(poly.min(axis=0) - ring[1] - 2, 0)
    x1, y1 = np.minimum(poly.max(axis=0) + ring[1] + 2, [w, h])
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [poly - [x0, y0]], 1)
    inner = cv2.dilate(m, np.ones((2 * ring[0] + 1,) * 2, np.uint8))
    outer = cv2.dilate(m, np.ones((2 * ring[1] + 1,) * 2, np.uint8))
    ys, xs = np.nonzero(outer & (1 - inner))
    xyz = np.asarray(scene.cloud[y0 + ys, x0 + xs, :3], float)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 20:
        return 0.0
    T = scene.camera_to_table
    z = (xyz @ T[:3, :3].T + T[:3, 3])[:, 2]
    return float(np.median(z))


class _Mask:
    def __init__(self, polygon):
        self.polygon_full = np.asarray(polygon, float)


def fit_all(scene, dets, table_to_base_many, skip=("bin", "gripper")):
    """d.fit for every detection with a mask (not bins or the gripper), or None where the fit is poor."""
    import models
    import silhouette as S

    cam = S.Camera(scene.camera_to_table, intrinsics(scene.cloud))
    for d in dets:
        d.fit = None
        shape = models.CLASS_TO_STL.get(d.name)
        if d.name in skip or shape is None or d.polygon is None or len(d.polygon) < 3:
            continue
        if shape not in _CANDIDATES:
            _CANDIDATES[shape] = models.load_candidates(shape)
        tz = local_table_z(scene, d.polygon)
        m = S.SilhouetteMatcher(cam, _Mask(d.polygon))
        best = max((m.fit(c, tz) for c in _CANDIDATES[shape]), key=lambda f: f.iou)
        if best.iou < MIN_IOU:
            continue
        c = best.candidate
        verts = np.unique(np.round(c.tris.reshape(-1, 3), 6), axis=0)
        table = best.pose_points(verts)                                  # TABLE frame, on the table
        base = table_to_base_many(table, scene.transform)
        top = table_to_base_many(best.pose_points(c.com[None] * [1, 1, 0] + [0, 0, c.height]), scene.transform)[0]
        d.fit = Fit(shape, c.size, c.pose_name, float(best.iou), top, float(c.height), base[:, :2],
                    float(top[2] - c.height), float(np.degrees(best.yaw)))
    return dets
