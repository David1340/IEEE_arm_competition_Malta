"""Load point-cloud cases saved by dev/experiments/capture_pointcloud_case.py.

A case folder holds image.png, point_cloud_xyzrgba.npy, camera_transform.npy, roi.json,
metadata.json and objects/object_XXX.npz (per YOLO detection). This module turns it into
per-object data in the calibrated TABLE frame (metres), plus:

  * the local table plane around each object, fitted to a ring of pixels just outside the
    mask (the calibrated Z = 0 is not exactly the table surface, and the table is not flat
    in the calibration either);
  * the camera position in the table frame (needed to know which model faces are visible).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# dev/experiments/capture_pointcloud_case.py writes to point_cloud_mathing/ (sic).
CASE_ROOTS = [ROOT / "datasets" / "point_cloud_cases", ROOT / "point_cloud_mathing"]   # recorded cases (not in git)
SKIP_CLASSES = {"bin", "gripper"}


def find_cases(roots=CASE_ROOTS):
    cases = [p for r in roots if r.is_dir() for p in r.glob("case_*") if (p / "metadata.json").exists()]
    return sorted(cases, key=lambda p: p.name)


def to_table(T, xyz):
    return xyz @ T[:3, :3].T + T[:3, 3]


def fit_plane(xyz, iters=3):
    """Robust z = a x + b y + c fit. Returns (a, b, c) or None."""
    if len(xyz) < 30:
        return None
    keep = np.ones(len(xyz), bool)
    coef = None
    for _ in range(iters):
        A = np.column_stack((xyz[keep, 0], xyz[keep, 1], np.ones(keep.sum())))
        coef, *_ = np.linalg.lstsq(A, xyz[keep, 2], rcond=None)
        r = xyz[:, 2] - (xyz[:, 0] * coef[0] + xyz[:, 1] * coef[1] + coef[2])
        mad = np.median(np.abs(r[keep])) * 1.4826 + 1e-4
        keep = np.abs(r) < 2.5 * mad
        if keep.sum() < 30:
            break
    return coef


@dataclass
class CaseObject:
    object_id: str
    class_name: str
    confidence: float
    polygon_full: np.ndarray          # (k, 2) full-image px
    pixel_xy: np.ndarray              # (n, 2) full-image px of each point
    xyz_raw: np.ndarray               # (n, 3) table frame, every valid mask point
    rgb_raw: np.ndarray               # (n, 3) 0-1
    table_plane: tuple | None         # (a, b, c) of z = a x + b y + c, or None
    yolo_center: np.ndarray | None    # centre saved by the capture script (table frame)
    ring_xyz: np.ndarray = field(default=None, repr=False)

    def table_z_at(self, x, y):
        if self.table_plane is None:
            return 0.0
        a, b, c = self.table_plane
        return float(a * x + b * y + c)

    def cleaned(self, min_height=0.003, max_height=0.09, max_radius=0.06, erode_px=0):
        """Points of the object itself: above the local table, near the mask's median XY."""
        xyz, rgb = self.xyz_raw, self.rgb_raw
        h = xyz[:, 2] - np.array([self.table_z_at(x, y) for x, y in xyz[:, :2]]) \
            if self.table_plane is not None else xyz[:, 2]
        med = np.median(xyz[:, :2], axis=0)
        keep = (h >= min_height) & (h <= max_height) & (np.linalg.norm(xyz[:, :2] - med, axis=1) <= max_radius)
        if erode_px > 0 and len(self.polygon_full) >= 3:
            x0, y0 = self.polygon_full.min(axis=0).astype(int) - 2
            x1, y1 = self.polygon_full.max(axis=0).astype(int) + 3
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.rint(self.polygon_full - [x0, y0]).astype(np.int32)], 1)
            m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode_px)
            px = self.pixel_xy - [x0, y0]
            inside = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < m.shape[1]) & (px[:, 1] < m.shape[0])
            ok = np.zeros(len(px), bool)
            ok[inside] = m[px[inside, 1], px[inside, 0]] > 0
            keep &= ok
        return xyz[keep], rgb[keep], h[keep]


@dataclass
class Case:
    path: Path
    image: np.ndarray            # full BGR image
    roi: tuple                   # x, y, w, h
    T: np.ndarray                # camera -> table
    objects: list

    @property
    def camera_pos(self):
        return self.T[:3, 3].copy()

    @property
    def case_id(self):
        return self.path.name


def load_case(path, skip_classes=SKIP_CLASSES, ring_px=(8, 30)):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    image = cv2.imread(str(path / meta["files"]["image"]))
    T = np.load(path / meta["files"].get("camera_transform", "camera_transform.npy"))
    x0, y0, w, h = meta["roi_full_image_xywh_px"]
    cloud_file = path / meta["files"].get("point_cloud_xyzrgba", "point_cloud_xyzrgba.npy")
    cloud = np.load(cloud_file, mmap_mode="r") if cloud_file.exists() else None

    dets = meta["detections"]
    # Union of every detection's mask, so table rings avoid neighbouring objects
    all_masks = np.zeros((h, w), np.uint8)
    for d in dets:
        poly = np.asarray(d["polygon_roi_xy_px"], np.float32)
        if len(poly) >= 3:
            cv2.fillPoly(all_masks, [np.rint(poly).astype(np.int32)], 1)
    all_masks = cv2.dilate(all_masks, np.ones((3, 3), np.uint8), iterations=ring_px[0])

    objects = []
    for d in dets:
        if d["class_name"] in skip_classes:
            continue
        npz = np.load(path / d["point_cloud_file"])
        poly = np.asarray(d["polygon_roi_xy_px"], np.float32)
        pixel_xy = npz["pixel_xy_full"].astype(np.int64)
        xyz = npz["table_xyz_m"].astype(np.float64)
        rgb = image[pixel_xy[:, 1], pixel_xy[:, 0], ::-1].astype(np.float64) / 255.0

        plane, ring = None, None
        if cloud is not None and len(poly) >= 3:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.rint(poly).astype(np.int32)], 1)
            outer = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=ring_px[1])
            ys, xs = np.nonzero(outer & (1 - all_masks))
            pts = np.asarray(cloud[ys + y0, xs + x0, :3], dtype=np.float64)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts):
                ring = to_table(T, pts)
                plane = fit_plane(ring)

        c = d.get("center_table_xyz_m")
        objects.append(CaseObject(
            object_id=d["object_id"], class_name=d["class_name"], confidence=float(d["confidence"]),
            polygon_full=poly + [x0, y0], pixel_xy=pixel_xy, xyz_raw=xyz, rgb_raw=rgb,
            table_plane=tuple(plane) if plane is not None else None,
            yolo_center=np.asarray(c) if c is not None else None, ring_xyz=ring,
        ))
    return Case(path=path, image=image, roi=(x0, y0, w, h), T=T, objects=objects)
