"""Render-and-compare: fit the STL to the YOLO mask by simulating the camera.

For a candidate (size x resting pose) placed on the table at (X, Y, yaw), the whole STL is
projected through the calibrated camera. That gives the outline the camera *would* see (top
face plus the visible sides, at the right angle), which is compared with the YOLO mask by
IoU. X, Y, yaw (and size / pose) are searched for the best IoU. The top-face centre then
comes from the STL, straight above the object's centre.

Depth on the object is NOT used (it is unreliable on dark objects). The point cloud is only
used for the local table height around each object (the table depth is good), and to
estimate the camera lens parameters.

Usage:
    python -m vision.silhouette                 # newest case
    python -m vision.silhouette <case_dir>

Writes into the case folder:
    silhouette_centres.png   camera image with only the top-face centres (+ close-ups);
                             red = this method, cyan = depth top-face method (top_face.py)
    silhouette_fit.png       close-ups with the predicted outline (green) vs YOLO mask (white)
    silhouette.json          centres and fit scores
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision import cases, models
from vision.matching import rot_z
from vision.top_face import estimate_intrinsics, fit_top_face

SS = 2  # supersampling of the comparison raster


class Camera:
    def __init__(self, T_cam_to_table, K):
        Ti = np.linalg.inv(T_cam_to_table)
        self.R, self.t = Ti[:3, :3], Ti[:3, 3]
        self.K = K
        self.pos = T_cam_to_table[:3, 3]

    def project(self, table_xyz):
        p = np.asarray(table_xyz).reshape(-1, 3) @ self.R.T + self.t
        fx, fy, cx, cy = self.K
        return np.column_stack((fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy))

    def ray_to_plane(self, u, v, z):
        """Table point where the ray through pixel (u, v) meets the plane Z = z."""
        fx, fy, cx, cy = self.K
        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        d = self.R.T @ d_cam
        s = (z - self.pos[2]) / d[2]
        return self.pos + s * d


@dataclass
class SilhouetteFit:
    candidate: object
    iou: float
    x: float
    y: float
    yaw: float
    table_z: float

    def pose_points(self, local):
        return local @ rot_z(self.yaw).T + np.array([self.x, self.y, self.table_z])

    @property
    def top_center(self):
        return self.pose_points(self.candidate.top_center[None])[0]

    @property
    def center(self):
        return self.pose_points(self.candidate.com[None])[0]


class SilhouetteMatcher:
    def __init__(self, cam, obj, pad=30):
        self.cam = cam
        self.obj = obj
        lo = np.floor(obj.polygon_full.min(axis=0)).astype(int) - pad
        hi = np.ceil(obj.polygon_full.max(axis=0)).astype(int) + pad
        self.origin = lo
        self.shape = tuple(((hi - lo) * SS)[::-1])
        self.target = self._fill([obj.polygon_full])
        self.area_t = float(self.target.sum())

    def _fill(self, polys_px):
        img = np.zeros(self.shape, np.uint8)
        pts = [np.rint((p - self.origin) * SS * 4).astype(np.int32) for p in polys_px]  # 2 bits subpixel
        cv2.fillPoly(img, pts, 1, lineType=cv2.LINE_8, shift=2)
        return img

    def render(self, cand, x, y, yaw, tz):
        """Silhouette of the model as the camera sees it. All official shapes are convex, so
        it is exactly the convex hull of the projected vertices."""
        verts = getattr(cand, "_hull_verts", None)
        if verts is None:
            verts = cand._hull_verts = np.unique(np.round(cand.tris.reshape(-1, 3), 7), axis=0)
        uv = self.cam.project(verts @ rot_z(yaw).T + np.array([x, y, tz]))
        hull = cv2.convexHull(np.rint((uv - self.origin) * SS * 4).astype(np.int32))
        img = np.zeros(self.shape, np.uint8)
        cv2.fillConvexPoly(img, hull, 1, lineType=cv2.LINE_8, shift=2)
        return img

    def iou(self, cand, x, y, yaw, tz):
        m = self.render(cand, x, y, yaw, tz)
        inter = float(cv2.countNonZero(cv2.bitwise_and(m, self.target)))
        return inter / (self.area_t + float(cv2.countNonZero(m)) - inter + 1e-9)

    STEPS = [(0.004, np.radians(6)), (0.002, np.radians(3)), (0.001, np.radians(1.5)),
             (0.0005, np.radians(0.75)), (0.00025, np.radians(0.4))]

    def start_xy(self, cand, tz):
        """Start position: the ray through the mask centre meets the plane at half height."""
        u, v = self.obj.polygon_full.mean(axis=0)
        x0, y0, _ = self.cam.ray_to_plane(u, v, tz + cand.height / 2)
        return x0, y0

    def _search(self, cand, tz, p, steps, trace=None):
        """Pattern search on (x, y, yaw): try +-step moves, keep any that raise IoU, then halve."""
        n = cand.sym_order
        s = self.iou(cand, *p, tz)
        if trace is not None:
            trace.append((*p, s))
        for dxy, dyaw in steps:
            improved = True
            while improved:
                improved = False
                moves = [(dxy, 0, 0), (-dxy, 0, 0), (0, dxy, 0), (0, -dxy, 0)]
                if n != 0:
                    moves += [(0, 0, dyaw), (0, 0, -dyaw)]
                for m in moves:
                    q = p + m
                    sq = self.iou(cand, *q, tz)
                    if sq > s + 1e-6:
                        p, s, improved = q, sq, True
                        if trace is not None:
                            trace.append((*p, s))
        return p, s

    def _make_fit(self, cand, p, s, tz):
        n = cand.sym_order
        period = 2 * np.pi / max(n, 1)
        yaw = 0.0 if n == 0 else (p[2] + period / 2) % period - period / 2
        return SilhouetteFit(cand, s, p[0], p[1], yaw, tz)

    def fit(self, cand, tz, yaw_starts=6, trace=None):
        """Full search for one candidate from several start yaws. If `trace` is a list, the
        accepted steps of the winning start are appended to it as (x, y, yaw, iou)."""
        x0, y0 = self.start_xy(cand, tz)
        n = cand.sym_order
        period = 2 * np.pi / max(n, 1)
        starts = [0.0] if n == 0 else [period * (k / yaw_starts - 0.5) for k in range(yaw_starts)]
        best, best_trace = None, None
        for yaw0 in starts:
            tr = [] if trace is not None else None
            p, s = self._search(cand, tz, np.array([x0, y0, yaw0]), self.STEPS, tr)
            if best is None or s > best.iou:
                best, best_trace = self._make_fit(cand, p, s, tz), tr
        if trace is not None:
            trace.extend(best_trace)
        return best

    def refine(self, prev_fit, tz):
        """Tracking: start from last frame's pose, small steps only (a few ms)."""
        p = np.array([prev_fit.x, prev_fit.y, prev_fit.yaw])
        p, s = self._search(prev_fit.candidate, tz, p, self.STEPS[2:])
        return self._make_fit(prev_fit.candidate, p, s, tz)


def fit_object(cam, obj, candidates):
    xyz, _, _ = obj.cleaned()
    pts = xyz if len(xyz) else obj.xyz_raw
    tz = obj.table_z_at(*np.median(pts[:, :2], axis=0))
    m = SilhouetteMatcher(cam, obj)
    fits = sorted((m.fit(c, tz) for c in candidates), key=lambda f: -f.iou)
    return fits, m


# ------------------------------------------------------------------------------- drawing
def crop_tile(img, obj, marks, T=300, outline=None, mask_outline=False):
    lo, hi = obj.polygon_full.min(axis=0), obj.polygon_full.max(axis=0)
    cx, cy = ((lo + hi) / 2).astype(int)
    r = int(max(hi - lo) / 2) + 12
    ax0, ay0 = max(cx - r, 0), max(cy - r, 0)
    z = T / (2 * r)
    crop = cv2.resize(img[ay0:cy + r, ax0:cx + r], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)

    def P(p):
        return int((p[0] - ax0 + 0.5) * z), int((p[1] - ay0 + 0.5) * z)

    if mask_outline:
        cv2.polylines(crop, [np.array([P(p) for p in obj.polygon_full], np.int32)], True, (255, 255, 255), 1)
    if outline is not None:
        for c in outline:
            cv2.polylines(crop, [np.array([P(p) for p in c], np.int32)], True, (0, 230, 0), 1)
    for uv, colour in marks:
        pu, pv = P(uv)
        cv2.line(crop, (pu - 14, pv), (pu + 14, pv), colour, 1)
        cv2.line(crop, (pu, pv - 14), (pu, pv + 14), colour, 1)
        cv2.circle(crop, (pu, pv), 2, colour, -1)
    tile = np.full((T + 24, T, 3), 30, np.uint8)
    hh, ww = min(crop.shape[0], T), min(crop.shape[1], T)
    tile[24:24 + hh, :ww] = crop[:hh, :ww]
    return tile


def grid(tiles, width):
    per_row = max(1, width // tiles[0].shape[1])
    rows = []
    for k in range(0, len(tiles), per_row):
        row = tiles[k:k + per_row] + [np.full_like(tiles[0], 30)] * (per_row - len(tiles[k:k + per_row]))
        rows.append(np.hstack(row))
    g = np.vstack(rows)
    return np.hstack((g, np.full((g.shape[0], max(0, width - g.shape[1]), 3), 30, np.uint8)))


def label(tile, text):
    cv2.putText(tile, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return tile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", type=Path)
    ap.add_argument("--no-compare", action="store_true", help="do not run/draw the depth top-face method")
    args = ap.parse_args()

    path = args.case or cases.find_cases()[-1]
    case = cases.load_case(path)
    cam = Camera(case.T, estimate_intrinsics(path))
    RED, CYAN = (0, 0, 255), (255, 255, 0)

    full = case.image.copy()
    tiles_c, tiles_f, report = [], [], []
    for i, o in enumerate(case.objects, 1):
        cands = models.candidates_for_class(o.class_name)
        if not cands:
            continue
        fits, matcher = fit_object(cam, o, cands)
        best = fits[0]
        uv_new = cam.project(best.top_center)[0]
        marks = [(uv_new, RED)]
        old = None
        if not args.no_compare:
            old, _ = fit_top_face(o, case, cands)
            if old is not None:
                marks.insert(0, (cam.project(old.center)[0], CYAN))
        for uv, colour in marks:
            cv2.drawMarker(full, tuple(int(round(c)) for c in uv), colour, cv2.MARKER_CROSS, 22, 2)
        u, v = (int(round(c)) for c in uv_new)
        cv2.putText(full, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(full, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        tiles_c.append(label(crop_tile(case.image, o, marks), f"{i}. {o.class_name}"))
        # predicted silhouette outline for the fit check
        m = matcher.render(best.candidate, best.x, best.y, best.yaw, best.table_z)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = [c.reshape(-1, 2) / SS + matcher.origin for c in cnts]
        tiles_f.append(label(crop_tile(case.image, o, [(uv_new, RED)], outline=outline, mask_outline=True),
                             f"{i}. {best.candidate.size} {best.candidate.pose_name}  IoU {best.iou:.2f}"))

        tc = best.top_center * 1000
        diff = None if old is None else float(np.linalg.norm(best.top_center[:2] - old.center[:2]) * 1000)
        print(f"{i}. {o.object_id} {o.class_name:12s} -> {best.candidate.label:34s} IoU {best.iou:.3f} "
              f"yaw {np.degrees(best.yaw):+5.1f}  top centre X {tc[0]:.1f} Y {tc[1]:.1f} Z {tc[2]:.1f} mm"
              + ("" if diff is None else f"  (vs depth method {diff:.1f} mm)"))
        report.append({"object_id": o.object_id, "class": o.class_name, "model": best.candidate.label,
                       "iou": best.iou, "yaw_deg": float(np.degrees(best.yaw)),
                       "top_center_table_xyz_m": best.top_center.tolist(),
                       "center_table_xyz_m": best.center.tolist(),
                       "candidates": {f.candidate.label: round(f.iou, 3) for f in fits},
                       "xy_diff_vs_depth_method_mm": diff})

    x0, y0, w, h = case.roi
    top = full[y0:y0 + h, x0:x0 + w]
    legend = np.full((34, w, 3), 255, np.uint8)
    cv2.putText(legend, "red = render-and-compare top-face centre", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    if not args.no_compare:
        cv2.putText(legend, "cyan = depth top-face method", (480, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 160, 0), 2)
    cv2.imwrite(str(path / "silhouette_centres.png"), np.vstack((legend, top, grid(tiles_c, w))))
    legend2 = np.full((34, w, 3), 255, np.uint8)
    cv2.putText(legend2, "white = YOLO mask   green = STL outline as the camera would see it   red = top-face centre",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(path / "silhouette_fit.png"), np.vstack((legend2, grid(tiles_f, w))))
    (path / "silhouette.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {path / 'silhouette_centres.png'}, {path / 'silhouette_fit.png'}, {path / 'silhouette.json'}")


if __name__ == "__main__":
    main()
