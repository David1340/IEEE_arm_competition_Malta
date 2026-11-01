"""Top-face matching: find the object's top face in the point cloud, compare it with the
STL top faces, and mark the orientation and centre.

Steps per object:
  1. Height of every mask point above the local table.
  2. Top face = the points within `band` mm of the highest part (the drop-off to the sides
     separates the top face from the side faces).
  3. Snap to STL height (optional, default on): the camera ray through each pixel is accurate
     even when the ZED depth is not, so each top point is moved along its ray to the STL's top
     height. This removes the sideways shift that a wrong depth causes.
  4. Rasterise the top points (seen from above, 0.5 mm grid) into a filled region.
  5. For every STL candidate (size x resting pose), take the STL surface within the same band
     below its top (the top face; for a prism / cylinder on its side, the strip along the top
     ridge), rotate it through its symmetry period and slide it over the region (all shifts at once with cross-correlation);
     score = IoU (overlap / union). Best IoU wins -> size, pose, yaw, centre.

Usage:
    python point_cloud_matching/top_face.py                       # newest case
    python point_cloud_matching/top_face.py <case_dir> --band 5 --out top_faces.png
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cases  # noqa: E402
import models  # noqa: E402

GRID = 0.5e-3  # m per raster cell


@dataclass
class TopFaceResult:
    candidate: object
    iou: float
    yaw: float                 # radians
    center_xy: np.ndarray      # table frame, metres (centre of the matched STL top face)
    top_z: float               # table z of the top face (local table + STL height)
    measured_top_mm: float     # height of the plateau in the ZED data
    top_points: np.ndarray     # (n, 3) top-face points used (after snapping)
    region: np.ndarray         # raster of the top region
    polygon_xy: np.ndarray     # (k, 3, 2) matched STL top triangles in table XY

    @property
    def center(self):
        return np.array([self.center_xy[0], self.center_xy[1], self.top_z])


def top_points(obj, camera_pos, band_mm=4.0, min_height_mm=3.0):
    """Mask points on the top plateau, plus the plateau height (m) and local table z."""
    xyz, _, h = obj.cleaned(min_height=min_height_mm / 1000)
    if len(xyz) < 20:
        return None, 0.0, 0.0
    tz = obj.table_z_at(*np.median(xyz[:, :2], axis=0))
    h_top = np.percentile(h, 95)
    return xyz[h >= h_top - band_mm / 1000], float(h_top), tz


def snap_to_height(points, camera_pos, z_plane):
    """Move each point along its camera ray onto the plane z = z_plane."""
    d = points - camera_pos
    s = (z_plane - camera_pos[2]) / d[:, 2]
    return camera_pos + s[:, None] * d


def rasterise_points(xy, origin, shape, close_px=5):
    img = np.zeros(shape, np.uint8)
    ij = np.rint((xy - origin) / GRID).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    img[ij[ok, 1], ij[ok, 0]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(img)
    if n > 1:  # keep the largest blob, fill its holes
        img = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, 1, -1)
    return img


def rasterise_polygon(tri_xy):
    """STL top triangles (k, 3, 2) in metres around (0,0) -> template + its origin."""
    lo = tri_xy.reshape(-1, 2).min(axis=0) - 2 * GRID
    hi = tri_xy.reshape(-1, 2).max(axis=0) + 2 * GRID
    shape = tuple((np.ceil((hi - lo) / GRID).astype(int) + 1)[::-1])
    img = np.zeros(shape, np.uint8)
    for t in tri_xy:
        cv2.fillPoly(img, [np.rint((t - lo) / GRID).astype(np.int32)], 1)
    return img, lo


def template_points(cand, band_mm):
    """Model surface points within the same band below the model top as the data selection
    (upward facing). For a flat top this is the top face; for a ridge (prism / cylinder on its
    side) it is the strip along the ridge, which is what the camera data contains."""
    dense = getattr(cand, "_dense_top", None)
    if dense is None:
        pts, nrm = models.sample_surface(cand.tris, _tri_normals(cand.tris), 0.4e-3, np.random.default_rng(0))
        keep = nrm[:, 2] > 0.3
        dense = cand._dense_top = pts[keep]
    sel = dense[dense[:, 2] >= cand.height - band_mm / 1000]
    return sel[:, :2] - cand.top_center[:2]


def _tri_normals(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)


def match_candidate(region, region_origin, cand, band_mm, yaw_step_deg=2.0):
    """Best yaw and position of the candidate's top template over the region (IoU)."""
    tpl = template_points(cand, band_mm)
    n = cand.sym_order
    period = 2 * np.pi / max(n, 1)
    yaws = [0.0] if n == 0 else np.arange(-period / 2, period / 2, np.radians(yaw_step_deg))
    pad = 60
    reg = cv2.copyMakeBorder(region, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0).astype(np.float32)
    area_r = float(region.sum())
    best = (-1.0, 0.0, None)
    for yaw in yaws:
        c, s = np.cos(yaw), np.sin(yaw)
        rt = tpl @ np.array([[c, s], [-s, c]])                # rotate by +yaw
        lo = rt.min(axis=0) - 4 * GRID
        shape = tuple((np.ceil((rt.max(axis=0) + 4 * GRID - lo) / GRID).astype(int) + 1)[::-1])
        tmpl = rasterise_points(rt, lo, shape, close_px=3)
        if tmpl.shape[0] >= reg.shape[0] or tmpl.shape[1] >= reg.shape[1]:
            continue
        inter = cv2.matchTemplate(reg, tmpl.astype(np.float32), cv2.TM_CCORR)
        iou = inter / (area_r + float(tmpl.sum()) - inter)
        j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[j] > best[0]:
            # template top-left at reg pixel (j[1], j[0]) -> top centre in table XY
            centre = region_origin + (np.array([j[1], j[0]]) - pad) * GRID - lo
            best = (float(iou[j]), float(yaw), centre)
    return best


def fit_top_face(obj, case, candidates, band_mm=4.0, snap=True):
    pts, h_top, tz = top_points(obj, case.camera_pos, band_mm)
    if pts is None or len(pts) < 15:
        return None, []
    results = []
    for cand in candidates:
        p = snap_to_height(pts, case.camera_pos, tz + cand.height) if snap else pts
        xy = p[:, :2]
        origin = xy.min(axis=0) - 25 * GRID
        shape = tuple((np.ceil((xy.max(axis=0) + 25 * GRID - origin) / GRID).astype(int) + 1)[::-1])
        region = rasterise_points(xy, origin, shape)
        iou, yaw, centre = match_candidate(region, origin, cand, band_mm)
        if centre is None:
            continue
        c, s = np.cos(yaw), np.sin(yaw)
        poly = (cand.top_tris[:, :, :2] - cand.top_center[:2]) @ np.array([[c, s], [-s, c]]) + centre
        results.append(TopFaceResult(cand, iou, yaw, centre, tz + cand.height, h_top * 1000, p,
                                     region, poly))
    results.sort(key=lambda r: -r.iou)
    return (results[0] if results else None), results


# ------------------------------------------------------------------------------- drawing
def estimate_intrinsics(case_path, step=8):
    cloud = np.load(Path(case_path) / "point_cloud_xyzrgba.npy", mmap_mode="r")
    h, w = cloud.shape[:2]
    v, u = np.mgrid[0:h:step, 0:w:step]
    xyz = np.asarray(cloud[::step, ::step, :3], dtype=np.float64)
    ok = np.isfinite(xyz).all(axis=2) & (xyz[..., 2] > 0.1)
    X, Y, Z = xyz[ok, 0], xyz[ok, 1], xyz[ok, 2]
    fx, cx = np.linalg.lstsq(np.column_stack((X / Z, np.ones_like(Z))), u[ok], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.column_stack((Y / Z, np.ones_like(Z))), v[ok], rcond=None)[0]
    return fx, fy, cx, cy


def project(case, K, table_xyz):
    Ti = np.linalg.inv(case.T)
    p = np.atleast_2d(table_xyz) @ Ti[:3, :3].T + Ti[:3, 3]
    fx, fy, cx, cy = K
    return np.column_stack((fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy))


def render_centres_only(case, K, fits, out):
    """Camera image with only a thin crosshair at each top-face centre, plus zoomed crops."""
    img = case.image.copy()
    marks = []
    for i, (o, best) in enumerate(fits, 1):
        if best is None:
            continue
        u, v = project(case, K, best.center)[0]
        marks.append((i, o, int(round(u)), int(round(v))))
    x0, y0, w, h = case.roi
    full = img.copy()
    for i, o, u, v in marks:
        cv2.drawMarker(full, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(full, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(full, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    top = full[y0:y0 + h, x0:x0 + w]
    tiles = []
    T = 300  # tile size in px
    for i, o, u, v in marks:
        lo, hi = o.polygon_full.min(axis=0), o.polygon_full.max(axis=0)
        cx, cy = ((lo + hi) / 2).astype(int)
        r = int(max(hi - lo) / 2) + 12   # square crop around the whole object
        ax0, ay0 = max(cx - r, 0), max(cy - r, 0)
        src = img[ay0:cy + r, ax0:cx + r]
        z = T / (2 * r)
        crop = cv2.resize(src, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        pu, pv = int((u - ax0 + 0.5) * z), int((v - ay0 + 0.5) * z)
        cv2.line(crop, (pu - 14, pv), (pu + 14, pv), (0, 0, 255), 1)   # thin crosshair
        cv2.line(crop, (pu, pv - 14), (pu, pv + 14), (0, 0, 255), 1)
        cv2.circle(crop, (pu, pv), 2, (0, 0, 255), -1)
        tile = np.full((T + 24, T, 3), 30, np.uint8)
        hh, ww = min(crop.shape[0], T), min(crop.shape[1], T)
        tile[24:24 + hh, :ww] = crop[:hh, :ww]
        cv2.putText(tile, f"{i}. {o.class_name}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(tile)
    per_row = max(1, top.shape[1] // tiles[0].shape[1]) if tiles else 1
    rows = []
    for k in range(0, len(tiles), per_row):
        row = tiles[k:k + per_row]
        row += [np.full_like(tiles[0], 30)] * (per_row - len(row))
        rows.append(np.hstack(row))
    grid = np.vstack(rows) if rows else np.zeros((0, top.shape[1], 3), np.uint8)
    width = max(top.shape[1], grid.shape[1])
    pad = lambda a: np.hstack((a, np.full((a.shape[0], width - a.shape[1], 3), 30, np.uint8)))  # noqa: E731
    cv2.imwrite(str(out), np.vstack((pad(top), pad(grid))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", type=Path)
    ap.add_argument("--band", type=float, default=4.0, help="top-face band below the highest points (mm)")
    ap.add_argument("--no-snap", action="store_true", help="do not move top points to the STL height")
    ap.add_argument("--out", type=Path, default=None, help="default: <case>/top_faces.png")
    ap.add_argument("--centres-only", action="store_true",
                    help="also write <case>/top_face_centres.png: just the centres on the camera image")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    path = args.case or cases.find_cases()[-1]
    case = cases.load_case(path)
    K = estimate_intrinsics(path)
    n = len(case.objects)
    fig, axs = plt.subplots(3, n, figsize=(3.1 * n, 9.6), squeeze=False)
    report, all_best = [], []
    for j, o in enumerate(case.objects):
        cands = models.candidates_for_class(o.class_name)
        best, allres = fit_top_face(o, case, cands, args.band, not args.no_snap)
        all_best.append((o, best))
        ax_img, ax_top, ax_txt = axs[0, j], axs[1, j], axs[2, j]
        for a in (ax_img, ax_top, ax_txt):
            a.set_xticks([])
            a.set_yticks([])
        x0, y0 = o.polygon_full.min(axis=0).astype(int) - 20
        x1, y1 = o.polygon_full.max(axis=0).astype(int) + 20
        x0, y0 = max(x0, 0), max(y0, 0)
        ax_img.imshow(case.image[y0:y1, x0:x1, ::-1], extent=(x0, x1, y1, y0))
        ax_img.set_title(f"{j + 1}. {o.object_id}\nYOLO {o.class_name}", fontsize=9)
        if best is None:
            ax_txt.text(0.02, 0.95, "no fit", va="top", fontsize=8)
            continue
        # (a) image: top-face points + matched STL top face + centre, projected
        uv = project(case, K, best.top_points)
        ax_img.scatter(uv[:, 0], uv[:, 1], s=1, c="magenta", alpha=0.4)
        for t in best.polygon_xy:
            p3 = np.column_stack((t, np.full(3, best.top_z)))
            q = project(case, K, p3)
            ax_img.fill(q[:, 0], q[:, 1], facecolor=(0, 1, 0, 0.25), edgecolor="none")
        edges = best.candidate.top_edges
        c, s = np.cos(best.yaw), np.sin(best.yaw)
        e_xy = (edges[:, :, :2] - best.candidate.top_center[:2]) @ np.array([[c, s], [-s, c]]) + best.center_xy
        for e in e_xy:
            q = project(case, K, np.column_stack((e, np.full(2, best.top_z))))
            ax_img.plot(q[:, 0], q[:, 1], "-", color="lime", lw=2)
        cu = project(case, K, best.center)[0]
        ax_img.plot(*cu, "r+", ms=14, mew=3)
        ax_img.set_xlim(x0, x1)
        ax_img.set_ylim(y1, y0)
        # (b) top-down: top region raster vs matched STL polygon
        xy = best.top_points[:, :2] * 1000
        ax_top.scatter(xy[:, 0], xy[:, 1], s=1, c="magenta", alpha=0.5)
        ax_top.add_collection(PolyCollection(best.polygon_xy * 1000, facecolor=(0, 1, 0, 0.3), edgecolor="none"))
        for e in e_xy * 1000:
            ax_top.plot(*e.T, "-", color="green", lw=1.5)
        ax_top.plot(*(best.center_xy * 1000), "r+", ms=14, mew=3)
        ax_top.set_aspect("equal")
        ax_top.autoscale()
        ax_top.set_title(f"top-down: top pts vs STL top\nIoU {best.iou:.2f}", fontsize=8)
        lines = [f"match: {best.candidate.size} {best.candidate.pose_name}",
                 f"top {best.candidate.top_kind}  yaw {np.degrees(best.yaw):+.0f} deg",
                 f"ZED plateau {best.measured_top_mm:.0f} mm",
                 f"STL height  {best.candidate.height * 1000:.0f} mm",
                 "centre (table mm):",
                 "  X {:.1f}\n  Y {:.1f}\n  Z {:.1f}".format(*(best.center * 1000)), "", "all candidates (IoU):"]
        lines += [f" {r.candidate.size[0].upper()} {r.candidate.pose_name[:8]:8s} {r.iou:.2f}" for r in allres]
        ax_txt.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.5)
        ax_txt.axis("off")
        report.append({"object_id": o.object_id, "class": o.class_name, "model": best.candidate.label,
                       "iou": best.iou, "yaw_deg": float(np.degrees(best.yaw)),
                       "top_center_table_xyz_m": best.center.tolist(),
                       "zed_plateau_mm": best.measured_top_mm, "stl_height_mm": best.candidate.height * 1000})
        print(f"{j + 1}. {o.object_id} {o.class_name:12s} -> {best.candidate.label:34s} IoU {best.iou:.2f} "
              f"yaw {np.degrees(best.yaw):+5.0f}  centre {np.round(best.center * 1000, 1)} mm  "
              f"(ZED top {best.measured_top_mm:.0f} / STL {best.candidate.height * 1000:.0f} mm)")
    fig.suptitle("magenta = ZED top-face points | green = matched STL top face | red + = centre", fontsize=10)
    fig.tight_layout()
    out = args.out or path / "top_faces.png"
    fig.savefig(out, dpi=80)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")
    if args.centres_only:
        c_out = out.with_name("top_face_centres.png")
        render_centres_only(case, K, all_best, c_out)
        print(f"wrote {c_out}")


if __name__ == "__main__":
    main()
