"""Draw the fitted model centres back onto the camera image, to check them by eye.

For every object in a case: fit the STL candidates (4-DoF ICP, both 3D and top-down
distances), then project into the image
    - the fitted model's silhouette (its points, faint),
    - the model centre (3D mode: red, top-down mode: orange),
    - the top-face centre (green),
    - the capture script's centre (blue).

The camera intrinsics are not saved with a case, so the pinhole model is estimated from the
saved point cloud itself (every pixel stores its camera-frame XYZ).

Usage:
    python dev/experiments/render_centres.py                 # newest case
    python dev/experiments/render_centres.py <case_dir> --out centres.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perception import cases, matching, object_models as models  # noqa: E402

RED, ORANGE, GREEN, BLUE, WHITE = (0, 0, 255), (0, 140, 255), (0, 220, 0), (255, 120, 0), (255, 255, 255)


def estimate_intrinsics(case_path, step=8):
    """fx, fy, cx, cy from the saved camera-frame point cloud (u = fx X/Z + cx, ...)."""
    cloud = np.load(Path(case_path) / "point_cloud_xyzrgba.npy", mmap_mode="r")
    h, w = cloud.shape[:2]
    v, u = np.mgrid[0:h:step, 0:w:step]
    xyz = np.asarray(cloud[::step, ::step, :3], dtype=np.float64)
    ok = np.isfinite(xyz).all(axis=2) & (xyz[..., 2] > 0.1)
    X, Y, Z, u, v = xyz[ok, 0], xyz[ok, 1], xyz[ok, 2], u[ok], v[ok]
    fx, cx = np.linalg.lstsq(np.column_stack((X / Z, np.ones_like(Z))), u, rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.column_stack((Y / Z, np.ones_like(Z))), v, rcond=None)[0]
    err = np.hypot(fx * X / Z + cx - u, fy * Y / Z + cy - v)
    return (fx, fy, cx, cy), float(np.median(err))


class Projector:
    def __init__(self, T_cam_to_table, K):
        self.T_inv = np.linalg.inv(T_cam_to_table)
        self.K = K

    def __call__(self, table_xyz):
        p = np.atleast_2d(table_xyz) @ self.T_inv[:3, :3].T + self.T_inv[:3, 3]
        fx, fy, cx, cy = self.K
        return np.column_stack((fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy))


def fit_object(case, o, mode):
    xyz, _, _ = o.cleaned()
    if len(xyz) < 20:
        return None
    tz = o.table_z_at(*np.median(xyz[:, :2], axis=0))
    cands = models.candidates_for_class(o.class_name)
    if not cands:
        return None
    return matching.Matcher(xyz, case.camera_pos, tz, mode=mode, icp="4dof").fit_all(cands)[0]


def marker(img, p, colour, kind, size=7):
    p = tuple(int(round(c)) for c in p)
    if kind == "cross":
        cv2.drawMarker(img, p, (0, 0, 0), cv2.MARKER_CROSS, size * 2 + 4, 4)
        cv2.drawMarker(img, p, colour, cv2.MARKER_CROSS, size * 2, 2)
    elif kind == "ring":
        cv2.circle(img, p, size, (0, 0, 0), 3)
        cv2.circle(img, p, size, colour, 2)
    else:
        cv2.circle(img, p, size // 2 + 1, (0, 0, 0), -1)
        cv2.circle(img, p, size // 2, colour, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <case>/fitted_centres.png")
    ap.add_argument("--zoom", type=int, default=4, help="zoom factor of the per-object close-ups")
    args = ap.parse_args()

    path = args.case or cases.find_cases()[-1]
    case = cases.load_case(path)
    K, err = estimate_intrinsics(path)
    print(f"{path.name}: intrinsics fx={K[0]:.1f} fy={K[1]:.1f} cx={K[2]:.1f} cy={K[3]:.1f} "
          f"(reprojection {err:.2f} px)")
    proj = Projector(case.T, K)

    img = case.image.copy()
    tiles, report = [], []
    for i, o in enumerate(case.objects, 1):
        f3, f2 = fit_object(case, o, "3d"), fit_object(case, o, "2d")
        cv2.polylines(img, [np.rint(o.polygon_full).astype(np.int32)], True, WHITE, 1)
        if f3 is not None:  # faint silhouette of the 3D-mode fit
            for u, v in proj(f3.pose.apply(f3.candidate.points[::3])):
                if 0 <= int(v) < img.shape[0] and 0 <= int(u) < img.shape[1]:
                    img[int(v), int(u)] = (0.5 * img[int(v), int(u)] + 0.5 * np.array(RED)).astype(np.uint8)
        pts = {}
        if o.yolo_center is not None:
            pts["capture"] = proj(o.yolo_center)[0]
        if f3 is not None:
            pts["model3d"], pts["top3d"] = proj(f3.center)[0], proj(f3.top_center)[0]
        if f2 is not None:
            pts["model2d"] = proj(f2.center)[0]
        mask_c = o.polygon_full.mean(axis=0)
        for key, colour, kind in (("capture", BLUE, "dot"), ("top3d", GREEN, "ring"),
                                  ("model2d", ORANGE, "cross"), ("model3d", RED, "cross")):
            if key in pts:
                marker(img, pts[key], colour, kind)
        lab = f"{i} {o.class_name}"
        if f3 is not None:
            lab += f" -> {f3.candidate.size} {f3.candidate.pose_name}"
        p = o.polygon_full.min(axis=0).astype(int)
        cv2.putText(img, lab, (p[0], p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, lab, (p[0], p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        row = {"object": o.object_id, "class": o.class_name, "mask_centre_px": mask_c.round(1).tolist()}
        for k, v in pts.items():
            row[k + "_px"] = v.round(1).tolist()
            row[k + "_offset_from_mask_centre_px"] = round(float(np.linalg.norm(v - mask_c)), 1)
        if f3 is not None:
            row["model3d"] = f3.candidate.label
            row["center3d_table_mm"] = (f3.center * 1000).round(1).tolist()
        if f2 is not None:
            row["model2d"] = f2.candidate.label
            row["center2d_table_mm"] = (f2.center * 1000).round(1).tolist()
        report.append(row)
        print(f"  {i}. {o.object_id} {o.class_name:12s} "
              + "  ".join(f"{k}: {row[k + '_offset_from_mask_centre_px']:5.1f}px" for k in pts)
              + "  (distance from mask centre)")

    # Close-ups after all drawing is done
    for i, o in enumerate(case.objects, 1):
        x0, y0 = o.polygon_full.min(axis=0).astype(int) - 25
        x1, y1 = o.polygon_full.max(axis=0).astype(int) + 25
        x0, y0 = max(x0, 0), max(y0, 0)
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
        tile = np.full((260, 260, 3), 40, np.uint8)
        ch, cw = min(crop.shape[0], 240), min(crop.shape[1], 260)
        oy, ox = (crop.shape[0] - ch) // 2, (crop.shape[1] - cw) // 2
        tile[20:20 + ch, (260 - cw) // 2:(260 - cw) // 2 + cw] = crop[oy:oy + ch, ox:ox + cw]
        cv2.putText(tile, f"{i}. {o.class_name}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        tiles.append(tile)

    x0, y0, w, h = case.roi
    roi_img = img[y0:y0 + h, x0:x0 + w]
    per_row = max(1, roi_img.shape[1] // 260)
    rows = []
    for s in range(0, len(tiles), per_row):
        r = tiles[s:s + per_row] + [np.full((260, 260, 3), 40, np.uint8)] * (per_row - len(tiles[s:s + per_row]))
        rows.append(np.hstack(r))
    grid = np.vstack(rows)
    grid = np.hstack((grid, np.full((grid.shape[0], roi_img.shape[1] - grid.shape[1], 3), 40, np.uint8)))
    legend = np.full((40, roi_img.shape[1], 3), 255, np.uint8)
    x = 10
    for colour, kind, text in ((RED, "cross", "model centre (3D fit)"), (ORANGE, "cross", "model centre (top-down fit)"),
                               (GREEN, "ring", "top-face centre (3D fit)"), (BLUE, "dot", "capture-script centre"),
                               (RED, "dot", "faint red = fitted model silhouette")):
        marker(legend, (x + 8, 20), colour, kind, 6)
        cv2.putText(legend, text, (x + 22, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x += 22 + 9 * len(text) + 20
    out_img = np.vstack((legend, roi_img, grid))
    out = args.out or path / "fitted_centres.png"
    cv2.imwrite(str(out), out_img)
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
