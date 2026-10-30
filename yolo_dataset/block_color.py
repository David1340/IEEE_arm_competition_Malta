"""Predict the colour (red / green / blue / yellow) of every labelled object from its HSV values.

Takes the YOLO data (image + .txt) and, for each object, decides which of the four
competition colours it is. Works with both YOLO label formats:
    detection:     class cx cy w h                (bounding box)
    segmentation:  class x1 y1 x2 y2 ...           (polygon, used as an exact mask)

Pipeline per object (based on the HSV thresholding / morphology approach of the speed sign
detector, with a few additions for accuracy). Only the pixels of that object are used:
    1. Light Gaussian blur, then convert to HSV (hue in degrees 0-360, S and V in 0-1).
    2. Object mask: the label polygon (or the box), eroded so edge pixels mixing with the
       table/neighbours are dropped. With plain boxes the saturated pixels are cleaned with
       an opening and only the blob filling the box centre is kept, so a neighbour block poking into the
       box does not vote. Pixels of smaller labelled objects lying on top (blocks inside a
       bin, the gripper) are removed too.
    3. Pixel filter: drop low saturation (table, highlights) and very dark pixels (shadows).
    4. Vote: every pixel votes for each colour by its circular hue distance to the colour's
       centre (Gaussian, so red wraps across 0/360), weighted by saturation. Highest score
       wins; its share of the total vote is reported as the confidence.

Hue centres can be fitted on your own images with --calibrate (circular k-means over the
dataset, saved to colors.json and loaded automatically afterwards).

Optional: --white-balance corrects a colour cast from the room lights first, using the white
table as a grey reference (see white_balance.py). Off by default.

Usage:
    python block_color.py                                   # all images in images/ + labels/
    python block_color.py --image images/xxx.jpg            # one image (label found in labels/)
    python block_color.py --image a.jpg --label a.txt --show
    python block_color.py --save-vis colour_vis --save-txt colour_labels
    python block_color.py --calibrate                       # fit hue centres -> colors.json
    python block_color.py --white-balance                   # any of the above + white balance
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png"}
COLORS_FILE = HERE / "colors.json"

# Hue centre and spread (degrees, 0-360) of each block colour, fitted on the dataset with
# --calibrate. colors.json, when present, overrides these.
DEFAULT_COLORS = {
    "red": {"hue": 353.0, "sigma": 12.0},
    "yellow": {"hue": 56.0, "sigma": 10.0},
    "green": {"hue": 181.0, "sigma": 12.0},
    "blue": {"hue": 235.0, "sigma": 12.0},
}
DRAW_BGR = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 160, 0), "blue": (255, 0, 0)}

# Default object classes to colour (classes.txt: cube cylinder dodecahedron hexagonal bin gripper).
DEFAULT_CLASSES = [0, 1, 2, 3, 4]


def load_colors(path: Path = COLORS_FILE) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return DEFAULT_COLORS


def to_hsv(bgr: np.ndarray) -> np.ndarray:
    """HSV with H in degrees [0, 360) and S, V in [0, 1]."""
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    return cv2.cvtColor(blurred.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)


def parse_labels(label_path: Path, w: int, h: int):
    """Yields (class_id, polygon Nx2 int32 or None, box (x1, y1, x2, y2))."""
    if not label_path.exists():
        return
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        vals = np.array(parts[1:], float)
        if len(vals) == 4:
            cx, cy, bw, bh = vals * [w, h, w, h]
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            poly = None
        else:
            poly = (vals[: len(vals) // 2 * 2].reshape(-1, 2) * [w, h]).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(poly)
            box = (x, y, x + bw, y + bh)
        x1, y1, x2, y2 = box
        box = (max(0, int(x1)), max(0, int(y1)), min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2))))
        if box[2] > box[0] and box[3] > box[1]:
            yield c, poly, box


def object_pixels(hsv, poly, box, s_min, v_min, v_max, exclude=None, stages=None):
    """Returns the H, S of the pixels that belong to the object and are reliably coloured.
    If a dict is passed as stages, the box-sized mask after each step is stored in it."""
    x1, y1, x2, y2 = box
    roi = hsv[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    k = max(3, int(0.08 * min(rh, rw)) | 1)  # erosion size scales with object size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if poly is not None:
        shape = np.zeros((rh, rw), np.uint8)
        cv2.fillPoly(shape, [poly - [x1, y1]], 1)
        eroded = cv2.erode(shape, kernel)
        shape = eroded if eroded.sum() >= 30 else shape
        if stages is not None:
            stages["label"] = cv2.fillPoly(np.zeros((rh, rw), np.uint8), [poly - [x1, y1]], 1)
    else:
        shape = np.ones((rh, rw), np.uint8)
        if stages is not None:
            stages["label"] = shape.copy()
    if stages is not None:
        stages["eroded"] = shape.copy()

    if exclude is not None:
        shape = shape & (1 - exclude)
    s, v = roi[..., 1], roi[..., 2]
    good = ((s > s_min) & (v > v_min) & (v < v_max)).astype(np.uint8) & shape
    if stages is not None:
        stages["no_overlap"] = shape.copy()
        stages["filter"] = good.copy()

    if poly is None:
        # Box only: clean the saturated pixels and keep the blob that best fills the box centre
        # (the object is centred in its box; a neighbour poking in sits at the edge).
        good = cv2.morphologyEx(good, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(good)
        if n > 2:
            yy, xx = np.mgrid[0:rh, 0:rw]
            centre = np.exp(-0.5 * (((xx - rw / 2) / (rw / 4)) ** 2 + ((yy - rh / 2) / (rh / 4)) ** 2))
            score = np.bincount(lab.ravel(), weights=centre.ravel(), minlength=n)[1:]
            good = (lab == 1 + np.argmax(score)).astype(np.uint8)
            good = cv2.erode(good, kernel) if cv2.erode(good, kernel).sum() >= 30 else good

    sel = good.astype(bool)
    if stages is not None:
        stages["final"] = sel
    return roi[..., 0][sel], roi[..., 1][sel]


def hue_dist(h, centre):
    d = np.abs(h - centre) % 360.0
    return np.minimum(d, 360.0 - d)


def classify(hue, sat, colors):
    """Saturation-weighted Gaussian hue vote. Returns (colour, confidence, scores)."""
    names = list(colors)
    scores = np.array([
        np.sum(sat * np.exp(-0.5 * (hue_dist(hue, colors[n]["hue"]) / colors[n]["sigma"]) ** 2))
        for n in names
    ])
    if scores.sum() <= 0:
        # Hue far from every centre: fall back to the closest centre of the circular mean hue.
        mean = circular_mean(hue, sat)
        i = int(np.argmin([hue_dist(mean, colors[n]["hue"]) for n in names]))
        return names[i], 0.0, dict(zip(names, scores.tolist()))
    i = int(np.argmax(scores))
    return names[i], float(scores[i] / scores.sum()), dict(zip(names, scores.tolist()))


def circular_mean(hue, weight=None):
    rad = np.deg2rad(hue)
    w = np.ones_like(rad) if weight is None else weight
    return float(np.rad2deg(np.arctan2(np.sum(w * np.sin(rad)), np.sum(w * np.cos(rad)))) % 360.0)


def overlap_mask(objects, i):
    """Mask (in object i's box) of smaller labelled objects lying on top of it, e.g. blocks in a bin."""
    _, poly, (x1, y1, x2, y2) = objects[i]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    for j, (_, p2, b2) in enumerate(objects):
        if j == i or area(b2) >= area(objects[i][2]):
            continue
        if b2[0] >= x2 or b2[2] <= x1 or b2[1] >= y2 or b2[3] <= y1:
            continue
        if p2 is not None:
            cv2.fillPoly(mask, [p2 - [x1, y1]], 1)
        else:
            cv2.rectangle(mask, (b2[0] - x1, b2[1] - y1), (b2[2] - x1 - 1, b2[3] - y1 - 1), 1, -1)
    return mask if mask.any() else None


def predict_image(bgr, label_path, colors, classes, s_min=0.35, v_min=0.15, v_max=1.01, min_pixels=30,
                  debug=False, use_white_balance=False):
    """Returns a list of dicts: class, box, polygon, colour, confidence, hue, n_pixels, scores.
    With debug=True each dict also holds the masks of every step and the pixels that voted."""
    h, w = bgr.shape[:2]
    if use_white_balance:
        from white_balance import white_balance
        bgr = white_balance(bgr)
    hsv = to_hsv(bgr)
    objects = list(parse_labels(label_path, w, h))
    results = []
    for i, (c, poly, box) in enumerate(objects):
        if classes is not None and c not in classes:
            continue
        exclude = overlap_mask(objects, i)
        stages = {} if debug else None
        hue, sat = object_pixels(hsv, poly, box, s_min, v_min, v_max, exclude, stages)
        relaxed = len(hue) < min_pixels
        if relaxed:  # badly lit / tiny object: relax the thresholds
            stages = {} if debug else None
            hue, sat = object_pixels(hsv, poly, box, s_min / 2, v_min / 2, 1.01, exclude, stages)
        r = dict(cls=c, box=box, poly=poly, color="unknown", conf=0.0, hue=None, n=int(len(hue)), scores={})
        if len(hue):
            r["color"], r["conf"], r["scores"] = classify(hue, sat, colors)
            r["hue"] = circular_mean(hue, sat)
        if debug:
            r.update(stages=stages, exclude=exclude, relaxed=relaxed, hue_px=hue, sat_px=sat)
        results.append(r)
    return results


def draw(bgr, results, class_names):
    vis = bgr.copy()
    for r in results:
        col = DRAW_BGR.get(r["color"], (255, 255, 255))
        x1, y1, x2, y2 = r["box"]
        if r["poly"] is not None:
            cv2.polylines(vis, [r["poly"]], True, col, 3)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
        name = class_names[r["cls"]] if r["cls"] < len(class_names) else str(r["cls"])
        text = f"{name} {r['color']} {r['conf']:.2f}"
        cv2.putText(vis, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return vis


def calibrate(files, label_dir, classes, out_path, k_iters=20, use_white_balance=False):
    """Circular k-means (k = 4) over per-object hues, seeded with the default centres."""
    hues, weights = [], []
    for f in files:
        bgr = cv2.imread(str(f))
        for r in predict_image(bgr, label_dir / f"{f.stem}.txt", DEFAULT_COLORS, classes,
                               use_white_balance=use_white_balance):
            if r["hue"] is not None:
                hues.append(r["hue"])
                weights.append(r["n"])
    if not hues:
        raise SystemExit("no labelled objects found to calibrate on")
    hues, weights = np.array(hues), np.sqrt(np.array(weights, float))
    names = list(DEFAULT_COLORS)
    centres = np.array([DEFAULT_COLORS[n]["hue"] for n in names])
    for _ in range(k_iters):
        assign = np.argmin([hue_dist(hues, c) for c in centres], axis=0)
        centres = np.array([
            circular_mean(hues[assign == i], weights[assign == i]) if np.any(assign == i) else centres[i]
            for i in range(len(names))
        ])
    colors = {}
    for i, n in enumerate(names):
        d = hue_dist(hues[assign == i], centres[i])
        # Spread: object-to-object scatter plus pixel noise, never narrower than 8 degrees.
        sigma = float(max(8.0, np.sqrt(np.mean(d ** 2)) + 6.0)) if len(d) else DEFAULT_COLORS[n]["sigma"]
        colors[n] = {"hue": round(float(centres[i]), 1), "sigma": round(sigma, 1)}
        print(f"{n:>7}: hue {centres[i]:6.1f} deg  sigma {sigma:5.1f}  ({int(np.sum(assign == i))} objects)")
    out_path.write_text(json.dumps(colors, indent=2))
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="single image (default: every image in --images)")
    ap.add_argument("--label", default=None, help="label file for --image (default: --labels/<stem>.txt)")
    ap.add_argument("--images", default=str(HERE / "images"))
    ap.add_argument("--labels", default=str(HERE / "labels"))
    ap.add_argument("--classes-file", default=str(HERE / "classes.txt"))
    ap.add_argument("--classes", type=int, nargs="*", default=DEFAULT_CLASSES,
                    help="class ids to colour; pass with no ids for all classes")
    ap.add_argument("--colors", default=str(COLORS_FILE), help="hue centres json (from --calibrate)")
    ap.add_argument("--s-min", type=float, default=0.35, help="min saturation 0-1")
    ap.add_argument("--v-min", type=float, default=0.15, help="min value 0-1")
    ap.add_argument("--show", action="store_true", help="show each annotated image (any key = next, q = quit)")
    ap.add_argument("--save-vis", default=None, help="folder for annotated images")
    ap.add_argument("--save-txt", default=None, help="folder for '<line> <class> <colour> <conf>' files")
    ap.add_argument("--calibrate", action="store_true", help="fit hue centres on the dataset -> --colors")
    ap.add_argument("--white-balance", action="store_true",
                    help="correct the room light's colour cast first (white_balance.py)")
    args = ap.parse_args()

    classes = set(args.classes) if args.classes else None
    cf = Path(args.classes_file)
    class_names = [c.strip() for c in cf.read_text().splitlines() if c.strip()] if cf.exists() else []

    if args.image:
        files = [Path(args.image)]
    else:
        files = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in IMG_EXT)
    label_dir = Path(args.labels)

    if args.calibrate:
        calibrate(files, label_dir, classes, Path(args.colors), use_white_balance=args.white_balance)
        return

    colors = load_colors(Path(args.colors))
    for d in (args.save_vis, args.save_txt):
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

    totals = {n: 0 for n in colors}
    for f in files:
        label = Path(args.label) if args.label and args.image else label_dir / f"{f.stem}.txt"
        bgr = cv2.imread(str(f))
        if bgr is None:
            print(f"could not read {f}")
            continue
        results = predict_image(bgr, label, colors, classes, s_min=args.s_min, v_min=args.v_min,
                                use_white_balance=args.white_balance)

        print(f"\n{f.name}: {len(results)} objects")
        for i, r in enumerate(results):
            name = class_names[r["cls"]] if r["cls"] < len(class_names) else str(r["cls"])
            hue = f"{r['hue']:5.1f}" if r["hue"] is not None else "  -  "
            print(f"  {i:3d} {name:<13} {r['color']:<7} conf {r['conf']:.2f}  hue {hue}  px {r['n']}")
            totals[r["color"]] = totals.get(r["color"], 0) + 1

        if args.save_txt:
            lines = [f"{i} {r['cls']} {r['color']} {r['conf']:.3f}" for i, r in enumerate(results)]
            (Path(args.save_txt) / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if args.save_vis or args.show:
            vis = draw(bgr, results, class_names)
            if args.save_vis:
                cv2.imwrite(str(Path(args.save_vis) / f.name), vis)
            if args.show:
                scale = min(1.0, 1280 / vis.shape[1])
                cv2.imshow("block colours", cv2.resize(vis, None, fx=scale, fy=scale))
                if cv2.waitKey(0) in (ord("q"), 27):
                    break
    if args.show:
        cv2.destroyAllWindows()
    print("\ntotals:", ", ".join(f"{k} {v}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
