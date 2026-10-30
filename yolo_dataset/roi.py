"""Region of interest (ROI) shared by capture, training and inference.

    python roi.py select [--image images/xxx.jpg]    # drag a rectangle, ENTER to confirm -> roi.json
    python roi.py apply                              # crop images + convert labels -> images_roi/, labels_roi/
    python roi.py preview                            # show the ROI over an image

In other code:
    from roi import load_roi, crop, to_full
    crop_img = crop(frame)              # use for training data, YOLO input
    x, y = to_full(px_x, px_y)          # map a point from the crop back to the full image
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROI_FILE = HERE / "roi.json"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_roi(path: Path = ROI_FILE) -> tuple[int, int, int, int]:
    """Returns (x, y, w, h) in full-resolution pixels."""
    r = json.loads(path.read_text())
    return r["x"], r["y"], r["w"], r["h"]


def crop(img: np.ndarray, roi=None) -> np.ndarray:
    x, y, w, h = roi or load_roi()
    return img[y : y + h, x : x + w]


def to_full(px: float, py: float, roi=None) -> tuple[float, float]:
    x, y, _, _ = roi or load_roi()
    return px + x, py + y


def first_image(images: Path) -> Path:
    return next(f for f in sorted(images.iterdir()) if f.suffix.lower() in IMG_EXT)


def select(args):
    img = cv2.imread(str(args.image or first_image(Path(args.images))))
    h, w = img.shape[:2]
    scale = min(1.0, args.max_width / w)
    view = cv2.resize(img, None, fx=scale, fy=scale)
    x, y, rw, rh = cv2.selectROI("Drag the ROI, ENTER confirms, c cancels", view, showCrosshair=False)
    cv2.destroyAllWindows()
    if rw == 0 or rh == 0:
        print("cancelled")
        return
    # Width and height rounded down to a multiple of 32, which YOLO handles without padding.
    roi = {
        "x": int(x / scale),
        "y": int(y / scale),
        "w": int(rw / scale) // 32 * 32,
        "h": int(rh / scale) // 32 * 32,
        "ref_size": [w, h],
    }
    ROI_FILE.write_text(json.dumps(roi, indent=2))
    print(f"saved {ROI_FILE}: {roi}")


def preview(args):
    x, y, rw, rh = load_roi()
    img = cv2.imread(str(args.image or first_image(Path(args.images))))
    cv2.rectangle(img, (x, y), (x + rw, y + rh), (0, 255, 0), 4)
    scale = min(1.0, args.max_width / img.shape[1])
    cv2.imshow("ROI (any key closes)", cv2.resize(img, None, fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_label(line: str, size, roi, min_area_px: float):
    """Maps one YOLO polygon line from the full image to the ROI crop; None if outside/too small."""
    W, H = size
    x0, y0, rw, rh = roi
    parts = line.split()
    if len(parts) < 7:
        return None
    pts = (np.array(parts[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    mask = mask[y0 : y0 + rh, x0 : x0 + rw]
    if mask.sum() < min_area_px:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    cnt = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2)
    if len(cnt) < 3:
        return None
    norm = cnt / [rw, rh]
    return f"{parts[0]} " + " ".join(f"{v:.6f}" for v in norm.flatten())


def apply(args):
    roi = load_roi()
    img_dir, lbl_dir = Path(args.images), Path(args.labels)
    out_img, out_lbl = Path(args.out_images), Path(args.out_labels)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_img = n_obj = n_drop = 0
    for f in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        cv2.imwrite(str(out_img / f.name), crop(img, roi))
        n_img += 1
        lbl = lbl_dir / f"{f.stem}.txt"
        if not lbl.exists():
            continue
        lines = []
        for line in lbl.read_text().splitlines():
            new = convert_label(line, (w, h), roi, args.min_area)
            if new:
                lines.append(new)
            elif line.strip():
                n_drop += 1
        (out_lbl / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_obj += len(lines)
    print(f"{n_img} images -> {out_img}; {n_obj} objects kept, {n_drop} dropped (outside ROI or < {args.min_area} px)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(HERE / "images"))
    ap.add_argument("--max-width", type=int, default=1280)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--image", default=None)
    s.set_defaults(fn=select)

    p = sub.add_parser("preview")
    p.add_argument("--image", default=None)
    p.set_defaults(fn=preview)

    a = sub.add_parser("apply")
    a.add_argument("--labels", default=str(HERE / "labels"))
    a.add_argument("--out-images", default=str(HERE / "images_roi"))
    a.add_argument("--out-labels", default=str(HERE / "labels_roi"))
    a.add_argument("--min-area", type=float, default=100, help="drop objects with fewer pixels left in the crop")
    a.set_defaults(fn=apply)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
