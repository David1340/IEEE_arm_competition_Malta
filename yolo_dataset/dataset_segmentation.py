"""Semi-automatic YOLO segmentation labels using SAM point prompts.

Per image: click on an object (left = include, right = exclude) and SAM draws the mask.
Keys (click the window first):
    0-9     select class index
    n       confirm current object, start a new one
    u       undo last click
    r       reset current object
    [ / ]   select previous / next confirmed object
    d       delete selected object (or last object if none selected)
    b       go back: reopen the last confirmed object for editing
    ENTER   save labels for this image and go to the next
    k       skip image without saving
    q/ESC   quit
Usage:
    python dataset_segmentation.py --images images            # classes from classes.txt
    python dataset_segmentation.py --classes cube ball        # also rewrites classes.txt
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
from ultralytics import SAM

COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
IMG_EXT = {".jpg", ".jpeg", ".png"}
HERE = Path(__file__).resolve().parent


def segment(model, img, points, labels):
    """Returns the largest polygon (normalized xy, Nx2) for one object, or None."""
    r = model(img, points=[points], labels=[labels], verbose=False)[0]
    if r.masks is None or len(r.masks.xyn) == 0:
        return None
    polys = [p for p in r.masks.xyn if len(p) >= 3]
    return max(polys, key=len) if polys else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(HERE / "images"))
    ap.add_argument("--labels", default=None, help="default: sibling 'labels' folder")
    ap.add_argument("--classes", nargs="+", default=None, help="overrides classes.txt")
    ap.add_argument("--classes-file", default=str(HERE / "classes.txt"), help="one class name per line, line order = class id")
    ap.add_argument("--model", default="sam_b.pt")
    ap.add_argument("--redo", action="store_true", help="relabel images that already have a label file")
    ap.add_argument("--add-missing", action="store_true", help="open labeled images and preserve their existing objects")
    ap.add_argument("--max-width", type=int, default=1280)
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels) if args.labels else img_dir.parent / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    classes_path = Path(args.classes_file)
    if args.classes:
        classes_path.write_text("\n".join(args.classes) + "\n")
    elif not classes_path.exists():
        ap.error(f"{classes_path} not found; create it or pass --classes")
    args.classes = [c.strip() for c in classes_path.read_text().splitlines() if c.strip()]
    print(f"{len(args.classes)} classes: {args.classes}")

    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXT)
    if args.add_missing:
        files = [f for f in files if (lbl_dir / f"{f.stem}.txt").exists()]
    elif not args.redo:
        files = [f for f in files if not (lbl_dir / f"{f.stem}.txt").exists()]
    print(f"{len(files)} images to label/edit")

    model = SAM(args.model)
    state = {"points": [], "labels": [], "poly": None, "scale": 1.0, "dirty": True}
    win = "SAM labeling"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        s = state["scale"]
        state["points"].append([x / s, y / s])
        state["labels"].append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
        state["dirty"] = True

    cv2.setMouseCallback(win, on_mouse)

    quit_all = False
    for file_i, f in enumerate(files, start=1):
        if quit_all:
            break
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        state["scale"] = min(1.0, args.max_width / w)
        objects = []  # (class_id, polygon, points, labels)
        cls = 0
        existing_label = lbl_dir / f"{f.stem}.txt"
        if args.add_missing and existing_label.exists():
            for label_line in existing_label.read_text().splitlines():
                parts = label_line.split()
                if len(parts) >= 7:
                    polygon = np.asarray(parts[1:], dtype=float).reshape(-1, 2)
                    objects.append((int(parts[0]), polygon, [], []))
        selected = 0 if objects else None
        quit_armed = False
        state.update(points=[], labels=[], poly=None, dirty=True)

        while True:
            if state["dirty"]:
                has_positive = 1 in state["labels"]
                state["poly"] = segment(model, img, state["points"], state["labels"]) if has_positive else None
                state["dirty"] = False

            vis = img.copy()
            for object_i, (c, poly, _, _) in enumerate(objects):
                color = (255, 255, 255) if object_i == selected else COLORS[c % len(COLORS)]
                cv2.polylines(vis, [(poly * [w, h]).astype(np.int32)], True, color, 5 if object_i == selected else 3)
            if state["poly"] is not None:
                pts = (state["poly"] * [w, h]).astype(np.int32)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], COLORS[cls % len(COLORS)])
                vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
            # Points are stored in full-resolution coordinates.
            for (x, y), l in zip(state["points"], state["labels"]):
                cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0) if l else (0, 0, 255), -1)
            if state["scale"] < 1.0:
                vis = cv2.resize(vis, None, fx=state["scale"], fy=state["scale"])
            text = f"[{file_i}/{len(files)}] {f.name} | class {cls}:{args.classes[cls]} | objects={len(objects)}"
            if args.add_missing:
                selected_text = f"selected {selected + 1}/{len(objects)}" if selected is not None else "no object selected"
                text = f"[{file_i}/{len(files)}] {f.name} | class {cls}:{args.classes[cls]} | {selected_text} | objects={len(objects)}"
            if quit_armed:
                text = "Press q again to quit (unsaved work on this image is lost), any other key cancels"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, vis)

            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                if quit_armed:
                    quit_all = True
                    break
                quit_armed = True
                continue
            quit_armed = False
            if ord("0") <= key <= ord("9") and key - ord("0") < len(args.classes):
                cls = key - ord("0")
            elif key == ord("n"):
                if state["poly"] is not None:
                    objects.append((cls, state["poly"], list(state["points"]), list(state["labels"])))
                    selected = len(objects) - 1
                state.update(points=[], labels=[], poly=None)
            elif key == ord("u") and state["points"]:
                state["points"].pop()
                state["labels"].pop()
                state["dirty"] = True
            elif key == ord("r"):
                state.update(points=[], labels=[], poly=None)
            elif key == ord("d") and objects:
                remove_i = selected if selected is not None else len(objects) - 1
                objects.pop(remove_i)
                selected = min(remove_i, len(objects) - 1) if objects else None
            elif key == ord("b") and objects:
                reopen_i = selected if selected is not None else len(objects) - 1
                cls, poly, pts, lbls = objects.pop(reopen_i)
                selected = min(reopen_i, len(objects) - 1) if objects else None
                state.update(points=pts, labels=lbls, poly=poly)
            elif key == ord("[") and objects:
                selected = (selected - 1) % len(objects) if selected is not None else len(objects) - 1
            elif key == ord("]") and objects:
                selected = (selected + 1) % len(objects) if selected is not None else 0
            elif key == ord("k"):
                break
            elif key in (13, 10):
                if state["poly"] is not None:
                    objects.append((cls, state["poly"], [], []))
                lines = [f"{c} " + " ".join(f"{v:.6f}" for v in poly.flatten()) for c, poly, _, _ in objects]
                (lbl_dir / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                print(f"saved {f.stem}.txt ({len(objects)} objects)")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
