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
    p       go back to the previous image (it opens with its saved objects; ENTER saves it again)
    l       legends: all objects -> selected object only -> none -> all ... (kept for the next images)
    h       hide / show the legend of the selected object
Every object shows its class name next to its outline (also the saved ones --add-missing opens), so wrong
labels are easy to spot: select it with [ / ], then d deletes it or b reopens it to label again. When the
legends cover the objects, l or h hides them (the outlines stay).
    q/ESC   quit
Usage:
    python dataset_segmentation.py --images images            # classes from classes.txt
    python dataset_segmentation.py --classes cube ball        # also rewrites classes.txt
    python dataset_segmentation.py --image pour_20261107_1530  # open that image (name or part of it); then
                                                               # ENTER / k / p move on from there
"""

import argparse
import sys
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
from ultralytics import SAM

LEGEND_MODES = ["all", "selected", "none"]   # which objects' class names are drawn (key l)
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
IMG_EXT = {".jpg", ".jpeg", ".png"}
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from roi import DATA  # noqa: E402  (the dataset folder: yolo_dataset/ or YOLO_DATASET)


def segment(model, img, points, labels):
    """Returns the largest polygon (normalized xy, Nx2) for one object, or None."""
    r = model(img, points=[points], labels=[labels], verbose=False)[0]
    if r.masks is None or len(r.masks.xyn) == 0:
        return None
    polys = [p for p in r.masks.xyn if len(p) >= 3]
    return max(polys, key=len) if polys else None


def draw_label(img, text, pts, color):
    """The class name just above an object's outline (black edge so it reads on any background)."""
    x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 10
    y = max(y, 30)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(DATA / "images"))
    ap.add_argument("--labels", default=None, help="default: sibling 'labels' folder")
    ap.add_argument("--classes", nargs="+", default=None, help="overrides classes.txt")
    ap.add_argument("--classes-file", default=str(DATA / "classes.txt"), help="one class name per line, line order = class id")
    ap.add_argument("--model", default="sam_b.pt")
    ap.add_argument("--redo", action="store_true", help="relabel images that already have a label file")
    ap.add_argument("--add-missing", action="store_true", help="open labeled images and preserve their existing objects")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--image", default=None, help="start at this image (its name, or part of it)")
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

    def to_do(f):
        """Images ENTER / k move on to: unlabelled ones (default), labelled ones (--add-missing), or all (--redo).
        p (back) goes to the previous image whatever it is."""
        labelled = (lbl_dir / f"{f.stem}.txt").exists()
        return labelled if args.add_missing else (True if args.redo else not labelled)

    def next_to_do(i):
        while i < len(files) and not to_do(files[i]):
            i += 1
        return i

    if args.image:
        matches = [i for i, f in enumerate(files) if args.image in f.name]
        if not matches:
            ap.error(f"no image matching {args.image!r} in {img_dir}")
        index = matches[0]
    else:
        index = next_to_do(0)
    print(f"{sum(map(to_do, files))} images to label/edit (of {len(files)}); p goes back to the previous one")

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
    legend_mode = 0                  # index in LEGEND_MODES; kept from image to image
    while 0 <= index < len(files) and not quit_all:
        f = files[index]
        file_i = index + 1
        step = 1                     # where to go after this image: +1 next to do, -1 back
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        state["scale"] = min(1.0, args.max_width / w)
        objects = []  # (class_id, polygon, points, labels)
        cls = 0
        existing_label = lbl_dir / f"{f.stem}.txt"
        if existing_label.exists():      # a labelled image (gone back to, or --add-missing / --image): edit it
            for label_line in existing_label.read_text().splitlines():
                parts = label_line.split()
                if len(parts) >= 7:
                    polygon = np.asarray(parts[1:], dtype=float).reshape(-1, 2)
                    objects.append((int(parts[0]), polygon, [], []))
        selected = 0 if objects else None
        hidden = set()                   # id() of the polygons whose legend h hid (this image only)
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
                pts = (poly * [w, h]).astype(np.int32)
                cv2.polylines(vis, [pts], True, color, 5 if object_i == selected else 3)
                mode = LEGEND_MODES[legend_mode]
                if id(poly) in hidden or mode == "none" or (mode == "selected" and object_i != selected):
                    continue
                name = args.classes[c] if c < len(args.classes) else "?"
                draw_label(vis, f"{c}: {name}" + ("  (selected)" if object_i == selected else ""), pts, color)
            if state["poly"] is not None:
                pts = (state["poly"] * [w, h]).astype(np.int32)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], COLORS[cls % len(COLORS)])
                vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
                if LEGEND_MODES[legend_mode] != "none":
                    draw_label(vis, f"{cls}: {args.classes[cls]}  (n confirms)", pts, COLORS[cls % len(COLORS)])
            # Points are stored in full-resolution coordinates.
            for (x, y), l in zip(state["points"], state["labels"]):
                cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0) if l else (0, 0, 255), -1)
            if state["scale"] < 1.0:
                vis = cv2.resize(vis, None, fx=state["scale"], fy=state["scale"])
            text = (f"[{file_i}/{len(files)}] {f.name} | class {cls}:{args.classes[cls]} | objects={len(objects)}"
                    + (" | labelled" if existing_label.exists() else ""))
            if args.add_missing:
                selected_text = f"selected {selected + 1}/{len(objects)}" if selected is not None else "no object selected"
                text = f"[{file_i}/{len(files)}] {f.name} | class {cls}:{args.classes[cls]} | {selected_text} | objects={len(objects)}"
            text += f" | legends: {LEGEND_MODES[legend_mode]} (l)" + (f", {len(hidden)} hidden (h)" if hidden else "")
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
            elif key == ord("l"):
                legend_mode = (legend_mode + 1) % len(LEGEND_MODES)
            elif key == ord("h") and selected is not None:
                hidden ^= {id(objects[selected][1])}
            elif key == ord("k"):
                break
            elif key == ord("p"):
                step = -1                 # unsaved changes on this image are dropped
                break
            elif key in (13, 10):
                if state["poly"] is not None:
                    objects.append((cls, state["poly"], [], []))
                lines = [f"{c} " + " ".join(f"{v:.6f}" for v in poly.flatten()) for c, poly, _, _ in objects]
                (lbl_dir / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                print(f"saved {f.stem}.txt ({len(objects)} objects)")
                break
        if step < 0:
            if index == 0:
                print("this is the first image")
            index = max(index - 1, 0)
        else:
            index = next_to_do(index + 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
