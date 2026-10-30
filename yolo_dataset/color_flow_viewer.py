"""Step-by-step viewer for block_color.py: click through how every block's colour is decided.

Left: the whole image at the selected step. Right: the selected object, its mask after every
step, its hue histogram against the 4 colours, the vote and its real vs ideal colour.

Keys (click the window first):
    d / RIGHT = next image            a / LEFT = previous image
    1-9 = step shown on the left      (1 original ... 9 colour groups)
    w = white balance on / off (steps 2-3; off by default, like block_color.py)
    click an object, or n / p = next / previous object
    s = save a screenshot             q / ESC = quit
Usage:
    python color_flow_viewer.py
    python color_flow_viewer.py --image images/img_20261029_105748_714449.jpg
    python color_flow_viewer.py --white-balance       # start with white balance on
    python color_flow_viewer.py --save flow_out      # no window: write every step as PNGs
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

import block_color as bc

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png"}

CANVAS_W, CANVAS_H = 1600, 900
LEFT_W = 1100  # image area; the object panel uses the rest
BG = (32, 32, 32)
TEXT = (235, 235, 235)
DIM = (150, 140, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX

STEPS = [
    ("1  Original + predictions",
     "The raw camera image. Each labelled object is outlined in the colour it was predicted as, with "
     "its confidence. Click any object to follow it through the steps in the right panel."),
    ("2  White reference pixels (optional)",
     "Only used with white balance on (press w). Magenta = pixels used as 'white': almost no colour (saturation < 0.15) and in the brightest 40% "
     "of the image. This is mostly the table. Under neutral light their B, G and R would be equal."),
    ("3  White balanced (optional)",
     "Only used with white balance on (press w). Each channel is multiplied by its gain so the white pixels average out grey. Left half = before, "
     "right half = after. This cancels a colour cast from the room lights before any hue is measured."),
    ("4  Hue",
     "Hue drawn at full strength, so you see the colour angle the classifier reads. Pixels too grey or "
     "too dark to have a reliable hue are shown in grey. Red sits at both ends of the circle (0/360)."),
    ("5  Saturation",
     "Brighter = more strongly coloured. The table, glare and shadows are dark here; blocks are bright. "
     "Pixels below the saturation threshold are dropped and stronger ones count more in the vote."),
    ("6  Value (brightness)",
     "Brightness of each pixel. Very dark pixels (deep shadows, gaps between blocks) are dropped "
     "because their hue is mostly noise."),
    ("7  Pixel filter",
     "White = pixel passes both saturation and brightness tests, so it may vote. Black = dropped. "
     "Notice the table and glare are removed while the blocks survive."),
    ("8  Pixel votes",
     "Every pixel that passed the filter, painted the colour its hue is closest to. Inside each object "
     "the majority wins. Things outside the labels (robot arm, tape) also vote here but are ignored."),
    ("9  Colour groups",
     "Every object cropped and grouped by predicted colour with its confidence. A block in the wrong "
     "column, or a low number, is the one to inspect. Click a thumbnail to select it."),
]


def put(img, text, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def wrap(text, width, scale=0.5):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, 1)[0][0] > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


def ideal_bgr(hue, s=0.85, v=0.85):
    px = np.array([[[hue, s, v]]], np.float32)
    return tuple(int(c) for c in (cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0] * 255))


def fit(img, w, h):
    """Scale img to fit w x h, keeping the aspect ratio. Returns image and scale."""
    scale = min(w / img.shape[1], h / img.shape[0])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def vote_map(hsv, colors):
    """Index of the closest colour for every pixel (by the same Gaussian score as the vote)."""
    names = list(colors)
    scores = np.stack([np.exp(-0.5 * (bc.hue_dist(hsv[..., 0], colors[n]["hue"]) / colors[n]["sigma"]) ** 2)
                       for n in names])
    return np.argmax(scores, axis=0), names


class ImageFlow:
    """Everything computed for one image: each full-image step and the per-object debug data."""

    def __init__(self, path, label_path, colors, classes, s_min, v_min, use_wb=False):
        self.path = path
        self.bgr = cv2.imread(str(path))
        if self.bgr is None:
            raise ValueError(f"could not read {path}")
        self.colors, self.s_min, self.v_min, self.use_wb = colors, s_min, v_min, use_wb
        self.white, self.gains, self.balanced = None, None, self.bgr
        if use_wb:
            from white_balance import white_balance, white_reference
            self.white, self.gains = white_reference(self.bgr)
            self.balanced = white_balance(self.bgr)
        self.hsv = bc.to_hsv(self.balanced)
        self.passed = (self.hsv[..., 1] > s_min) & (self.hsv[..., 2] > v_min)
        self.results = bc.predict_image(self.bgr, label_path, colors, classes,
                                        s_min=s_min, v_min=v_min, debug=True, use_white_balance=use_wb)
        self.votes, self.names = vote_map(self.hsv, colors)
        self._cache = {}

    def outlines(self, img, selected=None, color_by_pred=True, thick=3):
        for i, r in enumerate(self.results):
            col = bc.DRAW_BGR.get(r["color"], (255, 255, 255)) if color_by_pred else (255, 255, 255)
            t = thick + 3 if i == selected else thick
            if i == selected:
                col = (255, 0, 255) if color_by_pred else (0, 0, 255)
            if r["poly"] is not None:
                cv2.polylines(img, [r["poly"]], True, col, t)
            else:
                x1, y1, x2, y2 = r["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), col, t)
        return img

    def step_image(self, step):
        if step in self._cache:
            return self._cache[step].copy()
        h, w = self.bgr.shape[:2]
        if step == 0:
            img = self.bgr.copy()
        elif step in (1, 2) and not self.use_wb:
            img = self.bgr.copy()
        elif step == 1:
            img = self.bgr.copy()
            img[self.white] = (255, 0, 255)
        elif step == 2:
            img = self.bgr.copy()
            img[:, w // 2:] = self.balanced[:, w // 2:]
            cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 3)
        elif step == 3:
            full = self.hsv.copy()
            full[..., 1:] = 1.0
            img = (cv2.cvtColor(full, cv2.COLOR_HSV2BGR) * 255).astype(np.uint8)
            grey = (self.hsv[..., 2] * 120).astype(np.uint8)
            img[~self.passed] = np.repeat(grey[~self.passed][:, None], 3, axis=1)
        elif step == 4:
            img = cv2.cvtColor((self.hsv[..., 1] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif step == 5:
            img = cv2.cvtColor((self.hsv[..., 2] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif step == 6:
            img = cv2.cvtColor(self.passed.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        elif step == 7:
            img = (cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)[..., None].repeat(3, 2) * 0.3).astype(np.uint8)
            palette = np.array([bc.DRAW_BGR.get(n, (255, 255, 255)) for n in self.names], np.uint8)
            img[self.passed] = palette[self.votes[self.passed]]
        else:
            img = self.bgr.copy()
        self._cache[step] = img
        return img.copy()


def draw_left(flow, step, selected):
    """Image area + step description. Returns the canvas part and (scale, image height) for clicks."""
    area = np.full((CANVAS_H, LEFT_W, 3), BG, np.uint8)
    if step == 8:
        return gallery(flow, selected), None
    img = flow.step_image(step)
    flow.outlines(img, selected, color_by_pred=step in (0, 2), thick=2 if step in (0, 2) else 2)
    if step == 0:
        for i, r in enumerate(flow.results):
            x1, y1 = r["box"][:2]
            put(img, f"{i}", (x1 + 2, y1 + 22), 0.8, (0, 0, 0), 4)
            put(img, f"{i}", (x1 + 2, y1 + 22), 0.8, (255, 255, 255), 2)
    view, scale = fit(img, LEFT_W, 620)
    area[:view.shape[0], :view.shape[1]] = view

    y = view.shape[0] + 28
    title, desc = STEPS[step]
    put(area, title, (12, y), 0.75, TEXT, 2)
    y += 28
    for line in wrap(desc, LEFT_W - 24):
        put(area, line, (12, y), 0.52, DIM)
        y += 22
    if step in (1, 2):
        if not flow.use_wb:
            put(area, "White balance is OFF: the raw image goes straight to HSV. Press w to turn it on.",
                (12, y + 6), 0.55, (0, 200, 255))
        elif flow.gains is None:
            put(area, "Too few white pixels: no correction applied.", (12, y + 6), 0.55, (0, 200, 255))
        else:
            means = flow.bgr[flow.white].mean(axis=0)
            put(area, f"white pixels: {int(flow.white.sum())}   average B G R before: "
                      f"{means[0]:.0f} {means[1]:.0f} {means[2]:.0f}   gains B G R: "
                      f"{flow.gains[0]:.3f} {flow.gains[1]:.3f} {flow.gains[2]:.3f}", (12, y + 6), 0.52)
        y += 30
    if step in (3, 4, 5, 6):
        put(area, f"thresholds: saturation > {flow.s_min:.2f}   brightness > {flow.v_min:.2f}", (12, y + 6), 0.52)
        y += 30

    # Per-colour summary for the image.
    y = max(y + 14, CANVAS_H - 110)
    put(area, "This image:", (12, y), 0.55, TEXT)
    x = 12
    for n, spec in flow.colors.items():
        rs = [r for r in flow.results if r["color"] == n]
        cv2.rectangle(area, (x, y + 12), (x + 40, y + 52), ideal_bgr(spec["hue"]), -1)
        put(area, f"{n}: {len(rs)}", (x + 50, y + 28), 0.55)
        if rs:
            mean_hue = bc.circular_mean(np.array([r["hue"] for r in rs]))
            put(area, f"hue {mean_hue:.0f} (centre {spec['hue']:.0f})", (x + 50, y + 48), 0.45, DIM)
            put(area, f"min conf {min(r['conf'] for r in rs):.2f}", (x + 50, y + 66), 0.45, DIM)
        x += 265
    put(area, f"{flow.path.name}", (12, CANVAS_H - 12), 0.45, DIM)
    return area, (scale, view.shape[0])


def gallery(flow, selected):
    """Objects grouped into one column per predicted colour."""
    area = np.full((CANVAS_H, LEFT_W, 3), BG, np.uint8)
    names = list(flow.colors)
    col_w = LEFT_W // len(names)
    thumb, pad = 82, 8
    per_row = max(1, (col_w - pad) // (thumb + pad))
    flow.gallery_rects = []
    for ci, n in enumerate(names):
        x0 = ci * col_w
        cv2.rectangle(area, (x0 + pad, 10), (x0 + col_w - pad, 40), ideal_bgr(flow.colors[n]["hue"]), -1)
        members = [i for i, r in enumerate(flow.results) if r["color"] == n]
        put(area, f"{n} ({len(members)})", (x0 + pad + 8, 32), 0.6, (0, 0, 0), 2)
        for k, i in enumerate(members):
            r = flow.results[i]
            x1, y1, x2, y2 = r["box"]
            crop, _ = fit(flow.bgr[y1:y2, x1:x2], thumb, thumb)
            tx = x0 + pad + (k % per_row) * (thumb + pad)
            ty = 52 + (k // per_row) * (thumb + 24)
            if ty + thumb + 20 > CANVAS_H:
                break
            area[ty:ty + crop.shape[0], tx:tx + crop.shape[1]] = crop
            border = (255, 0, 255) if i == selected else (90, 90, 90)
            cv2.rectangle(area, (tx - 2, ty - 2), (tx + thumb + 1, ty + thumb + 1), border, 2 if i == selected else 1)
            conf_col = TEXT if r["conf"] >= 0.8 else (0, 200, 255)
            put(area, f"#{i} {r['conf']:.2f}", (tx, ty + thumb + 15), 0.42, conf_col)
            flow.gallery_rects.append((tx, ty, tx + thumb, ty + thumb, i))
    unknown = [i for i, r in enumerate(flow.results) if r["color"] not in names]
    if unknown:
        put(area, f"no coloured pixels: objects {unknown}", (12, CANVAS_H - 34), 0.5, (0, 200, 255))
    put(area, STEPS[8][0] + "   (click a thumbnail)", (12, CANVAS_H - 12), 0.5, DIM)
    return area


def mask_tile(crop, mask, size, label, colored=None):
    """Crop with everything outside the mask darkened, scaled into a size x size tile."""
    if colored is not None:
        img = colored
    else:
        img = crop.copy()
        if mask is not None:
            img[mask == 0] = (img[mask == 0] * 0.2).astype(np.uint8)
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, (255, 0, 255), max(1, crop.shape[0] // 120))
    tile = np.full((size + 34, size, 3), BG, np.uint8)
    view, _ = fit(img, size, size)
    oy, ox = (size - view.shape[0]) // 2, (size - view.shape[1]) // 2
    tile[oy:oy + view.shape[0], ox:ox + view.shape[1]] = view
    for k, line in enumerate(label.split("\n")[:2]):
        put(tile, line, (2, size + 14 + k * 15), 0.4, TEXT if k == 0 else DIM)
    return tile


def hue_histogram(r, colors, w, h):
    """Saturation-weighted hue histogram of the voting pixels with each colour's acceptance curve."""
    plot = np.full((h, w, 3), (45, 45, 45), np.uint8)
    top, bottom, left, right = 8, h - 22, 6, w - 6
    pw = right - left
    if r["hue_px"] is not None and len(r["hue_px"]):
        hist, _ = np.histogram(r["hue_px"], bins=180, range=(0, 360), weights=r["sat_px"])
        hist = hist / max(hist.max(), 1e-9)
        for b, v in enumerate(hist):
            if v <= 0:
                continue
            x0 = left + int(b * pw / 180)
            x1 = left + int((b + 1) * pw / 180)
            cv2.rectangle(plot, (x0, bottom - int(v * (bottom - top))), (max(x0, x1 - 1), bottom),
                          ideal_bgr(b * 2 + 1, 1.0, 1.0), -1)
    xs = np.arange(0, 360, 1.0)
    for n, spec in colors.items():
        ys = np.exp(-0.5 * (bc.hue_dist(xs, spec["hue"]) / spec["sigma"]) ** 2)
        pts = np.stack([left + xs * pw / 360, bottom - ys * (bottom - top)], 1).astype(np.int32)
        col = bc.DRAW_BGR.get(n, (255, 255, 255))
        # Split where the curve wraps so red does not draw a line across the whole plot.
        cut = np.where(np.diff(pts[:, 0]) < 0)[0]
        for seg in np.split(pts, cut + 1):
            cv2.polylines(plot, [seg], False, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.polylines(plot, [seg], False, col, 1, cv2.LINE_AA)
    if r["hue"] is not None:
        x = left + int(r["hue"] * pw / 360)
        cv2.line(plot, (x, top), (x, bottom), (255, 255, 255), 1)
        put(plot, f"mean {r['hue']:.0f}", (min(x + 4, right - 70), top + 14), 0.42)
    for deg in (0, 60, 120, 180, 240, 300, 360):
        x = left + int(deg * pw / 360)
        cv2.line(plot, (x, bottom), (x, bottom + 4), DIM, 1)
        put(plot, str(deg), (max(0, min(x - 10, w - 28)), h - 5), 0.38, DIM)
    return plot


def draw_panel(flow, selected, class_names):
    w = CANVAS_W - LEFT_W
    panel = np.full((CANVAS_H, w, 3), (24, 24, 24), np.uint8)
    if selected is None or not flow.results:
        put(panel, "No labelled objects in this image.", (14, 40), 0.55)
        return panel
    r = flow.results[selected]
    name = class_names[r["cls"]] if r["cls"] < len(class_names) else str(r["cls"])
    col = bc.DRAW_BGR.get(r["color"], (255, 255, 255))
    put(panel, f"Object #{selected}  {name}", (14, 28), 0.65, TEXT, 2)
    cv2.rectangle(panel, (14, 38), (44, 64), col, -1)
    put(panel, f"{r['color'].upper()}", (54, 61), 0.8, TEXT, 2)
    put(panel, f"confidence {r['conf']:.2f}", (220, 56), 0.55)
    if r.get("relaxed"):
        put(panel, "few pixels passed: thresholds were halved", (14, 80), 0.45, (0, 200, 255))

    x1, y1, x2, y2 = r["box"]
    crop, bal_crop = flow.bgr[y1:y2, x1:x2], flow.balanced[y1:y2, x1:x2]
    st = r["stages"] or {}
    votes = np.zeros_like(crop)
    if "final" in st:
        names = list(flow.colors)
        palette = np.array([bc.DRAW_BGR.get(n, (255, 255, 255)) for n in names], np.uint8)
        v = flow.votes[y1:y2, x1:x2]
        votes[st["final"]] = palette[v[st["final"]]]
    count = lambda k: int(st[k].sum()) if k in st else 0
    excl = int(r["exclude"].sum()) if r.get("exclude") is not None else 0
    tiles = [
        mask_tile(crop, None, 140, "a. crop\nraw camera"),
        mask_tile(crop, st.get("label"), 140, f"b. label shape\n{count('label')} px"),
        mask_tile(crop, st.get("eroded"), 140, f"c. edges eroded\n{count('eroded')} px"),
        mask_tile(bal_crop, st.get("no_overlap"), 140, f"d. minus overlaps\n{excl} px removed" if excl
                  else "d. minus overlaps\nnone on top"),
        mask_tile(bal_crop, st.get("filter"), 140, f"e. S/V filter\n{count('filter')} px pass"),
        mask_tile(None, None, 140, f"f. pixel votes\n{r['n']} px vote", colored=votes),
    ]
    y = 96
    for k, t in enumerate(tiles):
        tx, ty = 14 + (k % 3) * 160, y + (k // 3) * 178
        panel[ty:ty + t.shape[0], tx:tx + t.shape[1]] = t
    y += 2 * 178 + 20

    put(panel, "Hue of voting pixels vs the 4 colour bands", (14, y), 0.48)
    plot = hue_histogram(r, flow.colors, w - 28, 130)
    panel[y + 8:y + 8 + plot.shape[0], 14:14 + plot.shape[1]] = plot
    y += plot.shape[0] + 30

    # Vote share + real vs ideal colour for every class.
    total = sum(r["scores"].values()) or 1.0
    sel = st.get("final")
    if sel is not None and sel.any():
        real_raw = tuple(int(c) for c in crop[sel].mean(axis=0))
        real_bal = tuple(int(c) for c in bal_crop[sel].mean(axis=0))
        mean_s = float(flow.hsv[y1:y2, x1:x2][..., 1][sel].mean())
        mean_v = float(flow.hsv[y1:y2, x1:x2][..., 2][sel].mean())
    else:
        real_raw = real_bal = (0, 0, 0)
        mean_s = mean_v = 0.0
    put(panel, "real (raw)", (14, y), 0.42, DIM)
    cv2.rectangle(panel, (14, y + 6), (94, y + 46), real_raw, -1)
    if flow.use_wb:
        put(panel, "real (balanced)", (108, y), 0.42, DIM)
        cv2.rectangle(panel, (108, y + 6), (188, y + 46), real_bal, -1)
    hue_txt = f"{r['hue']:.0f}" if r["hue"] is not None else "-"
    put(panel, f"mean hue {hue_txt}", (210, y + 18), 0.46)
    put(panel, f"mean sat {mean_s:.2f}  bright {mean_v:.2f}", (210, y + 40), 0.46)
    y += 66

    put(panel, "ideal   colour   hue off   vote share", (14, y), 0.42, DIM)
    y += 10
    for n, spec in flow.colors.items():
        share = r["scores"].get(n, 0.0) / total
        off = bc.hue_dist(r["hue"], spec["hue"]) if r["hue"] is not None else float("nan")
        cv2.rectangle(panel, (14, y), (54, y + 26), ideal_bgr(spec["hue"]), -1)
        if n == r["color"]:
            cv2.rectangle(panel, (12, y - 2), (56, y + 28), (255, 255, 255), 2)
        put(panel, n, (64, y + 19), 0.5, TEXT, 2 if n == r["color"] else 1)
        put(panel, f"{off:4.0f} deg", (130, y + 19), 0.46, DIM)
        bar_x, bar_w = 215, w - 215 - 60
        cv2.rectangle(panel, (bar_x, y + 6), (bar_x + bar_w, y + 22), (60, 60, 60), -1)
        cv2.rectangle(panel, (bar_x, y + 6), (bar_x + int(bar_w * share), y + 22),
                      bc.DRAW_BGR.get(n, (200, 200, 200)), -1)
        put(panel, f"{share * 100:3.0f}%", (bar_x + bar_w + 6, y + 20), 0.46)
        y += 34

    put(panel, "n/p or click: object   1-9: step   a/d: image", (14, CANVAS_H - 30), 0.42, DIM)
    put(panel, f"w: white balance ({'ON' if flow.use_wb else 'off'})   s: screenshot   q: quit",
        (14, CANVAS_H - 12), 0.42, DIM)
    return panel


def render(flow, step, selected, class_names):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
    left, geom = draw_left(flow, step, selected)
    canvas[:, :LEFT_W] = left
    canvas[:, LEFT_W:] = draw_panel(flow, selected, class_names)
    cv2.line(canvas, (LEFT_W, 0), (LEFT_W, CANVAS_H), (70, 70, 70), 1)
    return canvas, geom


def pick(flow, step, geom, x, y):
    """Object under a click, smallest first so a block inside a bin wins over the bin."""
    if step == 8:
        for x1, y1, x2, y2, i in getattr(flow, "gallery_rects", []):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None
    if geom is None or x >= LEFT_W or y >= geom[1]:
        return None
    px, py = x / geom[0], y / geom[0]
    hits = []
    for i, r in enumerate(flow.results):
        bx1, by1, bx2, by2 = r["box"]
        if not (bx1 <= px <= bx2 and by1 <= py <= by2):
            continue
        if r["poly"] is not None and cv2.pointPolygonTest(r["poly"].astype(np.float32), (px, py), False) < 0:
            continue
        hits.append(((bx2 - bx1) * (by2 - by1), i))
    return min(hits)[1] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="start at / only show this image")
    ap.add_argument("--images", default=str(HERE / "images"))
    ap.add_argument("--labels", default=str(HERE / "labels"))
    ap.add_argument("--classes-file", default=str(HERE / "classes.txt"))
    ap.add_argument("--classes", type=int, nargs="*", default=bc.DEFAULT_CLASSES)
    ap.add_argument("--colors", default=str(bc.COLORS_FILE))
    ap.add_argument("--s-min", type=float, default=0.35)
    ap.add_argument("--v-min", type=float, default=0.15)
    ap.add_argument("--white-balance", action="store_true", help="start with white balance on (toggle with w)")
    ap.add_argument("--save", default=None, help="write every step of every image to this folder instead")
    ap.add_argument("--object", type=int, default=0, help="object shown in the panel with --save")
    args = ap.parse_args()

    classes = set(args.classes) if args.classes else None
    cf = Path(args.classes_file)
    class_names = [c.strip() for c in cf.read_text().splitlines() if c.strip()] if cf.exists() else []
    colors = bc.load_colors(Path(args.colors))
    files = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in IMG_EXT)
    labels = Path(args.labels)
    i = 0
    if args.image:
        img = Path(args.image)
        if img.resolve() in [f.resolve() for f in files]:
            i = [f.resolve() for f in files].index(img.resolve())
        else:
            files = [img]
    if not files:
        raise SystemExit("no images found")

    use_wb = [args.white_balance]

    def load(k):
        print(f"[{k + 1}/{len(files)}] {files[k].name}" + ("  (white balance on)" if use_wb[0] else ""))
        return ImageFlow(files[k], labels / f"{files[k].stem}.txt", colors, classes, args.s_min, args.v_min,
                         use_wb[0])

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        for k in ([i] if args.image else range(len(files))):
            flow = load(k)
            sel = min(args.object, len(flow.results) - 1) if flow.results else None
            for step in range(len(STEPS)):
                canvas, _ = render(flow, step, sel, class_names)
                cv2.imwrite(str(out / f"{files[k].stem}_step{step + 1}.png"), canvas)
        print(f"saved to {out}")
        return

    win = "colour flow"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        raise SystemExit("This OpenCV has no window support (opencv-python-headless).\n"
                         "Fix: pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python\n"
                         "Or save the steps as images: python color_flow_viewer.py --save flow_out")
    cv2.resizeWindow(win, CANVAS_W, CANVAS_H)
    state = {"flow": load(i), "step": 0, "sel": 0, "geom": None, "dirty": True}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = pick(state["flow"], state["step"], state["geom"], x, y)
            if hit is not None:
                state["sel"], state["dirty"] = hit, True

    cv2.setMouseCallback(win, on_mouse)
    canvas = None
    while True:
        flow = state["flow"]
        if state["dirty"]:
            sel = state["sel"] if flow.results else None
            canvas, state["geom"] = render(flow, state["step"], sel, class_names)
            cv2.imshow(win, canvas)
            state["dirty"] = False
        key = cv2.waitKeyEx(50)
        if key == -1:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        n = len(flow.results)
        if key in (ord("q"), 27):
            break
        elif key in (ord("d"), 65363, 2555904) and i < len(files) - 1:
            i += 1
            state.update(flow=load(i), sel=0)
        elif key in (ord("a"), 65361, 2424832) and i > 0:
            i -= 1
            state.update(flow=load(i), sel=0)
        elif ord("1") <= key <= ord("9"):
            state["step"] = key - ord("1")
        elif key == ord("n") and n:
            state["sel"] = (state["sel"] + 1) % n
        elif key == ord("p") and n:
            state["sel"] = (state["sel"] - 1) % n
        elif key == ord("w"):
            use_wb[0] = not use_wb[0]
            state["flow"] = load(i)
            state["sel"] = min(state["sel"], max(len(state["flow"].results) - 1, 0))
        elif key == ord("s") and canvas is not None:
            name = f"flow_{files[i].stem}_step{state['step'] + 1}_obj{state['sel']}.png"
            cv2.imwrite(name, canvas)
            print(f"saved {name}")
        state["dirty"] = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
