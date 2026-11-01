"""GUI: see how render-and-compare finds each object's top-face centre.

    python point_cloud_matching/silhouette_gui.py                 # newest case
    python point_cloud_matching/silhouette_gui.py <case_dir>

What you see:
  * Close-up (middle): the camera image of the selected object.
      white line   YOLO mask (the target)
      green line   the STL outline as the camera would see it at the current pose
      colours      overlap = green, YOLO only = red, STL only = blue  ->  IoU = green / all coloured
      cyan         the STL top face at that pose      red +  top-face centre
  * Top-down view (right): the table seen from above, in mm. Grey = STL footprint on the table,
    cyan = top face, dashed = the YOLO outline traced back along the camera rays onto the table
    (dotted) and onto the top-face height (dashed); the arrow points at the camera. This is why
    the mask stretches towards the camera while the top face sits at the far end.
  * IoU plot (bottom right): IoU at every accepted step of the search.

Controls:
  click an object in the overview, or  < prev / next >     pick an object
  model candidates     pick size / pose (IoU of each fit shown)
  animate search       start the STL 10 mm off and rotated 25 deg, and watch the search slide it in
  sliders              move the STL by hand and watch the IoU change; "back to fit" resets
  keys: n / p next / previous, a animate, f re-fit
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider  # noqa: E402

import cases  # noqa: E402
import models  # noqa: E402
from matching import rot_z  # noqa: E402
from silhouette import SS, Camera, SilhouetteMatcher  # noqa: E402
from top_face import estimate_intrinsics  # noqa: E402

TOGGLES = ["overlap colours", "top face", "start point"]


class SilhouetteGUI:
    def __init__(self, case):
        self.case = case
        self.objects = case.objects
        self.cam = Camera(case.T, estimate_intrinsics(case.path))
        self.cand_cache = {}
        self.show = dict(zip(TOGGLES, [True, True, False]))
        self.idx = 0
        self.anim = None          # (trace, step index) while animating
        self._quiet = False
        self._build()
        self.select(0)

    # ------------------------------------------------------------------ layout
    def _build(self):
        self.fig = plt.figure(figsize=(17, 9.5))
        self.fig.canvas.manager.set_window_title(f"Render-and-compare - {self.case.case_id}")
        self.ax_over = self.fig.add_axes([0.01, 0.63, 0.30, 0.35])
        self.ax_info = self.fig.add_axes([0.01, 0.27, 0.30, 0.34])
        self.ax_info.axis("off")
        self.ax_main = self.fig.add_axes([0.32, 0.20, 0.40, 0.78])
        self.ax_top = self.fig.add_axes([0.75, 0.52, 0.23, 0.45])
        self.ax_iou = self.fig.add_axes([0.77, 0.22, 0.21, 0.22])

        x0, y0, w, h = self.case.roi
        self.ax_over.imshow(self.case.image[y0:y0 + h, x0:x0 + w, ::-1], extent=(x0, x0 + w, y0 + h, y0))
        self.ax_over.set_title("click an object", fontsize=9)
        self.ax_over.set_xticks([])
        self.ax_over.set_yticks([])
        self.poly_lines = []
        for i, o in enumerate(self.objects):
            p = np.vstack((o.polygon_full, o.polygon_full[:1]))
            (ln,) = self.ax_over.plot(p[:, 0], p[:, 1], "w-", lw=1)
            self.ax_over.text(*o.polygon_full.mean(axis=0), str(i + 1), color="w", fontsize=8,
                              ha="center", va="center", fontweight="bold")
            self.poly_lines.append(ln)

        def button(rect, label, cb):
            b = Button(self.fig.add_axes(rect), label)
            b.label.set_fontsize(8)
            b.on_clicked(cb)
            return b

        self.b_prev = button([0.01, 0.215, 0.06, 0.035], "< prev", lambda e: self.select(self.idx - 1))
        self.b_next = button([0.075, 0.215, 0.06, 0.035], "next >", lambda e: self.select(self.idx + 1))
        self.b_anim = button([0.14, 0.215, 0.085, 0.035], "animate search", lambda e: self.animate())
        self.b_fit = button([0.23, 0.215, 0.08, 0.035], "re-fit", lambda e: self.refit())

        self.ax_cand = self.fig.add_axes([0.01, 0.01, 0.2, 0.19])
        self.r_cand = None
        ax = self.fig.add_axes([0.215, 0.07, 0.095, 0.13])
        self.c_show = CheckButtons(ax, TOGGLES, [self.show[k] for k in TOGGLES])
        for t in self.c_show.labels:
            t.set_fontsize(8)
        self.c_show.on_clicked(self._on_toggle)

        def slider(rect, label, lo, hi):
            s = Slider(self.fig.add_axes(rect), label, lo, hi, valinit=0.0)
            s.label.set_fontsize(8)
            s.on_changed(self._on_slider)
            return s

        self.s_x = slider([0.38, 0.12, 0.3, 0.025], "move X mm", -15, 15)
        self.s_y = slider([0.38, 0.08, 0.3, 0.025], "move Y mm", -15, 15)
        self.s_yaw = slider([0.38, 0.04, 0.3, 0.025], "rotate deg", -45, 45)
        self.b_zero = button([0.70, 0.04, 0.07, 0.035], "back to fit", lambda e: self._zero(True))

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------------------------------------------ fitting
    def candidates(self, cls):
        if cls not in self.cand_cache:
            self.cand_cache[cls] = models.candidates_for_class(cls)
        return self.cand_cache[cls]

    def select(self, i):
        self.idx = i % len(self.objects)
        for k, ln in enumerate(self.poly_lines):
            ln.set_color("red" if k == self.idx else "w")
            ln.set_linewidth(2.2 if k == self.idx else 1)
        self.refit()

    def refit(self):
        self._stop_anim()
        o = self.objects[self.idx]
        xyz, _, _ = o.cleaned()
        pts = xyz if len(xyz) else o.xyz_raw
        self.tz = o.table_z_at(*np.median(pts[:, :2], axis=0))
        self.matcher = SilhouetteMatcher(self.cam, o)
        self.fits, self.traces = [], []
        for c in self.candidates(o.class_name):
            tr = []
            self.fits.append(self.matcher.fit(c, self.tz, trace=tr))
            self.traces.append(tr)
        order = np.argsort([-f.iou for f in self.fits])
        self.fits = [self.fits[k] for k in order]
        self.traces = [self.traces[k] for k in order]
        self.sel = 0
        self.anim_trace = None
        self._rebuild_cands()
        self._zero(False)
        self.draw()

    def pose(self):
        """(candidate, x, y, yaw, iou) currently shown: animation step, or fit + slider nudges."""
        f = self.fits[self.sel]
        if self.anim is not None:
            x, y, yaw, s = self.anim[0][self.anim[1]]
            return f.candidate, x, y, yaw, s
        x = f.x + self.s_x.val / 1000
        y = f.y + self.s_y.val / 1000
        yaw = f.yaw + np.radians(self.s_yaw.val)
        s = f.iou if (self.s_x.val, self.s_y.val, self.s_yaw.val) == (0, 0, 0) else \
            self.matcher.iou(f.candidate, x, y, yaw, self.tz)
        return f.candidate, x, y, yaw, s

    # ------------------------------------------------------------------ callbacks
    def _rebuild_cands(self):
        self._quiet = True
        self.ax_cand.clear()
        self.ax_cand.set_title("model candidates (best first)", fontsize=8, loc="left")
        labels = [f"{f.candidate.label}   IoU {f.iou:.3f}" for f in self.fits] or ["(no STL for this class)"]
        self.r_cand = RadioButtons(self.ax_cand, labels, active=0)
        for t in self.r_cand.labels:
            t.set_fontsize(8)
        self.r_cand.on_clicked(self._on_cand)
        self._quiet = False

    def _on_cand(self, label):
        if self._quiet:
            return
        self._stop_anim()
        self.sel = [t.get_text() for t in self.r_cand.labels].index(label)
        self.anim_trace = None
        self._zero(False)
        self.draw()

    def _on_toggle(self, label):
        self.show[label] = not self.show[label]
        self.draw()

    def _on_slider(self, _):
        if not self._quiet:
            self._stop_anim()
            self.draw()

    def _zero(self, redraw):
        self._quiet = True
        for s in (self.s_x, self.s_y, self.s_yaw):
            s.set_val(0.0)
        self._quiet = False
        if redraw:
            self.draw()

    def _on_click(self, e):
        if e.inaxes is self.ax_over and e.xdata is not None:
            for i, o in enumerate(self.objects):
                if MplPath(o.polygon_full).contains_point((e.xdata, e.ydata)):
                    self.select(i)
                    return

    def _on_key(self, e):
        acts = {"n": lambda: self.select(self.idx + 1), "p": lambda: self.select(self.idx - 1),
                "a": self.animate, "f": self.refit}
        if e.key in acts:
            acts[e.key]()

    # ------------------------------------------------------------------ animation
    def animate(self):
        self._stop_anim()
        self._zero(False)
        # Re-run the search from a deliberately wrong start so the convergence is visible
        f = self.fits[self.sel]
        start = np.array([f.x + 0.010, f.y - 0.008, f.yaw + (np.radians(25) if f.candidate.sym_order else 0)])
        trace = []
        self.matcher._search(f.candidate, self.tz, start, self.matcher.STEPS, trace)
        self.anim_trace = trace
        self.anim = [trace, 0]
        self.timer = self.fig.canvas.new_timer(interval=max(25, int(4000 / len(trace))))
        self.timer.add_callback(self._anim_step)
        self.timer.start()
        self.draw()

    def _anim_step(self):
        if self.anim is None:
            return
        self.anim[1] += 1
        if self.anim[1] >= len(self.anim[0]):
            self._stop_anim()
        self.draw()

    def _stop_anim(self):
        if getattr(self, "timer", None) is not None:
            self.timer.stop()
            self.timer = None
        self.anim = None

    # ------------------------------------------------------------------ drawing
    def draw(self):
        o = self.objects[self.idx]
        m = self.matcher
        if not self.fits:
            return
        cand, x, y, yaw, iou = self.pose()
        H, W = m.shape
        ox, oy = m.origin
        ext = (ox, ox + W / SS, oy + H / SS, oy)

        # ---- close-up
        ax = self.ax_main
        ax.clear()
        img = self.case.image[max(oy, 0):oy + H // SS, max(ox, 0):ox + W // SS, ::-1]
        ax.imshow(img, extent=(max(ox, 0), max(ox, 0) + img.shape[1], max(oy, 0) + img.shape[0], max(oy, 0)))
        model = m.render(cand, x, y, yaw, self.tz).astype(bool)
        target = m.target.astype(bool)
        if self.show["overlap colours"]:
            rgba = np.zeros((H, W, 4))
            rgba[model & target] = (0, 1, 0, 0.35)
            rgba[target & ~model] = (1, 0, 0, 0.55)
            rgba[model & ~target] = (0, 0.3, 1, 0.55)
            ax.imshow(rgba, extent=ext, interpolation="nearest")
        p = np.vstack((o.polygon_full, o.polygon_full[:1]))
        ax.plot(p[:, 0], p[:, 1], "w-", lw=1.5, label="YOLO mask")
        cnts, _ = cv2.findContours(model.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for k, c in enumerate(cnts):
            c = np.vstack((c.reshape(-1, 2), c.reshape(-1, 2)[:1])) / SS + m.origin
            ax.plot(c[:, 0], c[:, 1], "-", color="lime", lw=1.5, label="STL as camera sees it" if k == 0 else None)
        R, t = rot_z(yaw), np.array([x, y, self.tz])
        top_c = cand.top_center @ R.T + t
        if self.show["top face"]:
            for k, e in enumerate(cand.top_edges):
                q = self.cam.project(e @ R.T + t)
                ax.plot(q[:, 0], q[:, 1], "-", color="cyan", lw=1.5, label="STL top face" if k == 0 else None)
        u, v = self.cam.project(top_c)[0]
        ax.plot(u, v, "r+", ms=18, mew=2.5, label="top-face centre")
        if self.show["start point"]:
            sx, sy = m.start_xy(cand, self.tz)
            su, sv = self.cam.project([sx, sy, self.tz + cand.height / 2])[0]
            ax.plot(su, sv, "o", color="orange", ms=8, label="search start")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
        step = "" if self.anim is None else f"   search step {self.anim[1] + 1}/{len(self.anim[0])}"
        ax.set_title(f"{self.idx + 1}. {o.object_id}  YOLO: {o.class_name}   |   {cand.label}   IoU {iou:.3f}{step}",
                     fontsize=10)

        # ---- top-down view
        ax = self.ax_top
        ax.clear()
        mm = 1000
        verts = cand._hull_verts @ R.T + t
        hull = cv2.convexHull((verts[:, :2] * mm).astype(np.float32)).reshape(-1, 2)
        ax.fill(hull[:, 0], hull[:, 1], color="0.75", label="STL footprint")
        for k, e in enumerate(cand.top_edges):
            q = (e @ R.T + t)[:, :2] * mm
            ax.plot(q[:, 0], q[:, 1], "-", color="c", lw=2, label="top face" if k == 0 else None)
        for z, style, lab in ((self.tz, ":", "mask traced to table"), (self.tz + cand.height, "--",
                                                                          "mask traced to top height")):
            q = np.array([self.cam.ray_to_plane(a, b, z)[:2] for a, b in o.polygon_full]) * mm
            q = np.vstack((q, q[:1]))
            ax.plot(q[:, 0], q[:, 1], style, color="k", lw=1, label=lab)
        ax.plot(top_c[0] * mm, top_c[1] * mm, "r+", ms=14, mew=2.5)
        cdir = self.cam.pos[:2] - top_c[:2]
        cdir = cdir / np.linalg.norm(cdir) * 0.6 * np.ptp(hull, axis=0).max()
        head = (top_c[0] * mm + cdir[0], top_c[1] * mm + cdir[1])
        ax.annotate("", xy=head, xytext=(top_c[0] * mm, top_c[1] * mm),
                    arrowprops=dict(arrowstyle="->", color="purple"))
        ax.text(*head, " to camera", fontsize=8, color="purple", ha="center", va="bottom")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("table from above (mm)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.8)

        # ---- IoU trace
        ax = self.ax_iou
        ax.clear()
        tr = self.anim[0] if self.anim is not None else getattr(self, "anim_trace", None) or self.traces[self.sel]
        if tr:
            ax.plot([s[3] for s in tr], "-", color="0.4")
            k = self.anim[1] if self.anim is not None else len(tr) - 1
            ax.plot(k, tr[k][3], "o", color="red")
        ax.set_xlabel("accepted search step", fontsize=8)
        ax.set_ylabel("IoU", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

        # ---- info
        tc = top_c * mm
        lines = [
            "HOW IT WORKS",
            "1. YOLO gives the object's outline (white).",
            "2. Place an STL (each size / pose) on the",
            "   table at a guessed X, Y and rotation.",
            "3. Project it through the calibrated camera",
            "   -> the outline it WOULD make (green).",
            "4. Score overlap: IoU = green / (green+red+blue)",
            "5. Nudge X, Y, rotation while IoU improves,",
            "   then smaller steps. Best size/pose wins.",
            "6. Top-face centre comes from the STL,",
            "   straight above the object's centre.",
            "",
            f"model     {cand.label}",
            f"IoU       {iou:.3f}",
            f"rotation  {np.degrees(yaw):+.1f} deg",
            f"table Z   {self.tz * mm:+.1f} mm (from table depth)",
            "TOP-FACE CENTRE (table, mm)",
            f"  X {tc[0]:.1f}   Y {tc[1]:.1f}   Z {tc[2]:.1f}",
        ]
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", type=Path)
    ap.add_argument("--screenshot", type=Path, help="save a PNG of the window and exit (for testing)")
    args = ap.parse_args()
    if args.screenshot:
        matplotlib.use("Agg")
    path = args.case or cases.find_cases()[-1]
    print(f"Loading {path}")
    gui = SilhouetteGUI(cases.load_case(path))
    if args.screenshot:
        gui.show["start point"] = True
        gui.select(6)
        gui.fig.savefig(args.screenshot, dpi=85)
        print(f"wrote {args.screenshot}")
        return
    plt.show()


if __name__ == "__main__":
    main()
