"""GUI: compare STL model point clouds with the YOLO-segmented ZED point cloud.

Loads a case saved by yolo_dataset/capture_pointcloud_case.py, and for the selected object
fits every model candidate of its YOLO class (small / large x each resting pose) and shows
the data and the fitted model in 3D, in the calibrated table frame (millimetres).

Usage:
    python point_cloud_matching/match_gui.py                    # newest case
    python point_cloud_matching/match_gui.py point_cloud_mathing/case_20261101T165958_770293Z
    python point_cloud_matching/match_gui.py --list

Mouse / keys:
    click an object in the image     select it          n / p   next / previous object
    f   re-fit                       r   reset 3D view  s       save results JSON into the case
    drag in 3D view = rotate, right-drag / scroll = zoom

3D view legend:
    data points        coloured from the camera image (or by distance to the model)
    orange points      model faces the camera can see (what the fit is compared against)
    grey points        hidden model faces           faint mesh  the fitted model
    red star           fitted model centre          green dot   model top-face centre
    blue cross         centre saved by the capture script (mean of the mask points)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import cases  # noqa: E402
import matching  # noqa: E402
import models  # noqa: E402

CHECK_LABELS = ["data", "visible model", "hidden model", "mesh", "colour by distance", "lock base to table"]


class MatchGUI:
    def __init__(self, case, args):
        self.case = case
        self.args = args
        self.objects = case.objects
        if not self.objects:
            raise SystemExit("No shape objects in this case")
        self.idx = 0
        self.mode = "3d"
        self.checks = dict(zip(CHECK_LABELS, [True, True, True, True, False, True]))
        self.cand_cache = {}
        self.results = {}       # object_id -> chosen FitResult (for saving)
        self.fits = []
        self.sel = 0
        self._updating = False
        self._build()
        self.select(0)

    # ---------------------------------------------------------------- layout
    def _build(self):
        self.fig = plt.figure(figsize=(17, 9.5))
        self.fig.canvas.manager.set_window_title(f"Point cloud matching - {self.case.case_id}")
        self.ax_img = self.fig.add_axes([0.005, 0.42, 0.36, 0.56])
        self.ax3d = self.fig.add_axes([0.36, 0.05, 0.45, 0.93], projection="3d")
        self.ax_info = self.fig.add_axes([0.815, 0.36, 0.18, 0.62])
        self.ax_info.axis("off")
        self.ax_cbar = self.fig.add_axes([0.375, 0.12, 0.008, 0.3])
        self.ax_cbar.set_visible(False)

        x0, y0, w, h = self.case.roi
        self.ax_img.imshow(self.case.image[y0:y0 + h, x0:x0 + w, ::-1], extent=(x0, x0 + w, y0 + h, y0))
        self.ax_img.set_title("click an object to select it", fontsize=9)
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.poly_artists = []
        for i, o in enumerate(self.objects):
            p = np.vstack((o.polygon_full, o.polygon_full[:1]))
            (ln,) = self.ax_img.plot(p[:, 0], p[:, 1], "-", lw=1.2, color="w")
            c = o.polygon_full.mean(axis=0)
            self.ax_img.text(c[0], o.polygon_full[:, 1].min() - 4, f"{i + 1}", color="w", fontsize=8,
                             ha="center", va="bottom", fontweight="bold")
            self.poly_artists.append(ln)

        def button(rect, label, cb):
            b = Button(self.fig.add_axes(rect), label)
            b.on_clicked(cb)
            return b

        self.b_prev = button([0.01, 0.365, 0.06, 0.04], "< prev", lambda e: self.select(self.idx - 1))
        self.b_next = button([0.075, 0.365, 0.06, 0.04], "next >", lambda e: self.select(self.idx + 1))
        self.b_fit = button([0.14, 0.365, 0.07, 0.04], "re-fit", lambda e: self.refit())
        self.b_view = button([0.215, 0.365, 0.07, 0.04], "reset view", lambda e: self.draw(reset_view=True))
        self.b_save = button([0.29, 0.365, 0.07, 0.04], "save JSON", lambda e: self.save())

        self.ax_mode = self.fig.add_axes([0.01, 0.25, 0.12, 0.085])
        self.ax_mode.set_title("fit mode", fontsize=8, loc="left")
        self.r_mode = RadioButtons(self.ax_mode, ["3D (xyz)", "top-down (xy)"], active=0)
        self.r_mode.on_clicked(self._on_mode)

        self.ax_chk = self.fig.add_axes([0.01, 0.03, 0.12, 0.2])
        self.c_chk = CheckButtons(self.ax_chk, CHECK_LABELS, [self.checks[k] for k in CHECK_LABELS])
        for t in self.c_chk.labels:
            t.set_fontsize(8)
        self.c_chk.on_clicked(self._on_check)

        self.ax_cand = self.fig.add_axes([0.14, 0.03, 0.22, 0.305])
        self.r_cand = None

        def slider(y, label, lo, hi):
            s = Slider(self.fig.add_axes([0.85, y, 0.12, 0.025]), label, lo, hi, valinit=0.0)
            s.on_changed(self._on_slider)
            return s

        self.fig.text(0.82, 0.325, "manual nudge (relative to fit)", fontsize=8)
        self.s_yaw = slider(0.29, "yaw deg", -45, 45)
        self.s_x = slider(0.25, "dx mm", -15, 15)
        self.s_y = slider(0.21, "dy mm", -15, 15)
        self.s_z = slider(0.17, "dz mm", -15, 15)
        self.b_zero = button([0.85, 0.12, 0.12, 0.035], "zero sliders", lambda e: self._zero_sliders(True))

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _rebuild_cand_radio(self):
        self.ax_cand.clear()
        labels = [f"{f.candidate.label}\n   rms {f.rms_mm:4.1f} mm  cover {f.coverage * 100:3.0f}%"
                  for f in self.fits] or ["(no model for this class)"]
        self.ax_cand.set_title("model candidates (best first)", fontsize=8, loc="left")
        self.r_cand = RadioButtons(self.ax_cand, labels, active=0)
        for t in self.r_cand.labels:
            t.set_fontsize(7.5)
        self.r_cand.on_clicked(self._on_cand)

    # ---------------------------------------------------------------- state
    def current(self):
        return self.objects[self.idx]

    def candidates(self, class_name):
        if class_name not in self.cand_cache:
            self.cand_cache[class_name] = models.candidates_for_class(class_name, spacing_mm=self.args.spacing)
        return self.cand_cache[class_name]

    def select(self, i):
        self.idx = i % len(self.objects)
        for k, ln in enumerate(self.poly_artists):
            ln.set_color("red" if k == self.idx else "w")
            ln.set_linewidth(2.5 if k == self.idx else 1.2)
        self.refit(reset_view=True)

    def refit(self, reset_view=False):
        o = self.current()
        self.xyz, self.rgb, self.h = o.cleaned(min_height=self.args.min_height / 1000)
        if len(self.xyz) < 20:
            self.xyz, self.rgb, self.h = o.xyz_raw, o.rgb_raw, o.xyz_raw[:, 2]
        med = np.median(self.xyz[:, :2], axis=0)
        self.table_z = o.table_z_at(*med)
        self.matcher = matching.Matcher(self.xyz, self.case.camera_pos, self.table_z, mode=self.mode,
                                        lock_z=self.checks["lock base to table"])
        cands = self.candidates(o.class_name)
        self.fig.canvas.manager.set_window_title(f"fitting {o.object_id} ...")
        self.fits = self.matcher.fit_all(cands) if cands else []
        self.fig.canvas.manager.set_window_title(f"Point cloud matching - {self.case.case_id}")
        self.sel = 0
        self._rebuild_cand_radio()
        self._zero_sliders(False)
        self.draw(reset_view=reset_view)

    def adjusted(self):
        """Selected fit with the slider nudges applied (re-scored)."""
        if not self.fits:
            return None
        f = self.fits[self.sel]
        d = np.array([self.s_x.val, self.s_y.val, self.s_z.val]) / 1000
        if not np.any(d) and self.s_yaw.val == 0:
            return f
        pose = matching.Pose(f.pose.yaw + np.radians(self.s_yaw.val), f.pose.t + d)
        return self.matcher.evaluate(f.candidate, pose)

    # ---------------------------------------------------------------- callbacks
    def _on_mode(self, label):
        self.mode = "3d" if label.startswith("3D") else "2d"
        self.refit()

    def _on_check(self, label):
        self.checks[label] = not self.checks[label]
        if label == "lock base to table":
            self.refit()
        else:
            self.draw()

    def _on_cand(self, label):
        self.sel = self.r_cand.labels.index(next(t for t in self.r_cand.labels if t.get_text() == label))
        self._zero_sliders(False)
        self.draw()

    def _on_slider(self, _):
        if not self._updating:
            self.draw()

    def _zero_sliders(self, redraw):
        self._updating = True
        for s in (self.s_yaw, self.s_x, self.s_y, self.s_z):
            s.set_val(0.0)
        self._updating = False
        if redraw:
            self.draw()

    def _on_click(self, event):
        if event.inaxes is not self.ax_img or event.xdata is None:
            return
        for i, o in enumerate(self.objects):
            if MplPath(o.polygon_full).contains_point((event.xdata, event.ydata)):
                self.select(i)
                return

    def _on_key(self, event):
        actions = {"n": lambda: self.select(self.idx + 1), "p": lambda: self.select(self.idx - 1),
                   "f": self.refit, "r": lambda: self.draw(reset_view=True), "s": self.save}
        if event.key in actions:
            actions[event.key]()

    # ---------------------------------------------------------------- drawing
    def draw(self, reset_view=False):
        ax = self.ax3d
        elev, azim = (ax.elev, ax.azim) if not reset_view and hasattr(self, "_drawn") else (35, -60)
        self._drawn = True
        ax.clear()
        o = self.current()
        fit = self.adjusted()
        mm = 1000.0

        colour_by_distance = self.checks["data"] and self.checks["colour by distance"] and fit is not None
        self.ax_cbar.set_visible(colour_by_distance)
        if self.checks["data"]:
            data = self.xyz
            if colour_by_distance:
                r = self.matcher.residuals(fit.candidate, fit.pose) if len(data) == len(self.matcher.data) else None
                if r is None:  # matcher subsampled the data; colour that subset
                    data, r = self.matcher.data, self.matcher.residuals(fit.candidate, fit.pose)
                sc = ax.scatter(*(data * mm).T, c=np.clip(r * mm, 0, 10), cmap="turbo", s=3, vmin=0, vmax=10,
                                depthshade=False)
                self.ax_cbar.clear()
                self.fig.colorbar(sc, cax=self.ax_cbar, label="distance to model (mm)")
            else:
                ax.scatter(*(data * mm).T, c=self.rgb, s=3, depthshade=False, label="ZED data")

        if fit is not None:
            c = fit.candidate
            R = matching.rot_z(fit.pose.yaw)
            pw = c.points @ R.T + fit.pose.t
            vis = matching.visible_mask(pw, c.normals @ R.T, self.case.camera_pos)
            if self.checks["visible model"]:
                ax.scatter(*(pw[vis] * mm).T, c="#ff8c1a", s=1.5, depthshade=False, label="model (visible)")
            if self.checks["hidden model"]:
                ax.scatter(*(pw[~vis] * mm).T, c="0.6", s=0.6, alpha=0.35, depthshade=False,
                           label="model (hidden)")
            if self.checks["mesh"]:
                tw = (c.tris @ R.T + fit.pose.t) * mm
                ax.add_collection3d(Poly3DCollection(tw, facecolor=(1, 0.55, 0.1, 0.08),
                                                     edgecolor=(0.6, 0.3, 0, 0.35), linewidths=0.4))
            ctr, top = fit.center * mm, fit.top_center * mm
            ax.scatter(*ctr, marker="*", s=250, c="red", edgecolor="k", depthshade=False, label="model centre")
            ax.scatter(*top, marker="o", s=60, c="lime", edgecolor="k", depthshade=False, label="top-face centre")
            ax.plot([ctr[0], top[0]], [ctr[1], top[1]], [ctr[2], top[2]], "r--", lw=0.8)
        if o.yolo_center is not None:
            ax.scatter(*(o.yolo_center * mm), marker="X", s=90, c="royalblue", edgecolor="k",
                       depthshade=False, label="capture-script centre")

        # Local table patch
        centre = (fit.center if fit is not None else np.median(self.xyz, axis=0)) * mm
        half = max(35.0, (np.ptp(self.xyz[:, :2], axis=0).max() * mm) * 0.8)
        gx, gy = np.meshgrid(np.linspace(centre[0] - half, centre[0] + half, 2),
                             np.linspace(centre[1] - half, centre[1] + half, 2))
        gz = np.array([[o.table_z_at(x / mm, y / mm) * mm for x, y in zip(rx, ry)] for rx, ry in zip(gx, gy)])
        ax.plot_surface(gx, gy, gz, color="0.8", alpha=0.25, linewidth=0)

        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        zmid = self.table_z * mm + half * 0.6
        ax.set_zlim(zmid - half, zmid + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X table (mm)")
        ax.set_ylabel("Y table (mm)")
        ax.set_zlabel("Z table (mm)")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper right", fontsize=7, markerscale=2)
        ax.set_title(f"{self.idx + 1}/{len(self.objects)}  {o.object_id}  YOLO: {o.class_name} "
                     f"({o.confidence:.2f})", fontsize=10)
        self._draw_info(o, fit)
        self.fig.canvas.draw_idle()

    def _draw_info(self, o, fit):
        mm = 1000.0
        h95 = np.percentile(self.h, 95) * mm if len(self.h) else float("nan")
        lines = [
            f"object      {o.object_id}",
            f"YOLO class  {o.class_name}  ({o.confidence:.2f})",
            f"points      {len(self.xyz)} used / {len(o.xyz_raw)} in mask",
            f"local table Z {self.table_z * mm:+.1f} mm",
            f"measured height (p95) {h95:.1f} mm",
            f"mode        {'3D' if self.mode == '3d' else 'top-down XY'}"
            f"{'' if self.checks['lock base to table'] or self.mode == '2d' else ', free Z'}",
            "",
        ]
        if fit is None:
            lines.append("no STL model for this class")
        else:
            c = fit.candidate
            ctr, top = fit.center * mm, fit.top_center * mm
            yaw = np.degrees(fit.pose.yaw) % 360
            lines += [
                f"model       {c.label}",
                f"model height {c.height * mm:.1f} mm",
                f"rms         {fit.rms_mm:.2f} mm",
                f"coverage    {fit.coverage * 100:.0f}% of visible model",
                f"data fit    {fit.data_inliers * 100:.0f}% within 3 mm",
                f"yaw         {yaw:.1f} deg",
                "",
                "MODEL CENTRE (table, mm)",
                f"  X {ctr[0]:8.1f}  Y {ctr[1]:8.1f}  Z {ctr[2]:7.1f}",
                "TOP-FACE CENTRE (table, mm)",
                f"  X {top[0]:8.1f}  Y {top[1]:8.1f}  Z {top[2]:7.1f}",
            ]
            if o.yolo_center is not None:
                d = ctr - o.yolo_center * mm
                lines += ["capture-script centre (mm)",
                          "  X {:8.1f}  Y {:8.1f}  Z {:7.1f}".format(*(o.yolo_center * mm)),
                          f"  model - capture: XY {np.linalg.norm(d[:2]):.1f}  Z {d[2]:+.1f}"]
            if self.mode == "3d" and h95 < 0.6 * c.height * mm:
                lines += ["", "! measured height is well below", "  the model: depth looks flattened",
                          "  (dark/low-texture object?) - try", "  top-down mode"]
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)

    # ---------------------------------------------------------------- save
    def save(self):
        fit = self.adjusted()
        if fit is not None:
            self.results[self.current().object_id] = (fit, self.mode)
        out = []
        for oid, (f, mode) in self.results.items():
            out.append({
                "object_id": oid,
                "model": f.candidate.label, "shape": f.candidate.shape, "size": f.candidate.size,
                "pose": f.candidate.pose_name, "mode": mode,
                "center_table_xyz_m": f.center.tolist(),
                "top_center_table_xyz_m": f.top_center.tolist(),
                "yaw_deg": float(np.degrees(f.pose.yaw) % 360),
                "rms_mm": f.rms_mm, "coverage": f.coverage,
            })
        path = self.case.path / "matching_results.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"saved {len(out)} object(s) to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", type=Path, help="case folder (default: newest)")
    ap.add_argument("--list", action="store_true", help="list available cases and exit")
    ap.add_argument("--min-height", type=float, default=3.0,
                    help="drop points less than this many mm above the local table (table bleed)")
    ap.add_argument("--spacing", type=float, default=1.5, help="model point spacing in mm")
    ap.add_argument("--screenshot", type=Path, help="render the first object to this PNG and exit (no window)")
    args = ap.parse_args()

    found = cases.find_cases()
    if args.list:
        for c in found:
            print(c)
        return
    path = args.case or (found[-1] if found else None)
    if path is None:
        raise SystemExit("No cases found; capture one with yolo_dataset/capture_pointcloud_case.py")
    if args.screenshot:
        matplotlib.use("Agg")
    print(f"Loading {path}")
    gui = MatchGUI(cases.load_case(path), args)
    if args.screenshot:
        gui.fig.savefig(args.screenshot, dpi=90)
        print(f"wrote {args.screenshot}")
        return
    plt.show()


if __name__ == "__main__":
    main()
