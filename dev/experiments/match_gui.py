"""GUI: compare STL model point clouds with the YOLO-segmented ZED point cloud (ICP).

Loads a case saved by yolo_dataset/capture_pointcloud_case.py. For the selected object every
model candidate of its YOLO class (small / large x each resting pose) is fitted to the data
with ICP, and the data and fitted models are shown in 3D in the calibrated table frame (mm).

Usage:
    python dev/experiments/match_gui.py                    # newest case
    python dev/experiments/match_gui.py point_cloud_mathing/case_20261101T165958_770293Z
    python dev/experiments/match_gui.py --list

Panels:
    model candidates   tick a model to show it, untick to hide it (each has its own colour);
                       the round button on the right picks the ACTIVE model, which is the one
                       the info text, the nudge sliders, distance colouring and "save" use
    ICP                4-DoF (flat on the table: X, Y, yaw [+Z]) or 6-DoF (free rigid ICP)
    distances          3D, or top-down XY (for objects whose depth is unreliable; 4-DoF only)

Mouse / keys:
    click an object in the image     select it          n / p   next / previous object
    1-9 show/hide model N            a   show all       b       best only
    f   re-fit                       r   reset 3D view  s       save results JSON into the case
    drag in 3D view = rotate, right-drag / scroll = zoom

3D view: data points use the camera colours (or distance to the active model). For each
shown model: solid points = faces the camera can see, faint points = hidden faces, star =
model centre, ring = top-face centre. Blue X = centre saved by the capture script.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from perception import cases, matching, object_models as models  # noqa: E402

VIEW_LABELS = ["data", "visible faces", "hidden faces", "mesh", "colour by distance", "lock base to table"]
MODEL_COLOURS = ["#ff7f0e", "#e040fb", "#00bcd4", "#7cb342", "#8d6e63", "#ffd600"]
ICP_LABELS = ["4-DoF ICP (flat on table)", "6-DoF ICP (free)"]
DIST_LABELS = ["3D (xyz)", "top-down (xy)"]


class MatchGUI:
    def __init__(self, case, args):
        self.case = case
        self.args = args
        self.objects = case.objects
        if not self.objects:
            raise SystemExit("No shape objects in this case")
        self.idx = 0
        self.mode = "3d"
        self.icp = "4dof"
        self.view = dict(zip(VIEW_LABELS, [True, True, True, True, False, True]))
        self.cand_cache = {}
        self.results = {}       # object_id -> (FitResult, mode, icp) for saving
        self.fits = []
        self.shown = []         # per fit: drawn or not
        self.active = 0
        self._quiet = False     # suppress widget callbacks while rebuilding
        self._build()
        self.select(0)

    # ---------------------------------------------------------------- layout
    def _build(self):
        self.fig = plt.figure(figsize=(17, 9.5))
        self.fig.canvas.manager.set_window_title(f"Point cloud matching - {self.case.case_id}")
        self.ax_img = self.fig.add_axes([0.005, 0.47, 0.36, 0.51])
        self.ax3d = self.fig.add_axes([0.36, 0.04, 0.45, 0.94], projection="3d")
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
            b.label.set_fontsize(8)
            b.on_clicked(cb)
            return b

        row = 0.415
        self.b_prev = button([0.01, row, 0.06, 0.035], "< prev", lambda e: self.select(self.idx - 1))
        self.b_next = button([0.075, row, 0.06, 0.035], "next >", lambda e: self.select(self.idx + 1))
        self.b_fit = button([0.14, row, 0.07, 0.035], "re-fit", lambda e: self.refit())
        self.b_view = button([0.215, row, 0.07, 0.035], "reset view", lambda e: self.draw(reset_view=True))
        self.b_save = button([0.29, row, 0.07, 0.035], "save JSON", lambda e: self.save())

        def radio(rect, title, labels):
            ax = self.fig.add_axes(rect)
            ax.set_title(title, fontsize=8, loc="left")
            r = RadioButtons(ax, labels, active=0)
            for t in r.labels:
                t.set_fontsize(8)
            return r

        self.r_icp = radio([0.01, 0.315, 0.125, 0.07], "ICP", ICP_LABELS)
        self.r_icp.on_clicked(self._on_icp)
        self.r_dist = radio([0.01, 0.215, 0.125, 0.07], "distances", DIST_LABELS)
        self.r_dist.on_clicked(self._on_dist)

        ax = self.fig.add_axes([0.01, 0.02, 0.125, 0.17])
        ax.set_title("view", fontsize=8, loc="left")
        self.c_view = CheckButtons(ax, VIEW_LABELS, [self.view[k] for k in VIEW_LABELS])
        for t in self.c_view.labels:
            t.set_fontsize(8)
        self.c_view.on_clicked(self._on_view)

        # Candidate list: show/hide check boxes + "active" radio column, rebuilt per object
        self.ax_show = self.fig.add_axes([0.14, 0.07, 0.185, 0.315])
        self.ax_active = self.fig.add_axes([0.327, 0.07, 0.033, 0.315])
        self.c_show = self.r_active = None
        self.b_all = button([0.14, 0.02, 0.06, 0.035], "show all", lambda e: self.show_only(None))
        self.b_best = button([0.203, 0.02, 0.06, 0.035], "best only", lambda e: self.show_only(0))
        self.b_none = button([0.266, 0.02, 0.06, 0.035], "hide all", lambda e: self.show_only(-1))

        def slider(y, label, lo, hi):
            s = Slider(self.fig.add_axes([0.85, y, 0.12, 0.025]), label, lo, hi, valinit=0.0)
            s.on_changed(self._on_slider)
            return s

        self.fig.text(0.82, 0.325, "nudge ACTIVE model (relative to fit)", fontsize=8)
        self.s_yaw = slider(0.29, "yaw deg", -45, 45)
        self.s_x = slider(0.25, "dx mm", -15, 15)
        self.s_y = slider(0.21, "dy mm", -15, 15)
        self.s_z = slider(0.17, "dz mm", -15, 15)
        self.b_zero = button([0.85, 0.12, 0.12, 0.035], "zero sliders", lambda e: self._zero_sliders(True))

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _rebuild_candidate_panel(self):
        self._quiet = True
        self.ax_show.clear()
        self.ax_active.clear()
        if self.fits:
            labels = [f"{i + 1}. {f.candidate.label}\n    rms {f.rms_mm:4.1f} mm   cover {f.coverage * 100:3.0f}%"
                      for i, f in enumerate(self.fits)]
        else:
            labels = ["(no STL model for this class)"]
        self.ax_show.set_title("model candidates: tick = show (best first)", fontsize=8, loc="left")
        self.c_show = CheckButtons(self.ax_show, labels, self.shown or [False])
        for i, t in enumerate(self.c_show.labels):
            t.set_fontsize(7.5)
            if self.fits:
                t.set_color(MODEL_COLOURS[i % len(MODEL_COLOURS)])
                t.set_fontweight("bold")
        self.c_show.on_clicked(self._on_show)

        self.ax_active.set_title("active", fontsize=8)
        self.r_active = RadioButtons(self.ax_active, [str(i + 1) for i in range(max(1, len(self.fits)))],
                                     active=self.active)
        for t in self.r_active.labels:
            t.set_fontsize(7.5)
        self.r_active.on_clicked(self._on_active)
        self._quiet = False

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
        self.table_z = o.table_z_at(*np.median(self.xyz[:, :2], axis=0))
        self.matcher = matching.Matcher(self.xyz, self.case.camera_pos, self.table_z, mode=self.mode,
                                        lock_z=self.view["lock base to table"], icp=self.icp)
        cands = self.candidates(o.class_name)
        print(f"{o.object_id}: ICP-fitting {len(cands)} candidates ({self.icp}, {self.matcher.mode}) ...")
        self.fits = self.matcher.fit_all(cands) if cands else []
        # Keep the user's show/hide choice per candidate when re-fitting the same object
        prev = getattr(self, "_shown_by_label", {}) if getattr(self, "_fitted_obj", None) == o.object_id else {}
        self.shown = [prev.get(f.candidate.label, i == 0) for i, f in enumerate(self.fits)]
        self._fitted_obj = o.object_id
        self.active = 0
        self._rebuild_candidate_panel()
        self._zero_sliders(False)
        self.draw(reset_view=reset_view)

    def _remember_shown(self):
        self._shown_by_label = {f.candidate.label: s for f, s in zip(self.fits, self.shown)}

    def show_only(self, which):
        """None = all, -1 = none, k = only candidate k."""
        if not self.fits:
            return
        self.shown = [which is None or i == which for i in range(len(self.fits))]
        if which is not None and which >= 0:
            self.active = which
        self._remember_shown()
        self._rebuild_candidate_panel()
        self.draw()

    def toggle(self, k):
        if 0 <= k < len(self.fits):
            self.shown[k] = not self.shown[k]
            self._remember_shown()
            self._rebuild_candidate_panel()
            self.draw()

    def fit_for(self, k):
        """Fit k, with the slider nudges applied if it is the active one (re-scored)."""
        f = self.fits[k]
        if k != self.active:
            return f
        d = np.array([self.s_x.val, self.s_y.val, self.s_z.val]) / 1000
        if not np.any(d) and self.s_yaw.val == 0:
            return f
        return self.matcher.evaluate(f.candidate, f.pose.nudged(np.radians(self.s_yaw.val), d))

    # ---------------------------------------------------------------- callbacks
    def _on_icp(self, label):
        self.icp = "6dof" if label.startswith("6") else "4dof"
        self.refit()

    def _on_dist(self, label):
        self.mode = "3d" if label.startswith("3D") else "2d"
        self.refit()

    def _on_view(self, label):
        self.view[label] = not self.view[label]
        if label == "lock base to table":
            self.refit()
        else:
            self.draw()

    def _on_show(self, label):
        if self._quiet or not self.fits:
            return
        k = [t.get_text() for t in self.c_show.labels].index(label)
        self.shown[k] = not self.shown[k]
        self._remember_shown()
        self.draw()

    def _on_active(self, _label):
        if self._quiet or not self.fits:
            return
        self.active = self._radio_index(self.r_active)
        if not self.shown[self.active]:
            self.shown[self.active] = True
            self._remember_shown()
            self._rebuild_candidate_panel()
        self._zero_sliders(False)
        self.draw()

    @staticmethod
    def _radio_index(radio):
        return [t.get_text() for t in radio.labels].index(radio.value_selected)

    def _on_slider(self, _):
        if not self._quiet:
            self.draw()

    def _zero_sliders(self, redraw):
        quiet, self._quiet = self._quiet, True
        for s in (self.s_yaw, self.s_x, self.s_y, self.s_z):
            s.set_val(0.0)
        self._quiet = quiet
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
                   "f": self.refit, "r": lambda: self.draw(reset_view=True), "s": self.save,
                   "a": lambda: self.show_only(None), "b": lambda: self.show_only(0)}
        if event.key in actions:
            actions[event.key]()
        elif event.key and event.key.isdigit() and event.key != "0":
            self.toggle(int(event.key) - 1)

    # ---------------------------------------------------------------- drawing
    def draw(self, reset_view=False):
        ax = self.ax3d
        elev, azim = (ax.elev, ax.azim) if not reset_view and hasattr(self, "_drawn") else (35, -60)
        self._drawn = True
        ax.clear()
        o = self.current()
        mm = 1000.0
        active_fit = self.fit_for(self.active) if self.fits else None
        active_shown = bool(self.fits) and self.shown[self.active]

        colour_by_distance = self.view["data"] and self.view["colour by distance"] and active_shown
        self.ax_cbar.set_visible(colour_by_distance)
        if self.view["data"]:
            if colour_by_distance:
                data = self.matcher.data  # (possibly subsampled) points the matcher uses
                r = self.matcher.residuals(active_fit.candidate, active_fit.pose)
                sc = ax.scatter(*(data * mm).T, c=np.clip(r * mm, 0, 10), cmap="turbo", s=3, vmin=0, vmax=10,
                                depthshade=False)
                self.ax_cbar.clear()
                self.fig.colorbar(sc, cax=self.ax_cbar, label="distance to active model (mm)")
            else:
                ax.scatter(*(self.xyz * mm).T, c=self.rgb, s=3, depthshade=False, label="ZED data")

        for k, f in enumerate(self.fits):
            if not self.shown[k]:
                continue
            f = self.fit_for(k)
            col = MODEL_COLOURS[k % len(MODEL_COLOURS)]
            c = f.candidate
            pw = f.pose.apply(c.points)
            vis = matching.visible_mask(pw, c.normals @ f.pose.R.T, self.case.camera_pos)
            tag = f"{k + 1}. {c.size} {c.pose_name}"
            if self.view["visible faces"]:
                ax.scatter(*(pw[vis] * mm).T, color=col, s=1.5, depthshade=False, label=tag)
            if self.view["hidden faces"]:
                ax.scatter(*(pw[~vis] * mm).T, color=col, s=0.6, alpha=0.25, depthshade=False,
                           label=None if self.view["visible faces"] else tag)
            if self.view["mesh"]:
                tw = f.pose.apply(c.tris.reshape(-1, 3)).reshape(-1, 3, 3) * mm
                ax.add_collection3d(Poly3DCollection(tw, facecolor=matplotlib.colors.to_rgba(col, 0.06),
                                                     edgecolor=matplotlib.colors.to_rgba(col, 0.45),
                                                     linewidths=0.5 if k == self.active else 0.3))
            ctr, top = f.center * mm, f.top_center * mm
            big = k == self.active
            ax.scatter(*ctr, marker="*", s=260 if big else 140, color=col, edgecolor="k", depthshade=False)
            ax.scatter(*top, marker="o", s=70 if big else 40, facecolor="none", edgecolor=col, linewidths=2,
                       depthshade=False)
            ax.plot([ctr[0], top[0]], [ctr[1], top[1]], [ctr[2], top[2]], "--", color=col, lw=0.8)
        if o.yolo_center is not None:
            ax.scatter(*(o.yolo_center * mm), marker="X", s=90, c="royalblue", edgecolor="k",
                       depthshade=False, label="capture-script centre")

        # Local table patch and limits
        centre = (active_fit.center if active_fit is not None else np.median(self.xyz, axis=0)) * mm
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
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7, markerscale=3)
        n_shown = sum(self.shown)
        ax.set_title(f"{self.idx + 1}/{len(self.objects)}  {o.object_id}  YOLO: {o.class_name} "
                     f"({o.confidence:.2f})   showing {n_shown}/{len(self.fits)} models", fontsize=10)
        self._draw_info(o, active_fit, active_shown)
        self.fig.canvas.draw_idle()

    def _draw_info(self, o, fit, active_shown):
        mm = 1000.0
        h95 = np.percentile(self.h, 95) * mm if len(self.h) else float("nan")
        icp = "6-DoF (free)" if self.icp == "6dof" else "4-DoF (flat on table" + \
            (", base locked)" if self.view["lock base to table"] and self.mode == "3d" else ")")
        lines = [
            f"object      {o.object_id}",
            f"YOLO class  {o.class_name}  ({o.confidence:.2f})",
            f"points      {len(self.xyz)} used / {len(o.xyz_raw)} in mask",
            f"local table Z {self.table_z * mm:+.1f} mm",
            f"measured height (p95) {h95:.1f} mm",
            f"ICP         {icp}",
            f"distances   {'3D' if self.matcher.mode == '3d' else 'top-down XY'}",
            "",
        ]
        if fit is None:
            lines.append("no STL model for this class")
        else:
            c = fit.candidate
            ctr, top = fit.center * mm, fit.top_center * mm
            lines += [
                f"ACTIVE: {self.active + 1}. {c.label}" + ("" if active_shown else "  (hidden)"),
                f"model height {c.height * mm:.1f} mm",
                f"rms         {fit.rms_mm:.2f} mm",
                f"coverage    {fit.coverage * 100:.0f}% of visible model",
                f"data fit    {fit.data_inliers * 100:.0f}% within 3 mm",
                f"yaw         {np.degrees(fit.pose.yaw) % 360:.1f} deg"
                + (f"   tilt {fit.pose.tilt_deg:.1f} deg" if self.icp == "6dof" else ""),
                f"ICP iters   {fit.iterations}",
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
            if self.matcher.mode == "3d" and h95 < 0.6 * c.height * mm:
                lines += ["! measured height well below model:", "  depth looks flattened - try top-down"]
            shown = [k for k in range(len(self.fits)) if self.shown[k]]
            if len(shown) > 1:
                lines += ["", "SHOWN MODELS   rms  centre X / Y / Z (mm)"]
                for k in shown:
                    f = self.fit_for(k)
                    cc = f.center * mm
                    lines.append(f"{k + 1}. {f.candidate.size[0].upper()} {f.candidate.pose_name[:8]:8s}"
                                 f"{f.rms_mm:5.1f}  {cc[0]:.0f} / {cc[1]:.0f} / {cc[2]:.0f}")
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.5)

    # ---------------------------------------------------------------- save
    def save(self):
        if self.fits:
            self.results[self.current().object_id] = (self.fit_for(self.active), self.matcher.mode, self.icp)
        out = []
        for oid, (f, mode, icp) in self.results.items():
            out.append({
                "object_id": oid,
                "model": f.candidate.label, "shape": f.candidate.shape, "size": f.candidate.size,
                "pose": f.candidate.pose_name, "icp": icp, "distances": mode,
                "center_table_xyz_m": f.center.tolist(),
                "top_center_table_xyz_m": f.top_center.tolist(),
                "yaw_deg": float(np.degrees(f.pose.yaw) % 360),
                "tilt_deg": f.pose.tilt_deg,
                "rotation_matrix": f.pose.R.tolist(),
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
        gui.show_only(None)
        gui.fig.savefig(args.screenshot, dpi=90)
        print(f"wrote {args.screenshot}")
        return
    plt.show()


if __name__ == "__main__":
    main()
