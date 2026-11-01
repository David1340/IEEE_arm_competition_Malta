"""Real-time render-and-compare: YOLO masks -> STL silhouette fit -> top-face centres.

Each new object gets a full fit (every size x pose, ~0.1-0.2 s). After that it is TRACKED:
every frame only refines last frame's pose with small steps (~2-5 ms per object). A full
re-fit runs again when the overlap (IoU) drops, every --refit-every frames, or on 'r'.

Sources:
    python point_cloud_matching/silhouette_live.py                      # live ZED + YOLO
    python point_cloud_matching/silhouette_live.py --weights runs/segment/train-3/weights/best.pt
    python point_cloud_matching/silhouette_live.py --replay point_cloud_mathing/case_...   # no camera

Keys (click the window first):
    q / ESC quit      r  re-fit all objects      p  pause      c  centres only / full overlay
    s  save the current centres to point_cloud_matching/live_results/

Display: white = YOLO mask, green = STL outline as the camera should see it, red + = top-face
centre, label = track id, class, size/pose, table X Y Z (mm), IoU.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cases  # noqa: E402
import models  # noqa: E402
from silhouette import SS, Camera, SilhouetteMatcher  # noqa: E402
from top_face import estimate_intrinsics  # noqa: E402

SHAPE_CLASSES = set(models.CLASS_TO_STL)


class Detection:
    """What the matcher needs from a YOLO detection."""

    def __init__(self, class_name, confidence, polygon_full):
        self.class_name = class_name
        self.confidence = confidence
        self.polygon_full = np.asarray(polygon_full, np.float64)
        self.centroid = self.polygon_full.mean(axis=0)


def local_table_z(cloud_xyz, T, det, all_mask, ring=(8, 30), default=0.0):
    """Median table height in a ring of pixels just outside the mask (table depth is good)."""
    lo = np.floor(det.polygon_full.min(axis=0)).astype(int) - ring[1] - 2
    hi = np.ceil(det.polygon_full.max(axis=0)).astype(int) + ring[1] + 2
    h, w = cloud_xyz.shape[:2]
    lo, hi = np.maximum(lo, 0), np.minimum(hi, [w, h])
    m = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
    cv2.fillPoly(m, [np.rint(det.polygon_full - lo).astype(np.int32)], 1)
    outer = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=ring[1])
    ring_px = outer & (1 - all_mask[lo[1]:hi[1], lo[0]:hi[0]])
    ys, xs = np.nonzero(ring_px)
    if len(xs) < 20:
        return default
    xyz = np.asarray(cloud_xyz[ys + lo[1], xs + lo[0]], dtype=np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 20:
        return default
    z = (xyz @ T[:3, :3].T + T[:3, 3])[:, 2]
    return float(np.median(z))


def intrinsics_from_cloud(cloud_xyz, step=8):
    """fx, fy, cx, cy by fitting u = fx X/Z + cx, v = fy Y/Z + cy to a camera-frame cloud."""
    h, w = cloud_xyz.shape[:2]
    v, u = np.mgrid[0:h:step, 0:w:step]
    xyz = np.asarray(cloud_xyz[::step, ::step], dtype=np.float64)
    ok = np.isfinite(xyz).all(axis=2) & (xyz[..., 2] > 0.1)
    X, Y, Z = xyz[ok, 0], xyz[ok, 1], xyz[ok, 2]
    fx, cx = np.linalg.lstsq(np.column_stack((X / Z, np.ones_like(Z))), u[ok], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.column_stack((Y / Z, np.ones_like(Z))), v[ok], rcond=None)[0]
    return fx, fy, cx, cy


class Track:
    _next = 1

    def __init__(self, det):
        self.id = Track._next
        Track._next += 1
        self.class_name = det.class_name
        self.centroid = det.centroid
        self.fit = None
        self.top_center = None      # smoothed, table frame (m)
        self.since_full = 0
        self.missed = 0
        self.matcher = None


class LiveFitter:
    def __init__(self, cam, T, refit_every=60, min_iou=0.85, max_full_per_frame=2, smooth=0.5):
        self.cam, self.T = cam, T
        self.refit_every, self.min_iou = refit_every, min_iou
        self.max_full = max_full_per_frame
        self.smooth = smooth
        self.tracks = []
        self.cands = {}
        for cls in SHAPE_CLASSES:     # load every STL once up front
            shape = models.CLASS_TO_STL[cls]
            if shape not in self.cands:
                self.cands[shape] = models.load_candidates(shape)
        self.force_refit = False

    def candidates(self, class_name):
        return self.cands.get(models.CLASS_TO_STL.get(class_name), [])

    def _associate(self, dets):
        """Greedy nearest-centroid matching of detections to tracks of the same class."""
        pairs, used_t, used_d = [], set(), set()
        cost = []
        for i, d in enumerate(dets):
            for j, t in enumerate(self.tracks):
                if t.class_name == d.class_name:
                    dist = np.linalg.norm(t.centroid - d.centroid)
                    if dist < 40:
                        cost.append((dist, i, j))
        for dist, i, j in sorted(cost):
            if i not in used_d and j not in used_t:
                pairs.append((i, j))
                used_d.add(i)
                used_t.add(j)
        return pairs, used_d, used_t

    def update(self, dets, cloud_xyz):
        dets = [d for d in dets if d.class_name in SHAPE_CLASSES]
        h, w = cloud_xyz.shape[:2]
        all_mask = np.zeros((h, w), np.uint8)
        for d in dets:
            cv2.fillPoly(all_mask, [np.rint(d.polygon_full).astype(np.int32)], 1)
        all_mask = cv2.dilate(all_mask, np.ones((3, 3), np.uint8), iterations=8)

        pairs, _, used_t = self._associate(dets)
        matched = {i: self.tracks[j] for i, j in pairs}
        for j, t in enumerate(self.tracks):
            if j not in used_t:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= 5]   # forget objects gone >5 frames
        live = []
        for i, d in enumerate(dets):
            t = matched.get(i)
            if t is None:
                t = Track(d)
                self.tracks.append(t)
            live.append((d, t))

        full_budget = self.max_full
        timing = {"refine": 0.0, "full": 0.0}
        for d, t in live:
            t.centroid, t.missed = d.centroid, 0
            cands = self.candidates(d.class_name)
            if not cands:
                continue
            tz = local_table_z(cloud_xyz, self.T, d, all_mask,
                               default=t.fit.table_z if t.fit is not None else 0.0)
            t.matcher = SilhouetteMatcher(self.cam, d)
            need_full = (t.fit is None or self.force_refit or t.since_full >= self.refit_every)
            if t.fit is not None and not need_full:
                t0 = time.perf_counter()
                f = t.matcher.refine(t.fit, tz)
                timing["refine"] += time.perf_counter() - t0
                need_full = f.iou < self.min_iou
                if not need_full:
                    t.fit = f
                    t.since_full += 1
            if need_full and full_budget > 0:
                full_budget -= 1
                t0 = time.perf_counter()
                t.fit = max((t.matcher.fit(c, tz) for c in cands), key=lambda f: f.iou)
                timing["full"] += time.perf_counter() - t0
                t.since_full = 0
            if t.fit is not None:
                tc = t.fit.top_center
                t.top_center = tc if t.top_center is None or need_full else \
                    self.smooth * t.top_center + (1 - self.smooth) * tc
        self.force_refit = False
        return [t for _, t in live], timing


# ------------------------------------------------------------------------------- sources
class ReplaySource:
    """Feeds a saved case (image + cloud + saved YOLO polygons) as if it were live."""

    def __init__(self, case_dir):
        meta = json.loads((Path(case_dir) / "metadata.json").read_text())
        self.image = cv2.imread(str(Path(case_dir) / meta["files"]["image"]))
        self.cloud = np.load(Path(case_dir) / meta["files"]["point_cloud_xyzrgba"])[..., :3]
        self.T = np.load(Path(case_dir) / "camera_transform.npy")
        self.K = estimate_intrinsics(case_dir)
        x0, y0 = meta["roi_full_image_xywh_px"][:2]
        self.roi = tuple(meta["roi_full_image_xywh_px"])
        self.dets = [Detection(d["class_name"], d["confidence"], np.asarray(d["polygon_roi_xy_px"]) + [x0, y0])
                     for d in meta["detections"] if len(d["polygon_roi_xy_px"]) >= 3]

    def grab(self):
        return self.image.copy(), self.cloud, self.dets

    def close(self):
        pass


class ZedYoloSource:
    def __init__(self, args):
        import pyzed.sl as sl
        from ultralytics import YOLO
        sys.path.insert(0, str(ROOT / "yolo_dataset"))
        from object_3d_live import CALIBRATION_FILE, latest_weights, load_camera_transform, roi_for_frame

        self.sl = sl
        self.roi_for_frame = roi_for_frame
        self.roi_data = json.loads((ROOT / "yolo_dataset" / "roi.json").read_text())
        self.T = load_camera_transform(Path(args.calibration) if args.calibration else CALIBRATION_FILE)
        weights = Path(args.weights) if args.weights else latest_weights()
        print(f"YOLO weights: {weights}")
        self.model = YOLO(str(weights))
        self.conf, self.imgsz = args.conf, args.imgsz

        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
        init.camera_fps = args.fps
        init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        if args.serial is not None:
            init.set_from_serial_number(args.serial)
        else:
            init.set_from_camera_id(args.camera_id)
        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise SystemExit(f"Could not open ZED camera: {status}")
        self.K = self._intrinsics()
        print(f"Intrinsics fx={self.K[0]:.1f} fy={self.K[1]:.1f} cx={self.K[2]:.1f} cy={self.K[3]:.1f}")
        self.image_mat, self.cloud_mat = sl.Mat(), sl.Mat()
        self.runtime = sl.RuntimeParameters()

    def _intrinsics(self):
        try:  # ZED SDK 4/5: camera_configuration.calibration_parameters; SDK 3: calibration_parameters
            info = self.zed.get_camera_information()
            conf = getattr(info, "camera_configuration", info)
            cam = conf.calibration_parameters.left_cam
            return cam.fx, cam.fy, cam.cx, cam.cy
        except Exception as e:  # fall back to fitting the pinhole model to one point cloud
            print(f"Could not read ZED intrinsics ({e}); estimating them from the point cloud")
            sl = self.sl
            mat = sl.Mat()
            while self.zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
                pass
            self.zed.retrieve_measure(mat, sl.MEASURE.XYZRGBA)
            return intrinsics_from_cloud(mat.get_data()[..., :3])

    def grab(self):
        sl = self.sl
        while self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            pass
        self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.cloud_mat, sl.MEASURE.XYZRGBA)
        frame = cv2.cvtColor(self.image_mat.get_data(), cv2.COLOR_BGRA2BGR)
        cloud = self.cloud_mat.get_data()[..., :3]
        x0, y0, w, h = self.roi = self.roi_for_frame(frame.shape, self.roi_data)
        res = self.model.predict(frame[y0:y0 + h, x0:x0 + w], imgsz=self.imgsz or max(w, h),
                                 conf=self.conf, verbose=False)[0]
        dets = []
        if res.masks is not None:
            for poly, cls, cf in zip(res.masks.xy, res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
                if len(poly) >= 3:
                    dets.append(Detection(str(self.model.names[cls]), float(cf), np.asarray(poly) + [x0, y0]))
        return frame, cloud, dets

    def close(self):
        self.zed.close()


# ------------------------------------------------------------------------------- display
def draw(frame, tracks, fitter, roi, centres_only, fps, timing, paused):
    out = frame.copy()
    for t in tracks:
        if t.fit is None or t.matcher is None:
            continue
        m = t.matcher
        if not centres_only:
            cv2.polylines(out, [np.rint(m.obj.polygon_full).astype(np.int32)], True, (255, 255, 255), 1)
            sil = m.render(t.fit.candidate, t.fit.x, t.fit.y, t.fit.yaw, t.fit.table_z)
            cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cv2.polylines(out, [np.rint(c.reshape(-1, 2) / SS + m.origin).astype(np.int32)], True,
                              (0, 230, 0), 1)
        u, v = fitter.cam.project(t.top_center)[0]
        p = (int(round(u)), int(round(v)))
        cv2.drawMarker(out, p, (0, 0, 0), cv2.MARKER_CROSS, 20, 4)
        cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        tc = t.top_center * 1000
        c = t.fit.candidate
        lines = [f"#{t.id} {t.class_name} {c.size} {c.pose_name}",
                 f"X{tc[0]:.0f} Y{tc[1]:.0f} Z{tc[2]:.0f}mm IoU{t.fit.iou:.2f}"]
        top = m.obj.polygon_full.min(axis=0).astype(int)
        for k, txt in enumerate(lines):
            y = max(int(top[1]) - 22, 12) + 13 * k
            cv2.putText(out, txt, (int(top[0]), y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(out, txt, (int(top[0]), y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    x0, y0, w, h = roi
    view = out[y0:y0 + h, x0:x0 + w].copy()
    status = (f"{fps:5.1f} FPS | {len(tracks)} objects | refine {timing['refine'] * 1000:4.1f} ms"
              f" full-fit {timing['full'] * 1000:5.0f} ms" + ("  [PAUSED]" if paused else "")
              + "  | q quit  r refit  p pause  c centres  s save")
    cv2.rectangle(view, (0, view.shape[0] - 24), (view.shape[1], view.shape[0]), (0, 0, 0), -1)
    cv2.putText(view, status, (8, view.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


def save(tracks, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"track": t.id, "class": t.class_name, "model": t.fit.candidate.label, "iou": t.fit.iou,
             "yaw_deg": float(np.degrees(t.fit.yaw)), "top_center_table_xyz_m": t.top_center.tolist()}
            for t in tracks if t.fit is not None]
    path = out_dir / f"centres_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"saved {len(rows)} centres to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=Path, help="run on a saved case instead of the camera")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    ap.add_argument("--headless", action="store_true", help="no window; print timings (for testing)")
    ap.add_argument("--snapshot", type=Path, help="with --frames: save the display of the last frame here")
    ap.add_argument("--serial", type=int, default=None)
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--calibration", default=None, help="default: camera_transform.npy in the repo root")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD1080")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--depth-mode", choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL"], default="QUALITY")
    ap.add_argument("--refit-every", type=int, default=60, help="full re-fit every N frames per object")
    ap.add_argument("--min-iou", type=float, default=0.85, help="full re-fit when tracking IoU drops below")
    ap.add_argument("--max-width", type=int, default=1400)
    args = ap.parse_args()

    if args.replay:
        path = args.replay if (args.replay / "metadata.json").exists() else cases.find_cases()[-1]
        src = ReplaySource(path)
    else:
        src = ZedYoloSource(args)
    cam = Camera(src.T, src.K)
    t0 = time.perf_counter()
    fitter = LiveFitter(cam, src.T, args.refit_every, args.min_iou)
    print(f"loaded STL models in {time.perf_counter() - t0:.1f} s")

    win = "Render-and-compare live"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused, centres_only = False, False
    fps, n, t_fps, frame_i, last_print = 0.0, 0, time.time(), 0, 0.0
    tracks, timing, frame = [], {"refine": 0, "full": 0}, None
    try:
        while True:
            if not paused or frame is None:
                frame, cloud, dets = src.grab()
                t1 = time.perf_counter()
                tracks, timing = fitter.update(dets, cloud)
                fit_ms = (time.perf_counter() - t1) * 1000
                frame_i += 1
                n += 1
                if time.time() - t_fps >= 1.0:
                    fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()
                if args.headless or time.time() - last_print > 1.0:
                    last_print = time.time()
                    print(f"frame {frame_i:4d}: {len(tracks)} objects, fit {fit_ms:6.1f} ms "
                          f"(refine {timing['refine'] * 1000:5.1f} ms, full {timing['full'] * 1000:6.1f} ms)")
            if not args.headless:
                view = draw(frame, tracks, fitter, src.roi, centres_only, fps, timing, paused)
                scale = min(1.0, args.max_width / view.shape[1])
                cv2.imshow(win, cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)
                key = cv2.waitKey(1 if not args.replay else 30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    fitter.force_refit = True
                if key == ord("p"):
                    paused = not paused
                if key == ord("c"):
                    centres_only = not centres_only
                if key == ord("s"):
                    save(tracks, HERE / "live_results")
            if args.snapshot and frame_i == args.frames:
                cv2.imwrite(str(args.snapshot), draw(frame, tracks, fitter, src.roi, centres_only, fps, timing, False))
            if args.frames and frame_i >= args.frames:
                break
    finally:
        src.close()
        if not args.headless:
            cv2.destroyAllWindows()
    for t in tracks:
        if t.fit is not None:
            tc = t.top_center * 1000
            print(f"  #{t.id} {t.class_name:12s} {t.fit.candidate.label:34s} IoU {t.fit.iou:.3f} "
                  f"top centre X {tc[0]:.1f} Y {tc[1]:.1f} Z {tc[2]:.1f} mm")


if __name__ == "__main__":
    main()
