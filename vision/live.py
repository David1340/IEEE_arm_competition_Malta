"""ZED + YOLO + top-face silhouette detection of table objects, shared by the live viewer and the pipeline.

    vision = Vision(args)            # args from add_vision_args()
    frame, cloud = vision.grab()
    canvas, objects = vision.detect(frame, cloud)
    # objects: [(class_name, confidence, center_xyz_in_robot_frame_or_None, point_count), ...]

Centres: objects with an STL get the fitted top-face centre (else the point-cloud mean of the mask). Bins use
no depth at all: the mean pixel of the YOLO mask is turned into x, y by the camera ray through it meeting the
plane --bin-plane-z above the table (camera calibration + lens parameters only). The mask holds the rim and
the walls the camera sees, so its mean pixel lies between the rim and the base: half the bin's height works best
(simulated 15 x 15 x 8 cm bin across the field: at most ~17 mm off at 0.04 m; 47 mm at 0, 56 mm at the rim).
--bin-point-cloud goes back to the point-cloud mean for bins.

--centre rect (default here) finds an object's x, y without fitting the STL: the ZED points inside its YOLO mask
are put in the robot frame, the table and stray depth are dropped (only points between --rect-min-z above the table
and its STL height + 2 cm, within its STL half-width + 1 cm of their median), and the object's x, y is the centre of
a rectangle around them seen from above: only the top face's points when there are enough (--rect-top-band,
--rect-top-min), its sides from the middle 90 % of the points (--rect-trim), and no wider than the object's real
size from the STL (a window of that size where the most points are). The label says which: RECT[top] / RECT[all],
+size when the size limit moved it. z is not measured: the object
stands on the table, so its top is its STL height (returned, as the grasp needs it) and its middle half of that.
--centre fit is the silhouette / point-cloud method above. --centre parallax (default here) uses no depth at all: the
mask outline is cast onto the top's and the table's planes and the parallax is undone (parallax_center); simulated
it is within ~1.5 mm across the field, for upright objects.

An object with neither a silhouette fit nor --min-points depth points (plain or shiny surfaces, glare under the
camera) is not dropped: its x, y come the same way as a bin's (mean mask pixel, camera ray onto the plane at half
its STL height) and its top is its STL height. Labelled "PIXELS (no depth)"; --no-pixel-fallback drops it instead.

Objects inside a bin are left out (already sorted, or out of the gripper's way): an object whose mask lies at least
--in-bin-overlap inside a bin's outline (its convex hull, so the bin's open inside counts) is drawn grey, labelled
"in bin", and not returned. --keep-in-bin returns them as before.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

from vision.detection import (
    OBJECT_SIZES,
    OBJECT_STL_FILES,
    CALIBRATION_FILE,
    ROI_FILE,
    camera_intrinsics,
    latest_weights,
    load_camera_transform,
    roi_for_frame,
    visible_surface_center,
)

BIN_CLASS = "bin"
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
          (128, 0, 255), (0, 200, 120), (200, 120, 0), (120, 120, 255)]


def add_vision_args(ap):
    ap.add_argument("--serial", type=int, default=None, help="ZED serial number; default uses camera ID")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--weights", default=None, help="default: most recently modified best.pt under runs/segment")
    ap.add_argument("--calibration", type=Path, default=CALIBRATION_FILE,
                    help="camera-to-robot transform saved by qr_calibrate.py")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="default: longest ROI side")
    ap.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD720")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--depth-mode", choices=["QUALITY", "ULTRA", "NEURAL"], default="QUALITY")
    ap.add_argument("--min-depth", type=float, default=0.10, help="ignore points closer than this (meters)")
    ap.add_argument("--max-depth", type=float, default=2.00, help="ignore points farther than this (meters)")
    ap.add_argument("--min-points", type=int, default=5, help="minimum valid mask points for a center estimate")
    ap.add_argument("--centre", choices=["parallax", "rect", "fit"], default="parallax",
                    help="objects' x, y: parallax = from the mask outline with the parallax corrected, no depth "
                         "(upright objects; z from the STL); rect = centre of the smallest rectangle around their "
                         "points seen from above; fit = silhouette fit / point-cloud mean")
    ap.add_argument("--rect-min-z", type=float, default=0.008,
                    help="rect: points lower than this above the table [m] are table, not object")
    ap.add_argument("--smooth-frames", type=int, default=8,
                    help="parallax / rect: each object's centre is the median of its last this many frames (1 = off); "
                         "the scripts that take one look (pick_and_place, Full_pipeline) look this many frames")
    ap.add_argument("--rect-trim", type=float, default=0.05,
                    help="rect: share of the points ignored at each end of each side (0 = the extreme points)")
    ap.add_argument("--rect-top-band", type=float, default=0.01,
                    help="rect: points within this of the STL height [m] are the top face")
    ap.add_argument("--rect-top-min", type=int, default=30,
                    help="rect: use only the top face when it has at least this many points")
    ap.add_argument("--rect-partial", type=float, default=0.7,
                    help="rect: a rectangle side under this share of the object's STL size means part of it has no "
                         "depth; it is then placed from its mask's pixels")
    ap.add_argument("--no-pixel-fallback", action="store_true",
                    help="drop an object with no fit and too few depth points (default: place it from its pixels)")
    ap.add_argument("--no-silhouette", action="store_true",
                    help="use the point-cloud mean instead of the fitted STL top-face center")
    ap.add_argument("--silhouette-min-iou", type=float, default=0.7,
                    help="minimum mask/silhouette IoU to trust the top-face center")
    ap.add_argument("--bin-plane-z", type=float, default=0.04,
                    help="bins: height above the table [m] of the plane the mask's mean pixel is cast onto "
                         "(about half the bin's height)")
    ap.add_argument("--bin-point-cloud", action="store_true",
                    help="bins: use the point-cloud mean of the mask instead of the mean pixel")
    ap.add_argument("--in-bin-overlap", type=float, default=0.5,
                    help="an object with at least this share of its mask inside a bin's outline is in the bin and "
                         "left out")
    ap.add_argument("--keep-in-bin", action="store_true", help="do not leave out the objects inside bins")


def inside_bins(valid, names, roi_shape, min_overlap):
    """ids of the masks (non-bin detections) lying at least min_overlap inside a bin's outline. valid: [(polygon
    in ROI pixels, class_id, confidence)]. The bin's convex hull is used, so its open inside counts even when its
    mask is only the rim."""
    bins = [polygon for polygon, class_id, _ in valid if names[class_id].casefold() == BIN_CLASS]
    if not bins:
        return set()
    height, width = roi_shape
    bin_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in bins:
        cv2.fillPoly(bin_mask, [cv2.convexHull(np.rint(polygon).astype(np.int32))], 1)
    inside = set()
    for polygon, class_id, _ in valid:
        if names[class_id].casefold() == BIN_CLASS:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        area = int(mask.sum())
        if area and int((mask & bin_mask).sum()) >= min_overlap * area:
            inside.add(id(polygon))
    return inside


def _window_centre(values, length):
    """Centre of the window `length` long holding the most of these 1-D values (ties: the middle one)."""
    v = np.sort(values)
    ends = np.searchsorted(v, v + length, side="right")
    counts = ends - np.arange(len(v))
    best = np.flatnonzero(counts == counts.max())
    i = best[len(best) // 2]
    return v[i] + length / 2


def rect_center(cloud, polygon, roi, camera_to_robot, height, width, args, footprint=None, top_face=None):
    """x, y of an object from its point cloud, seen from above in the robot frame. Returns (centre_xy, corners
    4 x 2, (w, l), angle_deg, point count, how) in metres, or (None, count, None, None, None, None) with fewer than
    args.min_points points. Steps:
      1. the points inside the mask, without the table (below args.rect_min_z), stray depth (above height + 2 cm)
         and points farther than width / 2 + 1 cm from their median x, y;
      2. top face: when at least args.rect_top_min of them lie within args.rect_top_band of the STL height, only
         those are used (an upright object's top face is centred on it: no sides, no edge bleeding);
      3. the rectangle's angle from cv2.minAreaRect of the points nearest the median (the outer args.rect_trim
         dropped), then along each of its two sides the extent is the args.rect_trim .. 1 - args.rect_trim
         quantile range of the points, not their extremes;
      4. known size: where that extent is wider than the object can be (footprint / top_face (short, long) from
         the STL, + 4 mm), the window of the object's size holding the most points gives the centre on that side;
      5. where it is much narrower (a side under args.rect_partial of the STL size), part of the object has no depth
         and the rectangle cannot know where the rest is: (None, count, None, None, None, "partial") is returned,
         and the caller places the object from its mask's pixels instead (the mask covers the whole object)."""
    x0, y0, w, h = roi
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)], 1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    ys, xs = np.nonzero(mask)
    xyz = cloud[y0:y0 + h, x0:x0 + w, :3][ys, xs].astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= args.min_depth) & (xyz[:, 2] <= args.max_depth)]
    xyz = xyz @ camera_to_robot[:3, :3].T + camera_to_robot[:3, 3]
    xyz = xyz[(xyz[:, 2] >= args.rect_min_z) & (xyz[:, 2] <= height + 0.02)]
    if len(xyz) >= args.min_points:
        keep = np.linalg.norm(xyz[:, :2] - np.median(xyz[:, :2], axis=0), axis=1) <= width / 2 + 0.01
        xyz = xyz[keep]
    if len(xyz) < args.min_points:
        return None, len(xyz), None, None, None, None
    # 2. top face only, when there is enough of it
    top = xyz[xyz[:, 2] >= height - args.rect_top_band]
    if len(top) >= max(args.rect_top_min, args.min_points):
        xy, how, size = top[:, :2], "top", top_face
    else:
        xy, how, size = xyz[:, :2], "all", footprint
    # 3. angle from the inner points, then trimmed extents along the rectangle's two sides
    median = np.median(xy, axis=0)
    dist = np.linalg.norm(xy - median, axis=1)
    inner = xy[dist <= np.quantile(dist, 1 - args.rect_trim)] if args.rect_trim > 0 else xy
    if len(inner) < 3:
        inner = xy
    _, _, angle = cv2.minAreaRect((1000 * inner).astype(np.float32))
    a = np.radians(angle)
    axes = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])     # rows: the rectangle's two sides
    proj = xy @ axes.T
    lo, hi = np.quantile(proj, [args.rect_trim, 1 - args.rect_trim], axis=0)
    centre_uv, extent = (lo + hi) / 2, hi - lo
    # 5. much narrower than the object: depth missing on part of it
    if size is not None and np.min(np.sort(extent) / np.maximum(sorted(size), 1e-6)) < args.rect_partial:
        return None, len(xy), None, None, None, "partial"
    # 4. known size: an extent wider than the object can be -> the best window of its real size on that side
    if size is not None:
        order = np.argsort(extent)                                         # match short side to short side
        for axis, known in zip(order, sorted(size)):
            if extent[axis] > known + 0.004:
                centre_uv[axis] = _window_centre(proj[:, axis], known)
                extent[axis] = known
                how += "+size"
    centre = centre_uv @ axes
    half = extent / 2
    corners = np.array([centre + sx * half[0] * axes[0] + sy * half[1] * axes[1]
                        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return centre, len(xy), corners, tuple(extent), angle, how


class CentreSmoother:
    """Median of each object's last `frames` centres, so a jagged mask that changes a little every frame does not
    make the centre jump. A centre within `radius` of a track of the same class continues it; a track not seen for
    `frames` frames is dropped, so a moved object is followed within a few frames."""

    def __init__(self, frames=8, radius=0.03):
        self.frames, self.radius = frames, radius
        self.tracks = []                   # [{"cls", "xy": [recent centres], "missed"}]

    def begin(self):
        for t in self.tracks:
            t["missed"] += 1
            t["used"] = False

    def smooth(self, cls, xy):
        xy = np.asarray(xy, float)
        if self.frames <= 1:
            return xy
        best, best_d = None, self.radius
        for t in self.tracks:
            if t["cls"] == cls and not t["used"]:
                d = np.linalg.norm(np.median(t["xy"], axis=0) - xy)
                if d < best_d:
                    best, best_d = t, d
        if best is None:
            best = {"cls": cls, "xy": [], "missed": 0, "used": False}
            self.tracks.append(best)
        best["xy"] = (best["xy"] + [xy])[-self.frames:]
        best["missed"], best["used"] = 0, True
        return np.median(best["xy"], axis=0)

    def end(self):
        self.tracks = [t for t in self.tracks if t["missed"] <= self.frames]


def resample_outline(polygon, step=1.0):
    """The closed outline with points every `step` pixels along it (YOLO's vertices are unevenly spaced, so
    quantiles of them would weigh the detailed parts more)."""
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    length = np.concatenate([[0], np.cumsum(seg)])
    if length[-1] <= 0:
        return poly
    at = np.arange(0, length[-1], step)
    return np.column_stack([np.interp(at, length, closed[:, 0]), np.interp(at, length, closed[:, 1])])


def parallax_center(polygon_full, cam, height, quantile=0.02):
    """x, y of an upright object from its mask outline alone (no depth), correcting the parallax.

    Away from the point under the camera (the nadir), the mask is the top face plus the walls facing the camera, so
    its centre is pulled towards the nadir. Every outline point is on a camera ray; the outline is cast onto the
    plane of the top (z = height) and of the table (z = 0). With r the direction from the nadir to the object:
      far edge  (max along r)        - the top face's far edge: from the outline cast at z = height
      near edge (min along r)        - the bottom's near edge:  from the outline cast at z = 0
      sides     (min/max across r)   - the top face's sides:    from the outline cast at z = height
    (cast at the wrong plane, the other parts of the outline land inside the object, never beyond these.) The
    centre is the middle of both pairs: exact for shapes symmetric about their centre (cube, cylinder, hexagonal
    prism) standing upright; an approximation for the dodecahedron. The outline is resampled evenly and each edge
    is the `quantile` / 1 - `quantile` point along its direction, not the single extreme one, so one stray pixel of
    YOLO's jagged mask does not move it. Returns (xy, top outline, bottom outline, (far, near, side_a, side_b)
    points) in the robot frame, metres."""
    poly = resample_outline(polygon_full)
    top = np.array([cam.ray_to_plane(u, v, height)[:2] for u, v in poly])
    bottom = np.array([cam.ray_to_plane(u, v, 0.0)[:2] for u, v in poly])
    nadir = cam.pos[:2]
    r = top.mean(axis=0) - nadir
    if np.linalg.norm(r) < 1e-3:                        # right under the camera: no parallax to correct
        r = np.array([1.0, 0.0])
    r = r / np.linalg.norm(r)
    q = np.array([-r[1], r[0]])
    far, near = top @ r, bottom @ r
    side = top @ q
    f, n = np.quantile(far, 1 - quantile), np.quantile(near, quantile)
    s_lo, s_hi = np.quantile(side, [quantile, 1 - quantile])
    along, across = (f + n) / 2, (s_lo + s_hi) / 2
    pick = lambda values, target, pts: pts[np.argmin(np.abs(values - target))]
    points = (pick(far, f, top), pick(near, n, bottom), pick(side, s_lo, top), pick(side, s_hi, top))
    return along * r + across * q, top, bottom, points


_FOOTPRINT_CACHE = {}


def rendered_area(stl, cam, xy):
    """Pixel area of an object model standing on the table at xy (robot frame), seen through the camera."""
    if stl not in _FOOTPRINT_CACHE:
        from control.gripper import _stl_vertices

        v = _stl_vertices(stl).astype(float)
        if np.ptp(v[:, 2]) > 1.0:
            v = v / 1000.0                                     # STL in mm
        v[:, :2] -= v[:, :2].mean(axis=0)
        v[:, 2] -= v[:, 2].min()
        _FOOTPRINT_CACHE[stl] = v
    v = _FOOTPRINT_CACHE[stl] + [xy[0], xy[1], 0.0]
    px = cam.project(v)
    # the objects are convex, so their outline is the convex hull of their projected corners (filling the
    # triangles one call would cancel the overlapping faces: fillPoly's even-odd rule)
    lo = np.floor(px.min(axis=0)).astype(int) - 2
    size = np.ceil(px.max(axis=0)).astype(int) - lo + 3
    canvas = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillConvexPoly(canvas, cv2.convexHull(np.rint(px - lo).astype(np.int32)), 1)
    return int(canvas.sum())


def parallax_with_size(polygon_full, cam, class_name):
    """The parallax centre for each size of the object (small / large STL: different heights), keeping the size
    whose model, rendered at its centre, covers about as many pixels as the mask (a small object has about half
    the area of a large one, so this is clear-cut). Returns (size, height, parallax_center result)."""
    from control.gripper import object_height

    poly = np.asarray(polygon_full, dtype=float).reshape(-1, 2)
    mask = np.zeros((int(poly[:, 1].max()) + 3, int(poly[:, 0].max()) + 3), np.uint8)
    cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
    area = max(int(mask.sum()), 1)
    best = None
    for size, stl in OBJECT_SIZES.get(class_name, {}).items():
        height = object_height(stl)
        result = parallax_center(poly, cam, height)
        mismatch = abs(np.log(max(rendered_area(stl, cam, result[0]), 1) / area))
        if best is None or mismatch < best[0]:
            best = (mismatch, size, height, result)
    if best is None:                                           # no size list: the one STL
        height = object_height(OBJECT_STL_FILES[class_name])
        return "", height, parallax_center(poly, cam, height)
    return best[1], best[2], best[3]


def bin_center_from_pixels(polygon, roi, cam, plane_z):
    """A bin's centre without depth: the mean of the mask's pixels, then the camera ray through that pixel
    meets the plane Z = plane_z (robot frame, the table at 0). Returns (xyz, pixel count, (u, v)) or
    (None, count, None) for a mask with no pixels."""
    x0, y0, width, height = roi
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)], 1)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, 0, None
    u, v = xs.mean() + x0, ys.mean() + y0
    return cam.ray_to_plane(u, v, plane_z), len(xs), (int(round(u)), int(round(v)))


class Vision:
    def __init__(self, args):
        import pyzed.sl as sl
        from ultralytics import YOLO

        self.sl = sl
        self.args = args
        if not ROI_FILE.exists():
            raise SystemExit(f"{ROI_FILE} not found; run 'python yolo_dataset/roi.py select' first")
        self.roi_data = json.loads(ROI_FILE.read_text())
        roi = tuple(self.roi_data[key] for key in ("x", "y", "w", "h"))
        self.imgsz = args.imgsz or max(roi[2:])
        self.camera_to_robot = load_camera_transform(args.calibration)
        weights = Path(args.weights) if args.weights else latest_weights()
        print(f"Model: {weights} | imgsz={self.imgsz} | depth={args.depth_mode}")
        print(f"Camera calibration: {args.calibration}")
        self.model = YOLO(str(weights))
        self.names = self.model.names

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
        self.image_mat, self.cloud_mat = sl.Mat(), sl.Mat()
        self.runtime = sl.RuntimeParameters()

        from vision.silhouette import Camera
        from vision.silhouette_live import LiveFitter, intrinsics_from_cloud

        try:
            intrinsics = camera_intrinsics(self.zed)
        except Exception as error:
            print(f"Could not read ZED intrinsics ({error}); estimating them from the point cloud")
            self.grab()
            intrinsics = intrinsics_from_cloud(self.cloud_mat.get_data()[..., :3])
        self.cam = Camera(self.camera_to_robot, intrinsics)   # pixel <-> robot frame, for the bins' rays
        self.smoother = CentreSmoother(getattr(args, "smooth_frames", 8))
        self.fitter = (None if args.no_silhouette or args.centre != "fit"
                       else LiveFitter(self.cam, self.camera_to_robot))

    def grab(self):
        """Latest left image (BGR) and XYZRGBA point cloud, blocking until a frame arrives."""
        sl = self.sl
        while self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            cv2.waitKey(1)
        self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.cloud_mat, sl.MEASURE.XYZRGBA)
        frame = cv2.cvtColor(self.image_mat.get_data(), cv2.COLOR_BGRA2BGR)
        cloud = self.cloud_mat.get_data()
        if cloud.shape[:2] != frame.shape[:2]:
            raise RuntimeError(f"Image/point cloud sizes differ: {frame.shape[:2]} vs {cloud.shape[:2]}")
        return frame, cloud

    def detect(self, frame, cloud):
        from vision.silhouette_live import Detection, SHAPE_CLASSES

        args, names, fitter = self.args, self.names, self.fitter
        current_roi = roi_for_frame(frame.shape, self.roi_data)
        x0, y0, roi_w, roi_h = current_roi
        roi_image = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]
        result = self.model.predict(roi_image, imgsz=self.imgsz, conf=args.conf, verbose=False)[0]
        canvas = frame.copy()
        overlay = canvas.copy()
        objects = []
        self.last_polygons = []   # full-frame pixel outline of each entry of objects, in the same order
        self.last_sizes = []      # "small" / "large" (parallax centre) or "" (unknown), in the same order

        self.smoother.begin()
        if result.masks is not None:
            classes = result.boxes.cls.int().tolist()
            confidences = result.boxes.conf.tolist()
            valid = [(polygon, class_id, confidence)
                     for polygon, class_id, confidence in zip(result.masks.xy, classes, confidences)
                     if len(polygon) >= 3]
            in_bin = set() if args.keep_in_bin else inside_bins(valid, names, (roi_h, roi_w), args.in_bin_overlap)
            for polygon, class_id, confidence in valid:
                if id(polygon) in in_bin:
                    points = np.rint(polygon + [x0, y0]).astype(np.int32)
                    cv2.polylines(canvas, [points], True, (150, 150, 150), 1)
                    px, py = points.min(axis=0)
                    cv2.putText(canvas, f"{names[class_id]} in bin", (int(px), max(int(py) - 5, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            valid = [item for item in valid if id(item[0]) not in in_bin]
            track_by_detection = {}
            if fitter is not None:
                shape_items = [item for item in valid if names[item[1]] in SHAPE_CLASSES]
                tracks, _ = fitter.update(
                    [Detection(names[class_id], confidence, polygon + [x0, y0])
                     for polygon, class_id, confidence in shape_items],
                    cloud[..., :3],
                )
                track_by_detection = {id(item[0]): track for item, track in zip(shape_items, tracks)}
            for polygon, class_id, confidence in valid:
                points = np.rint(polygon + [x0, y0]).astype(np.int32)
                color = COLORS[class_id % len(COLORS)]
                size = ""                        # small / large, when the parallax centre works it out
                if names[class_id].casefold() == BIN_CLASS and not args.bin_point_cloud:
                    center, count, pixel_center = bin_center_from_pixels(polygon, current_roi, self.cam,
                                                                         args.bin_plane_z)
                    cv2.polylines(canvas, [points], True, color, 2)
                    if center is None:
                        text = f"{BIN_CLASS} {confidence:.2f} XY n/a (mask too small)"
                    else:
                        cv2.drawMarker(canvas, pixel_center, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
                        text = f"{BIN_CLASS} {confidence:.2f} PIXELS X{center[0]:+.3f} Y{center[1]:+.3f}m"
                    objects.append((BIN_CLASS, confidence, center, count))
                    self.last_polygons.append(points)
                    self.last_sizes.append(size)
                    px, py = points.min(axis=0)
                    cv2.putText(canvas, text, (int(px), max(int(py) - 5, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                color, 2)
                    continue
                class_name = names[class_id]
                rect_failed, no_depth_reason = False, "no depth"
                if args.centre == "parallax" and class_name.casefold() in OBJECT_STL_FILES:
                    size, height, (xy, top_xy, bottom_xy, (far, near, side_a, side_b)) = parallax_with_size(
                        polygon + [x0, y0], self.cam, class_name.casefold())
                    xy = self.smoother.smooth(f"{class_name} {size}", xy)  # median of its last frames
                    center = np.array([xy[0], xy[1], height])           # top of the object, from its STL
                    count = len(polygon)
                    cv2.polylines(canvas, [points], True, color, 2)
                    for a_, b_ in ((far, near), (side_a, side_b)):      # the two measured widths, at the top
                        pa, pb = self.cam.project(np.array([[*a_, height], [*b_, height]]))
                        cv2.line(canvas, tuple(np.rint(pa).astype(int)), tuple(np.rint(pb).astype(int)), color, 1)
                    cx, cy = (int(round(v)) for v in self.cam.project(center)[0])
                    cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                    text = (f"{size} {class_name} {confidence:.2f} PARALLAX X{center[0]:+.3f} Y{center[1]:+.3f} "
                            f"top {1000 * height:.0f}mm")
                    objects.append((class_name, confidence, center, count))
                    self.last_polygons.append(points)
                    self.last_sizes.append(size)
                    px, py = points.min(axis=0)
                    cv2.putText(canvas, text, (int(px), max(int(py) - 5, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                color, 2)
                    continue
                if args.centre == "rect" and class_name.casefold() in OBJECT_STL_FILES:
                    from control.gripper import object_footprint, object_height, object_top_face, object_width

                    stl = OBJECT_STL_FILES[class_name.casefold()]
                    height = object_height(stl)
                    xy, count, corners, size, angle, how = rect_center(
                        cloud, polygon, current_roi, self.camera_to_robot, height, object_width(stl), args,
                        footprint=object_footprint(stl), top_face=object_top_face(stl))
                    if xy is not None:
                        xy = self.smoother.smooth(class_name, xy)         # median of its last frames
                        center = np.array([xy[0], xy[1], height])        # top of the object, from its STL
                        rect_px = self.cam.project(np.column_stack([corners, np.full(4, height)]))
                        cv2.polylines(canvas, [np.rint(rect_px).astype(np.int32)], True, color, 2)
                        cv2.polylines(canvas, [points], True, color, 1)
                        cx, cy = (int(round(v)) for v in self.cam.project(center)[0])
                        cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                        text = (f"{class_name} {confidence:.2f} RECT[{how}] X{center[0]:+.3f} Y{center[1]:+.3f} "
                                f"{1000 * size[0]:.0f}x{1000 * size[1]:.0f}mm ({count} pts)")
                        objects.append((class_name, confidence, center, count))
                        self.last_polygons.append(points)
                        self.last_sizes.append(size)
                        px, py = points.min(axis=0)
                        cv2.putText(canvas, text, (int(px), max(int(py) - 5, 16)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
                        continue
                    rect_failed = True                     # too few points / partial depth: place it from pixels
                    no_depth_reason = "partial depth" if how == "partial" else "no depth"
                if rect_failed:
                    center, pixel_center, top_mask = None, None, None
                else:
                    center, count, pixel_center, top_mask = visible_surface_center(
                        cloud, polygon, current_roi, args.min_depth, args.max_depth, args.min_points,
                        self.camera_to_robot,
                    )
                track = track_by_detection.get(id(polygon))
                if track is not None and track.fit is not None and track.fit.iou >= args.silhouette_min_iou:
                    # Top face of the fitted STL: its centre sits above the object axis.
                    top = fitter.cam.project(track.fit.pose_points(
                        np.asarray(track.fit.candidate.top_tris, dtype=float).reshape(-1, 3)))
                    cv2.polylines(canvas, [cv2.convexHull(np.rint(top).astype(np.int32))], True, color, 2)
                    center = np.asarray(track.top_center, dtype=float)
                    count = len(track.fit.candidate.points)
                    cx, cy = (int(round(value)) for value in fitter.cam.project(center)[0])
                    cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                    text = (f"{class_name} {confidence:.2f} TOP X{center[0]:+.3f} Y{center[1]:+.3f} "
                            f"Z{center[2]:+.3f}m IoU{track.fit.iou:.2f}")
                    objects.append((class_name, confidence, center, count))
                    self.last_polygons.append(points)
                    self.last_sizes.append(size)
                elif center is None and not args.no_pixel_fallback and class_name.casefold() in OBJECT_STL_FILES:
                    # No fit and no depth: x, y from the mask's mean pixel and the camera ray at half the object's
                    # height (as for the bins); the top at its STL height (standing).
                    from control.gripper import object_height

                    height = object_height(OBJECT_STL_FILES[class_name.casefold()])
                    xy, _, pixel_center = bin_center_from_pixels(polygon, current_roi, self.cam, height / 2)
                    if xy is None:
                        center = None
                    else:
                        center = np.array([xy[0], xy[1], height])
                        cv2.polylines(canvas, [points], True, color, 2)
                        cv2.drawMarker(canvas, pixel_center, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
                        text = (f"{class_name} {confidence:.2f} PIXELS ({no_depth_reason}) X{center[0]:+.3f} "
                                f"Y{center[1]:+.3f} Z{center[2]:+.3f}m")
                    objects.append((class_name, confidence, center, count))
                    self.last_polygons.append(points)
                    self.last_sizes.append(size)
                    if center is None:
                        cv2.polylines(canvas, [points], True, (128, 128, 128), 1)
                        text = f"{class_name} {confidence:.2f} XYZ n/a ({count} pts)"
                elif center is None:
                    cv2.polylines(canvas, [points], True, (128, 128, 128), 1)
                    text = f"{class_name} {confidence:.2f} XYZ n/a ({count} pts)"
                    objects.append((class_name, confidence, None, count))
                    self.last_polygons.append(points)
                    self.last_sizes.append(size)
                else:
                    roi_overlay = overlay[y0 : y0 + roi_h, x0 : x0 + roi_w]
                    roi_overlay[top_mask > 0] = color
                    contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, color, 2, offset=(x0, y0))
                    cx, cy = pixel_center
                    cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)
                    text = f"{class_name} {confidence:.2f} TABLE X{center[0]:+.3f} Y{center[1]:+.3f} Z{center[2]:+.3f}m"
                    objects.append((class_name, confidence, center, count))
                    self.last_polygons.append(points)
                    self.last_sizes.append(size)
                px, py = points.min(axis=0)
                cv2.putText(canvas, text, (int(px), max(int(py) - 5, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        self.smoother.end()
        canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0)
        cv2.rectangle(canvas, (x0, y0), (x0 + roi_w, y0 + roi_h), (255, 255, 255), 2)
        return canvas, objects

    def detect_steady(self):
        """Look --smooth-frames frames and return the last detection (frame, cloud, canvas, objects): its centres
        are then the median over those frames. For the scripts that take a single look per decision."""
        for _ in range(max(1, getattr(self.args, "smooth_frames", 1)) - 1):
            self.detect(*self.grab())
        frame, cloud = self.grab()
        canvas, objects = self.detect(frame, cloud)
        return frame, cloud, canvas, objects

    def close(self):
        self.zed.close()
