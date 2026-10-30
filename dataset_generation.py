"""Capture images from one ZED camera for YOLO training.

Keys (preview window): s = save one image, a = toggle auto-capture, q/ESC = quit.
The current ROI is outlined when yolo_dataset/roi.json exists. Use --roi to save crops.
Usage:
    python dataset_generation.py --list
    python dataset_generation.py --serial 12345678 --out yolo_dataset/images --interval 1.0
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import pyzed.sl as sl

sys.path.insert(0, str(Path(__file__).resolve().parent / "yolo_dataset"))
from roi import ROI_FILE, crop, load_roi


def list_cameras():
    for d in sl.Camera.get_device_list():
        print(f"id={d.id} serial={d.serial_number} model={d.camera_model}")


def open_camera(serial, cam_id, resolution, fps):
    init = sl.InitParameters()
    init.camera_resolution = resolution
    init.camera_fps = fps
    init.depth_mode = sl.DEPTH_MODE.NONE
    if serial is not None:
        init.set_from_serial_number(serial)
    elif cam_id is not None:
        init.set_from_camera_id(cam_id)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED: {status}")
    return zed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list connected ZED cameras and exit")
    ap.add_argument("--serial", type=int, default=None)
    ap.add_argument("--id", type=int, default=0, help="camera id, used if --serial is not given")
    ap.add_argument("--out", default="yolo_dataset/images")
    ap.add_argument("--prefix", default="img")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between auto-captures")
    ap.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD1080")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--ext", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--roi", action="store_true", help="save only the ROI from yolo_dataset/roi.json")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return

    roi = load_roi() if ROI_FILE.exists() else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    zed = open_camera(args.serial, args.id, getattr(sl.RESOLUTION, args.resolution), args.fps)
    mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    auto, last_save, count = False, 0.0, 0
    existing = sum(1 for f in out.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"{existing} images already in {out}")

    def save(frame):
        nonlocal count
        name = f"{args.prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.{args.ext}"
        saved = crop(frame, roi) if args.roi and roi else frame
        cv2.imwrite(str(out / name), saved)
        count += 1

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                cv2.waitKey(1)
                continue
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            frame = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)

            now = time.time()
            if auto and now - last_save >= args.interval:
                save(frame)
                last_save = now

            view = frame.copy()
            if roi:
                x, y, rw, rh = roi
                cv2.rectangle(view, (x, y), (x + rw, y + rh), (0, 255, 0), 4)
            view = cv2.resize(view, None, fx=0.5, fy=0.5)
            roi_label = "ROI shown / crop save" if args.roi and roi else "ROI shown / full-frame save" if roi else "ROI not configured"
            label = f"saved={count} total={existing + count} auto={'ON' if auto else 'OFF'} | {roi_label}"
            cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("ZED capture", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save(frame)
            elif key == ord("a"):
                auto = not auto
                last_save = 0.0
    finally:
        zed.close()
        cv2.destroyAllWindows()
    print(f"Saved {count} images to {out} (total {existing + count})")


if __name__ == "__main__":
    main()
