"""Optional white balance for block_color.py, turned on with --white-balance.

Removes a colour cast from the room lights before hues are measured. The white table is used as
the reference: its bright, unsaturated pixels should be grey (equal B, G, R), so a gain per
channel is computed that makes them grey, and the same gains are applied to the whole image,
blocks included.

On the training photos the light is already neutral (gains ~1.00) and it changes no predictions.
It helps under warm/cool light (simulated warm tint: 18 wrong -> 9 wrong); re-running
block_color.py --calibrate on photos from the venue is the better fix when that is possible.

Standalone use, to see the gains and save a before/after image:
    python white_balance.py images/xxx.jpg [--out wb_compare.jpg]
"""

import argparse

import cv2
import numpy as np


def white_reference(bgr: np.ndarray):
    """Mask of the bright unsaturated pixels (the white table) and the B, G, R gains that make
    them grey. Gains are None when there are too few such pixels."""
    f = bgr.astype(np.float32)
    mx, mn = f.max(axis=2), f.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    white = (sat < 0.15) & (mx > np.percentile(mx, 60))
    if white.sum() < 500:
        return white, None
    means = f[white].mean(axis=0)
    return white, means.mean() / np.maximum(means, 1)


def white_balance(bgr: np.ndarray) -> np.ndarray:
    """Scale B, G, R so the bright unsaturated pixels (the white table) come out grey."""
    _, gains = white_reference(bgr)
    if gains is None:
        return bgr
    return np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default=None, help="save original | balanced side by side")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")
    white, gains = white_reference(bgr)
    print(f"white pixels: {int(white.sum())} ({100 * white.mean():.0f}% of the image)")
    if gains is None:
        print("too few white pixels: no correction applied")
        return
    means = bgr[white].mean(axis=0)
    print(f"average B G R of white pixels: {means[0]:.0f} {means[1]:.0f} {means[2]:.0f}")
    print(f"gains B G R: {gains[0]:.3f} {gains[1]:.3f} {gains[2]:.3f}")
    if args.out:
        cv2.imwrite(args.out, np.hstack([bgr, white_balance(bgr)]))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
