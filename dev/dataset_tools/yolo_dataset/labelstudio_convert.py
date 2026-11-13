"""Convert between YOLO segmentation labels and Label Studio (polygon) JSON.

1) Import SAM labels into Label Studio as pre-annotations:
    python labelstudio_convert.py to-ls            # writes labelstudio_tasks.json + labeling_config.xml
2) After reviewing in Label Studio, export as "JSON" and convert back:
    python labelstudio_convert.py from-ls export.json --out labels_reviewed

Label Studio setup for local images (set BEFORE starting it):
    export LOCAL_FILES_SERVING_ENABLED=true
    export LOCAL_FILES_DOCUMENT_ROOT=<this yolo_dataset folder>
    label-studio start
Then in the project: Create Project -> Labeling Setup -> Custom template, paste labeling_config.xml,
and Import labelstudio_tasks.json.
"""

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png"}
FROM_NAME, TO_NAME = "label", "image"


def read_classes(path: Path) -> list[str]:
    return [c.strip() for c in path.read_text().splitlines() if c.strip()]


def to_ls(args):
    classes = read_classes(Path(args.classes_file))
    img_dir, lbl_dir = Path(args.images), Path(args.labels)
    root = Path(args.root).resolve()

    tasks = []
    for img_path in sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXT):
        h, w = cv2.imread(str(img_path)).shape[:2]
        rel = img_path.resolve().relative_to(root)
        task = {"data": {"image": f"/data/local-files/?d={rel.as_posix()}"}}

        lbl = lbl_dir / f"{img_path.stem}.txt"
        results = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                cls = int(parts[0])
                xy = list(map(float, parts[1:]))
                points = [[xy[i] * 100, xy[i + 1] * 100] for i in range(0, len(xy) - 1, 2)]
                results.append(
                    {
                        "from_name": FROM_NAME,
                        "to_name": TO_NAME,
                        "type": "polygonlabels",
                        "original_width": w,
                        "original_height": h,
                        "image_rotation": 0,
                        "value": {"points": points, "closed": True, "polygonlabels": [classes[cls]]},
                    }
                )
        if results:
            task["predictions"] = [{"model_version": "sam", "result": results}]
        tasks.append(task)

    Path(args.out).write_text(json.dumps(tasks, indent=2))
    labels_xml = "\n".join(f'    <Label value="{c}"/>' for c in classes)
    config = (
        "<View>\n"
        f'  <Image name="{TO_NAME}" value="$image"/>\n'
        f'  <PolygonLabels name="{FROM_NAME}" toName="{TO_NAME}">\n{labels_xml}\n  </PolygonLabels>\n'
        "</View>\n"
    )
    (Path(args.out).parent / "labeling_config.xml").write_text(config)
    print(f"{len(tasks)} tasks -> {args.out}; config -> labeling_config.xml")


def from_ls(args):
    classes = read_classes(Path(args.classes_file))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(Path(args.export).read_text())
    n = 0
    for task in tasks:
        image = task["data"]["image"]
        q = parse_qs(urlparse(image).query)
        name = Path(q["d"][0] if "d" in q else urlparse(image).path).stem
        anns = task.get("annotations") or []
        if not anns:
            continue
        lines = []
        for r in anns[0]["result"]:
            if r.get("type") != "polygonlabels":
                continue
            cls = classes.index(r["value"]["polygonlabels"][0])
            flat = [f"{v / 100:.6f}" for pt in r["value"]["points"] for v in pt]
            lines.append(f"{cls} " + " ".join(flat))
        (out / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    print(f"{n} label files -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes-file", default=str(HERE / "classes.txt"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("to-ls")
    a.add_argument("--images", default=str(HERE / "images"))
    a.add_argument("--labels", default=str(HERE / "labels"))
    a.add_argument("--root", default=str(HERE), help="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT")
    a.add_argument("--out", default=str(HERE / "labelstudio_tasks.json"))
    a.set_defaults(fn=to_ls)

    b = sub.add_parser("from-ls")
    b.add_argument("export", help="Label Studio JSON export")
    b.add_argument("--out", default=str(HERE / "labels_reviewed"))
    b.set_defaults(fn=from_ls)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
