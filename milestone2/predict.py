"""Detect every object in an image: prints each Object ID, name, confidence and bounding box.

Usage
    python predict.py --weights yolov8s_multiobject_best.pt --source path/to/image.jpg
    python predict.py --weights yolov8s_multiobject_best.pt --source path/to/folder --save-dir predictions --csv detections.csv

Boxes are pixel coordinates (x1, y1, x2, y2) with the origin at the top-left corner of the image.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from object_ids import OBJECT_NAMES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained YOLOv8 weights (.pt)")
    ap.add_argument("--source", required=True, help="image file or folder of images")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default 0.25)")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (default 640)")
    ap.add_argument("--save-dir", help="folder to save annotated images")
    ap.add_argument("--csv", help="write all detections to this CSV file")
    args = ap.parse_args()

    model = YOLO(args.weights)
    rows = []
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for r in model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, iou=0.6, batch=1, stream=True, verbose=False):
        dets = []
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            oid = r.names[int(c)]
            dets.append({"image": Path(r.path).name, "object_id": oid, "object_name": OBJECT_NAMES.get(oid, ""),
                         "confidence": round(s, 3), "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
        dets.sort(key=lambda d: ((d["y1"] + d["y2"]) // 2, (d["x1"] + d["x2"]) // 2))
        print(f"\n{Path(r.path).name}: {len(dets)} objects detected")
        for d in dets:
            print(f"  {d['object_id']}  {d['object_name']:<24} conf {d['confidence']:.2f}  box ({d['x1']}, {d['y1']}, {d['x2']}, {d['y2']})")
        if args.save_dir:
            Image.fromarray(r.plot()[..., ::-1]).save(Path(args.save_dir) / Path(r.path).name)
        rows += dets

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nSaved {len(rows)} detections to {args.csv}")


if __name__ == "__main__":
    main()
