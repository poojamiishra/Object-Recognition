"""Build and optionally run inference on a standard gap-free 2x2 or 3x3 grid.

Every input tile must already be 224x224. Tiles are pasted directly beside one
another without resizing, spacing, padding, empty cells, or augmentation.

Examples
--------
Generate a deterministic 2x2 grid from a test-split directory and predict it::

    python standard_grid_demo.py --grid-size 2 --source-dir prep_224/test \
        --weights yolov8s_multiobject_best.pt --output-dir standard_grid_outputs

Build a 3x3 grid from nine explicitly selected images::

    python standard_grid_demo.py --grid-size 3 --images image1.jpg image2.jpg \
        image3.jpg image4.jpg image5.jpg image6.jpg image7.jpg image8.jpg image9.jpg
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps

from object_ids import OBJECT_NAMES


TILE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OBJECT_ID_RE = re.compile(r"OBJ\s*0*(\d+)", re.IGNORECASE)


def infer_object_id(path: Path) -> str | None:
    """Infer OBJ### from a file or parent-directory name when available."""
    for part in reversed(path.parts):
        match = OBJECT_ID_RE.search(part)
        if match:
            return f"OBJ{int(match.group(1)):03d}"
    return None


def discover_images(source_dir: Path) -> list[Path]:
    """Return supported images below ``source_dir`` in stable path order."""
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")
    images = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"no supported images found below: {source_dir}")
    return images


def select_distinct_images(paths: Iterable[Path], count: int, seed: int) -> list[Path]:
    """Select a deterministic sample, preferring distinct Object IDs."""
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    selected: list[Path] = []
    used_ids: set[str] = set()
    deferred: list[Path] = []
    for path in shuffled:
        object_id = infer_object_id(path)
        if object_id is not None and object_id in used_ids:
            deferred.append(path)
            continue
        selected.append(path)
        if object_id is not None:
            used_ids.add(object_id)
        if len(selected) == count:
            return selected

    for path in deferred:
        selected.append(path)
        if len(selected) == count:
            return selected
    raise ValueError(f"need {count} images, but only found {len(shuffled)}")


def build_standard_grid(image_paths: Sequence[Path], grid_size: int) -> tuple[Image.Image, list[dict]]:
    """Paste exactly ``grid_size**2`` 224x224 images into a gap-free grid."""
    if grid_size not in (2, 3):
        raise ValueError("grid_size must be 2 or 3")
    required = grid_size**2
    if len(image_paths) != required:
        raise ValueError(f"a {grid_size}x{grid_size} grid requires exactly {required} images")

    canvas = Image.new("RGB", (grid_size * TILE_SIZE, grid_size * TILE_SIZE))
    manifest: list[dict] = []
    for index, raw_path in enumerate(image_paths):
        path = Path(raw_path)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if image.size != (TILE_SIZE, TILE_SIZE):
            raise ValueError(
                f"{path} is {image.width}x{image.height}; every tile must already be 224x224"
            )
        row, column = divmod(index, grid_size)
        x, y = column * TILE_SIZE, row * TILE_SIZE
        canvas.paste(image, (x, y))
        object_id = infer_object_id(path)
        manifest.append(
            {
                "tile": index + 1,
                "row": row + 1,
                "column": column + 1,
                "source": str(path),
                "expected_object_id": object_id or "",
                "x1": x,
                "y1": y,
                "x2": x + TILE_SIZE,
                "y2": y + TILE_SIZE,
            }
        )
    return canvas, manifest


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict_grid(image_path: Path, weights: Path, confidence: float, image_size: int) -> tuple[Image.Image, list[dict]]:
    """Run YOLO and return an annotated image plus tabular detections."""
    from ultralytics import YOLO

    result = YOLO(str(weights)).predict(
        source=str(image_path), imgsz=image_size, conf=confidence,
        iou=0.6, verbose=False
    )[0]
    detections = []
    for box, class_index, score in zip(
        result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
    ):
        x1, y1, x2, y2 = box
        object_id = result.names[int(class_index)]
        detections.append(
            {
                "object_id": object_id,
                "object_name": OBJECT_NAMES.get(object_id, ""),
                "confidence": round(float(score), 4),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )
    detections.sort(key=lambda row: ((row["y1"] + row["y2"]) // 2, (row["x1"] + row["x2"]) // 2))
    annotated = Image.fromarray(result.plot()[..., ::-1])
    return annotated, detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-size", required=True, type=int, choices=(2, 3))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=Path, help="recursively select tiles from this directory")
    source.add_argument("--images", nargs="+", type=Path, help="exactly 4 or 9 tile paths in row-major order")
    parser.add_argument("--weights", type=Path, help="optional trained YOLOv8 .pt checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("standard_grid_outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = args.grid_size**2
    if args.images is not None:
        image_paths = args.images
    else:
        image_paths = select_distinct_images(discover_images(args.source_dir), required, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"standard_{args.grid_size}x{args.grid_size}"
    grid, manifest = build_standard_grid(image_paths, args.grid_size)
    input_path = args.output_dir / f"{stem}_input.jpg"
    grid.save(input_path, quality=95, subsampling=0)
    write_csv(args.output_dir / f"{stem}_tiles.csv", manifest)
    print(f"Saved {input_path} ({grid.width}x{grid.height}, {required} gap-free tiles)")

    if args.weights is not None:
        annotated, detections = predict_grid(input_path, args.weights, args.conf, args.imgsz)
        prediction_path = args.output_dir / f"{stem}_prediction.jpg"
        annotated.save(prediction_path, quality=95)
        write_csv(args.output_dir / f"{stem}_predictions.csv", detections)
        print(f"Saved {prediction_path} with {len(detections)} detections")


if __name__ == "__main__":
    main()
