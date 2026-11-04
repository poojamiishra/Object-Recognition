# Object Recognition

An end-to-end deep learning project for recognizing and locating everyday objects in images. The system covers single-object classification, single-object detection, and multi-object detection across 73 object classes.

## What the project does

- Cleans and validates a class-collected image dataset, removes exact duplicates, resizes images, and creates stratified train, validation, and test splits.
- Compares four approaches for single-object recognition: a custom CNN, EfficientNet-B0, YOLO11s, and Faster R-CNN.
- Builds synthetic multi-object scenes containing 2–9 objects with class-balanced sampling, randomized layouts, scaling, positioning, flipping, color jitter, and backgrounds.
- Verifies the trained detector on standard gap-free 2×2 and 3×3 grids made from unchanged 224×224 held-out images, matching the Milestone 2 clarification.
- Fine-tunes YOLOv8s to return each object's class ID, confidence score, and bounding box.
- Evaluates classification quality, localization quality, inference speed, and performance as scene complexity increases.

## Dataset

The processed dataset contains 7,315 images across 73 object classes:

- 5,120 training images
- 1,097 validation images
- 1,098 test images

The image files are not included in this repository. Class IDs and object names are available in [`object_ids.csv`](object_ids.csv).

## Single-object recognition

| Model | Approach | Test accuracy | Macro-F1 | Parameters |
|---|---|---:|---:|---:|
| Custom CNN | Classification from scratch | 89.6% | 0.897 | 2.4M |
| **EfficientNet-B0** | Transfer learning | **95.4%** | **0.953** | 4.1M |
| YOLO11s | One-stage detection | 93.6% | 0.936 | 9.5M |
| Faster R-CNN (R50-FPN v2) | Two-stage detection | 88.9% | 0.888 | 43.6M |

EfficientNet-B0 achieved the strongest classification performance and was selected as the final single-object model.

## Multi-object detection

The multi-object pipeline creates 640×640 composite images from the single-object dataset. It uses exact object placement to generate bounding-box labels and prevents the same class from appearing twice in one scene.

YOLOv8s was fine-tuned on:

- 3,000 training composites
- 500 validation composites
- 500 test composites

### Results

| Metric | Result |
|---|---:|
| mAP@0.5 | **0.821** |
| mAP@0.5:0.95 | **0.754** |
| Precision | 0.846 |
| Recall | 0.811 |
| F1 | 0.828 |
| ID accuracy for localized objects | **0.977** |
| 12-object stress test | **12/12 correct** |
| Inference time on NVIDIA T4 | 17.4 ms/image |

## Run the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

To run detection on an image with trained YOLOv8s weights:

```bash
cd milestone2
python predict.py \
  --weights yolov8s_multiobject_best.pt \
  --source my_image.jpg \
  --save-dir predictions
```

The output includes the detected class ID, confidence score, and bounding-box coordinates for every object.

## Standard 2×2 and 3×3 grid verification

The Milestone 2 notebook now contains a separate held-out verification section for the standard layouts expected on demo day. It generates 50 exact 2×2 grids and 50 exact 3×3 grids from test-split photos. Every tile remains 224×224 and is placed directly beside its neighbors, with no gaps, padding, empty cells, scaling, movement, flipping, or color jitter.

The section reports precision, recall, F1, localization recall, object-count accuracy, and exact-image accuracy by layout. It also saves one 2×2 and one 3×3 input, annotated model prediction, prediction CSV, and prediction-versus-ground-truth comparison under `standard_grid_evaluation/` in the configured Google Drive output directory.

For a quick standalone sample, provide either a directory of prepared 224×224 images or exactly 4/9 image paths:

```bash
cd milestone2

# Deterministically select four distinct-object tiles when Object IDs are present in directory names.
python standard_grid_demo.py \
  --grid-size 2 \
  --source-dir /path/to/test_224 \
  --weights yolov8s_multiobject_best.pt \
  --output-dir standard_grid_outputs

# Supply nine images explicitly in row-major order.
python standard_grid_demo.py \
  --grid-size 3 \
  --images image1.jpg image2.jpg image3.jpg image4.jpg image5.jpg \
           image6.jpg image7.jpg image8.jpg image9.jpg \
  --weights yolov8s_multiobject_best.pt \
  --output-dir standard_grid_outputs
```

The existing randomized layouts are intentionally retained: the clarification asks for a mixture of layouts and object counts, while the additional standard-grid evaluation confirms readiness for the primary 2×2 and 3×3 demo formats. The standard-grid results must be generated before adding numerical claims to the report.

## Repository structure

```text
├── object_ids.csv
├── requirements.txt
├── milestone1/
│   └── Milestone1_All_4_Models.ipynb
├── milestone2/
│   ├── Milestone2_YOLOv8_MultiObject.ipynb
│   ├── predict.py
│   ├── standard_grid_demo.py
│   └── object_ids.py
└── reports/
    ├── Milestone1_Report.pdf
    └── Milestone2_Report.pdf
```
