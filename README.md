# Floor & Safe Zone Detection — YOLOv8

A computer vision project that detects **floor** and **safe zones** in images using YOLOv8 instance segmentation. Built for AGV (Automated Guided Vehicle) navigation assistance.

---

## What it does

The model takes an image or camera frame and draws segmentation masks over:
- **Floor** — a area not for AGVs
- **Safe** — designated safe zones

---

## Dataset

- **Total images:** ~120 (including augmentation)
- **Classes:** 2 — `floor`, `safe`
- **Annotation type:** Polygon segmentation (exported in YOLOv8 format via Roboflow)
- **Split:** 102 train / 10 val / 5 test

### Folder structure

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### data.yaml

```yaml
train: train/images
val: valid/images
test: test/images
nc: 2
names: ['floor', 'safe']
```

---

## Annotation

Images were annotated using **polygon annotation** (not bounding boxes) to precisely trace floor and safe zone boundaries. Annotations were done in Roboflow and exported in YOLOv8 segmentation format.

Each label `.txt` file contains one object per line:
```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

---

## Model

| | |
|---|---|
| Architecture | YOLOv8n-seg (nano segmentation) |
| Pretrained on | COCO |
| Input size | 640×640 |
| Epochs | 100 |
| Hardware trained on | Nvidia RTX A3000 |

---

## Evaluation Results

Validated on 10 images, 21 instances.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **all** | 0.931 | 0.896 | 0.945 | 0.856 |
| floor | 0.898 | 0.882 | 0.968 | 0.950 |
| safe | 0.963 | 0.909 | 0.922 | 0.762 |

Inference speed: **18.2ms per image**

---

## Setup

```bash
pip install ultralytics torch torchvision opencv-python matplotlib pyyaml
```

---

## Commands

### 1. Check dataset
```bash
python code-files/check_dataset.py --data dataset/data.yaml
```

### 2. Train
```bash
python code-files/train.py --data dataset/data.yaml
```

### 3. Test on images
```bash
# Test split
python code-files/test_images.py --weights runs/segment/floor_safe_v1/weights/best.pt --source dataset/test/images/

# Single image
python code-files/test_images.py --weights runs/segment/floor_safe_v1/weights/best.pt --source image.jpg

# Side by side (original vs prediction)
python code-files/test_images.py --weights runs/segment/floor_safe_v1/weights/best.pt --source dataset/test/images/ --show-orig
```

### 4. Re-run evaluation on best model
```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/segment/floor_safe_v1/weights/best.pt')
metrics = model.val(data='dataset/data.yaml')
print(metrics.results_dict)
"
```

### 5. Live camera inference
```bash
python code-files/inference.py --weights runs/segment/floor_safe_v1/weights/best.pt
```

---

## Output

Trained weights and results are saved to:
```
runs/segment/floor_safe_v1/
├── weights/
│   ├── best.pt      ← use this for inference
│   └── last.pt
├── results.png      ← training curves
├── confusion_matrix.png
├── PR_curve.png
└── val_batch*.jpg   ← sample predictions
```

---

## Controls (test_images.py)

| Key | Action |
|-----|--------|
| `N` / Space | Next image |
| `P` | Previous image |
| `S` | Save screenshot |
| `Q` | Quit |
