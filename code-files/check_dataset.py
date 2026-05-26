"""
Script 1: Dataset Checker for YOLOv8 Segmentation
---------------------------------------------------
Run this BEFORE training to catch any issues early.
Usage: python check_dataset.py --data path/to/data.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from collections import defaultdict

# ── optional visualisation (only if matplotlib/cv2 are installed) ──────────
try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    VISUAL = True
except ImportError:
    VISUAL = False
    print("[WARN] cv2 / matplotlib not found – skipping visualisation.\n")


# ───────────────────────────── helpers ─────────────────────────────────────

COLORS = [
    (255, 56,  56),   # red   – class 0
    (56,  255, 56),   # green – class 1
    (56,  56,  255),  # blue  – class 2
]

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def resolve_path(yaml_dir: Path, rel: str) -> Path:
    """Resolve a path that may be relative to the yaml file."""
    p = Path(rel)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    return p

def count_files(folder: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    if not folder.exists():
        return [], False
    files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
    return files, True

def parse_label(txt_path: Path):
    """Return list of (class_id, [(x,y), ...]) for each object in a label file."""
    objects = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0 or len(coords) < 6:
                # malformed – flag but skip
                objects.append((cls, None))
                continue
            pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            objects.append((cls, pts))
    return objects

def draw_segmentation(img_path: Path, label_path: Path, class_names: list):
    """Return an annotated BGR image with polygon overlays."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    overlay = img.copy()

    if label_path.exists():
        for cls, pts in parse_label(label_path):
            if pts is None:
                continue
            color = COLORS[cls % len(COLORS)]
            pixel_pts = np.array(
                [(int(x * w), int(y * h)) for x, y in pts], dtype=np.int32
            )
            cv2.fillPoly(overlay, [pixel_pts], color)
            cv2.polylines(img, [pixel_pts], True, color, 2)
            cx, cy = pixel_pts.mean(axis=0).astype(int)
            label = class_names[cls] if cls < len(class_names) else str(cls)
            cv2.putText(img, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return cv2.addWeighted(overlay, 0.35, img, 0.65, 0)


# ───────────────────────────── main check ──────────────────────────────────

def check(yaml_path: str, preview: int = 4):
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found: {yaml_path}")
        sys.exit(1)

    cfg       = load_yaml(yaml_path)
    yaml_dir  = yaml_path.parent
    nc        = cfg.get("nc", 0)
    names     = cfg.get("names", [])

    print("=" * 60)
    print("  YOLO DATASET CHECKER")
    print("=" * 60)
    print(f"  yaml   : {yaml_path}")
    print(f"  classes: {nc} → {names}")
    print()

    splits = {
        "train": cfg.get("train"),
        "val"  : cfg.get("val"),
        "test" : cfg.get("test"),
    }

    total_images  = 0
    total_labels  = 0
    class_counts  = defaultdict(int)   # per-class object count
    issues        = []
    preview_data  = []                 # (img_path, lbl_path) for visualisation

    for split, rel in splits.items():
        if rel is None:
            print(f"  [{split:5s}] – not specified in yaml, skipping")
            continue

        img_dir = resolve_path(yaml_dir, rel)
        # Labels sit next to images: dataset/train/images → dataset/train/labels
        # Fallback: dataset/labels/train  (older Roboflow layout)
        lbl_dir_sibling = img_dir.parent / "labels"
        lbl_dir_legacy  = img_dir.parent.parent / "labels" / img_dir.parent.name
        lbl_dir = lbl_dir_sibling if lbl_dir_sibling.exists() else lbl_dir_legacy

        images, img_ok = count_files(img_dir)
        labels, lbl_ok = count_files(lbl_dir, exts=(".txt",))

        if not img_ok:
            print(f"  [{split:5s}] images dir NOT FOUND: {img_dir}")
            issues.append(f"Missing images dir for split '{split}'")
            continue

        print(f"  [{split:5s}] images : {len(images):>4}  → {img_dir}")
        print(f"  [{split:5s}] labels : {len(labels):>4}  → {lbl_dir}")

        # ── per-image checks ──
        label_names = {f.stem for f in labels}
        for img in images:
            total_images += 1
            lbl_path = lbl_dir / (img.stem + ".txt")

            if img.stem not in label_names:
                issues.append(f"[{split}] Missing label for: {img.name}")
                continue

            total_labels += 1
            objects = parse_label(lbl_path)

            if not objects:
                issues.append(f"[{split}] Empty label file: {lbl_path.name}")
                continue

            for cls, pts in objects:
                if cls >= nc:
                    issues.append(
                        f"[{split}] Class id {cls} >= nc({nc}) in {lbl_path.name}"
                    )
                class_counts[cls] += 1
                if pts is None:
                    issues.append(
                        f"[{split}] Bounding box annotation in {lbl_path.name}"
                    )

            if len(preview_data) < preview:
                preview_data.append((img, lbl_path))

        print()

    # ── summary ─────────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"  Total images : {total_images}")
    print(f"  Total labeled: {total_labels}")
    print()
    print("  Class distribution:")
    for cls_id, cnt in sorted(class_counts.items()):
        name = names[cls_id] if cls_id < len(names) else f"cls_{cls_id}"
        bar  = "█" * min(cnt, 40)
        print(f"    [{cls_id}] {name:15s} {cnt:>5}  {bar}")
    print()

    if issues:
        print(f"  ⚠  {len(issues)} issue(s) found:")
        for iss in issues[:20]:   # cap at 20 to avoid spam
            print(f"     • {iss}")
        if len(issues) > 20:
            print(f"     ... and {len(issues)-20} more")
    else:
        print("  ✅ No issues found! Dataset looks clean.")

    print("=" * 60)

    # ── visual preview ───────────────────────────────────────────────────────
    if VISUAL and preview_data:
        print(f"\n  Showing {len(preview_data)} sample annotation(s) …")
        cols = min(len(preview_data), 4)
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
        if cols == 1:
            axes = [axes]

        for ax, (img_path, lbl_path) in zip(axes, preview_data):
            annotated = draw_segmentation(img_path, lbl_path, names)
            if annotated is None:
                ax.axis("off")
                continue
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            ax.imshow(annotated_rgb)
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")

        # legend
        patches = [
            mpatches.Patch(
                color=[c/255 for c in COLORS[i % len(COLORS)]],
                label=names[i] if i < len(names) else f"cls_{i}"
            )
            for i in range(nc)
        ]
        fig.legend(handles=patches, loc="lower center",
                   ncol=nc, fontsize=10, frameon=False)
        plt.suptitle("Sample Annotations (polygon segmentation)", fontsize=12)
        plt.tight_layout()
        plt.show()
    elif not VISUAL:
        print("  (Install opencv-python + matplotlib to see visual previews)")


# ───────────────────────────── entry point ─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 dataset checker")
    parser.add_argument(
        "--data",
        default="data.yaml",
        help="Path to your data.yaml file"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=4,
        help="Number of annotated images to preview (default: 4)"
    )
    args = parser.parse_args()
    check(args.data, args.preview)