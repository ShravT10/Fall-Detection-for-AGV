"""
Test Script — Forklift & Person Detection (YOLOv8n)
----------------------------------------------------
Tests your trained bounding box detection model on images.

Usage:
  # Test split
  python test_forklift.py --weights runs/detect/forklift_person_v1/weights/best.pt --source dataset/test/images/

  # Single image
  python test_forklift.py --weights runs/detect/forklift_person_v1/weights/best.pt --source image.jpg

  # Side by side
  python test_forklift.py --weights runs/detect/forklift_person_v1/weights/best.pt --source dataset/test/images/ --show-orig

  # Just save, no popup windows
  python test_forklift.py --weights runs/detect/forklift_person_v1/weights/best.pt --source dataset/test/images/ --no-show

  # Custom confidence
  python test_forklift.py --weights runs/detect/forklift_person_v1/weights/best.pt --source dataset/test/images/ --conf 0.5
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from collections import Counter
from ultralytics import YOLO


# ── config ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
OUTPUT_DIR     = Path("forklift_results")

# Colors per class BGR
CLASS_COLORS = {
    0: (0,  140, 255),   # forklift → orange
    1: (0,  220, 80),    # person   → green
}


# ── drawing ────────────────────────────────────────────────────────────────

def draw_boxes(frame: np.ndarray, result) -> np.ndarray:
    """Draw bounding boxes with label + confidence."""
    out = frame.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out

    h, w = out.shape[:2]

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
        name   = result.names[cls_id]

        # xyxy absolute coords
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # filled label bg
        label = f"{name}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return out


def draw_info_bar(frame: np.ndarray, filename: str,
                  det_summary: list, ms: float) -> np.ndarray:
    """Attach a dark info bar at the bottom."""
    h, w   = frame.shape[:2]
    bar_h  = 40
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h] = frame
    canvas[h:] = (28, 28, 28)

    # filename
    cv2.putText(canvas, Path(filename).name, (10, h + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # detections
    det_text = "  |  ".join(f"{n}: {c}" for n, c in det_summary) \
               if det_summary else "no detections"
    (tw, _), _ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(canvas, det_text, (w // 2 - tw // 2, h + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 210, 160), 1)

    # ms
    t_text = f"{ms:.1f} ms"
    (tw2, _), _ = cv2.getTextSize(t_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(canvas, t_text, (w - tw2 - 10, h + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 190, 255), 1)

    return canvas


# ── helpers ────────────────────────────────────────────────────────────────

def collect_images(source: str) -> list:
    p = Path(source)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTS:
            return [p]
        raise ValueError(f"Unsupported file type: {p.suffix}")
    elif p.is_dir():
        imgs = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS)
        if not imgs:
            raise FileNotFoundError(f"No images found in: {p}")
        return imgs
    raise FileNotFoundError(f"Source not found: {p}")


# ── main ───────────────────────────────────────────────────────────────────

def run(weights: str, source: str, conf: float = 0.35,
        iou: float = 0.45, save: bool = True,
        show: bool = True, show_orig: bool = False):

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    images = collect_images(source)

    print(f"\n{'='*58}")
    print(f"  Model   : {weights_path.name}")
    print(f"  Images  : {len(images)}")
    print(f"  Conf    : {conf}   IOU: {iou}")
    print(f"  Save    : {save}")
    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"  Output  : {OUTPUT_DIR.resolve()}")
    print(f"{'='*58}\n")

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"  Device  : {'GPU' if device == 0 else 'CPU'}\n")

    model        = YOLO(str(weights_path))
    class_names  = model.names
    total_counts = Counter()
    inf_times    = []

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        # ── inference ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = model.predict(
            source  = frame,
            conf    = conf,
            iou     = iou,
            imgsz   = 640,
            device  = device,
            verbose = False,
        )
        ms = (time.perf_counter() - t0) * 1000
        inf_times.append(ms)
        result = results[0]

        # ── count ──────────────────────────────────────────────────────
        det_summary = []
        if result.boxes is not None and len(result.boxes):
            cls_counts = Counter(
                class_names[int(c)] for c in result.boxes.cls.cpu().numpy()
            )
            total_counts.update(cls_counts)
            det_summary = list(cls_counts.items())

        status = ", ".join(f"{n}:{c}" for n, c in det_summary) or "none"
        print(f"  [{idx+1:>4}/{len(images)}]  {img_path.name:<40} {status:<25} {ms:6.1f}ms")

        # ── annotate ───────────────────────────────────────────────────
        annotated = draw_boxes(frame, result)
        annotated = draw_info_bar(annotated, img_path.name, det_summary, ms)

        if show_orig:
            orig_bar = draw_info_bar(frame, "(original)", [], ms)
            # pad to same height
            display  = np.hstack([orig_bar, annotated])
        else:
            display = annotated

        # ── save ───────────────────────────────────────────────────────
        if save:
            out_path = OUTPUT_DIR / f"pred_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), display)

        # ── show ───────────────────────────────────────────────────────
        if show:
            win = f"[{idx+1}/{len(images)}]  {img_path.name}  —  SPACE=next  P=prev  S=save  Q=quit"
            cv2.imshow(win, display)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("\n  Quit.")
                    cv2.destroyAllWindows()
                    _summary(total_counts, inf_times)
                    return
                elif key in (ord('n'), ord(' '), 13):
                    break
                elif key == ord('p') and idx > 0:
                    idx -= 2
                    break
                elif key == ord('s'):
                    shot = OUTPUT_DIR / f"snap_{img_path.stem}.jpg"
                    OUTPUT_DIR.mkdir(exist_ok=True)
                    cv2.imwrite(str(shot), display)
                    print(f"  Saved: {shot}")

            cv2.destroyAllWindows()

    _summary(total_counts, inf_times)


def _summary(counts: Counter, times: list):
    print(f"\n{'='*58}")
    print("  SUMMARY")
    print(f"{'='*58}")
    for name, cnt in counts.most_common():
        bar = "█" * min(cnt, 40)
        print(f"  {name:<15} {cnt:>5}  {bar}")
    if times:
        avg = sum(times) / len(times)
        print(f"\n  Avg inference : {avg:.1f} ms  (~{1000/avg:.0f} FPS equivalent)")
        print(f"  Min / Max     : {min(times):.1f} ms / {max(times):.1f} ms")
    print(f"{'='*58}\n")
    print(f"  Results saved to: {OUTPUT_DIR.resolve()}")


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv8 forklift/person detector")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--source",  required=True, help="Image or folder path")
    parser.add_argument("--conf",    type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--no-show", action="store_true", help="Don't display windows")
    parser.add_argument("--show-orig", action="store_true", help="Side by side with original")
    args = parser.parse_args()

    run(
        weights   = args.weights,
        source    = args.source,
        conf      = args.conf,
        iou       = args.iou,
        save      = not args.no_save,
        show      = not args.no_show,
        show_orig = args.show_orig,
    )
