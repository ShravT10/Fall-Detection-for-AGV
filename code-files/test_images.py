"""
Script 4: Image Inference — Floor & Safe Segmentation
------------------------------------------------------
Test your trained model on single images, a folder, or your test split.

Usage examples:
  # Single image
  python test_images.py --weights best.pt --source image.jpg

  # Folder of images
  python test_images.py --weights best.pt --source path/to/images/

  # Your test split directly
  python test_images.py --weights best.pt --source dataset/test/images/

  # With custom confidence
  python test_images.py --weights best.pt --source images/ --conf 0.4

  # Don't save, just show
  python test_images.py --weights best.pt --source images/ --no-save
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


# ───────────────────────────── config ──────────────────────────────────────

CLASS_COLORS = {
    0: (0,  80,  255),    # floor → orange-red  (BGR)
    1: (0,  200, 80),     # safe  → green        (BGR)
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

OUTPUT_DIR = Path("inference_results")


# ───────────────────────────── drawing ─────────────────────────────────────

def draw_masks(frame: np.ndarray, result, alpha: float = 0.45) -> np.ndarray:
    overlay = frame.copy()
    if result.masks is None:
        return frame

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    h, w  = frame.shape[:2]

    for i, mask in enumerate(masks):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))

        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask  = (mask_resized > 0.5).astype(np.uint8)

        # filled mask
        colored = np.zeros_like(frame, dtype=np.uint8)
        colored[binary_mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

        # contour
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)

        # label at centroid
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                name = result.names[cls_id]
                text = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    overlay,
                    (cx - 5, cy - th - 10),
                    (cx + tw + 5, cy + 5),
                    color, -1
                )
                cv2.putText(
                    overlay, text, (cx, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

    return overlay


def draw_info_bar(frame: np.ndarray, filename: str, detections: list,
                  inference_ms: float) -> np.ndarray:
    """Draw a bottom bar with filename, detections, and inference time."""
    h, w   = frame.shape[:2]
    bar_h  = 40
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h] = frame

    # dark bar
    canvas[h:] = (30, 30, 30)

    # filename (left)
    cv2.putText(canvas, Path(filename).name, (10, h + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # detections summary (centre)
    det_text = "  |  ".join(
        [f"{name}: {cnt}" for name, cnt in detections]
    ) if detections else "no detections"
    (tw, _), _ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(canvas, det_text, (w // 2 - tw // 2, h + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 180), 1)

    # inference time (right)
    t_text = f"{inference_ms:.1f} ms"
    (tw2, _), _ = cv2.getTextSize(t_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(canvas, t_text, (w - tw2 - 10, h + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 1)

    return canvas


# ───────────────────────────── core logic ──────────────────────────────────

def collect_images(source: str) -> list[Path]:
    p = Path(source)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTS:
            return [p]
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")
    elif p.is_dir():
        images = sorted([
            f for f in p.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ])
        if not images:
            raise FileNotFoundError(f"No images found in: {p}")
        return images
    else:
        raise FileNotFoundError(f"Source not found: {p}")


def run(weights: str, source: str, conf: float = 0.35,
        save: bool = True, show: bool = True, show_orig: bool = False):

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Train first with train.py, then point --weights to best.pt"
        )

    images = collect_images(source)

    print(f"\n{'='*55}")
    print(f"  Model   : {weights_path.name}")
    print(f"  Images  : {len(images)}")
    print(f"  Conf    : {conf}")
    print(f"  Save    : {save}")
    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"  Output  : {OUTPUT_DIR.resolve()}")
    print(f"{'='*55}\n")

    model = YOLO(str(weights_path))
    class_names = model.names   # {0: 'floor', 1: 'safe', ...}

    total_detections = {name: 0 for name in class_names.values()}
    inference_times  = []

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        # ── inference ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = model.predict(
            source  = frame,
            conf    = conf,
            iou     = 0.45,
            imgsz   = 640,
            verbose = False,
        )
        ms = (time.perf_counter() - t0) * 1000
        inference_times.append(ms)

        result = results[0]

        # ── count detections ───────────────────────────────────────────────
        det_summary = []
        if result.boxes is not None and len(result.boxes):
            from collections import Counter
            cls_counts = Counter(
                class_names[int(c)] for c in result.boxes.cls.cpu().numpy()
            )
            for name, cnt in cls_counts.items():
                total_detections[name] += cnt
                det_summary.append((name, cnt))
        
        status = ", ".join(f"{n}:{c}" for n, c in det_summary) or "none"
        print(f"  [{idx+1:>3}/{len(images)}] {img_path.name:<35} {status:<25} {ms:6.1f}ms")

        # ── annotate ───────────────────────────────────────────────────────
        annotated = draw_masks(frame, result)
        annotated = draw_info_bar(annotated, img_path.name, det_summary, ms)

        if show_orig:
            # pad original to same height as annotated (which has info bar)
            orig_padded = draw_info_bar(frame, "(original)", [], ms)
            display = np.hstack([orig_padded, annotated])
        else:
            display = annotated

        # ── save ───────────────────────────────────────────────────────────
        if save:
            out_path = OUTPUT_DIR / f"pred_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), display)

        # ── show ───────────────────────────────────────────────────────────
        if show:
            win_title = f"[{idx+1}/{len(images)}] {img_path.name}  —  Q=quit  N=next  P=prev"
            cv2.imshow(win_title, display)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    print("\n  Quit.")
                    cv2.destroyAllWindows()
                    _print_summary(total_detections, inference_times)
                    return
                elif key in (ord("n"), ord(" "), 13):  # n / space / enter
                    break
                elif key == ord("p") and idx > 0:
                    idx -= 2   # step back (loop will +1)
                    break
                elif key == ord("s"):
                    shot = OUTPUT_DIR / f"screenshot_{img_path.stem}.jpg"
                    OUTPUT_DIR.mkdir(exist_ok=True)
                    cv2.imwrite(str(shot), display)
                    print(f"  Saved: {shot}")

            cv2.destroyAllWindows()

    _print_summary(total_detections, inference_times)


def _print_summary(total_detections: dict, inference_times: list):
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    for name, cnt in total_detections.items():
        print(f"  {name:<15} total detections: {cnt}")
    if inference_times:
        avg = sum(inference_times) / len(inference_times)
        print(f"\n  Avg inference  : {avg:.1f} ms  ({1000/avg:.1f} FPS equivalent)")
        print(f"  Min / Max      : {min(inference_times):.1f} ms / {max(inference_times):.1f} ms")
    print(f"{'='*55}\n")


# ───────────────────────────── entry point ─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test YOLOv8n-seg on images"
    )
    parser.add_argument(
        "--weights", required=True,
        help="Path to best.pt from training"
    )
    parser.add_argument(
        "--source", required=True,
        help="Single image path OR folder of images"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save annotated results"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display images (useful for batch runs)"
    )
    parser.add_argument(
        "--show-orig", action="store_true",
        help="Show original image side-by-side with prediction"
    )
    args = parser.parse_args()

    run(
        weights   = args.weights,
        source    = args.source,
        conf      = args.conf,
        save      = not args.no_save,
        show      = not args.no_show,
        show_orig = args.show_orig,
    )
