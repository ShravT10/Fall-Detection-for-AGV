"""
Script 3: Live Camera Inference — Floor & Safe Segmentation
------------------------------------------------------------
Runs your trained YOLOv8n-seg model on a webcam feed.
Usage: python inference.py --weights runs/segment/floor_safe_v1/weights/best.pt
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


# ───────────────────────────── config ──────────────────────────────────────

# Semi-transparent overlay colors per class (BGR)
CLASS_COLORS = {
    0: (0,  80,  255),    # floor → orange-red
    1: (0,  200, 80),     # safe  → green
}

# Confidence threshold – lower = more detections, higher = fewer false positives
CONF_THRESHOLD = 0.35

# How many past FPS samples to average for smooth display
FPS_SMOOTHING = 20


# ───────────────────────────── drawing helpers ─────────────────────────────

def draw_masks(frame: np.ndarray, result, alpha: float = 0.45) -> np.ndarray:
    """Overlay segmentation masks on the frame."""
    overlay = frame.copy()

    if result.masks is None:
        return frame

    masks  = result.masks.data.cpu().numpy()   # (N, H, W) float32 0-1
    boxes  = result.boxes
    h, w   = frame.shape[:2]

    for i, mask in enumerate(masks):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))

        # resize mask to frame size
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask  = (mask_resized > 0.5).astype(np.uint8)

        # fill polygon
        colored = np.zeros_like(frame, dtype=np.uint8)
        colored[binary_mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

        # draw contour
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)

        # label at centroid
        if contours:
            M   = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                name = result.names[cls_id]
                text = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    overlay,
                    (cx - 4, cy - th - 8),
                    (cx + tw + 4, cy + 4),
                    color, -1
                )
                cv2.putText(
                    overlay, text, (cx, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

    return overlay


def draw_hud(frame: np.ndarray, fps: float, n_floor: int, n_safe: int) -> np.ndarray:
    """Draw heads-up display: FPS + detection counts."""
    h, w = frame.shape[:2]

    # semi-transparent top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 42), (20, 20, 20), -1)
    frame = cv2.addWeighted(bar, 0.55, frame, 0.45, 0)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    floor_color = CLASS_COLORS[0]
    safe_color  = CLASS_COLORS[1]

    cv2.putText(frame, f"floor:{n_floor}", (w // 2 - 80, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, floor_color, 2)
    cv2.putText(frame, f"safe:{n_safe}", (w // 2 + 40, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, safe_color, 2)

    cv2.putText(frame, "Q to quit", (w - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    return frame


# ───────────────────────────── main loop ───────────────────────────────────

def run(weights: str, source: int = 0, conf: float = CONF_THRESHOLD,
        show_orig: bool = False):
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Run train.py first, then pass the best.pt path."
        )

    # auto-detect GPU, fall back to CPU gracefully
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"\n[INFO] Loading model: {weights}")
    print(f"[INFO] Device       : {'GPU (CUDA)' if device == 0 else 'CPU'}")
    model = YOLO(str(weights))
    print(f"[INFO] Classes      : {model.names}")
    print(f"[INFO] Opening camera source: {source}")
    print("[INFO] Press Q to quit, S to save screenshot\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {source}")

    # try to set a decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_history = []
    screenshot_count = 0

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying …")
            continue

        # ── run inference ──────────────────────────────────────────────────
        results = model.predict(
            source    = frame,
            conf      = conf,
            iou       = 0.45,
            imgsz     = 640,
            device    = device,
            verbose   = False,
        )
        result = results[0]

        # ── count detections per class ─────────────────────────────────────
        n_floor = n_safe = 0
        if result.boxes is not None:
            for cls_id in result.boxes.cls.cpu().numpy().astype(int):
                if cls_id == 0: n_floor += 1
                elif cls_id == 1: n_safe += 1

        # ── draw ───────────────────────────────────────────────────────────
        annotated = draw_masks(frame, result)

        # compute FPS
        elapsed = time.perf_counter() - t0
        fps_history.append(1.0 / max(elapsed, 1e-6))
        if len(fps_history) > FPS_SMOOTHING:
            fps_history.pop(0)
        fps = sum(fps_history) / len(fps_history)

        annotated = draw_hud(annotated, fps, n_floor, n_safe)

        if show_orig:
            display = np.hstack([frame, annotated])
        else:
            display = annotated

        cv2.imshow("Floor & Safe Detection — YOLOv8n-seg", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit.")
            break
        elif key == ord("s"):
            path = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(path, annotated)
            print(f"[INFO] Saved screenshot: {path}")
            screenshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


# ───────────────────────────── entry point ─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live camera inference with YOLOv8n-seg"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to best.pt from training"
    )
    parser.add_argument(
        "--source",
        default=0,
        help="Camera index (0=default webcam) or path to video file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--show-orig",
        action="store_true",
        help="Show original frame side-by-side with annotated frame"
    )
    args = parser.parse_args()

    run(
        weights   = args.weights,
        source    = int(args.source) if str(args.source).isdigit() else args.source,
        conf      = args.conf,
        show_orig = args.show_orig,
    )