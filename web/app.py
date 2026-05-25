"""
FastAPI backend — Floor & Safe Zone Detection
Receives JPEG frames from the browser, runs YOLOv8n-seg inference
in a thread pool (so the async server never blocks), returns
annotated JPEG + JSON detections.
"""

import os
import time
import base64
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from ultralytics import YOLO

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── model config ───────────────────────────────────────────────────────────
WEIGHTS = os.environ.get("MODEL_WEIGHTS", "runs/segment/floor_safe_v1/weights/best.pt")
CONF    = float(os.environ.get("CONF_THRESHOLD", "0.35"))
DEVICE  = 0 if torch.cuda.is_available() else "cpu"

log.info(f"Loading model : {WEIGHTS}")
log.info(f"Device        : {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
log.info(f"Conf threshold: {CONF}")

if not Path(WEIGHTS).exists():
    raise FileNotFoundError(
        f"Weights not found: '{WEIGHTS}'\n"
        "Set MODEL_WEIGHTS env var or place best.pt at the expected path."
    )

model = YOLO(WEIGHTS)
log.info(f"Model ready. Classes: {model.names}")

# Thread pool — inference is CPU-bound, so we offload it here
# so FastAPI's event loop stays free for other requests
executor = ThreadPoolExecutor(max_workers=2)

# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Floor Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="web/templates")

# ── colors per class (BGR) ─────────────────────────────────────────────────
CLASS_COLORS = {
    0: (0,  80,  255),   # floor → orange-red
    1: (0,  200, 80),    # safe  → green
}

# ── schemas ────────────────────────────────────────────────────────────────
class FrameRequest(BaseModel):
    frame: str   # base64 data URL

class Detection(BaseModel):
    cls:  str
    conf: float

class DetectResponse(BaseModel):
    image:        str              # base64 JPEG
    detections:   list[Detection]
    inference_ms: float


# ── helpers ────────────────────────────────────────────────────────────────

def decode_frame(data_url: str) -> np.ndarray:
    _, encoded = data_url.split(",", 1)
    arr   = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


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

        resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary  = (resized > 0.5).astype(np.uint8)

        colored = np.zeros_like(frame)
        colored[binary == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{result.names[cls_id]} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(overlay, (cx-4, cy-th-10), (cx+tw+4, cy+4), color, -1)
                cv2.putText(overlay, label, (cx, cy-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return overlay


def run_inference(frame: np.ndarray) -> tuple[np.ndarray, list, float]:
    """Runs YOLO inference — called inside thread pool, not on event loop."""
    t0 = time.perf_counter()
    results = model.predict(
        source  = frame,
        conf    = CONF,
        iou     = 0.45,
        imgsz   = 640,
        device  = DEVICE,
        verbose = False,
    )
    ms     = (time.perf_counter() - t0) * 1000
    result = results[0]

    annotated  = draw_masks(frame, result)

    detections = []
    if result.boxes is not None:
        for cls_t, conf_t in zip(result.boxes.cls, result.boxes.conf):
            detections.append(Detection(
                cls  = result.names[int(cls_t.item())],
                conf = round(float(conf_t.item()), 3),
            ))

    return annotated, detections, round(ms, 1)


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect", response_model=DetectResponse)
async def detect(body: FrameRequest):
    # decode frame (fast — stays on event loop)
    try:
        frame = decode_frame(body.frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame decode failed: {e}")

    # run inference off the event loop so we don't block other requests
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        annotated, detections, ms = await loop.run_in_executor(
            executor, run_inference, frame
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # encode result
    _, buf   = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64  = base64.b64encode(buf).decode("utf-8")

    return DetectResponse(image=img_b64, detections=detections, inference_ms=ms)


@app.get("/health")
async def health():
    return {
        "status" : "ok",
        "device" : str(DEVICE),
        "model"  : WEIGHTS,
        "classes": model.names,
    }


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1)
