"""
FastAPI backend — Forklift & Person Detection
Receives JPEG frames from the browser, runs YOLOv8n inference
in a thread pool, returns annotated JPEG + JSON detections.
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ultralytics import YOLO

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── model config ───────────────────────────────────────────────────────────
WEIGHTS = os.environ.get("MODEL_WEIGHTS", "runs/forklift_person_v1-2/weights/best.pt")
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

model    = YOLO(WEIGHTS)
executor = ThreadPoolExecutor(max_workers=2)
log.info(f"Model ready. Classes: {model.names}")

# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Forklift & Person Detection API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="web/templates")

# ── colors per class BGR ───────────────────────────────────────────────────
CLASS_COLORS = {
    0: (0,  140, 255),   # forklift → orange
    1: (0,  220, 80),    # person   → green
}

# ── schemas ────────────────────────────────────────────────────────────────
class FrameRequest(BaseModel):
    frame: str

class Detection(BaseModel):
    cls:  str
    conf: float

class DetectResponse(BaseModel):
    image:        str
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


def draw_boxes(frame: np.ndarray, result) -> np.ndarray:
    out = frame.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
        name   = result.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{name}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return out


def run_inference(frame: np.ndarray) -> tuple:
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

    annotated  = draw_boxes(frame, result)
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
    try:
        frame = decode_frame(body.frame)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame decode failed: {e}")

    import asyncio
    loop = asyncio.get_event_loop()
    try:
        annotated, detections, ms = await loop.run_in_executor(
            executor, run_inference, frame
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    _, buf  = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return DetectResponse(image=img_b64, detections=detections, inference_ms=ms)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE),
            "model": WEIGHTS, "classes": model.names}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1)
