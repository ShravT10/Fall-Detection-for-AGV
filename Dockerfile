# ── base ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── workdir ───────────────────────────────────────────────────────────────
WORKDIR /app

# ── python deps ───────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy app ──────────────────────────────────────────────────────────────
COPY web/           ./web/
COPY runs/          ./runs/

# ── env defaults ─────────────────────────────────────────────────────────
ENV MODEL_WEIGHTS=runs/segment/floor_safe_v1/weights/best.pt
ENV CONF_THRESHOLD=0.35
ENV PORT=5000

# ── run ───────────────────────────────────────────────────────────────────
EXPOSE 5000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
