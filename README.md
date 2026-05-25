# Floor & Safe Zone Detection — YOLOv8

A computer vision project that detects **floor** and **safe zones** using YOLOv8 instance segmentation, with a web interface accessible from any device on the same network.

---

## What it does

Points a phone camera at a scene and draws segmentation masks over:
- **Floor** — an area AGVs should avoid
- **Safe** — designated safe zones

---

## Project Structure

```
Fall-Detection-for-AGV/
├── code-files/
│   ├── check_dataset.py     ← validate dataset before training
│   ├── train.py             ← train YOLOv8n-seg
│   ├── test_images.py       ← test on images
│   └── inference.py         ← local webcam inference
├── web/
│   ├── app.py               ← FastAPI backend
│   └── templates/
│       └── index.html       ← phone/browser frontend
├── dataset/
│   ├── data.yaml
│   ├── train/images+labels
│   ├── valid/images+labels
│   └── test/images+labels
├── runs/segment/floor_safe_v1/weights/
│   ├── best.pt              ← use this for inference
│   └── last.pt
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Dataset

- **Total images:** ~120 (including augmentation)
- **Classes:** 2 — `floor`, `safe`
- **Annotation:** Polygon segmentation via Roboflow, exported in YOLOv8 format
- **Split:** 102 train / 10 val / 5 test

### data.yaml
```yaml
train: train/images
val: valid/images
test: test/images
nc: 2
names: ['floor', 'safe']
```

---

## Model

| | |
|---|---|
| Architecture | YOLOv8n-seg (nano segmentation) |
| Pretrained on | COCO |
| Input size | 640×640 |
| Epochs | 100 |
| Training hardware | Nvidia RTX A3000 |

---

## Evaluation Results

Validated on 10 images, 21 instances.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **all** | 0.931 | 0.896 | 0.945 | 0.856 |
| floor | 0.898 | 0.882 | 0.968 | 0.950 |
| safe  | 0.963 | 0.909 | 0.922 | 0.762 |

Inference speed: **18.2ms per image** (local GPU)

---

## Setup

```bash
pip install -r requirements.txt
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

# Side by side view
python code-files/test_images.py --weights runs/segment/floor_safe_v1/weights/best.pt --source dataset/test/images/ --show-orig
```

### 4. Local webcam inference
```bash
python code-files/inference.py --weights runs/segment/floor_safe_v1/weights/best.pt
```

### 5. Re-run evaluation
```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/segment/floor_safe_v1/weights/best.pt')
metrics = model.val(data='dataset/data.yaml')
print(metrics.results_dict)
"
```

---

## Web App (Phone + Browser)

FastAPI backend + browser frontend. Phone camera streams frames to the laptop, YOLO runs inference, annotated frames stream back.

### Run (HTTP)
```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### Run (HTTPS — required for camera on phone)

Generate a self-signed cert once:
```bash
python -c "
from OpenSSL import crypto
k = crypto.PKey(); k.generate_key(crypto.TYPE_RSA, 2048)
c = crypto.X509(); c.get_subject().CN = 'localhost'
c.set_serial_number(1); c.gmtime_adj_notBefore(0); c.gmtime_adj_notAfter(365*24*60*60)
c.set_issuer(c.get_subject()); c.set_pubkey(k); c.sign(k, 'sha256')
open('cert.pem','wb').write(crypto.dump_certificate(crypto.FILETYPE_PEM, c))
open('key.pem','wb').write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
"
```

Then start with HTTPS:
```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000 --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

Open on phone: `https://<your-laptop-ip>:8000` → tap Advanced → Proceed.

### Windows Firewall (run once as Administrator)
```powershell
netsh advfirewall firewall add rule name="YOLOv8 App" dir=in action=allow protocol=TCP localport=8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web frontend |
| `/detect` | POST | Run inference on a frame |
| `/health` | GET | Server status |
| `/docs` | GET | Auto Swagger UI |

### Frontend Features
- Live video stream from phone rear camera
- Real-time FPS and inference MS display in header
- Interval slider (80–800ms) — tune speed vs smoothness live
- Snap button — saves current annotated frame to downloads
- Stale warning if server stops responding

---

## Docker Deploy (EC2)

```bash
# Build and run locally
docker-compose up --build

# Deploy to EC2 (Ubuntu)
sudo apt update && sudo apt install -y docker.io docker-compose
scp -r . ubuntu@<EC2-IP>:~/floor-detect/
ssh ubuntu@<EC2-IP>
cd floor-detect && sudo docker-compose up -d
```

Open EC2 security group inbound rule: TCP port 8000.

> **Note:** EC2 t3.medium has no GPU. Set the interval slider to 500–800ms for smooth experience on CPU. For real-time performance use a g4dn.xlarge GPU instance.

---

## Training Outputs

```
runs/segment/floor_safe_v1/
├── weights/best.pt          ← best checkpoint
├── weights/last.pt          ← last checkpoint  
├── results.png              ← training curves
├── confusion_matrix.png
├── PR_curve.png
└── val_batch*.jpg           ← sample predictions
```