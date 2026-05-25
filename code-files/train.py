"""
Script 2: Train YOLOv8n-seg — Floor & Safe Detection
------------------------------------------------------
Optimised for: Nvidia RTX A3000, 2-class segmentation, ~120 images
Usage: python train.py --data path/to/data.yaml
"""

import os
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO


# ───────────────────────────── config ──────────────────────────────────────

DEFAULT_CFG = {
    # ── model ──
    "model"      : "yolov8n-seg.pt",   # nano-seg; pre-trained on COCO
    "epochs"     : 100,
    "imgsz"      : 640,                 # full res – A3000 can handle it

    # ── batch & hardware ──
    "batch"      : 8,                  # safe for ~4 GB VRAM; bump to 16 if no OOM
    "device"     : "0",                # GPU 0 (RTX A3000); set "cpu" to force CPU
    "workers"    : 4,                  # dataloader threads

    # ── optimiser ──
    "optimizer"  : "AdamW",
    "lr0"        : 0.001,              # initial LR
    "lrf"        : 0.01,               # final LR = lr0 * lrf
    "momentum"   : 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,               # gentle warmup – helps small datasets

    # ── augmentation ──
    "hsv_h"      : 0.015,             # hue jitter
    "hsv_s"      : 0.7,               # saturation jitter
    "hsv_v"      : 0.4,               # brightness jitter
    "degrees"    : 5.0,               # rotation (floors are mostly level)
    "translate"  : 0.1,
    "scale"      : 0.5,
    "shear"      : 2.0,
    "perspective": 0.0005,            # slight perspective warp
    "flipud"     : 0.0,               # don't flip upside-down (floors are below)
    "fliplr"     : 0.5,               # horizontal flip is fine
    "mosaic"     : 1.0,               # mosaic – great for small datasets
    "mixup"      : 0.1,               # light mixup
    "copy_paste" : 0.3,               # copy-paste augmentation for segmentation

    # ── training behaviour ──
    "patience"   : 30,                # early stopping patience
    "save_period": 10,                # checkpoint every N epochs
    "val"        : True,
    "plots"      : True,              # save training plots
    "project"    : "runs/segment",
    "name"       : "floor_safe_v1",
}


# ───────────────────────────── helpers ─────────────────────────────────────

def check_gpu():
    print("\n" + "=" * 55)
    print("  HARDWARE CHECK")
    print("=" * 55)
    if torch.cuda.is_available():
        idx  = 0
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"  ✅ GPU found : {name}")
        print(f"     VRAM     : {vram:.1f} GB")
        print(f"     CUDA     : {torch.version.cuda}")

        # adjust batch size based on VRAM
        if vram < 4:
            DEFAULT_CFG["batch"] = 4
            print("     [INFO] Low VRAM – batch size set to 4")
        elif vram >= 8:
            DEFAULT_CFG["batch"] = 16
            print("     [INFO] Good VRAM – batch size set to 16")
    else:
        print("  ⚠  No GPU detected – falling back to CPU")
        print("     Training will be much slower (~2-4 hrs for 100 epochs)")
        DEFAULT_CFG["device"] = "cpu"
        DEFAULT_CFG["batch"]  = 4
        DEFAULT_CFG["imgsz"]  = 416
    print("=" * 55 + "\n")


def train(data_yaml: str, resume: bool = False, overrides: dict = None):
    check_gpu()

    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    cfg = {**DEFAULT_CFG, **(overrides or {})}
    cfg["data"] = str(data_yaml)

    print("  TRAINING CONFIG")
    print("=" * 55)
    for k, v in cfg.items():
        print(f"  {k:<20} {v}")
    print("=" * 55 + "\n")

    # ── load model ──────────────────────────────────────────────────────────
    if resume:
        # resume from last checkpoint
        last_ckpt = Path(cfg["project"]) / cfg["name"] / "weights" / "last.pt"
        if not last_ckpt.exists():
            print("[WARN] No checkpoint found to resume from. Starting fresh.")
            model = YOLO(cfg["model"])
        else:
            print(f"[INFO] Resuming from {last_ckpt}")
            model = YOLO(str(last_ckpt))
    else:
        model = YOLO(cfg["model"])   # downloads weights if not cached

    # ── train ───────────────────────────────────────────────────────────────
    results = model.train(
        data        = cfg["data"],
        epochs      = cfg["epochs"],
        imgsz       = cfg["imgsz"],
        batch       = cfg["batch"],
        device      = cfg["device"],
        workers     = cfg["workers"],
        optimizer   = cfg["optimizer"],
        lr0         = cfg["lr0"],
        lrf         = cfg["lrf"],
        momentum    = cfg["momentum"],
        weight_decay= cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        hsv_h       = cfg["hsv_h"],
        hsv_s       = cfg["hsv_s"],
        hsv_v       = cfg["hsv_v"],
        degrees     = cfg["degrees"],
        translate   = cfg["translate"],
        scale       = cfg["scale"],
        shear       = cfg["shear"],
        perspective = cfg["perspective"],
        flipud      = cfg["flipud"],
        fliplr      = cfg["fliplr"],
        mosaic      = cfg["mosaic"],
        mixup       = cfg["mixup"],
        copy_paste  = cfg["copy_paste"],
        patience    = cfg["patience"],
        save_period = cfg["save_period"],
        val         = cfg["val"],
        plots       = cfg["plots"],
        project     = cfg["project"],
        name        = cfg["name"],
        exist_ok    = resume,
    )

    # ── post-training summary ───────────────────────────────────────────────
    save_dir = Path(results.save_dir)
    best_pt  = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Results saved to : {save_dir}")
    print(f"  Best weights     : {best_pt}")

    # print final metrics if available
    try:
        box = results.results_dict
        print(f"\n  Final metrics:")
        print(f"    mAP50 (box)  : {box.get('metrics/mAP50(B)',   'N/A'):.4f}")
        print(f"    mAP50 (mask) : {box.get('metrics/mAP50(M)',   'N/A'):.4f}")
        print(f"    mAP50-95 (M) : {box.get('metrics/mAP50-95(M)','N/A'):.4f}")
    except Exception:
        pass

    print("\n  Next step → run inference.py with your trained weights:")
    print(f"    python inference.py --weights {best_pt}")
    print("=" * 55)

    return results


# ───────────────────────────── entry point ─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n-seg")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    args = parser.parse_args()

    overrides = {}
    if args.epochs: overrides["epochs"] = args.epochs
    if args.batch:  overrides["batch"]  = args.batch

    train(args.data, resume=args.resume, overrides=overrides)
