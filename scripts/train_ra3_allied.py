"""Train an experimental YOLOv8 model on the Allied-only RA3 dataset.

This script does not deploy the model. It writes a separate training run
named ``ra3_allied`` so the result can be evaluated before touching
``config/perception.py`` or ``data/models``.

Run:
    M:/ra3_bot/venv/Scripts/python.exe scripts/train_ra3_allied.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_DATA = Path("data/datasets/ra3_allied/dataset.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Ultralytics dataset.yaml path")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="GPU index like 0, or cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/yolo_train")
    parser.add_argument("--name", default="ra3_allied")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,

        # Keep V2/V3 color robustness. Allied blue/cyan still benefits from
        # full hue rotation when maps, lighting, and team colors shift.
        hsv_h=0.5,
        hsv_s=0.9,
        hsv_v=0.6,

        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.15,

        degrees=8.0,
        translate=0.2,
        scale=0.5,
    )


if __name__ == "__main__":
    main()
