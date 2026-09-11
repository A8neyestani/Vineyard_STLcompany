"""Fine-tune a YOLOv8 segmentation model for the MAR Rover dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 segmentation model.")
    parser.add_argument("--data", default="configs/stl2.yaml", help="Dataset YAML file.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Base segmentation model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="mar-rover-seg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = Path(args.data)
    if not data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
