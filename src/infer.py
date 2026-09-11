"""Run MAR Rover scene segmentation on an image or a video.

The script intentionally keeps the prototype's four semantic classes visible:
plant, sky, soil, and trunk.  It writes annotated media to disk so it can run on
a headless computer as well as on a development laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 segmentation for MAR Rover scenes.")
    parser.add_argument("--source", required=True, help="Path to an image or video.")
    parser.add_argument("--weights", default="last.pt", help="Path to a trained YOLOv8 segmentation checkpoint.")
    parser.add_argument("--output", default="outputs", help="Directory for annotated results.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum confidence threshold.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth video frame.")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 2, 3],
        help="Class IDs to render (default: plant, soil, trunk).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    weights = Path(args.weights)

    if not source.is_file():
        raise FileNotFoundError(f"Source not found: {source}")
    if not weights.is_file():
        raise FileNotFoundError(
            f"Weights not found: {weights}. Place your trained checkpoint locally and pass --weights."
        )

    model = YOLO(weights)
    model.predict(
        source=str(source),
        conf=args.confidence,
        classes=args.classes,
        vid_stride=args.stride,
        save=True,
        project=args.output,
        name="segmentation",
        exist_ok=True,
        verbose=True,
    )
    print(f"Annotated result saved under {Path(args.output) / 'segmentation'}")


if __name__ == "__main__":
    main()
