"""Create a crop-row guidance overlay from a YOLOv8 segmentation checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from row_guidance import draw_guidance, estimate_row_guidance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate an orchard-row centreline from a still image.")
    parser.add_argument("--source", required=True, help="Input image path.")
    parser.add_argument("--weights", required=True, help="YOLOv8 segmentation checkpoint.")
    parser.add_argument("--output", default="outputs/guidance.jpg", help="Annotated output path.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--soil-class", type=int, default=2, help="Class ID used for soil.")
    return parser.parse_args()


def merged_soil_mask(result, soil_class: int) -> np.ndarray | None:
    """Merge all masks predicted as soil into original image coordinates."""
    if result.masks is None or result.boxes is None:
        return None
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks = result.masks.data.detach().cpu().numpy()
    selected = masks[classes == soil_class]
    if not len(selected):
        return None
    mask = np.any(selected > 0.5, axis=0).astype(np.uint8)
    height, width = result.orig_img.shape[:2]
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def main() -> None:
    args = parse_args()
    source, weights, output = Path(args.source), Path(args.weights), Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Source not found: {source}")
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    result = YOLO(weights).predict(str(source), conf=args.confidence, verbose=False)[0]
    mask = merged_soil_mask(result, args.soil_class)
    if mask is None:
        raise RuntimeError("The model did not produce a soil mask for this image.")
    guidance = estimate_row_guidance(mask)
    if guidance is None:
        raise RuntimeError("A stable crop-row path could not be estimated from this soil mask.")

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), draw_guidance(result.orig_img, guidance)):
        raise RuntimeError(f"Could not write {output}")
    print(f"Saved {output}")
    print(f"Image-space lateral error: {guidance.lateral_error:+.3f}")


if __name__ == "__main__":
    main()