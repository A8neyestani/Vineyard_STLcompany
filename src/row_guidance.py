"""Estimate an image-space centreline from the segmented soil region.

The result is a perception signal for the MAR Rover project. A vehicle-specific
controller is still responsible for calibration, safety, and actuator commands.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RowGuidance:
    """A path through the soil corridor in image coordinates."""

    centerline: tuple[tuple[int, int], ...]
    lookahead: tuple[int, int]
    lateral_error: float
    confidence: float


def _ground_connected_soil(mask: np.ndarray) -> np.ndarray:
    """Keep the largest soil component that reaches the lower half of the image."""
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    height = mask.shape[0]
    candidates = [
        label for label in range(1, labels_count)
        if stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] >= height * 0.55
    ]
    if not candidates:
        return np.zeros_like(mask, dtype=bool)
    selected = max(candidates, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    return labels == selected


def _row_center(row: np.ndarray, previous_x: float | None) -> float | None:
    """Find the most plausible continuous soil interval in one scan line."""
    edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
    runs = list(zip(edges[::2], edges[1::2]))
    if not runs:
        return None

    def score(run: tuple[int, int]) -> tuple[float, float]:
        left, right = run
        centre = (left + right - 1) / 2
        width = right - left
        distance = 0 if previous_x is None else abs(centre - previous_x)
        return width - 0.7 * distance, width

    left, right = max(runs, key=score)
    return (left + right - 1) / 2


def estimate_row_guidance(
    soil_mask: np.ndarray,
    *,
    lookahead_ratio: float = 0.72,
    samples: int = 9,
) -> RowGuidance | None:
    """Estimate a centreline and image-space lateral offset from a soil mask.

    ``lateral_error`` is normalised around the camera centre: negative is left,
    positive is right. It is intentionally not a motor or steering command.
    """
    if soil_mask.ndim != 2 or soil_mask.size == 0:
        raise ValueError("soil_mask must be a non-empty, two-dimensional array")
    if not 0.5 <= lookahead_ratio <= 0.95:
        raise ValueError("lookahead_ratio must be between 0.5 and 0.95")

    height, width = soil_mask.shape
    cleaned = cv2.morphologyEx(
        soil_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    ).astype(bool)
    corridor = _ground_connected_soil(cleaned)
    if not corridor.any():
        return None

    scan_lines = np.linspace(int(height * 0.48), int(height * 0.92), samples, dtype=int)[::-1]
    path: list[tuple[int, int]] = []
    previous_x: float | None = None
    for y in scan_lines:
        centre_x = _row_center(corridor[y], previous_x)
        if centre_x is not None:
            previous_x = centre_x
            path.append((round(centre_x), int(y)))

    if len(path) < max(3, samples // 2):
        return None

    lookahead = min(path, key=lambda point: abs(point[1] - round(height * lookahead_ratio)))
    camera_centre = (width - 1) / 2
    return RowGuidance(
        centerline=tuple(reversed(path)),
        lookahead=lookahead,
        lateral_error=float((lookahead[0] - camera_centre) / max(camera_centre, 1)),
        confidence=len(path) / samples,
    )


def draw_guidance(image: np.ndarray, guidance: RowGuidance) -> np.ndarray:
    """Return a copy of the image with the visual route overlaid."""
    annotated = image.copy()
    points = np.array(guidance.centerline, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [points], False, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.circle(annotated, guidance.lookahead, 8, (255, 80, 0), -1, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"lateral error: {guidance.lateral_error:+.2f}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return annotated