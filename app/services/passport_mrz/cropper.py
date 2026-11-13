from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MrzLineCrop:
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    projection_score: float


def clamp_box(
    box: list[float], width: int, height: int, padding_ratio: float = 0.08
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    pad_x = max(2.0, (right - left) * padding_ratio)
    pad_y = max(2.0, (bottom - top) * padding_ratio)
    return (
        max(0, int(round(left - pad_x))),
        max(0, int(round(top - pad_y))),
        min(width, int(round(right + pad_x))),
        min(height, int(round(bottom + pad_y))),
    )


def binarize_for_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _estimate_skew(binary: np.ndarray, max_angle: float = 15.0) -> float:
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=30,
        minLineLength=max(60, int(binary.shape[1] * 0.12)),
        maxLineGap=20,
    )
    if lines is not None:
        angles = [
            float(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
            for x1, y1, x2, y2 in lines.reshape(-1, 4)
            if abs(int(x2) - int(x1)) >= 30
        ]
        angles = [angle for angle in angles if abs(angle) <= max_angle]
        if len(angles) >= 3:
            return float(np.median(angles))

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    points: list[tuple[float, float]] = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if 3 <= width <= 45 and 6 <= height <= 55 and area >= 10:
            points.append((float(centroids[index][0]), float(centroids[index][1])))
    if len(points) < 8:
        return 0.0
    slope, _ = np.polyfit(
        np.asarray([point[0] for point in points], dtype=np.float32),
        np.asarray([point[1] for point in points], dtype=np.float32),
        1,
    )
    angle = float(np.degrees(np.arctan(float(slope))))
    return angle if abs(angle) <= max_angle else 0.0


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
        borderValue=(255, 255, 255),
    )


def _component_bands(
    binary: np.ndarray, expected_lines: int
) -> list[tuple[int, int, float]]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    image_height, image_width = binary.shape[:2]
    components: list[dict[str, float]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if (
            2 <= width <= min(70, image_width)
            and 6 <= height <= min(70, image_height)
            and area >= 8
        ):
            components.append(
                {
                    "x1": float(x),
                    "x2": float(x + width),
                    "y1": float(y),
                    "y2": float(y + height),
                    "area": float(area),
                    "height": float(height),
                    "cy": float(centroids[index][1]),
                }
            )
    if len(components) < expected_lines * 5:
        return []
    threshold = max(10.0, float(np.median([item["height"] for item in components])) * 1.4)
    groups: list[dict[str, Any]] = []
    for component in sorted(components, key=lambda item: item["cy"]):
        if not groups or abs(component["cy"] - groups[-1]["cy_mean"]) > threshold:
            groups.append({"items": [component], "cy_mean": component["cy"]})
        else:
            groups[-1]["items"].append(component)
            groups[-1]["cy_mean"] = float(
                np.mean([item["cy"] for item in groups[-1]["items"]])
            )
    candidates: list[tuple[float, int, int, float]] = []
    for group in groups:
        items = group["items"]
        if len(items) < 3:
            continue
        left, right = min(item["x1"] for item in items), max(item["x2"] for item in items)
        if right - left < image_width * 0.20:
            continue
        top = int(max(0, round(min(item["y1"] for item in items))))
        bottom = int(min(image_height, round(max(item["y2"] for item in items))))
        score = float((right - left) * len(items) + sum(item["area"] for item in items))
        candidates.append((score, top, bottom, score / max(1, bottom - top)))
    selected = sorted(candidates, reverse=True)[:expected_lines]
    return sorted([(top, bottom, score) for _, top, bottom, score in selected])


def _projection_bands(
    binary: np.ndarray, expected_lines: int
) -> list[tuple[int, int, float]]:
    bands = _component_bands(binary, expected_lines)
    if len(bands) >= expected_lines:
        return bands
    clean = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2)),
        iterations=1,
    )
    projection = clean.sum(axis=1).astype(np.float32)
    if projection.max() <= 0:
        return []
    active = np.where(projection >= projection.max() * 0.03)[0]
    if not active.size:
        return []
    top, bottom = int(active.min()), int(active.max()) + 1
    if bottom - top < expected_lines * 6:
        return []
    boundaries = np.linspace(top, bottom, expected_lines + 1, dtype=int)
    return [
        (
            int(boundaries[index]),
            int(boundaries[index + 1]),
            float(projection[boundaries[index] : boundaries[index + 1]].mean()),
        )
        for index in range(expected_lines)
    ]


def crop_mrz_lines(
    image: np.ndarray,
    bbox: list[float],
    *,
    expected_lines: int = 2,
    padding_ratio: float = 0.08,
) -> tuple[np.ndarray, list[MrzLineCrop], float]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(bbox, width, height, padding_ratio)
    raw_crop = image[y1:y2, x1:x2].copy()
    if raw_crop.size == 0:
        raise ValueError("MRZ crop is empty")
    angle = _estimate_skew(binarize_for_text(raw_crop))
    crop = _rotate(raw_crop, angle)
    binary = binarize_for_text(crop)
    bands = _projection_bands(binary, expected_lines)
    if len(bands) < expected_lines:
        raise ValueError(f"Expected {expected_lines} MRZ lines, detected {len(bands)}")
    pad_y = max(2, int(round(crop.shape[0] * 0.06)))
    line_crops: list[MrzLineCrop] = []
    for top, bottom, score in bands:
        line_top = max(0, top - pad_y)
        line_bottom = min(crop.shape[0], bottom + pad_y)
        line_image = crop[line_top:line_bottom, :].copy()
        if line_image.size:
            line_crops.append(
                MrzLineCrop(
                    image=line_image,
                    bbox=(0, line_top, crop.shape[1], line_bottom),
                    projection_score=score,
                )
            )
    return crop, line_crops, angle


def enhance_lines(lines: list[MrzLineCrop]) -> list[MrzLineCrop]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: list[MrzLineCrop] = []
    for line in lines:
        gray = cv2.cvtColor(line.image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        enhanced.append(
            MrzLineCrop(
                image=image,
                bbox=line.bbox,
                projection_score=line.projection_score,
            )
        )
    return enhanced
