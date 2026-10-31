from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.models.paddle_ocr_engine import prepare_paddlex_runtime


class LayoutModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class LayoutRegionPrediction:
    region_type: str
    confidence: float
    bbox: tuple[int, int, int, int]


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR returned an unsupported layout result object.")
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _rect_from_coordinate(
    coordinate: Any, *, width: int, height: int
) -> tuple[int, int, int, int] | None:
    if hasattr(coordinate, "tolist"):
        coordinate = coordinate.tolist()
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 4:
        return None
    if len(coordinate) == 4 and all(isinstance(value, (int, float)) for value in coordinate):
        xs = [float(coordinate[0]), float(coordinate[2])]
        ys = [float(coordinate[1]), float(coordinate[3])]
    else:
        points = [item.tolist() if hasattr(item, "tolist") else item for item in coordinate]
        pairs = [item for item in points if isinstance(item, (list, tuple)) and len(item) >= 2]
        if not pairs:
            return None
        xs = [float(item[0]) for item in pairs]
        ys = [float(item[1]) for item in pairs]
    x1 = max(0, min(width - 1, math.floor(min(xs))))
    y1 = max(0, min(height - 1, math.floor(min(ys))))
    x2 = max(1, min(width, math.ceil(max(xs))))
    y2 = max(1, min(height, math.ceil(max(ys))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class PaddleLayoutDetector:
    """Detect exclusion regions without loading the table-structure model."""

    def __init__(self, settings: Settings, *, device: str) -> None:
        self._settings = settings
        self.effective_device = device
        prepare_paddlex_runtime(settings)
        try:
            from paddleocr import LayoutDetection
        except ImportError as exc:
            raise LayoutModelUnavailableError(
                "PaddleOCR LayoutDetection is unavailable. Install requirements-ocr-gpu.txt."
            ) from exc

        model_dir = Settings.resolve_optional_project_path(settings.ocr_layout_model_dir)
        if model_dir is not None and not model_dir.is_dir():
            raise LayoutModelUnavailableError(
                f"Configured OCR layout model directory does not exist: {model_dir}"
            )
        try:
            self._model = LayoutDetection(
                model_name=settings.ocr_layout_model_name,
                model_dir=str(model_dir) if model_dir else None,
                threshold=settings.ocr_layout_threshold,
                device=device,
                enable_hpi=settings.ocr_enable_hpi,
                cpu_threads=settings.ocr_cpu_threads,
            )
        except Exception as exc:
            raise LayoutModelUnavailableError(
                f"Could not initialize {settings.ocr_layout_model_name}: {exc}"
            ) from exc

    @property
    def model_name(self) -> str:
        return self._settings.ocr_layout_model_name

    def predict(self, image_path: Path) -> list[LayoutRegionPrediction]:
        if not image_path.is_file():
            raise FileNotFoundError(f"Layout page image does not exist: {image_path}")
        from PIL import Image, ImageOps

        with Image.open(image_path) as source:
            width, height = ImageOps.exif_transpose(source).size
        output = self._model.predict(
            input=str(image_path), batch_size=1, threshold=self._settings.ocr_layout_threshold
        )
        try:
            payload = _result_payload(next(iter(output)))
        except StopIteration as exc:
            raise RuntimeError("PaddleOCR returned no layout result.") from exc

        allowed = set(self._settings.ocr_exclusion_region_type_list)
        regions: list[LayoutRegionPrediction] = []
        for item in payload.get("boxes", []):
            if not isinstance(item, dict):
                continue
            region_type = str(item.get("label", "")).strip().upper()
            confidence = float(item.get("score", 0.0))
            if region_type not in allowed or confidence < self._settings.ocr_layout_threshold:
                continue
            bbox = _rect_from_coordinate(item.get("coordinate"), width=width, height=height)
            if bbox is not None:
                regions.append(LayoutRegionPrediction(region_type, confidence, bbox))
        regions.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
        return regions


def token_overlap_ratio(
    token_bbox: tuple[int, int, int, int], region_bbox: tuple[int, int, int, int]
) -> float:
    tx1, ty1, tx2, ty2 = token_bbox
    rx1, ry1, rx2, ry2 = region_bbox
    intersection_width = max(0, min(tx2, rx2) - max(tx1, rx1))
    intersection_height = max(0, min(ty2, ry2) - max(ty1, ry1))
    token_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    return (intersection_width * intersection_height) / token_area


def token_exclusion_reason(
    token_bbox: tuple[int, int, int, int],
    regions: list[LayoutRegionPrediction],
    *,
    threshold: float,
) -> str | None:
    matches = [
        region
        for region in regions
        if token_overlap_ratio(token_bbox, region.bbox) >= threshold
    ]
    if not matches:
        return None
    best = max(matches, key=lambda region: token_overlap_ratio(token_bbox, region.bbox))
    return f"INSIDE_{best.region_type}"
