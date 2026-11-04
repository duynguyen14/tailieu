from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from app.core.config import DevicePreference, Settings


class OcrModelUnavailableError(RuntimeError):
    pass


def prepare_paddlex_runtime(settings: Settings) -> None:
    """Prevent PaddleX's optional ModelScope downloader from importing Torch.

    Paddle and Torch GPU can bind incompatible cuDNN DLLs on Windows. OCR uses
    Paddle's BOS model host, so ModelScope is unnecessary even though PaddleX
    3.7 imports it unconditionally.
    """

    source = settings.paddle_pdx_model_source.strip() or "BOS"
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = source
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = (
        "True" if settings.paddle_pdx_disable_model_source_check else "False"
    )
    if source.casefold() != "bos" or "modelscope" in sys.modules:
        return

    stub = ModuleType("modelscope")
    stub.__path__ = []  # type: ignore[attr-defined]

    def unavailable_snapshot_download(*_args, **_kwargs):
        raise RuntimeError(
            "ModelScope is disabled for Paddle OCR; use PADDLE_PDX_MODEL_SOURCE=BOS."
        )

    stub.snapshot_download = unavailable_snapshot_download  # type: ignore[attr-defined]
    sys.modules["modelscope"] = stub


@dataclass(frozen=True)
class OcrTokenPrediction:
    token_id: str
    token_index: int
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    polygon: list[list[int]]


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR returned an unsupported result object.")
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _bbox_from_polygon(polygon: Any) -> tuple[int, int, int, int] | None:
    points = _as_list(polygon)
    coordinates: list[tuple[float, float]] = []
    for point in points:
        pair = _as_list(point)
        if len(pair) >= 2:
            coordinates.append((float(pair[0]), float(pair[1])))
    if not coordinates:
        return None
    xs = [item[0] for item in coordinates]
    ys = [item[1] for item in coordinates]
    x1, y1 = math.floor(min(xs)), math.floor(min(ys))
    x2, y2 = math.ceil(max(xs)), math.ceil(max(ys))
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_ocr_tokens(
    payload: dict[str, Any], *, min_confidence: float
) -> list[OcrTokenPrediction]:
    texts = _as_list(payload.get("rec_texts"))
    scores = _as_list(payload.get("rec_scores"))
    boxes = _as_list(payload.get("rec_boxes"))
    polygons = _as_list(payload.get("rec_polys"))
    candidates: list[tuple[str, float, tuple[int, int, int, int], list[list[int]]]] = []

    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        confidence = float(scores[index]) if index < len(scores) else 0.0
        if not text or confidence < min_confidence:
            continue
        polygon_value = polygons[index] if index < len(polygons) else []
        polygon = [
            [int(round(float(pair[0]))), int(round(float(pair[1])))]
            for point in _as_list(polygon_value)
            if len(pair := _as_list(point)) >= 2
        ]
        bbox: tuple[int, int, int, int] | None = None
        if index < len(boxes):
            raw_box = _as_list(boxes[index])
            if len(raw_box) >= 4:
                bbox = tuple(int(round(float(value))) for value in raw_box[:4])  # type: ignore[assignment]
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    bbox = None
        # Calculate the fallback box from the original floating-point polygon
        # so floor/ceil still encloses every source pixel.
        bbox = bbox or _bbox_from_polygon(polygon_value)
        if bbox is not None:
            candidates.append((text, confidence, bbox, polygon))

    # Stable geometry order works for paragraphs and preserves left-to-right
    # cells on the same table row. The model still receives every source bbox.
    candidates.sort(key=lambda item: (item[2][1], item[2][0], item[2][3]))
    return [
        OcrTokenPrediction(
            token_id=f"t_{index:04d}",
            token_index=index,
            text=text,
            confidence=confidence,
            bbox=bbox,
            polygon=polygon,
        )
        for index, (text, confidence, bbox, polygon) in enumerate(candidates, start=1)
    ]


class PaddleOcrEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        device_override: DevicePreference | None = None,
    ) -> None:
        self._settings = settings
        requested = device_override or settings.ocr_runtime_device
        self.effective_device = self._resolve_device(requested)
        prepare_paddlex_runtime(settings)
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OcrModelUnavailableError(
                "PaddleOCR is not installed. Install requirements-ocr-gpu.txt "
                "for the GPU-first OCR pipeline."
            ) from exc

        detection_dir = Settings.resolve_optional_project_path(
            settings.ocr_detection_model_dir
        )
        recognition_dir = Settings.resolve_optional_project_path(
            settings.ocr_recognition_model_dir
        )
        for name, path in (("detection", detection_dir), ("recognition", recognition_dir)):
            if path is not None and not path.is_dir():
                raise OcrModelUnavailableError(
                    f"Configured OCR {name} model directory does not exist: {path}"
                )

        self._pipeline = PaddleOCR(
            text_detection_model_name=settings.ocr_detection_model_name,
            text_detection_model_dir=str(detection_dir) if detection_dir else None,
            text_recognition_model_name=settings.ocr_recognition_model_name,
            text_recognition_model_dir=str(recognition_dir) if recognition_dir else None,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.effective_device,
            enable_hpi=settings.ocr_enable_hpi,
            cpu_threads=settings.ocr_cpu_threads,
        )

    @property
    def model_name(self) -> str:
        return (
            f"{self._settings.ocr_detection_model_name}+"
            f"{self._settings.ocr_recognition_model_name}"
        )

    def predict(self, image_path: Path) -> list[OcrTokenPrediction]:
        if not image_path.is_file():
            raise FileNotFoundError(f"OCR page image does not exist: {image_path}")
        output = self._pipeline.predict(str(image_path))
        try:
            result = next(iter(output))
        except StopIteration as exc:
            raise RuntimeError("PaddleOCR returned no result.") from exc
        return parse_ocr_tokens(
            _result_payload(result),
            min_confidence=self._settings.ocr_min_confidence,
        )

    def predict_many(
        self, image_paths: list[Path]
    ) -> list[list[OcrTokenPrediction]]:
        if not image_paths:
            return []
        for image_path in image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(f"OCR page image does not exist: {image_path}")
        output = self._pipeline.predict([str(path) for path in image_paths])
        results = [
            parse_ocr_tokens(
                _result_payload(result),
                min_confidence=self._settings.ocr_min_confidence,
            )
            for result in output
        ]
        if len(results) != len(image_paths):
            raise RuntimeError(
                "PaddleOCR returned a different number of results "
                f"({len(results)}) than input images ({len(image_paths)})."
            )
        return results

    def predict_refinement_many(
        self, image_paths: list[Path]
    ) -> list[list[OcrTokenPrediction]]:
        return self.predict_many(image_paths)

    def _resolve_device(self, requested: DevicePreference) -> str:
        try:
            import paddle
        except ImportError as exc:
            raise OcrModelUnavailableError("PaddlePaddle is not installed.") from exc

        gpu_available = bool(
            paddle.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > self._settings.ocr_runtime_gpu_device_id
        )
        if requested in {DevicePreference.AUTO, DevicePreference.CUDA} and gpu_available:
            return f"gpu:{self._settings.ocr_runtime_gpu_device_id}"
        if not self._settings.ocr_runtime_allow_cpu_fallback:
            if requested == DevicePreference.CPU:
                raise OcrModelUnavailableError(
                    "CPU OCR is disabled by OCR_RUNTIME_ALLOW_CPU_FALLBACK=false."
                )
            raise OcrModelUnavailableError(
                "GPU OCR is required but CUDA-enabled Paddle is unavailable."
            )
        return "cpu"
