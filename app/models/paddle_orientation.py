from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.config import Settings
from app.models.paddle_ocr_engine import prepare_paddlex_runtime


class OrientationModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrientationPrediction:
    angle: int
    confidence: float
    predicted_angle: int = 0


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR returned an unsupported orientation result object.")
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _first(value: Any) -> Any | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


class PaddleOrientationClassifier:
    def __init__(self, settings: Settings, *, device: str) -> None:
        self._settings = settings
        self.effective_device = device
        prepare_paddlex_runtime(settings)
        try:
            from paddleocr import DocImgOrientationClassification
        except ImportError as exc:
            raise OrientationModelUnavailableError(
                "PaddleOCR DocImgOrientationClassification is unavailable. "
                "Install requirements-ocr-gpu.txt."
            ) from exc

        model_dir = Settings.resolve_optional_project_path(
            settings.ocr_orientation_model_dir
        )
        if model_dir is not None and not model_dir.is_dir():
            raise OrientationModelUnavailableError(
                f"Configured orientation model directory does not exist: {model_dir}"
            )
        try:
            self._model = DocImgOrientationClassification(
                model_name=settings.ocr_orientation_model_name,
                model_dir=str(model_dir) if model_dir else None,
                device=device,
                enable_hpi=settings.ocr_enable_hpi,
                cpu_threads=settings.ocr_cpu_threads,
            )
        except Exception as exc:
            raise OrientationModelUnavailableError(
                f"Could not initialize {settings.ocr_orientation_model_name}: {exc}"
            ) from exc

    @property
    def model_name(self) -> str:
        return self._settings.ocr_orientation_model_name

    def predict(self, image_path: Path) -> OrientationPrediction:
        if not image_path.is_file():
            raise FileNotFoundError(f"Orientation input image does not exist: {image_path}")
        return self._predict_input(str(image_path))

    def predict_image(self, image: Any) -> OrientationPrediction:
        """Classify an in-memory RGB image used as a content-focused probe."""

        try:
            with TemporaryDirectory(prefix="orientation-probe-") as temp_dir:
                probe_path = Path(temp_dir) / "probe.png"
                image.convert("RGB").save(probe_path, format="PNG", optimize=True)
                return self._predict_input(str(probe_path))
        except Exception as exc:
            raise RuntimeError(f"Could not prepare orientation probe image: {exc}") from exc

    def _predict_input(self, value: Any) -> OrientationPrediction:
        output = self._model.predict(input=value, batch_size=1)
        try:
            payload = _result_payload(next(iter(output)))
        except StopIteration as exc:
            raise RuntimeError("PaddleOCR returned no orientation result.") from exc
        raw_label = _first(payload.get("label_names"))
        raw_score = _first(payload.get("scores"))
        try:
            predicted_angle = int(str(raw_label).replace("°", "").strip())
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Unsupported orientation label: {raw_label!r}") from exc
        if predicted_angle not in {0, 90, 180, 270}:
            raise RuntimeError(f"Unsupported orientation angle: {predicted_angle}")
        confidence = float(raw_score) if raw_score is not None else 0.0
        applied_angle = (
            predicted_angle
            if confidence >= self._settings.ocr_orientation_min_confidence
            else 0
        )
        return OrientationPrediction(
            angle=applied_angle,
            confidence=confidence,
            predicted_angle=predicted_angle,
        )


def normalized_png(
    image_path: Path, prediction: OrientationPrediction
) -> tuple[bytes, int, int]:
    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if prediction.angle:
            image = image.rotate(prediction.angle, expand=True, fillcolor="white")
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        width, height = image.size
    return buffer.getvalue(), width, height
