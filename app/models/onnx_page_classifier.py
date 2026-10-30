from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from app.core.config import DevicePreference
from app.domain.types import RenderedPage
from app.models.interfaces import PageClassifier
from app.schemas.document import PageClassification, PageType


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: PageType
    confidence: float


@dataclass(frozen=True, slots=True)
class DetailedPagePrediction:
    label: PageType
    confidence: float
    top_predictions: tuple[ClassScore, ...]


class OnnxPageClassifier(PageClassifier):
    def __init__(
        self,
        *,
        model_path: Path,
        labels_path: Path,
        device: DevicePreference,
        gpu_device_id: int,
        allow_cpu_fallback: bool,
    ) -> None:
        if not model_path.is_file():
            raise RuntimeError(f"ONNX classifier model was not found: {model_path}")
        if not labels_path.is_file():
            raise RuntimeError(f"Classifier labels file was not found: {labels_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is not installed. Install requirements-inference.txt "
                "or the matching onnxruntime-gpu package."
            ) from exc

        metadata = json.loads(labels_path.read_text(encoding="utf-8-sig"))
        self._labels = tuple(PageType(label) for label in metadata["labels"])
        self._image_size = int(metadata["image_size"])
        preprocessing = metadata["preprocessing"]
        self._mean = np.asarray(preprocessing["mean"], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(preprocessing["std"], dtype=np.float32).reshape(3, 1, 1)
        self._input_name = str(metadata.get("input_name", "images"))
        self._output_name = str(metadata.get("output_name", "logits"))

        available = set(ort.get_available_providers())
        wants_cuda = device in {DevicePreference.CUDA, DevicePreference.AUTO}
        if wants_cuda and "CUDAExecutionProvider" in available:
            providers = [
                ("CUDAExecutionProvider", {"device_id": gpu_device_id}),
                "CPUExecutionProvider",
            ]
            self.effective_device = f"cuda:{gpu_device_id}"
        elif device == DevicePreference.CUDA and not allow_cpu_fallback:
            raise RuntimeError(
                "CUDA was requested but ONNX Runtime CUDAExecutionProvider is unavailable."
            )
        else:
            providers = ["CPUExecutionProvider"]
            self.effective_device = "cpu"
        self._session = ort.InferenceSession(str(model_path), providers=providers)

        input_names = {item.name for item in self._session.get_inputs()}
        output_names = {item.name for item in self._session.get_outputs()}
        if self._input_name not in input_names:
            raise RuntimeError(f"ONNX input {self._input_name!r} was not found in the model.")
        if self._output_name not in output_names:
            raise RuntimeError(f"ONNX output {self._output_name!r} was not found in the model.")

    def classify(self, pages: list[RenderedPage]) -> list[PageClassification]:
        detailed = self.predict_bytes([page.image_bytes for page in pages], top_k=1)
        return [
            PageClassification(page_type=item.label, confidence=item.confidence)
            for item in detailed
        ]

    def predict_paths(
        self, image_paths: Sequence[Path], *, top_k: int = 3
    ) -> list[DetailedPagePrediction]:
        tensors = [self._preprocess_path(path) for path in image_paths]
        return self._predict_tensors(tensors, top_k=top_k)

    def predict_bytes(
        self, images: Sequence[bytes], *, top_k: int = 3
    ) -> list[DetailedPagePrediction]:
        tensors = [self._preprocess_bytes(data) for data in images]
        return self._predict_tensors(tensors, top_k=top_k)

    def _preprocess_path(self, path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Classifier input image was not found: {path}")
        with Image.open(path) as image:
            return self._preprocess_image(ImageOps.exif_transpose(image).convert("RGB"))

    def _preprocess_bytes(self, data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as image:
            return self._preprocess_image(ImageOps.exif_transpose(image).convert("RGB"))

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        prepared = ImageOps.pad(
            image,
            (self._image_size, self._image_size),
            method=Image.Resampling.BICUBIC,
            color=255,
            centering=(0.5, 0.5),
        )
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        chw = np.transpose(array, (2, 0, 1))
        return (chw - self._mean) / self._std

    def _predict_tensors(
        self, tensors: Sequence[np.ndarray], *, top_k: int
    ) -> list[DetailedPagePrediction]:
        if not tensors:
            return []
        batch = np.stack(tensors).astype(np.float32, copy=False)
        logits = np.asarray(
            self._session.run([self._output_name], {self._input_name: batch})[0],
            dtype=np.float32,
        )
        if logits.ndim != 2 or logits.shape[1] != len(self._labels):
            raise RuntimeError(
                f"Unexpected classifier output shape {logits.shape}; "
                f"expected [batch, {len(self._labels)}]."
            )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        result: list[DetailedPagePrediction] = []
        count = min(max(1, top_k), len(self._labels))
        for row in probabilities:
            indexes = np.argsort(row)[::-1][:count]
            top = tuple(
                ClassScore(label=self._labels[index], confidence=float(row[index]))
                for index in indexes
            )
            result.append(
                DetailedPagePrediction(
                    label=top[0].label,
                    confidence=top[0].confidence,
                    top_predictions=top,
                )
            )
        return result
