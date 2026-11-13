from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.core.config import DevicePreference, Settings
from app.services.passport_mrz.parser import normalize_mrz_text


class PaddleMrzRecognizer:
    def __init__(self, settings: Settings) -> None:
        try:
            import paddle.inference as paddle_infer
        except ImportError as exc:
            raise RuntimeError("paddlepaddle is required by the MRZ recognizer") from exc

        self.model_dir = settings.passport_mrz_ocr_model_resolved_path
        self.dict_path = settings.passport_mrz_ocr_dict_resolved_path
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Passport MRZ OCR model was not found: {self.model_dir}")
        if not self.dict_path.is_file():
            raise FileNotFoundError(f"Passport MRZ dictionary was not found: {self.dict_path}")
        model_file = _first_file(
            self.model_dir / "inference.json", self.model_dir / "inference.pdmodel"
        )
        params_file = self.model_dir / "inference.pdiparams"
        config_file = self.model_dir / "inference.yml"
        for path in (model_file, params_file, config_file):
            if not path.is_file():
                raise FileNotFoundError(f"Passport MRZ OCR model file was not found: {path}")

        inference_config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        self.image_shape = _read_image_shape(inference_config)
        self.characters = [
            character
            for line in self.dict_path.read_text(encoding="utf-8").splitlines()
            if (character := line.rstrip("\r\n"))
        ]
        if not self.characters:
            raise ValueError("Passport MRZ OCR dictionary is empty")
        self.device = settings.passport_mrz_ocr_device
        self.batch_size = settings.passport_mrz_ocr_batch_size
        self.min_confidence = settings.passport_mrz_ocr_min_confidence
        started = time.perf_counter()
        config = paddle_infer.Config(str(model_file), str(params_file))
        if self.device == DevicePreference.CUDA:
            config.enable_use_gpu(
                settings.passport_mrz_ocr_gpu_memory_mb,
                settings.passport_mrz_ocr_gpu_device_id,
            )
            self.effective_device = f"gpu:{settings.passport_mrz_ocr_gpu_device_id}"
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(settings.passport_mrz_ocr_cpu_threads)
            self.effective_device = "cpu"
        config.disable_glog_info()
        self.predictor = paddle_infer.create_predictor(config)
        self.input_name = self.predictor.get_input_names()[0]
        self.output_name = self.predictor.get_output_names()[0]
        self.load_ms = (time.perf_counter() - started) * 1000

    def warmup(self) -> float:
        image = np.full((self.image_shape[1], self.image_shape[2], 3), 255, dtype=np.uint8)
        started = time.perf_counter()
        self.recognize([image])
        return (time.perf_counter() - started) * 1000

    def recognize(self, images: list[np.ndarray]) -> tuple[list[dict[str, Any]], float]:
        if not images:
            return [], 0.0
        all_results: list[dict[str, Any]] = []
        started = time.perf_counter()
        for offset in range(0, len(images), self.batch_size):
            chunk = images[offset : offset + self.batch_size]
            batch = np.stack([_preprocess(image, self.image_shape) for image in chunk])
            input_handle = self.predictor.get_input_handle(self.input_name)
            input_handle.reshape(batch.shape)
            input_handle.copy_from_cpu(batch)
            self.predictor.run()
            logits = self.predictor.get_output_handle(self.output_name).copy_to_cpu()
            for text, confidence in _decode_ctc(logits, self.characters):
                normalized = normalize_mrz_text(text)
                all_results.append(
                    {
                        "text": text,
                        "normalized_text": normalized,
                        "confidence": round(confidence, 6),
                        "accepted": bool(normalized) and confidence >= self.min_confidence,
                    }
                )
        return all_results[: len(images)], (time.perf_counter() - started) * 1000


def _first_file(*paths: Path) -> Path:
    return next((path for path in paths if path.is_file()), paths[0])


def _read_image_shape(config: dict[str, Any]) -> tuple[int, int, int]:
    for transform in config.get("PreProcess", {}).get("transform_ops", []):
        if isinstance(transform, dict) and "RecResizeImg" in transform:
            values = transform["RecResizeImg"].get("image_shape", [3, 48, 640])
            if len(values) >= 3:
                return int(values[0]), int(values[1]), int(values[2])
    return 3, 48, 640


def _preprocess(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    channels, target_height, target_width = shape
    if channels != 3:
        raise ValueError(f"Passport MRZ recognizer expects 3 channels, got {channels}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    resized_width = min(target_width, max(1, round(target_height * width / max(1, height))))
    resized = cv2.resize(image, (resized_width, target_height), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    canvas = np.zeros((target_height, target_width, 3), dtype=np.float32)
    canvas[:, :resized_width] = normalized
    return canvas.transpose(2, 0, 1)


def _decode_ctc(logits: np.ndarray, characters: list[str]) -> list[tuple[str, float]]:
    predictions = np.asarray(logits)
    if predictions.ndim != 3:
        raise RuntimeError(f"MRZ OCR output must be [batch,time,class], got {predictions.shape}")
    class_sums = predictions.sum(axis=2)
    is_probability = (
        float(predictions.min()) >= 0.0
        and float(predictions.max()) <= 1.0
        and float(np.mean(np.abs(class_sums - 1.0))) < 1e-3
    )
    probabilities = predictions if is_probability else _softmax(predictions)
    indices = probabilities.argmax(axis=2)
    scores = probabilities.max(axis=2)
    decoded: list[tuple[str, float]] = []
    for row_indices, row_scores in zip(indices, scores, strict=False):
        parts: list[str] = []
        character_scores: list[float] = []
        previous = -1
        for class_index, score in zip(row_indices.tolist(), row_scores.tolist(), strict=False):
            if class_index != 0 and class_index != previous:
                dictionary_index = class_index - 1
                if 0 <= dictionary_index < len(characters):
                    parts.append(characters[dictionary_index])
                    character_scores.append(float(score))
            previous = class_index
        confidence = sum(character_scores) / len(character_scores) if character_scores else 0.0
        decoded.append(("".join(parts), confidence))
    return decoded


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=2, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(exponent.sum(axis=2, keepdims=True), 1e-12)
