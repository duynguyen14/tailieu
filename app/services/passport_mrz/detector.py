from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import DevicePreference, Settings


@dataclass(frozen=True)
class MrzDetection:
    bbox: list[float]
    confidence: float
    rotation_angle: int


class OnnxMrzDetector:
    """Minimal YOLOv8/YOLO11 ONNX detector for a single MRZ class."""

    def __init__(self, settings: Settings) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required by the passport MRZ detector") from exc

        model_path = settings.passport_mrz_detector_model_resolved_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Passport MRZ detector was not found: {model_path}")
        available = set(ort.get_available_providers())
        requested = settings.passport_mrz_detector_device
        if requested == DevicePreference.CUDA:
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDAExecutionProvider is unavailable for passport MRZ ONNX")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device = "cuda"
        else:
            providers = ["CPUExecutionProvider"]
            self.device = "cpu"
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = min(4, max(1, settings.passport_mrz_ocr_cpu_threads))
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        started = time.perf_counter()
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )
        self.load_ms = (time.perf_counter() - started) * 1000
        self.model_path = model_path
        self.input_name = self.session.get_inputs()[0].name
        self.image_size = settings.passport_mrz_detector_image_size
        self.confidence = settings.passport_mrz_detector_confidence
        self.rotation_fallback = settings.passport_mrz_detector_rotation_fallback
        self.rotation_min_confidence = (
            settings.passport_mrz_detector_rotation_min_confidence
        )

    def warmup(self) -> float:
        started = time.perf_counter()
        self.detect(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        return (time.perf_counter() - started) * 1000

    def detect(self, image: np.ndarray) -> tuple[MrzDetection | None, float]:
        started = time.perf_counter()
        height, width = image.shape[:2]
        best = self._detect_once(image, 0, width, height)
        if self.rotation_fallback and best is None:
            for angle, candidate_image in (
                (90, np.ascontiguousarray(np.rot90(image, 1))),
                (180, np.ascontiguousarray(np.rot90(image, 2))),
                (270, np.ascontiguousarray(np.rot90(image, 3))),
            ):
                candidate = self._detect_once(candidate_image, angle, width, height)
                if candidate is not None and (
                    best is None or candidate.confidence > best.confidence
                ):
                    best = candidate
        return best, (time.perf_counter() - started) * 1000

    def _detect_once(
        self,
        image: np.ndarray,
        rotation_angle: int,
        original_width: int,
        original_height: int,
    ) -> MrzDetection | None:
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0]
        boxes, scores = self._decode_output(output)
        if boxes.size == 0:
            return None
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        rotated_height, rotated_width = image.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, rotated_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, rotated_height)
        nms_boxes = [
            [
                float(x1),
                float(y1),
                float(max(0.0, x2 - x1)),
                float(max(0.0, y2 - y1)),
            ]
            for x1, y1, x2, y2 in boxes.tolist()
        ]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, scores.tolist(), self.confidence, 0.45
        )
        if len(indices) == 0:
            return None
        index = int(np.asarray(indices).reshape(-1)[0])
        mapped = _map_rotated_bbox(
            boxes[index].tolist(),
            rotation_angle=rotation_angle,
            original_width=original_width,
            original_height=original_height,
        )
        return MrzDetection(mapped, float(scores[index]), rotation_angle)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.image_size - resized_width) // 2
        pad_y = (self.image_size - resized_height) // 2
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    def _decode_output(self, raw_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        output = np.asarray(raw_output)
        if output.ndim == 3:
            output = output[0]
        if output.ndim != 2:
            raise RuntimeError(f"Unsupported MRZ detector output shape: {raw_output.shape}")
        if output.shape[0] < output.shape[1] and output.shape[0] <= 128:
            output = output.T

        # End-to-end NMS exports commonly return x1,y1,x2,y2,score,class.
        if output.shape[1] == 6 and np.mean(output[:, 2] > output[:, 0]) > 0.5:
            scores = output[:, 4]
            keep = scores >= self.confidence
            return output[keep, :4].astype(np.float32), scores[keep].astype(np.float32)

        if output.shape[1] < 5:
            raise RuntimeError(f"Unsupported MRZ detector output shape: {raw_output.shape}")
        xywh = output[:, :4]
        class_scores = output[:, 4:]
        scores = class_scores.max(axis=1)
        keep = scores >= self.confidence
        xywh, scores = xywh[keep], scores[keep]
        if xywh.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        boxes = np.empty_like(xywh, dtype=np.float32)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        return boxes, scores.astype(np.float32)


def _map_rotated_bbox(
    bbox: list[float],
    *,
    rotation_angle: int,
    original_width: int,
    original_height: int,
) -> list[float]:
    x1, y1, x2, y2 = bbox
    points = [
        _map_rotated_point(
            x,
            y,
            rotation_angle=rotation_angle,
            original_width=original_width,
            original_height=original_height,
        )
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    ]
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    return [
        round(max(0.0, min(float(original_width), min(xs))), 2),
        round(max(0.0, min(float(original_height), min(ys))), 2),
        round(max(0.0, min(float(original_width), max(xs))), 2),
        round(max(0.0, min(float(original_height), max(ys))), 2),
    ]


def _map_rotated_point(
    x: float,
    y: float,
    *,
    rotation_angle: int,
    original_width: int,
    original_height: int,
) -> tuple[float, float]:
    if rotation_angle == 90:
        return float(original_width - y), float(x)
    if rotation_angle == 180:
        return float(original_width - x), float(original_height - y)
    if rotation_angle == 270:
        return float(y), float(original_height - x)
    return float(x), float(y)
