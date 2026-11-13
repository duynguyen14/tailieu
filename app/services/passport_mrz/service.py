from __future__ import annotations

import base64
import binascii
import logging
import threading
import time

import cv2
import numpy as np

from app.core.config import Settings
from app.schemas.production_extraction import (
    PassportMrzBatchRequest,
    PassportMrzBatchResponse,
    PassportMrzReadinessResponse,
)
from app.services.passport_mrz.detector import OnnxMrzDetector
from app.services.passport_mrz.pipeline import PassportMrzPipeline
from app.services.passport_mrz.recognizer import PaddleMrzRecognizer

LOGGER = logging.getLogger(__name__)


class PassportMrzService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._loading = False
        self._ready = False
        self._error: str | None = None
        self._detector: OnnxMrzDetector | None = None
        self._recognizer: PaddleMrzRecognizer | None = None
        self._pipeline: PassportMrzPipeline | None = None

    def initialize(self) -> None:
        if not self.settings.passport_mrz_service_enabled or self._ready:
            return
        self._loading = True
        self._error = None
        try:
            self._detector = OnnxMrzDetector(self.settings)
            self._recognizer = PaddleMrzRecognizer(self.settings)
            self._pipeline = PassportMrzPipeline(
                self._detector,
                self._recognizer,
                max_workers=self.settings.passport_mrz_detector_max_workers,
            )
            if self.settings.passport_mrz_warmup_models:
                detector_ms = self._detector.warmup()
                recognizer_ms = self._recognizer.warmup()
                LOGGER.info(
                    "Passport MRZ warmup completed detector_ms=%.1f recognizer_ms=%.1f",
                    detector_ms,
                    recognizer_ms,
                )
            self._ready = True
            LOGGER.info(
                "Passport MRZ runtime ready detector=%s recognizer=%s",
                self._detector.device,
                self._recognizer.effective_device,
            )
        except Exception as exc:
            self._error = str(exc)
            raise
        finally:
            self._loading = False

    def readiness(self) -> PassportMrzReadinessResponse:
        return PassportMrzReadinessResponse(
            enabled=self.settings.passport_mrz_service_enabled,
            ready=self._ready,
            loading=self._loading,
            detector_model=(
                str(self._detector.model_path) if self._detector is not None else None
            ),
            recognizer_model=(
                str(self._recognizer.model_dir) if self._recognizer is not None else None
            ),
            detector_device=(self._detector.device if self._detector is not None else None),
            recognizer_device=(
                self._recognizer.effective_device
                if self._recognizer is not None
                else None
            ),
            error=self._error,
        )

    def read(self, request: PassportMrzBatchRequest) -> PassportMrzBatchResponse:
        if not self._ready or self._pipeline is None:
            raise RuntimeError(self._error or "Passport MRZ models are not ready")
        if not request.pages:
            return PassportMrzBatchResponse(request_id=request.request_id, pages=[], total_ms=0)
        if len(request.pages) > self.settings.passport_mrz_max_batch_pages:
            raise ValueError(
                f"Passport MRZ batch exceeds {self.settings.passport_mrz_max_batch_pages} pages"
            )
        if not self._lock.acquire(blocking=False):
            raise TimeoutError("Passport MRZ worker is busy")
        started = time.perf_counter()
        try:
            decoded = [
                (page.page_number, _decode_image(page.image_base64))
                for page in request.pages
            ]
            results = self._pipeline.read_batch(decoded)
            return PassportMrzBatchResponse(
                request_id=request.request_id,
                pages=results,
                total_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        finally:
            self._lock.release()

    def close(self) -> None:
        self._ready = False
        self._pipeline = None
        self._recognizer = None
        self._detector = None


def _decode_image(value: str) -> np.ndarray:
    encoded = (
        value.split(",", 1)[1]
        if value.lower().startswith("data:") and "," in value
        else value
    )
    try:
        image_bytes = base64.b64decode("".join(encoded.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid passport page base64 image") from exc
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Passport page image could not be decoded")
    return image

