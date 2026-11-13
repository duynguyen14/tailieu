from __future__ import annotations

import base64
import logging
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from app.core.config import DevicePreference, Settings
from app.domain.types import RenderedPage
from app.schemas.production_extraction import (
    PassportMrzBatchRequest,
    PassportMrzBatchResponse,
    PassportMrzPageInput,
    PassportMrzPageResult,
    PassportMrzReadinessResponse,
    ProductionModelStatus,
    ProductionOcrResponse,
    ProductionOcrTiming,
    ProductionPageClassificationPreview,
    ProductionPageClassificationResponse,
    ProductionPageClassificationTiming,
    ProductionPageClassScore,
    ProductionPageResult,
    ProductionReadinessResponse,
)
from app.services.exceptions import ProductionExtractionError
from app.services.pdf_renderer import PdfRenderer
from app.services.text_reconstruction import (
    PageReconstructionInput,
    ReconstructionToken,
    reconstruct_case,
)

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image

    from app.models.ocr_preprocessing import ContentDetection
    from app.models.onnx_page_classifier import OnnxPageClassifier
    from app.models.paddle_ocr_engine import OcrTokenPrediction, PaddleOcrEngine
    from app.models.paddle_orientation import PaddleOrientationClassifier


@dataclass
class _PreparedPage:
    page_number: int
    image: Image.Image
    image_bytes: bytes | None
    width: int
    height: int
    rotation_angle: int
    orientation_confidence: float
    content_detection: ContentDetection | None = None

    def close(self) -> None:
        self.image.close()


@dataclass
class _OcrPage:
    prepared: _PreparedPage
    page_type: str
    tokens: list[OcrTokenPrediction]


class ProductionOcrService:
    """Private single-flight PDF -> reconstructed text service.

    This process owns ONNX classification and Paddle orientation/OCR only. It
    deliberately never imports Torch, Transformers, the student model, or seal
    detection, allowing it to run in a Paddle-specific virtual environment.
    """

    def __init__(self, settings: Settings, renderer: PdfRenderer) -> None:
        self.settings = settings
        self.renderer = renderer
        self._lock = threading.Lock()
        self._loading = False
        self._ready = False
        self._error: str | None = None
        self._classifier: OnnxPageClassifier | None = None
        self._orientation: PaddleOrientationClassifier | None = None
        self._ocr: PaddleOcrEngine | None = None
        self._passport_client: httpx.Client | None = None
        self._passport_executor: ThreadPoolExecutor | None = None
        self._models: list[ProductionModelStatus] = []

    def initialize(self) -> None:
        if not self.settings.production_ocr_service_enabled or self._ready:
            return
        self._loading = True
        self._error = None
        try:
            from app.models.onnx_page_classifier import OnnxPageClassifier
            from app.models.paddle_ocr_engine import PaddleOcrEngine
            from app.models.paddle_orientation import PaddleOrientationClassifier

            model_path = self.settings.page_classifier_model_resolved_path
            labels_path = self.settings.page_classifier_labels_resolved_path
            if model_path is None or labels_path is None:
                raise RuntimeError(
                    "PAGE_CLASSIFIER_MODEL_PATH and PAGE_CLASSIFIER_LABELS_PATH are required."
                )
            classifier_device = self.settings.production_ocr_classifier_device
            self._classifier = OnnxPageClassifier(
                model_path=model_path,
                labels_path=labels_path,
                device=classifier_device,
                gpu_device_id=self.settings.ocr_runtime_gpu_device_id,
                allow_cpu_fallback=classifier_device != DevicePreference.CUDA,
            )
            self._ocr = PaddleOcrEngine(
                self.settings,
                device_override=DevicePreference.CUDA,
            )
            self._orientation = PaddleOrientationClassifier(
                self.settings,
                device=self._ocr.effective_device,
            )
            paddle_devices = {
                self._ocr.effective_device,
                self._orientation.effective_device,
            }
            if self.settings.production_require_gpu and any(
                device == "cpu" for device in paddle_devices
            ):
                raise RuntimeError(
                    f"A Paddle production model fell back to CPU: {sorted(paddle_devices)}"
                )
            self._models = [
                ProductionModelStatus(
                    name="page_classifier",
                    loaded=True,
                    device=self._classifier.effective_device,
                    model=str(model_path),
                ),
                ProductionModelStatus(
                    name="orientation",
                    loaded=True,
                    device=self._orientation.effective_device,
                    model=self._orientation.model_name,
                ),
                ProductionModelStatus(
                    name="ocr_detection_recognition",
                    loaded=True,
                    device=self._ocr.effective_device,
                    model=self._ocr.model_name,
                    message="Seal/layout exclusion model is not loaded.",
                ),
            ]
            if self.settings.production_passport_service_enabled:
                timeout_seconds = (
                    self.settings.production_passport_service_timeout_seconds
                )
                self._passport_client = httpx.Client(
                    timeout=httpx.Timeout(
                        timeout_seconds,
                        connect=min(1.0, timeout_seconds),
                    )
                )
                self._passport_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="passport-mrz",
                )
                self._models.append(
                    ProductionModelStatus(
                        name="passport_mrz",
                        loaded=True,
                        device="remote",
                        model=self.settings.production_passport_service_base_url,
                        message="Called only for PASSPORT_FRONT pages; timeout is 5 seconds.",
                    )
                )
            if self.settings.production_warmup_models:
                self._warmup()
            self._ready = True
            LOGGER.info(
                "Production OCR runtime is ready; classifier=%s paddle=%s",
                self._classifier.effective_device,
                sorted(paddle_devices),
            )
        except Exception as exc:
            self._error = str(exc)
            self._models = []
            raise
        finally:
            self._loading = False

    def _warmup(self) -> None:
        from PIL import Image

        assert self._classifier is not None
        assert self._orientation is not None
        assert self._ocr is not None
        image = Image.new("RGB", (640, 640), "white")
        try:
            self._classifier.predict_images([image], top_k=1)
            self._orientation.predict_image(image)
            self._ocr.predict_image(image)
        finally:
            image.close()

    def readiness(self) -> ProductionReadinessResponse:
        models = list(self._models)
        ready = self._ready
        error = self._error
        if (
            self.settings.production_passport_service_enabled
            and self._passport_client is not None
        ):
            try:
                response = self._passport_client.get(
                    f"{self.settings.production_passport_service_base_url}/readiness",
                    timeout=min(
                        1.0,
                        self.settings.production_passport_service_timeout_seconds,
                    ),
                )
                response.raise_for_status()
                remote = PassportMrzReadinessResponse.model_validate(response.json())
                ready = ready and remote.enabled and remote.ready
                models = [item for item in models if item.name != "passport_mrz"]
                models.append(
                    ProductionModelStatus(
                        name="passport_mrz",
                        loaded=remote.ready,
                        device=remote.recognizer_device,
                        model=remote.recognizer_model,
                        message=remote.error,
                    )
                )
                if not remote.ready:
                    error = remote.error or "Passport MRZ service is not ready."
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                ready = False
                error = f"Passport MRZ service is unavailable: {exc}"
        return ProductionReadinessResponse(
            enabled=self.settings.production_ocr_service_enabled,
            ready=ready,
            loading=self._loading,
            seal_detection_enabled=False,
            models=models,
            error=error,
        )

    def close(self) -> None:
        self._ready = False
        if self._passport_client is not None:
            self._passport_client.close()
            self._passport_client = None
        if self._passport_executor is not None:
            self._passport_executor.shutdown(wait=False, cancel_futures=True)
            self._passport_executor = None

    def _submit_passport_pages(
        self,
        *,
        pages: list[_PreparedPage],
        request_id: str,
    ) -> Future[PassportMrzBatchResponse] | None:
        if (
            not pages
            or self._passport_client is None
            or self._passport_executor is None
        ):
            return None
        from app.models.ocr_preprocessing import png_bytes

        payload = PassportMrzBatchRequest(
            request_id=request_id,
            pages=[
                PassportMrzPageInput(
                    page_number=page.page_number,
                    image_base64=base64.b64encode(
                        page.image_bytes or png_bytes(page.image)
                    ).decode("ascii"),
                )
                for page in pages
            ],
        )
        return self._passport_executor.submit(self._read_passport_pages, payload)

    def _read_passport_pages(
        self, payload: PassportMrzBatchRequest
    ) -> PassportMrzBatchResponse:
        if self._passport_client is None:
            raise RuntimeError("Passport MRZ HTTP client is not initialized.")
        response = self._passport_client.post(
            f"{self.settings.production_passport_service_base_url}/read",
            json=payload.model_dump(mode="json"),
            timeout=self.settings.production_passport_service_timeout_seconds,
        )
        response.raise_for_status()
        return PassportMrzBatchResponse.model_validate(response.json())

    def reconstruct(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> ProductionOcrResponse:
        if not self._ready:
            raise ProductionExtractionError(
                self._error or "Production OCR models are not loaded yet."
            )
        if not self._lock.acquire(blocking=False):
            raise ProductionExtractionError(
                "The OCR GPU worker is busy. Retry this request shortly."
            )
        try:
            LOGGER.info(
                "OCR pipeline started filename=%r sha256=%s",
                filename,
                sha256[:12],
            )
            result = self._reconstruct_locked(
                pdf_path=pdf_path,
                filename=filename,
                sha256=sha256,
            )
            LOGGER.info(
                "OCR pipeline completed filename=%r pages=%d render_ms=%.1f "
                "orientation_ms=%.1f classification_ms=%.1f ocr_ms=%.1f "
                "reconstruction_ms=%.1f total_ms=%.1f",
                filename,
                result.page_count,
                result.timings.render_ms,
                result.timings.orientation_ms,
                result.timings.classification_ms,
                result.timings.ocr_ms,
                result.timings.reconstruction_ms,
                result.timings.total_ms,
            )
            return result
        except Exception:
            LOGGER.exception(
                "OCR pipeline failed filename=%r sha256=%s",
                filename,
                sha256[:12],
            )
            raise
        finally:
            self._lock.release()

    def preview_page_classifications(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> ProductionPageClassificationResponse:
        if not self._ready:
            raise ProductionExtractionError(
                self._error or "Production OCR models are not loaded yet."
            )
        if not self._lock.acquire(blocking=False):
            raise ProductionExtractionError(
                "The OCR GPU worker is busy. Retry this request shortly."
            )
        try:
            return self._preview_page_classifications_locked(
                pdf_path=pdf_path,
                filename=filename,
                sha256=sha256,
            )
        finally:
            self._lock.release()

    def _preview_page_classifications_locked(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> ProductionPageClassificationResponse:
        from app.models.ocr_preprocessing import png_bytes

        assert self._classifier is not None
        assert self._orientation is not None
        total_started = time.perf_counter()

        render_started = time.perf_counter()
        rendered = self.renderer.render(pdf_path)
        render_ms = (time.perf_counter() - render_started) * 1000
        LOGGER.info(
            "OCR stage render filename=%r pages=%d elapsed_ms=%.1f",
            filename,
            len(rendered),
            render_ms,
        )

        prepared_pages: list[_PreparedPage] = []
        try:
            orientation_started = time.perf_counter()
            prepared_pages = self._orient_pages(rendered, include_png_bytes=True)
            orientation_ms = (time.perf_counter() - orientation_started) * 1000
            LOGGER.info(
                "OCR stage orientation filename=%r pages=%d elapsed_ms=%.1f rotations=%s",
                filename,
                len(prepared_pages),
                orientation_ms,
                [page.rotation_angle for page in prepared_pages],
            )

            classification_started = time.perf_counter()
            predictions = self._classifier.predict_images(
                [page.image for page in prepared_pages], top_k=3
            )
            classification_ms = (time.perf_counter() - classification_started) * 1000
            LOGGER.info(
                "Classification preview filename=%r pages=%d elapsed_ms=%.1f predictions=%s",
                filename,
                len(predictions),
                classification_ms,
                [
                    f"p{page.page_number}:{prediction.label.value}:{prediction.confidence:.3f}"
                    for page, prediction in zip(prepared_pages, predictions, strict=True)
                ],
            )

            target_types = set(self.settings.production_ocr_page_type_list)
            pages = [
                ProductionPageClassificationPreview(
                    page_number=page.page_number,
                    image_base64=base64.b64encode(
                        page.image_bytes or png_bytes(page.image)
                    ).decode("ascii"),
                    width=page.width,
                    height=page.height,
                    rotation_angle=page.rotation_angle,
                    orientation_confidence=page.orientation_confidence,
                    predicted_type=prediction.label.value,
                    confidence=prediction.confidence,
                    top_predictions=[
                        ProductionPageClassScore(
                            label=item.label.value,
                            confidence=item.confidence,
                        )
                        for item in prediction.top_predictions
                    ],
                    is_extraction_target=prediction.label.value in target_types,
                )
                for page, prediction in zip(prepared_pages, predictions, strict=True)
            ]
        finally:
            for page in prepared_pages:
                page.close()

        total_ms = (time.perf_counter() - total_started) * 1000
        return ProductionPageClassificationResponse(
            filename=filename,
            sha256=sha256,
            page_count=len(rendered),
            target_page_types=self.settings.production_ocr_page_type_list,
            pages=pages,
            timings=ProductionPageClassificationTiming(
                render_ms=round(render_ms, 3),
                orientation_ms=round(orientation_ms, 3),
                classification_ms=round(classification_ms, 3),
                total_ms=round(total_ms, 3),
            ),
        )

    def _reconstruct_locked(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> ProductionOcrResponse:
        assert self._classifier is not None
        assert self._orientation is not None
        assert self._ocr is not None
        total_started = time.perf_counter()
        warnings: list[str] = []
        passport_results: list[PassportMrzPageResult] = []
        passport_mrz_ms = 0.0
        passport_mrz_status = "NOT_REQUESTED"
        reconstructed_text = ""
        reconstruction_ms = 0.0
        page_results: list[ProductionPageResult] = []

        render_started = time.perf_counter()
        try:
            rendered = self.renderer.render(pdf_path, include_png_bytes=False)
        except TypeError:
            rendered = self.renderer.render(pdf_path)
        render_ms = (time.perf_counter() - render_started) * 1000
        LOGGER.info(
            "OCR stage render filename=%r pages=%d elapsed_ms=%.1f",
            filename,
            len(rendered),
            render_ms,
        )

        prepared_pages: list[_PreparedPage] = []
        try:
            orientation_started = time.perf_counter()
            prepared_pages = self._orient_pages(rendered, include_png_bytes=False)
            orientation_ms = (time.perf_counter() - orientation_started) * 1000
            LOGGER.info(
                "OCR stage orientation filename=%r pages=%d elapsed_ms=%.1f rotations=%s",
                filename,
                len(prepared_pages),
                orientation_ms,
                [page.rotation_angle for page in prepared_pages],
            )

            classification_started = time.perf_counter()
            predictions = self._classifier.predict_images(
                [page.image for page in prepared_pages], top_k=1
            )
            classification_ms = (time.perf_counter() - classification_started) * 1000
            LOGGER.info(
                "OCR stage classification filename=%r pages=%d elapsed_ms=%.1f predictions=%s",
                filename,
                len(predictions),
                classification_ms,
                [
                    f"p{page.page_number}:{prediction.label.value}:{prediction.confidence:.3f}"
                    for page, prediction in zip(prepared_pages, predictions, strict=True)
                ],
            )

            passport_types = set(self.settings.production_passport_page_type_list)
            passport_rows = [
                (page, prediction)
                for page, prediction in zip(prepared_pages, predictions, strict=True)
                if self.settings.production_passport_service_enabled
                and prediction.label.value in passport_types
            ]
            passport_started = time.perf_counter()
            passport_future = self._submit_passport_pages(
                pages=[page for page, _prediction in passport_rows],
                request_id=sha256,
            )
            if passport_rows and passport_future is None:
                passport_mrz_status = "DISABLED"
                warnings.append(
                    "PASSPORT_FRONT page(s) were found, but the passport MRZ service "
                    "is disabled."
                )
            elif passport_future is not None:
                passport_mrz_status = "RUNNING"

            ocr_started = time.perf_counter()
            ocr_pages: list[_OcrPage] = []
            target_types = set(self.settings.production_ocr_page_type_list)
            has_target_page = any(
                prediction.label.value in target_types for prediction in predictions
            )
            fallback_all_pages = (
                bool(prepared_pages) and not has_target_page and not passport_rows
            )
            if fallback_all_pages:
                fallback_msg = (
                    "No page matched configured target page types "
                    f"({', '.join(self.settings.production_ocr_page_type_list)}); "
                    f"falling back to OCR on all {len(prepared_pages)} page(s)."
                )
                warnings.append(fallback_msg)
                LOGGER.warning(
                    "OCR stage classification fallback filename=%r pages=%d message=%s",
                    filename,
                    len(prepared_pages),
                    fallback_msg,
                )
            target_rows = [
                (page, prediction)
                for page, prediction in zip(prepared_pages, predictions, strict=True)
                if fallback_all_pages or prediction.label.value in target_types
            ]
            batch_result = None
            if (
                target_rows
                and not self.settings.production_ocr_enable_small_text_upscale
                and not self.settings.production_ocr_enable_refinement
            ):
                from app.services.ocr_batch import recognize_pages_in_memory

                batch_result = recognize_pages_in_memory(
                    [row[0] for row in target_rows],
                    settings=self.settings,
                    ocr=self._ocr,
                )
                LOGGER.info(
                    "OCR stage in-memory batch filename=%r pages=%d inference_ms=%.1f",
                    filename,
                    len(target_rows),
                    batch_result.inference_time_ms,
                )
            batch_index_by_page = {
                page.page_number: index
                for index, (page, _prediction) in enumerate(target_rows)
            }
            for page, prediction in zip(prepared_pages, predictions, strict=True):
                page_type = prediction.label.value
                use_ocr = fallback_all_pages or (page_type in target_types)
                use_passport = (
                    self.settings.production_passport_service_enabled
                    and page_type in passport_types
                )
                use_page = use_ocr or use_passport
                tokens: list[OcrTokenPrediction] = []
                page_ocr_ms = 0.0
                if use_ocr:
                    if batch_result is not None:
                        batch_index = batch_index_by_page[page.page_number]
                        tokens = batch_result.tokens_by_page[batch_index]
                        page_ocr_ms = batch_result.processing_time_ms_by_page[batch_index]
                    else:
                        page_ocr_started = time.perf_counter()
                        tokens = self._recognize_page(page)
                        page_ocr_ms = (time.perf_counter() - page_ocr_started) * 1000
                    ocr_pages.append(
                        _OcrPage(prepared=page, page_type=page_type, tokens=tokens)
                    )
                    LOGGER.info(
                        "OCR stage page filename=%r page=%d type=%s target=%s "
                        "tokens=%d elapsed_ms=%.1f",
                        filename,
                        page.page_number,
                        page_type,
                        "fallback" if fallback_all_pages else "true",
                        len(tokens),
                        page_ocr_ms,
                    )
                else:
                    LOGGER.info(
                        "OCR stage page filename=%r page=%d type=%s target=%s "
                        "tokens=0 elapsed_ms=0",
                        filename,
                        page.page_number,
                        page_type,
                        "passport" if use_passport else "false",
                    )
                if prediction.confidence < self.settings.low_confidence_threshold:
                    warnings.append(
                        f"Page {page.page_number} classification confidence "
                        f"{prediction.confidence:.3f} is below "
                        f"{self.settings.low_confidence_threshold:.3f}."
                    )
                page_results.append(
                    ProductionPageResult(
                        page_number=page.page_number,
                        rotation_angle=page.rotation_angle,
                        orientation_confidence=page.orientation_confidence,
                        page_type=page_type,
                        classification_confidence=prediction.confidence,
                        used_for_extraction=use_page,
                        ocr_token_count=len(tokens),
                        ocr_processing_time_ms=round(page_ocr_ms, 3),
                    )
                )
            ocr_ms = (time.perf_counter() - ocr_started) * 1000
            LOGGER.info(
                "OCR stage recognition filename=%r target_pages=%d elapsed_ms=%.1f",
                filename,
                len(ocr_pages),
                ocr_ms,
            )

            if passport_future is not None:
                timeout_seconds = (
                    self.settings.production_passport_service_timeout_seconds
                )
                elapsed_seconds = time.perf_counter() - passport_started
                remaining_seconds = max(0.0, timeout_seconds - elapsed_seconds)
                try:
                    passport_response = passport_future.result(
                        timeout=remaining_seconds
                    )
                    passport_results = passport_response.pages
                    passport_mrz_status = "COMPLETED"
                except (TimeoutError, httpx.TimeoutException):
                    passport_mrz_status = "TIMEOUT"
                    passport_future.cancel()
                    warnings.append(
                        "Passport MRZ service exceeded the 5-second batch timeout; "
                        "the partial non-passport result was retained."
                    )
                    LOGGER.warning(
                        "Passport MRZ timed out filename=%r pages=%d timeout_seconds=%.1f",
                        filename,
                        len(passport_rows),
                        timeout_seconds,
                    )
                except (httpx.HTTPError, RuntimeError, TypeError, ValueError) as exc:
                    passport_mrz_status = "ERROR"
                    warnings.append(
                        f"Passport MRZ service failed; partial result retained: {exc}"
                    )
                    LOGGER.warning(
                        "Passport MRZ failed filename=%r pages=%d error=%s",
                        filename,
                        len(passport_rows),
                        exc,
                    )
                passport_mrz_ms = (time.perf_counter() - passport_started) * 1000
                LOGGER.info(
                    "Passport MRZ stage filename=%r pages=%d status=%s elapsed_ms=%.1f",
                    filename,
                    len(passport_rows),
                    passport_mrz_status,
                    passport_mrz_ms,
                )

            if not ocr_pages and not passport_rows:
                raise ProductionExtractionError(
                    "No page was classified as an OCR/extraction target."
                )
            if ocr_pages and not any(page.tokens for page in ocr_pages):
                raise ProductionExtractionError("OCR returned no usable text tokens.")

            if ocr_pages:
                reconstruction_started = time.perf_counter()
                reconstruction_inputs = [
                    PageReconstructionInput(
                        page_id=f"page_{page.prepared.page_number}",
                        page_number=page.prepared.page_number,
                        page_type=page.page_type,
                        page_role="NOTE",
                        tokens=tuple(
                            ReconstructionToken(
                                id=f"p{page.prepared.page_number}_{token.token_id}",
                                text=token.text,
                                bbox=token.bbox,
                                order=token.token_index,
                            )
                            for token in page.tokens
                        ),
                    )
                    for page in ocr_pages
                ]
                reconstructed_text, reconstruction_results = reconstruct_case(
                    reconstruction_inputs,
                    tolerance_ratio=self.settings.text_reconstruction_line_tolerance,
                )
                reconstruction_ms = (
                    time.perf_counter() - reconstruction_started
                ) * 1000
                LOGGER.info(
                    "OCR stage reconstruction filename=%r chars=%d elapsed_ms=%.1f",
                    filename,
                    len(reconstructed_text),
                    reconstruction_ms,
                )
                for result in reconstruction_results:
                    warnings.extend(result.warnings)
        finally:
            for page in prepared_pages:
                page.close()

        total_ms = (time.perf_counter() - total_started) * 1000
        return ProductionOcrResponse(
            filename=filename,
            sha256=sha256,
            page_count=len(rendered),
            reconstructed_text=reconstructed_text,
            passport_results=passport_results,
            pages=page_results,
            timings=ProductionOcrTiming(
                render_ms=round(render_ms, 3),
                orientation_ms=round(orientation_ms, 3),
                classification_ms=round(classification_ms, 3),
                ocr_ms=round(ocr_ms, 3),
                reconstruction_ms=round(reconstruction_ms, 3),
                passport_mrz_ms=round(passport_mrz_ms, 3),
                passport_mrz_status=passport_mrz_status,
                total_ms=round(total_ms, 3),
            ),
            warnings=list(dict.fromkeys(warnings)),
        )

    def _orient_pages(
        self,
        pages: list[RenderedPage],
        *,
        include_png_bytes: bool = False,
    ) -> list[_PreparedPage]:
        from app.models.ocr_preprocessing import (
            load_rgb_bytes,
            orientation_probe,
            png_bytes,
        )

        assert self._orientation is not None
        output: list[_PreparedPage] = []
        for page in pages:
            source = (
                page.pil_image.convert("RGB")
                if page.pil_image is not None
                else load_rgb_bytes(page.image_bytes)
            )
            probe, detection = orientation_probe(
                source,
                thumbnail_max_side=self.settings.ocr_content_thumbnail_max_side,
                padding_ratio=self.settings.ocr_content_padding_ratio,
                min_components=self.settings.ocr_content_min_components,
                min_content_area_ratio=self.settings.ocr_content_min_area_ratio,
                max_crop_area_ratio=self.settings.ocr_content_max_crop_area_ratio,
            )
            prediction = self._orientation.predict_image(probe)
            probe.close()
            if prediction.angle:
                rotated = source.rotate(prediction.angle, expand=True, fillcolor="white")
                source.close()
                encoded_bytes = png_bytes(rotated) if include_png_bytes else None
                cached_detection = None
            else:
                rotated = source
                encoded_bytes = (
                    (page.image_bytes or png_bytes(rotated))
                    if include_png_bytes
                    else None
                )
                cached_detection = detection
            width, height = rotated.size
            output.append(
                _PreparedPage(
                    page_number=page.page_number,
                    image=rotated,
                    image_bytes=encoded_bytes,
                    width=width,
                    height=height,
                    rotation_angle=prediction.angle,
                    orientation_confidence=prediction.confidence,
                    content_detection=cached_detection,
                )
            )
        return output

    def _recognize_page(self, page: _PreparedPage) -> list[OcrTokenPrediction]:
        from app.models.ocr_preprocessing import (
            ocr_quality_score,
            prepare_ocr_image,
            recommended_upscale,
        )
        from app.models.ocr_region_refinement import refine_ocr_tokens

        assert self._ocr is not None
        prepare_kwargs = {
            "rotation": 0,
            "enable_content_crop": self.settings.ocr_enable_content_crop,
            "thumbnail_max_side": self.settings.ocr_content_thumbnail_max_side,
            "padding_ratio": self.settings.ocr_content_padding_ratio,
            "min_components": self.settings.ocr_content_min_components,
            "min_content_area_ratio": self.settings.ocr_content_min_area_ratio,
            "max_crop_area_ratio": self.settings.ocr_content_max_crop_area_ratio,
            "max_processed_side": self.settings.ocr_upscale_max_processed_side,
            "precomputed_detection": page.content_detection,
        }
        selected = prepare_ocr_image(page.image, scale=1.0, **prepare_kwargs)
        tokens = self._ocr.predict_image(selected.image)
        selected_score = ocr_quality_score(tokens)
        if self.settings.production_ocr_enable_small_text_upscale:
            scale = recommended_upscale(
                tokens,
                target_text_height=self.settings.ocr_upscale_target_text_height,
                max_factor=self.settings.ocr_upscale_max_factor,
            )
            if scale > 1.05:
                upscaled = prepare_ocr_image(page.image, scale=scale, **prepare_kwargs)
                upscaled_tokens = self._ocr.predict_image(upscaled.image)
                if ocr_quality_score(upscaled_tokens) >= selected_score - 0.05:
                    selected.image.close()
                    selected = upscaled
                    tokens = upscaled_tokens
                else:
                    upscaled.image.close()
        if self.settings.production_ocr_enable_refinement and tokens:
            with tempfile.TemporaryDirectory(prefix="production-ocr-refine-") as refine_dir:
                refinement = refine_ocr_tokens(
                    selected.image,
                    tokens,
                    temp_dir=Path(refine_dir),
                    predict_many=self._ocr.predict_refinement_many,
                    scales=self.settings.ocr_refinement_scale_list,
                    confidence_threshold=self.settings.ocr_refinement_confidence_threshold,
                    candidate_min_confidence=self.settings.ocr_refinement_candidate_min_confidence,
                    agreement_similarity=self.settings.ocr_refinement_agreement_similarity,
                    min_confidence_gain=self.settings.ocr_refinement_min_confidence_gain,
                    max_regions=self.settings.ocr_refinement_max_regions,
                    max_seconds=self.settings.ocr_refinement_max_seconds,
                    padding_ratio=self.settings.ocr_refinement_padding_ratio,
                    max_crop_side=self.settings.ocr_refinement_max_crop_side,
                    protect_digits=self.settings.ocr_refinement_protect_digits,
                    version=self.settings.ocr_refinement_version,
                )
                tokens = refinement.tokens
        selected.image.close()
        return tokens
