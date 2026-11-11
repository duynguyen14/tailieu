from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL import Image

    from app.core.config import Settings
    from app.models.ocr_preprocessing import ContentDetection
    from app.models.paddle_ocr_engine import OcrTokenPrediction, PaddleOcrEngine


class InMemoryOcrPage(Protocol):
    image: Image.Image
    content_detection: ContentDetection | None


@dataclass(frozen=True)
class InMemoryBatchOcrResult:
    tokens_by_page: list[list[OcrTokenPrediction]]
    processing_time_ms_by_page: list[float]
    inference_time_ms: float


def recognize_pages_in_memory(
    pages: list[InMemoryOcrPage],
    *,
    settings: Settings,
    ocr: PaddleOcrEngine,
) -> InMemoryBatchOcrResult:
    """Prepare and recognize multiple pages without writing temporary images.

    PaddleOCR receives one list of BGR arrays. Per-page timings include each
    page's own preprocessing plus a pixel-weighted share of the shared batch
    inference time, so their sum remains meaningful in the review UI.
    """

    from app.models.ocr_preprocessing import prepare_ocr_image

    if not pages:
        return InMemoryBatchOcrResult([], [], 0.0)

    prepared = []
    preparation_times_ms: list[float] = []
    try:
        for page in pages:
            started = time.perf_counter()
            prepared.append(
                prepare_ocr_image(
                    page.image,
                    scale=1.0,
                    rotation=0,
                    enable_content_crop=settings.ocr_enable_content_crop,
                    thumbnail_max_side=settings.ocr_content_thumbnail_max_side,
                    padding_ratio=settings.ocr_content_padding_ratio,
                    min_components=settings.ocr_content_min_components,
                    min_content_area_ratio=settings.ocr_content_min_area_ratio,
                    max_crop_area_ratio=settings.ocr_content_max_crop_area_ratio,
                    max_processed_side=settings.ocr_upscale_max_processed_side,
                    precomputed_detection=page.content_detection,
                )
            )
            preparation_times_ms.append((time.perf_counter() - started) * 1000)

        inference_started = time.perf_counter()
        predict_images = getattr(ocr, "predict_images", None)
        if callable(predict_images):
            tokens_by_page = predict_images([item.image for item in prepared])
        else:
            # Test doubles and older engines remain compatible. Production's
            # PaddleOcrEngine implements predict_images as one batch call.
            tokens_by_page = [ocr.predict_image(item.image) for item in prepared]
        inference_time_ms = (time.perf_counter() - inference_started) * 1000

        if len(tokens_by_page) != len(pages):
            raise RuntimeError(
                "PaddleOCR returned a different number of results "
                f"({len(tokens_by_page)}) than in-memory page images ({len(pages)})."
            )

        pixel_counts = [max(1, item.image.width * item.image.height) for item in prepared]
        total_pixels = sum(pixel_counts)
        processing_times_ms = [
            preparation_ms + inference_time_ms * pixels / total_pixels
            for preparation_ms, pixels in zip(
                preparation_times_ms, pixel_counts, strict=True
            )
        ]
        return InMemoryBatchOcrResult(
            tokens_by_page=tokens_by_page,
            processing_time_ms_by_page=processing_times_ms,
            inference_time_ms=inference_time_ms,
        )
    finally:
        for item in prepared:
            item.image.close()
