from __future__ import annotations

from PIL import Image, ImageDraw

from app.models.ocr_preprocessing import (
    detect_content_bbox,
    ocr_quality_score,
    prepare_ocr_image,
    recommended_upscale,
)
from app.models.paddle_ocr_engine import OcrTokenPrediction
from app.models.paddle_orientation import OrientationPrediction
from tools.run_grounded_ocr import candidate_angles


def _token(
    index: int,
    *,
    text: str,
    bbox: tuple[int, int, int, int],
    confidence: float = 0.9,
) -> OcrTokenPrediction:
    return OcrTokenPrediction(
        token_id=f"t_{index:04d}",
        token_index=index,
        text=text,
        confidence=confidence,
        bbox=bbox,
        polygon=[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ],
    )


def _content_image() -> Image.Image:
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    # A compact document-like strip surrounded by a large white canvas.
    for row in range(20):
        y = 180 + row * 42
        draw.rectangle((150, y, 430, y + 10), fill="black")
        draw.rectangle((450, y, 590, y + 10), fill="black")
    return image


def test_content_detection_is_conservative_but_removes_large_blank_margin() -> None:
    detection = detect_content_bbox(
        _content_image(),
        thumbnail_max_side=1000,
        padding_ratio=0.05,
        min_components=8,
        min_content_area_ratio=0.0001,
        max_crop_area_ratio=0.95,
    )

    assert detection.bbox is not None
    x1, y1, x2, y2 = detection.bbox
    assert x1 < 150 and y1 < 180
    assert x2 > 590 and y2 > 180 + 19 * 42 + 10
    assert (x2 - x1) < 800


def test_prepare_rotates_full_source_then_crops_and_upscales() -> None:
    prepared = prepare_ocr_image(
        _content_image(),
        rotation=90,
        enable_content_crop=True,
        thumbnail_max_side=1000,
        padding_ratio=0.05,
        min_components=8,
        min_content_area_ratio=0.0001,
        max_crop_area_ratio=0.95,
        scale=2.0,
        max_processed_side=5000,
    )

    assert prepared.source_size == (1000, 1400)
    assert prepared.rotated_size == (1400, 1000)
    assert prepared.rotation == 90
    assert prepared.scale == 2.0
    assert prepared.changed is True
    assert prepared.processed_size[0] < prepared.rotated_size[0] * 2
    prepared.image.close()


def test_low_confidence_orientation_evaluates_every_angle() -> None:
    prediction = OrientationPrediction(angle=0, confidence=0.62, predicted_angle=90)

    assert candidate_angles(
        prediction,
        min_confidence=0.80,
        fallback_all_angles=True,
    ) == [90, 0, 180, 270]


def test_quality_score_prefers_horizontal_text_geometry() -> None:
    horizontal = [
        _token(index, text="Applicant name", bbox=(10, index * 40, 260, index * 40 + 25))
        for index in range(1, 8)
    ]
    vertical = [
        _token(index, text="Applicant name", bbox=(index * 40, 10, index * 40 + 25, 260))
        for index in range(1, 8)
    ]

    assert ocr_quality_score(horizontal) > ocr_quality_score(vertical)


def test_small_text_recommends_bounded_upscale() -> None:
    tokens = [
        _token(index, text="small", bbox=(10, index * 20, 90, index * 20 + 12))
        for index in range(1, 12)
    ]

    assert recommended_upscale(
        tokens,
        target_text_height=30,
        max_factor=2.5,
    ) == 2.5
