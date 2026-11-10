from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.models.paddle_ocr_engine import OcrTokenPrediction


@dataclass(frozen=True)
class ContentDetection:
    bbox: tuple[int, int, int, int] | None
    component_count: int
    ink_ratio: float


@dataclass(frozen=True)
class PreparedOcrImage:
    image: Image.Image
    source_size: tuple[int, int]
    rotated_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    rotation: int
    scale: float
    content_component_count: int

    @property
    def processed_size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def changed(self) -> bool:
        full_crop = (0, 0, self.rotated_size[0], self.rotated_size[1])
        return self.rotation != 0 or self.crop_box != full_crop or self.scale != 1.0

    def metadata(self) -> dict[str, object]:
        return {
            "source_size": list(self.source_size),
            "rotation": self.rotation,
            "rotated_size": list(self.rotated_size),
            "crop_box": list(self.crop_box),
            "scale": round(self.scale, 6),
            "processed_size": list(self.processed_size),
            "content_component_count": self.content_component_count,
        }


def load_rgb_image(image_path: Path) -> Image.Image:
    if not image_path.is_file():
        raise FileNotFoundError(f"OCR source image does not exist: {image_path}")
    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def load_rgb_bytes(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def rgb_to_bgr_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _odd(value: int) -> int:
    value = max(3, value)
    return value if value % 2 else value + 1


def _binary_ink_mask(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    block_size = _odd(min(51, max(15, round(min(width, height) * 0.025))))
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        15,
    )


def detect_content_bbox(
    image: Image.Image,
    *,
    thumbnail_max_side: int,
    padding_ratio: float,
    min_components: int,
    min_content_area_ratio: float,
    max_crop_area_ratio: float,
) -> ContentDetection:
    """Find a conservative foreground box without interpreting the text.

    The detector deliberately unions every meaningful connected component.
    That can retain extra whitespace, but it avoids silently cutting a header,
    signature or isolated table value. Tiny scan speckles and page-border lines
    are ignored.
    """

    source = image.convert("L")
    source_width, source_height = source.size
    reduction = min(1.0, thumbnail_max_side / max(source_width, source_height))
    thumb_width = max(1, round(source_width * reduction))
    thumb_height = max(1, round(source_height * reduction))
    thumbnail = source.resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
    gray = np.asarray(thumbnail, dtype=np.uint8)
    ink = _binary_ink_mask(gray)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    thumb_area = thumb_width * thumb_height
    minimum_area = max(5, round(thumb_area * 0.000003))
    components: list[tuple[int, int, int, int, int]] = []
    valid_mask = np.zeros_like(ink)
    valid_ink_pixels = 0

    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < minimum_area or width <= 1 or height <= 1:
            continue
        # Scanner shadows and page borders often appear as almost full-length,
        # very thin components. They are not useful for orientation.
        if width >= thumb_width * 0.90 and height <= max(4, thumb_height * 0.006):
            continue
        if height >= thumb_height * 0.90 and width <= max(4, thumb_width * 0.006):
            continue
        if (
            (x <= thumb_width * 0.01 or x + width >= thumb_width * 0.99)
            and width <= max(5, thumb_width * 0.012)
            and height >= thumb_height * 0.10
        ):
            continue
        if (
            (y <= thumb_height * 0.01 or y + height >= thumb_height * 0.99)
            and height <= max(5, thumb_height * 0.012)
            and width >= thumb_width * 0.10
        ):
            continue
        # A single huge dark region is normally a scan artefact/photo, not text.
        if area >= thumb_area * 0.35:
            continue
        components.append((x, y, x + width, y + height, area))
        valid_mask[labels == index] = 255
        valid_ink_pixels += area

    ink_ratio = valid_ink_pixels / thumb_area if thumb_area else 0.0
    if len(components) < min_components or ink_ratio < min_content_area_ratio:
        return ContentDetection(None, len(components), ink_ratio)

    ink_y, ink_x = np.nonzero(valid_mask)
    if not len(ink_x):
        return ContentDetection(None, len(components), ink_ratio)
    # Robust bounds ignore isolated dust while the generous proportional
    # padding below restores a safety margin around true document content.
    left = math.floor(float(np.quantile(ink_x, 0.005)))
    top = math.floor(float(np.quantile(ink_y, 0.005)))
    right = math.ceil(float(np.quantile(ink_x, 0.995))) + 1
    bottom = math.ceil(float(np.quantile(ink_y, 0.995))) + 1

    pad_x = round(thumb_width * padding_ratio)
    pad_y = round(thumb_height * padding_ratio)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(thumb_width, right + pad_x)
    bottom = min(thumb_height, bottom + pad_y)

    crop_area_ratio = ((right - left) * (bottom - top)) / thumb_area
    if crop_area_ratio >= max_crop_area_ratio:
        return ContentDetection(None, len(components), ink_ratio)

    inverse_scale = 1.0 / reduction
    source_box = (
        max(0, math.floor(left * inverse_scale)),
        max(0, math.floor(top * inverse_scale)),
        min(source_width, math.ceil(right * inverse_scale)),
        min(source_height, math.ceil(bottom * inverse_scale)),
    )
    if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
        return ContentDetection(None, len(components), ink_ratio)
    return ContentDetection(source_box, len(components), ink_ratio)


def orientation_probe(
    image: Image.Image,
    *,
    thumbnail_max_side: int,
    padding_ratio: float,
    min_components: int,
    min_content_area_ratio: float,
    max_crop_area_ratio: float,
) -> tuple[Image.Image, ContentDetection]:
    detection = detect_content_bbox(
        image,
        thumbnail_max_side=thumbnail_max_side,
        padding_ratio=padding_ratio,
        min_components=min_components,
        min_content_area_ratio=min_content_area_ratio,
        max_crop_area_ratio=max_crop_area_ratio,
    )
    return (
        image.crop(detection.bbox) if detection.bbox else image.copy(),
        detection,
    )


def prepare_ocr_image(
    source: Image.Image,
    *,
    rotation: int,
    enable_content_crop: bool,
    thumbnail_max_side: int,
    padding_ratio: float,
    min_components: int,
    min_content_area_ratio: float,
    max_crop_area_ratio: float,
    scale: float = 1.0,
    max_processed_side: int = 5000,
    precomputed_detection: ContentDetection | None = None,
) -> PreparedOcrImage:
    if rotation not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported OCR rotation: {rotation}")
    source = source.convert("RGB")
    rotated = (
        source.rotate(rotation, expand=True, fillcolor="white")
        if rotation
        else source
    )
    rotated_size = rotated.size
    detection = ContentDetection(None, 0, 0.0)
    if enable_content_crop:
        if precomputed_detection is not None and rotation == 0:
            detection = precomputed_detection
        else:
            detection = detect_content_bbox(
                rotated,
                thumbnail_max_side=thumbnail_max_side,
                padding_ratio=padding_ratio,
                min_components=min_components,
                min_content_area_ratio=min_content_area_ratio,
                max_crop_area_ratio=max_crop_area_ratio,
            )
    crop_box = detection.bbox or (0, 0, rotated.width, rotated.height)
    processed = rotated.crop(crop_box)

    requested_scale = max(1.0, scale)
    side_limited_scale = min(
        requested_scale,
        max_processed_side / max(processed.width, processed.height),
    )
    effective_scale = max(1.0, side_limited_scale)
    if effective_scale > 1.001:
        processed = processed.resize(
            (
                max(1, round(processed.width * effective_scale)),
                max(1, round(processed.height * effective_scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        effective_scale = 1.0

    return PreparedOcrImage(
        image=processed,
        source_size=source.size,
        rotated_size=rotated_size,
        crop_box=crop_box,
        rotation=rotation,
        scale=effective_scale,
        content_component_count=detection.component_count,
    )


def png_bytes(image: Image.Image, *, compress_level: int = 1) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level)
    return buffer.getvalue()


def median_text_height(tokens: list[OcrTokenPrediction]) -> float | None:
    heights = sorted(token.bbox[3] - token.bbox[1] for token in tokens)
    if not heights:
        return None
    # Ignore extreme boxes (page borders or tall merged columns) when enough
    # samples exist. The median of the core is stable for mixed font sizes.
    if len(heights) >= 10:
        trim = max(1, len(heights) // 10)
        heights = heights[trim:-trim]
    return float(median(heights)) if heights else None


def recommended_upscale(
    tokens: list[OcrTokenPrediction],
    *,
    target_text_height: int,
    max_factor: float,
) -> float:
    text_height = median_text_height(tokens)
    if text_height is None:
        return max_factor
    if text_height >= target_text_height * 0.85:
        return 1.0
    return min(max_factor, max(1.0, target_text_height / max(text_height, 1.0)))


def ocr_quality_score(tokens: list[OcrTokenPrediction]) -> float:
    if not tokens:
        return -1000.0
    average_confidence = sum(token.confidence for token in tokens) / len(tokens)
    horizontal = sum(
        1
        for token in tokens
        if (token.bbox[2] - token.bbox[0]) >= (token.bbox[3] - token.bbox[1]) * 1.15
    ) / len(tokens)
    character_count = sum(len(token.text.strip()) for token in tokens)
    # Orientation is dominated by horizontal text geometry and confidence;
    # logarithms keep pages with many tiny false detections from winning.
    return (
        average_confidence * 5.0
        + horizontal * 3.0
        + math.log1p(len(tokens)) * 0.35
        + math.log1p(character_count) * 0.20
    )
