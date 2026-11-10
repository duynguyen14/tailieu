from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.teacher_extraction import CompactDocumentExtraction


class ProductionModelStatus(BaseModel):
    name: str
    loaded: bool
    device: str | None = None
    model: str | None = None
    message: str | None = None


class ProductionReadinessResponse(BaseModel):
    enabled: bool
    ready: bool
    loading: bool
    seal_detection_enabled: bool = False
    models: list[ProductionModelStatus] = Field(default_factory=list)
    error: str | None = None


class ProductionPageResult(BaseModel):
    page_number: int = Field(ge=1)
    rotation_angle: int
    orientation_confidence: float = Field(ge=0.0, le=1.0)
    page_type: str
    classification_confidence: float = Field(ge=0.0, le=1.0)
    used_for_extraction: bool
    ocr_token_count: int = Field(ge=0)
    ocr_processing_time_ms: float = Field(ge=0.0)


class ProductionTiming(BaseModel):
    render_ms: float = Field(ge=0.0)
    orientation_ms: float = Field(ge=0.0)
    classification_ms: float = Field(ge=0.0)
    ocr_ms: float = Field(ge=0.0)
    reconstruction_ms: float = Field(ge=0.0)
    extraction_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class ProductionOcrTiming(BaseModel):
    render_ms: float = Field(ge=0.0)
    orientation_ms: float = Field(ge=0.0)
    classification_ms: float = Field(ge=0.0)
    ocr_ms: float = Field(ge=0.0)
    reconstruction_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class ProductionOcrResponse(BaseModel):
    filename: str
    sha256: str
    page_count: int = Field(ge=1)
    reconstructed_text: str
    pages: list[ProductionPageResult]
    timings: ProductionOcrTiming
    warnings: list[str] = Field(default_factory=list)


class ProductionPageClassScore(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class ProductionPageClassificationPreview(BaseModel):
    page_number: int = Field(ge=1)
    image_content_type: str = "image/png"
    image_base64: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    rotation_angle: int
    orientation_confidence: float = Field(ge=0.0, le=1.0)
    predicted_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[ProductionPageClassScore]
    is_extraction_target: bool


class ProductionPageClassificationTiming(BaseModel):
    render_ms: float = Field(ge=0.0)
    orientation_ms: float = Field(ge=0.0)
    classification_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class ProductionPageClassificationResponse(BaseModel):
    filename: str
    sha256: str
    page_count: int = Field(ge=1)
    target_page_types: list[str]
    pages: list[ProductionPageClassificationPreview]
    timings: ProductionPageClassificationTiming


class ProductionExtractionResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    request_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=1)
    extraction: CompactDocumentExtraction
    compact_model_output: dict[str, Any]
    reconstructed_text: str | None = None
    pages: list[ProductionPageResult]
    timings: ProductionTiming
    warnings: list[str] = Field(default_factory=list)

