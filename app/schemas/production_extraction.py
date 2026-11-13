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
    passport_mrz_ms: float = Field(default=0.0, ge=0.0)
    passport_mrz_status: str = "NOT_REQUESTED"
    extraction_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class ProductionOcrTiming(BaseModel):
    render_ms: float = Field(ge=0.0)
    orientation_ms: float = Field(ge=0.0)
    classification_ms: float = Field(ge=0.0)
    ocr_ms: float = Field(ge=0.0)
    reconstruction_ms: float = Field(ge=0.0)
    passport_mrz_ms: float = Field(default=0.0, ge=0.0)
    passport_mrz_status: str = "NOT_REQUESTED"
    total_ms: float = Field(ge=0.0)


class PassportMrzCheck(BaseModel):
    name: str
    expected: str
    actual: str | None = None
    passed: bool


class PassportMrzPageInput(BaseModel):
    page_number: int = Field(ge=1)
    image_base64: str = Field(min_length=1)


class PassportMrzBatchRequest(BaseModel):
    request_id: str
    pages: list[PassportMrzPageInput]


class PassportMrzPageResult(BaseModel):
    page_number: int = Field(ge=1)
    found: bool
    document_type: str | None = None
    checksum_pass: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detector_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    average_ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    lines: list[str] = Field(default_factory=list)
    fields: dict[str, str | None] = Field(default_factory=dict)
    checks: list[PassportMrzCheck] = Field(default_factory=list)
    processing: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PassportMrzBatchResponse(BaseModel):
    request_id: str
    pages: list[PassportMrzPageResult]
    total_ms: float = Field(ge=0.0)


class PassportMrzReadinessResponse(BaseModel):
    enabled: bool
    ready: bool
    loading: bool
    detector_model: str | None = None
    recognizer_model: str | None = None
    detector_device: str | None = None
    recognizer_device: str | None = None
    error: str | None = None


class ProductionOcrResponse(BaseModel):
    filename: str
    sha256: str
    page_count: int = Field(ge=1)
    reconstructed_text: str
    passport_results: list[PassportMrzPageResult] = Field(default_factory=list)
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


class ProductionCatalogMappingEntry(BaseModel):
    field: str
    applicant_index: int | None = None
    raw_value: str | None = None
    canonical_value: str | None = None
    status: str


class ProductionCatalogMappingReport(BaseModel):
    catalog_version: str
    entries: list[ProductionCatalogMappingEntry] = Field(default_factory=list)
    unmapped_count: int = Field(default=0, ge=0)


def _null_value_list() -> list[str | None]:
    return [None]


class MultiValueBusinessNote(BaseModel):
    note_number: list[str | None] = Field(default_factory=_null_value_list)
    sender_organization: list[str | None] = Field(default_factory=_null_value_list)
    request_type_code: list[str | None] = Field(default_factory=_null_value_list)


class MultiValueBusinessApplicant(BaseModel):
    full_name: list[str | None] = Field(default_factory=_null_value_list)
    date_of_birth: list[str | None] = Field(default_factory=_null_value_list)
    gender: list[str | None] = Field(default_factory=_null_value_list)
    nationality: list[str | None] = Field(default_factory=_null_value_list)
    passport_number: list[str | None] = Field(default_factory=_null_value_list)
    passport_type: list[str | None] = Field(default_factory=_null_value_list)
    passport_issue_date: list[str | None] = Field(default_factory=_null_value_list)
    passport_expiration: list[str | None] = Field(default_factory=_null_value_list)
    visa_type: list[str | None] = Field(default_factory=_null_value_list)
    number_of_entries: list[str | None] = Field(default_factory=_null_value_list)
    entry_date: list[str | None] = Field(default_factory=_null_value_list)
    exit_date: list[str | None] = Field(default_factory=_null_value_list)


class MultiValueBusinessOutput(BaseModel):
    note: MultiValueBusinessNote
    applicants: list[MultiValueBusinessApplicant] = Field(default_factory=list)


class ProductionExtractionResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    request_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=1)
    extraction: CompactDocumentExtraction
    compact_model_output: dict[str, Any]
    catalog_mapped_output: dict[str, Any] | None = None
    catalog_mapping: ProductionCatalogMappingReport | None = None
    multi_source_output: MultiValueBusinessOutput | None = None
    passport_results: list[PassportMrzPageResult] = Field(default_factory=list)
    reconstructed_text: str | None = None
    pages: list[ProductionPageResult]
    timings: ProductionTiming
    warnings: list[str] = Field(default_factory=list)

