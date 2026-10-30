from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

BBox = Annotated[
    tuple[float, float, float, float],
    Field(description="[x_min, y_min, x_max, y_max] in rendered-page pixels"),
]


class PageType(StrEnum):
    DIPLOMATIC_NOTE = "DIPLOMATIC_NOTE"
    APPLICANT_TABLE = "APPLICANT_TABLE"
    APPLICANT_LIST = "APPLICANT_LIST"
    VISA_APPLICATION_FORM = "VISA_APPLICATION_FORM"
    TEMPORARY_RESIDENCE_EXTENSION_FORM = "TEMPORARY_RESIDENCE_EXTENSION_FORM"
    PASSPORT_FRONT = "PASSPORT_FRONT"
    PASSPORT_BACK = "PASSPORT_BACK"
    VISA_STICKER = "VISA_STICKER"
    OTHER = "OTHER"


class ExtractionStatus(StrEnum):
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ImageSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SourceLocation(BaseModel):
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    bbox: BBox
    image_size: ImageSize


class ExtractedField(BaseModel):
    value: str | int | float | bool | None = None
    raw_text: str | None = None
    normalized_value: str | int | float | bool | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: SourceLocation | None = None
    sources: list[SourceLocation] = Field(default_factory=list)


class OCRToken(BaseModel):
    token_id: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    reading_order: int = Field(ge=0)


class PageClassification(BaseModel):
    page_type: PageType
    confidence: float = Field(ge=0.0, le=1.0)


class NoteExtraction(BaseModel):
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    applicants: list[dict[str, ExtractedField]] = Field(default_factory=list)


class TableCell(BaseModel):
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    field_name: str | None = None
    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: SourceLocation


class TableRow(BaseModel):
    row_index: int = Field(ge=0)
    fields: dict[str, ExtractedField] = Field(default_factory=dict)


class TableExtraction(BaseModel):
    table_bbox: SourceLocation | None = None
    headers: list[str] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)


class PageExtraction(BaseModel):
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    page_type: PageType
    classification_confidence: float = Field(ge=0.0, le=1.0)
    image_size: ImageSize
    ocr_tokens: list[OCRToken] = Field(default_factory=list)
    note: NoteExtraction | None = None
    table: TableExtraction | None = None
    warnings: list[str] = Field(default_factory=list)


class DocumentData(BaseModel):
    note_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    applicants: list[dict[str, ExtractedField]] = Field(default_factory=list)


class RuntimeInfo(BaseModel):
    backend: str
    requested_device: str
    effective_device: str


class DocumentExtractionResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    document_id: str
    filename: str
    sha256: str
    status: ExtractionStatus
    processing_time_ms: float = Field(ge=0.0)
    runtime: RuntimeInfo
    pages: list[PageExtraction]
    data: DocumentData
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    processing_mode: str


class ModelComponentStatus(BaseModel):
    name: str
    enabled: bool
    configured_path: str | None
    path_exists: bool
    loaded: bool
    message: str


class ModelsStatusResponse(BaseModel):
    backend: str
    requested_device: str
    effective_device: str
    gpu_available: bool
    runtime_reason: str
    components: list[ModelComponentStatus]
