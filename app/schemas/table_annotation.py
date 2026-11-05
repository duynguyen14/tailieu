from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from app.compat import StrEnum


BBox = Annotated[tuple[int, int, int, int], Field(description="[x1, y1, x2, y2]")]


class TableAnnotationSource(StrEnum):
    MODEL = "MODEL"
    HUMAN = "HUMAN"


class TableAnnotationStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PREDICTED = "PREDICTED"
    REVIEWED = "REVIEWED"
    FAILED = "FAILED"


class TableElementType(StrEnum):
    HEADER_ROW = "HEADER_ROW"
    DATA_ROW = "DATA_ROW"
    COLUMN = "COLUMN"
    CELL = "CELL"


def validate_bbox(bbox: BBox) -> BBox:
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x1 >= 0, y1 >= 0, x2 > x1 and y2 > y1")
    return bbox


class TableElementPayload(BaseModel):
    element_type: TableElementType
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    bbox: BBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_geometry_and_indexes(self) -> TableElementPayload:
        validate_bbox(self.bbox)
        if self.element_type in {TableElementType.HEADER_ROW, TableElementType.DATA_ROW}:
            if self.row_index is None or self.column_index is not None:
                raise ValueError("row elements require row_index and no column_index")
        elif self.element_type == TableElementType.COLUMN:
            if self.row_index is not None or self.column_index is None:
                raise ValueError("COLUMN requires column_index and no row_index")
        elif self.row_index is None or self.column_index is None:
            raise ValueError("CELL requires both row_index and column_index")
        return self


class TableRegionPayload(BaseModel):
    table_index: int = Field(ge=0)
    bbox: BBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    elements: list[TableElementPayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_geometry(self) -> TableRegionPayload:
        validate_bbox(self.bbox)
        return self


class TableReviewRequest(BaseModel):
    reviewed_by: str = Field(min_length=1, max_length=100)
    review_note: str | None = Field(default=None, max_length=2000)
    tables: list[TableRegionPayload] = Field(default_factory=list)


class TablePageInfo(BaseModel):
    id: str
    document_id: str
    original_filename: str
    document_page_count: int = Field(gt=0)
    page_number: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str


class TableAnnotationInfo(BaseModel):
    id: str
    page_id: str
    parent_annotation_id: str | None
    source: TableAnnotationSource
    status: TableAnnotationStatus
    is_active: bool
    model_name: str | None
    model_version: str | None
    processing_time_ms: int | None
    error_message: str | None
    reviewed_by: str | None
    review_note: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class TableAnnotationDetail(BaseModel):
    annotation: TableAnnotationInfo
    page: TablePageInfo
    image_base64: str
    tables: list[TableRegionPayload]


class TableReviewStats(BaseModel):
    pending: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    failed: int = Field(ge=0)
    total: int = Field(ge=0)


class TableReviewQueueResponse(BaseModel):
    current: TableAnnotationDetail | None
    stats: TableReviewStats
