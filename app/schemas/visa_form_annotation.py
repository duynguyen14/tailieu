from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from app.compat import StrEnum

BBox = Annotated[tuple[int, int, int, int], Field(description="[x1, y1, x2, y2]")]


class VisaFormAnnotationSource(StrEnum):
    TEMPLATE = "TEMPLATE"
    HUMAN = "HUMAN"


class VisaFormAnnotationStatus(StrEnum):
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"
    REJECTED = "REJECTED"


class VisaFormReviewDecision(StrEnum):
    REVIEWED = "REVIEWED"
    REJECTED = "REJECTED"


class VisaFormFieldDefinition(BaseModel):
    code: str
    label: str
    order: int = Field(ge=1)


class VisaFormFieldPayload(BaseModel):
    field_code: str = Field(min_length=1, max_length=64)
    field_order: int = Field(ge=1)
    value_text: str | None = Field(default=None, max_length=4000)
    bbox: BBox
    normalized_bbox: BBox

    @model_validator(mode="after")
    def validate_geometry(self) -> VisaFormFieldPayload:
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox is invalid")
        nx1, ny1, nx2, ny2 = self.normalized_bbox
        if nx1 < 0 or ny1 < 0 or nx2 > 1000 or ny2 > 1000 or nx2 <= nx1 or ny2 <= ny1:
            raise ValueError("normalized_bbox must use the 0..1000 coordinate space")
        return self


class VisaFormReviewRequest(BaseModel):
    reviewed_by: str = Field(min_length=1, max_length=100)
    decision: VisaFormReviewDecision = VisaFormReviewDecision.REVIEWED
    review_note: str | None = Field(default=None, max_length=2000)
    fields: list[VisaFormFieldPayload]


class VisaFormPageInfo(BaseModel):
    id: str
    document_id: str
    original_filename: str
    document_page_count: int = Field(gt=0)
    page_number: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str


class VisaFormAnnotationInfo(BaseModel):
    id: str
    page_id: str
    parent_annotation_id: str | None
    source: VisaFormAnnotationSource
    status: VisaFormAnnotationStatus
    template_version: str
    is_active: bool
    reviewed_by: str | None
    review_note: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class VisaFormAnnotationDetail(BaseModel):
    annotation: VisaFormAnnotationInfo
    page: VisaFormPageInfo
    image_base64: str
    fields: list[VisaFormFieldPayload]


class VisaFormReviewStats(BaseModel):
    pending: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    rejected: int = Field(ge=0)
    total: int = Field(ge=0)


class VisaFormReviewQueueResponse(BaseModel):
    current: VisaFormAnnotationDetail | None
    current_index: int | None = Field(default=None, ge=0)
    stats: VisaFormReviewStats
