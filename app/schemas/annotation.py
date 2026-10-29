from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AnnotationLabel(StrEnum):
    DIPLOMATIC_NOTE = "DIPLOMATIC_NOTE"
    APPLICANT_TABLE = "APPLICANT_TABLE"
    APPLICANT_LIST = "APPLICANT_LIST"
    OTHER = "OTHER"


class ReviewStatus(StrEnum):
    UNLABELED = "UNLABELED"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEWED = "REVIEWED"


class LabelDefinition(BaseModel):
    code: AnnotationLabel
    display_name: str
    description: str


class AnnotationPageSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str
    label: AnnotationLabel | None
    review_status: ReviewStatus
    reviewed_by: str | None
    review_note: str | None
    reviewed_at: datetime | None


class AnnotationDocumentSummary(BaseModel):
    id: str
    original_filename: str
    sha256: str
    page_count: int = Field(gt=0)
    reviewed_pages: int = Field(ge=0)
    review_status: ReviewStatus
    created_at: datetime
    updated_at: datetime


class AnnotationDocumentDetail(AnnotationDocumentSummary):
    pages: list[AnnotationPageSummary]


class AnnotationDocumentList(BaseModel):
    items: list[AnnotationDocumentSummary]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)


class AnnotationImportResponse(BaseModel):
    created: bool
    document: AnnotationDocumentDetail


class AnnotationPageImageResponse(BaseModel):
    document_id: str
    original_filename: str
    document_page_count: int = Field(gt=0)
    page: AnnotationPageSummary
    image_base64: str


class AnnotationReviewStats(BaseModel):
    pending: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    total: int = Field(ge=0)


class AnnotationReviewQueueResponse(BaseModel):
    current: AnnotationPageImageResponse | None
    stats: AnnotationReviewStats


class PageReviewRequest(BaseModel):
    label: AnnotationLabel
    reviewed_by: str = Field(min_length=1, max_length=100)
    review_note: str | None = Field(default=None, max_length=2000)


class PageReviewResponse(BaseModel):
    document_id: str
    document_review_status: ReviewStatus
    page: AnnotationPageSummary
