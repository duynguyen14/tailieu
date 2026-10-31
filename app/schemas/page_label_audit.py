from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.annotation import AnnotationLabel, AnnotationPageSummary


class PageLabelAuditStatus(StrEnum):
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"


class PageLabelAuditItem(BaseModel):
    audit_id: str
    original_label: AnnotationLabel
    audited_label: AnnotationLabel | None
    audit_status: PageLabelAuditStatus
    audit_note: str | None
    page: AnnotationPageSummary
    image_base64: str


class PageLabelAuditDocument(BaseModel):
    document_id: str
    original_filename: str
    document_page_count: int = Field(gt=0)
    candidate_pages: list[PageLabelAuditItem]


class PageLabelAuditOverviewItem(BaseModel):
    page: AnnotationPageSummary
    image_base64: str
    original_label: AnnotationLabel | None = None
    audited_label: AnnotationLabel | None = None
    audit_status: PageLabelAuditStatus | None = None


class PageLabelAuditDocumentOverview(BaseModel):
    document_id: str
    original_filename: str
    document_page_count: int = Field(gt=0)
    pages: list[PageLabelAuditOverviewItem]


class PageLabelAuditStats(BaseModel):
    pending_pages: int = Field(ge=0)
    reviewed_pages: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    pending_documents: int = Field(ge=0)
    total_documents: int = Field(ge=0)


class PageLabelAuditQueueResponse(BaseModel):
    current: PageLabelAuditDocument | None
    stats: PageLabelAuditStats


class PageLabelAuditDecision(BaseModel):
    page_id: str = Field(min_length=1)
    label: AnnotationLabel
    audit_note: str | None = Field(default=None, max_length=2000)


class PageLabelAuditReviewRequest(BaseModel):
    reviewed_by: str = Field(min_length=1, max_length=100)
    pages: list[PageLabelAuditDecision] = Field(min_length=1)


class PageLabelAuditReviewResponse(BaseModel):
    document_id: str
    reviewed_pages: int = Field(ge=1)
