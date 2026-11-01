from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TextReconstructionReviewRequest(BaseModel):
    reviewed_by: str = Field(min_length=1, max_length=100)
    review_note: str | None = Field(default=None, max_length=2000)
    reconstructed_text: str = Field(min_length=1, max_length=500_000)


class TextReconstructionInfo(BaseModel):
    id: str
    case_id: str
    parent_reconstruction_id: str | None
    algorithm_version: str
    source_fingerprint: str
    status: str
    is_active: bool
    page_count: int = Field(ge=1)
    token_count: int = Field(ge=0)
    reviewed_by: str | None
    review_note: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class TextReconstructionPage(BaseModel):
    id: str
    page_number: int = Field(ge=1)
    page_type: str | None
    page_role: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str
    image_base64: str


class TextReconstructionCase(BaseModel):
    case_id: str
    document_id: str
    original_filename: str
    case_index: int = Field(ge=1)
    case_status: str


class TextReconstructionDetail(BaseModel):
    reconstruction: TextReconstructionInfo
    case: TextReconstructionCase
    pages: list[TextReconstructionPage]
    reconstructed_text: str
    structure: dict[str, Any]


class TextReconstructionStats(BaseModel):
    pending: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    total: int = Field(ge=0)


class TextReconstructionQueueResponse(BaseModel):
    current: TextReconstructionDetail | None
    stats: TextReconstructionStats
