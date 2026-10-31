from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from app.schemas.annotation import AnnotationLabel


BBox = Annotated[tuple[int, int, int, int], Field(description="[x1, y1, x2, y2]")]


class OcrReviewTokenPayload(BaseModel):
    source_token_id: str | None = None
    reviewed_text: str = Field(min_length=1, max_length=4000)
    reviewed_bbox: BBox

    @model_validator(mode="after")
    def validate_bbox(self) -> OcrReviewTokenPayload:
        x1, y1, x2, y2 = self.reviewed_bbox
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("reviewed_bbox must satisfy x2 > x1 and y2 > y1")
        return self


class OcrReviewRequest(BaseModel):
    reviewed_by: str = Field(min_length=1, max_length=100)
    review_note: str | None = Field(default=None, max_length=2000)
    tokens: list[OcrReviewTokenPayload] = Field(default_factory=list)


class OcrReviewRunInfo(BaseModel):
    id: str
    page_id: str
    status: str
    model_name: str
    model_version: str
    device: str
    processing_time_ms: int | None
    review_id: str | None
    reviewed_by: str | None
    review_note: str | None
    reviewed_at: datetime | None


class OcrReviewPageInfo(BaseModel):
    id: str
    document_id: str
    original_filename: str
    document_page_count: int = Field(gt=0)
    page_number: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str
    page_type: AnnotationLabel | None


class OcrReviewTokenView(BaseModel):
    source_token_id: str | None
    raw_token_code: str | None
    raw_text: str | None
    raw_bbox: BBox | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reviewed_text: str
    reviewed_bbox: BBox
    is_human_added: bool
    is_changed: bool


class OcrReviewDetail(BaseModel):
    run: OcrReviewRunInfo
    page: OcrReviewPageInfo
    image_base64: str
    tokens: list[OcrReviewTokenView]


class OcrReviewStats(BaseModel):
    pending: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    failed: int = Field(ge=0)
    total: int = Field(ge=0)


class OcrReviewQueueResponse(BaseModel):
    current: OcrReviewDetail | None
    stats: OcrReviewStats
