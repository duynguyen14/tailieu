from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


KieReviewMode = Literal["SER", "RE"]


class KieReviewStats(BaseModel):
    ser_pending: int = 0
    ser_approved: int = 0
    re_blocked: int = 0
    re_pending: int = 0
    re_approved: int = 0
    total_pages: int = 0


class KieTokenView(BaseModel):
    id: str
    order: int
    text: str
    bbox: list[int]


class KieFieldProposalView(BaseModel):
    id: str
    field_path: str
    entity_type: str
    entity_index: int | None
    field_name: str
    expected_value: str | None
    match_status: str
    chosen: dict | None
    candidates: list[dict]


class KieEntityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    label: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1)
    field_path: str | None = None
    applicant_index: int | None = Field(default=None, ge=0)
    token_ids: list[str] = Field(default_factory=list)
    bbox: list[int] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def bbox_is_valid(self):
        x1, y1, x2, y2 = self.bbox
        if min(self.bbox) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be [x1,y1,x2,y2] with positive area")
        return self


class KieEntityView(KieEntityInput):
    id: str
    normalized_bbox: list[int]
    confidence: float | None
    source: str
    candidate_rank: int | None


class KieRelationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    head_entity_id: str
    tail_entity_id: str
    relation_type: str = Field(min_length=1, max_length=100)
    applicant_index: int | None = Field(default=None, ge=0)


class KieRelationView(KieRelationInput):
    id: str
    confidence: float | None
    source: str


class KieCaseView(BaseModel):
    run_id: str
    case_id: str
    document_id: str
    original_filename: str
    page_annotation_id: str
    page_id: str
    page_number: int
    page_order: int
    image_width: int
    image_height: int
    image_mime_type: str
    image_base64: str
    ser_status: str
    re_status: str
    current_index: int | None = None
    tokens: list[KieTokenView]
    proposals: list[KieFieldProposalView]
    entities: list[KieEntityView]
    relations: list[KieRelationView]


class KieQueueResponse(BaseModel):
    current: KieCaseView | None
    stats: KieReviewStats


class KieSerReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewed_by: str = Field(min_length=1, max_length=100)
    review_note: str | None = Field(default=None, max_length=4000)
    entities: list[KieEntityInput]


class KieReReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewed_by: str = Field(min_length=1, max_length=100)
    review_note: str | None = Field(default=None, max_length=4000)
    relations: list[KieRelationInput]


class KieReviewSaveResponse(BaseModel):
    page_annotation_id: str
    mode: KieReviewMode
    review_id: str
    parent_review_id: str | None
    reviewed_at: datetime
    ser_status: str
    re_status: str
