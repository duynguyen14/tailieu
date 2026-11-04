from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TEACHER_SCHEMA_VERSION = "compact-document-v1"


class CompactDiplomaticNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_number: str | None = None
    reference_number: str | None = None
    issue_date: str | None = None
    issue_place: str | None = None
    sender_organization: str | None = None
    recipient_organization: str | None = None
    request_type: str | None = None


class CompactApplicant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: str = Field(min_length=1)
    sequence_number: str | int | None = None
    full_name: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None
    nationality: str | None = None
    passport_number: str | None = None
    visa_type: str | None = None
    entry_date: str | None = None
    exit_date: str | None = None
    occupation: str | None = None
    job_title: str | None = None
    organization: str | None = None
    purpose_of_entry: str | None = None
    relationship: str | None = None
    related_to_person_id: str | None = None


class CompactDocumentExtraction(BaseModel):
    """Business-only teacher target used to train the small text-to-JSON model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["compact-document-v1"] = TEACHER_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    note: CompactDiplomaticNote = Field(default_factory=CompactDiplomaticNote)
    applicants: list[CompactApplicant] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_person_links(self) -> CompactDocumentExtraction:
        person_ids = [applicant.person_id for applicant in self.applicants]
        if len(person_ids) != len(set(person_ids)):
            raise ValueError("Applicant person_id values must be unique.")
        known = set(person_ids)
        invalid_links = sorted(
            {
                applicant.related_to_person_id
                for applicant in self.applicants
                if applicant.related_to_person_id is not None
                and applicant.related_to_person_id not in known
            }
        )
        if invalid_links:
            raise ValueError(
                "related_to_person_id must reference another applicant; "
                f"invalid={invalid_links}"
            )
        return self


TeacherReviewDecision = Literal["PENDING", "APPROVED", "REJECTED"]


class TeacherExtractionReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed_by: str = Field(min_length=1, max_length=100)
    decision: TeacherReviewDecision
    reviewed_output: dict[str, Any]
    review_note: str | None = Field(default=None, max_length=4000)


class TeacherExtractionReviewInfo(BaseModel):
    id: str
    parent_review_id: str | None
    review_status: TeacherReviewDecision
    is_active: bool
    reviewed_by: str | None
    review_note: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class TeacherExtractionJobInfo(BaseModel):
    id: str
    reconstruction_id: str
    input_hash: str
    schema_version: str
    prompt_version: str
    teacher_model: str
    teacher_endpoint: str
    status: str
    attempt_count: int = Field(ge=0)
    completed_at: datetime | None
    created_at: datetime


class TeacherExtractionAttemptInfo(BaseModel):
    id: str
    attempt_number: int = Field(ge=1)
    validation_status: str
    http_status: int | None
    latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    error_message: str | None
    created_at: datetime


class TeacherExtractionCaseInfo(BaseModel):
    case_id: str
    document_id: str
    original_filename: str


class TeacherExtractionReviewDetail(BaseModel):
    review: TeacherExtractionReviewInfo
    job: TeacherExtractionJobInfo
    attempt: TeacherExtractionAttemptInfo
    case: TeacherExtractionCaseInfo
    input_text: str
    reviewed_output: dict[str, Any]
    raw_response: dict[str, Any] | list[Any] | str | None
    validation_errors: dict[str, Any] | list[Any] | str | None


class TeacherExtractionReviewStats(BaseModel):
    pending: int = Field(ge=0)
    approved: int = Field(ge=0)
    rejected: int = Field(ge=0)
    total: int = Field(ge=0)


class TeacherExtractionReviewQueueResponse(BaseModel):
    current: TeacherExtractionReviewDetail | None
    stats: TeacherExtractionReviewStats
