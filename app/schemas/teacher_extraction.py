from __future__ import annotations

from typing import Literal

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
