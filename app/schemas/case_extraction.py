from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CASE_SCHEMA_VERSION = "document-case-v1"


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    token_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def token_prefix_matches_page(self) -> EvidenceReference:
        expected = f"p{self.page_number}_t_"
        invalid = [token_id for token_id in self.token_ids if not token_id.startswith(expected)]
        if invalid:
            raise ValueError(
                f"Evidence token IDs must start with {expected!r}; invalid={invalid}"
            )
        return self


class GroundedValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | int | None = None
    normalized_value: str | int | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def non_null_value_requires_evidence(self) -> GroundedValue:
        if self.value is not None and not self.evidence:
            raise ValueError("Every non-null extracted value must cite OCR evidence.")
        if self.value is None and self.evidence:
            raise ValueError("A null extracted value cannot cite OCR evidence.")
        return self


class DiplomaticNoteData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_number: GroundedValue | None = None
    reference_number: GroundedValue | None = None
    issue_date: GroundedValue | None = None
    issue_place: GroundedValue | None = None
    sender_organization: GroundedValue | None = None
    recipient_organization: GroundedValue | None = None
    request_type: GroundedValue | None = None


class ApplicantData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: str = Field(min_length=1)
    sequence_number: GroundedValue | None = None
    full_name: GroundedValue | None = None
    date_of_birth: GroundedValue | None = None
    gender: GroundedValue | None = None
    nationality: GroundedValue | None = None
    passport_number: GroundedValue | None = None
    visa_type: GroundedValue | None = None
    entry_date: GroundedValue | None = None
    exit_date: GroundedValue | None = None
    occupation: GroundedValue | None = None
    job_title: GroundedValue | None = None
    organization: GroundedValue | None = None
    purpose_of_entry: GroundedValue | None = None
    relationship: GroundedValue | None = None
    related_to_person_id: str | None = None


class CaseExtractionTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["document-case-v1"] = "document-case-v1"
    case_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    note: DiplomaticNoteData = Field(default_factory=DiplomaticNoteData)
    applicants: list[ApplicantData] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
