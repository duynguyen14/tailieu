from __future__ import annotations

import json

import pytest

from app.schemas.student_extraction import (
    STUDENT_APPLICANT_FIELDS,
    STUDENT_NOTE_FIELDS,
    build_student_user_prompt,
    compact_document_to_student_target,
    student_target_to_compact_document,
)
from app.schemas.teacher_extraction import CompactDocumentExtraction
from tools.export_student_extraction_dataset import (
    ExportExample,
    normalize_teacher_payload_for_student,
    split_name,
    training_record,
)


def extraction() -> CompactDocumentExtraction:
    return CompactDocumentExtraction.model_validate(
        {
            "schema_version": "compact-document-v1",
            "case_id": "case-1",
            "document_id": "document-1",
            "note": {
                "note_number": "850-2026",
                "issue_date": "2026-09-30",
                "sender_organization": "Embassy of Australia",
            },
            "applicants": [
                {
                    "person_id": "person_primary",
                    "sequence_number": "1",
                    "full_name": "Nicholas John Hogarth",
                    "date_of_birth": "1976-10-14",
                    "passport_number": "RF0210530",
                    "passport_type": "Ordinary",
                    "passport_expiration": "2030-11-13",
                    "nationality": "Australian",
                    "number_of_entries": "MULTIPLE",
                },
                {
                    "person_id": "person_spouse",
                    "sequence_number": "2",
                    "full_name": "Jennifer May Blair",
                    "relationship": "SPOUSE",
                    "related_to_person_id": "person_primary",
                },
            ],
            "warnings": ["teacher warning not exported"],
        }
    )


def test_student_target_is_reduced_positional_json() -> None:
    target = compact_document_to_student_target(extraction())

    assert len(target) == 2
    assert len(target[0]) == len(STUDENT_NOTE_FIELDS)
    assert all(len(row) == len(STUDENT_APPLICANT_FIELDS) for row in target[1])
    assert target[0][0] == "850-2026"
    assert target[1][0][0] == "Nicholas John Hogarth"
    assert target[1][0][4] == "RF0210530"
    assert target[1][0][9] == "MULTIPLE"


def test_student_target_round_trip_restores_canonical_backend_fields() -> None:
    target = compact_document_to_student_target(extraction())
    restored = student_target_to_compact_document(
        target,
        case_id="case-1",
        document_id="document-1",
    )

    assert restored.note.note_number == "850-2026"
    assert restored.applicants[0].person_id == "person_1"
    assert restored.applicants[0].sequence_number == 1
    assert restored.applicants[0].number_of_entries == "MULTIPLE"
    assert restored.note.recipient_organization is None
    assert restored.applicants[0].port_of_entry is None
    assert restored.applicants[0].proposed_place_of_visa_collection is None
    assert restored.applicants[1].person_id == "person_2"
    assert restored.applicants[1].related_to_person_id is None
    assert restored.warnings == []


def test_student_target_rejects_invalid_root_and_extra_fields() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        student_target_to_compact_document(
            {"not": "an array"}, case_id="case-1", document_id="document-1"
        )

    target = compact_document_to_student_target(extraction())
    target[1][0].append("extra position")
    with pytest.raises(ValueError, match="exactly 12 values"):
        student_target_to_compact_document(
            target, case_id="case-1", document_id="document-1"
        )


def test_split_is_deterministic_by_document() -> None:
    assert split_name("document-1") == split_name("document-1")
    assert split_name("document-1") in {"train", "val", "test"}


def test_training_record_uses_user_and_assistant_messages() -> None:
    target = compact_document_to_student_target(extraction())
    example = ExportExample(
        review_id="review-1",
        job_id="job-1",
        case_id="case-1",
        document_id="document-1",
        original_filename="000001.pdf",
        input_hash="a" * 64,
        reviewed_by="thai",
        reviewed_at=None,
        input_text="No: 850-2026",
        target=target,
        split="train",
        record_hash="b" * 64,
    )

    record = training_record(example)

    assert [message["role"] for message in record["messages"]] == [
        "user",
        "assistant",
    ]
    assert "DOCUMENT_TEXT:\nNo: 850-2026" in record["messages"][0]["content"]
    assert json.loads(record["messages"][1]["content"]) == target
    assert "system" not in {message["role"] for message in record["messages"]}


def test_prompt_contains_locked_compact_contract() -> None:
    prompt = build_student_user_prompt("source")
    assert "positional JSON array" in prompt
    assert "exactly 5 values" in prompt
    assert "exactly 12 values" in prompt
    assert "Do not add, remove, or reorder positions" in prompt


def test_export_normalizes_internal_person_ids_and_stale_links() -> None:
    payload = extraction().model_dump(mode="json")
    payload["applicants"] = [payload["applicants"][1]]
    payload["applicants"][0]["person_id"] = "person_spouse"
    payload["applicants"][0]["sequence_number"] = 99
    payload["applicants"][0]["related_to_person_id"] = "missing_person"

    normalized = normalize_teacher_payload_for_student(payload)
    validated = CompactDocumentExtraction.model_validate(normalized)

    assert validated.applicants[0].person_id == "person_1"
    assert validated.applicants[0].sequence_number == 1
    assert validated.applicants[0].related_to_person_id is None
    assert validated.applicants[0].relationship == "SPOUSE"
