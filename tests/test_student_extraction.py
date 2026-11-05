from __future__ import annotations

import json

import pytest

from app.schemas.student_extraction import (
    APPLICANT_FIELDS,
    NOTE_FIELDS,
    build_student_user_prompt,
    compact_document_to_student_target,
    student_target_to_compact_document,
)
from app.schemas.teacher_extraction import CompactDocumentExtraction
from tools.export_student_extraction_dataset import split_name, training_record, ExportExample


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


def test_student_target_has_fixed_lengths_and_one_based_relationship() -> None:
    target = compact_document_to_student_target(extraction())

    assert len(target) == 2
    assert len(target[0]) == len(NOTE_FIELDS) == 7
    assert all(len(applicant) == len(APPLICANT_FIELDS) == 21 for applicant in target[1])
    assert target[0][0] == "850-2026"
    assert target[1][0][4] == "RF0210530"
    assert target[1][0][10] == "MULTIPLE"
    assert target[1][1][19] == "SPOUSE"
    assert target[1][1][20] == 1


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
    assert restored.applicants[1].person_id == "person_2"
    assert restored.applicants[1].related_to_person_id == "person_1"
    assert restored.warnings == []


def test_student_target_rejects_invalid_shape_and_relationship() -> None:
    with pytest.raises(ValueError, match="exactly 7"):
        student_target_to_compact_document(
            [["too short"], []], case_id="case-1", document_id="document-1"
        )

    target = compact_document_to_student_target(extraction())
    target[1][1][20] = 2
    with pytest.raises(ValueError, match="invalid related"):
        student_target_to_compact_document(
            target, case_id="case-1", document_id="document-1"
        )

    target = compact_document_to_student_target(extraction())
    target[1][0][0] = {"unexpected": "object"}
    with pytest.raises(ValueError, match="string or null"):
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


def test_prompt_contains_locked_positional_contract() -> None:
    prompt = build_student_user_prompt("source")
    assert "exactly 7 positions" in prompt
    assert "exactly 21 positions" in prompt
    assert "one-based applicant position" in prompt
