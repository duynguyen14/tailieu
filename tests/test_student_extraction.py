from __future__ import annotations

import json

import pytest

from app.schemas.student_extraction import (
    STUDENT_APPLICANT_FIELDS,
    STUDENT_NOTE_FIELDS,
    build_student_user_prompt,
    compact_document_to_student_target,
    normalize_review_output_for_student_v8,
    review_form_schema_descriptor,
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
                "request_type_code": "CAP_MOI",
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


def test_student_target_is_reduced_short_key_json() -> None:
    target = compact_document_to_student_target(extraction())

    assert set(target) == {"n", "a"}
    assert len(target["n"]) == len(STUDENT_NOTE_FIELDS)
    assert all(
        len(applicant) == len(STUDENT_APPLICANT_FIELDS)
        for applicant in target["a"]
    )
    assert target["n"]["no"] == "850-2026"
    assert set(target["n"]) == {"no", "snd", "rt"}
    assert target["n"]["rt"] == "CAP_MOI"
    assert "dt" not in target["n"]
    assert "pl" not in target["n"]
    assert target["a"][0]["nm"] == "Nicholas John Hogarth"
    assert target["a"][0]["pp"] == "RF0210530"
    assert "pt" in target["a"][0]
    assert target["a"][0]["ne"] == "MULTIPLE"


def test_student_target_round_trip_restores_canonical_backend_fields() -> None:
    target = compact_document_to_student_target(extraction())
    restored = student_target_to_compact_document(
        target,
        case_id="case-1",
        document_id="document-1",
    )

    assert restored.note.note_number == "850-2026"
    assert restored.note.issue_date is None
    assert restored.note.issue_place is None
    assert restored.note.request_type_code == "CAP_MOI"
    assert restored.applicants[0].person_id == "person_1"
    assert restored.applicants[0].sequence_number == 1
    assert restored.applicants[0].number_of_entries == "MULTIPLE"
    assert restored.note.recipient_organization is None
    assert restored.applicants[0].port_of_entry is None
    assert restored.applicants[0].proposed_place_of_visa_collection is None
    assert restored.applicants[1].person_id == "person_2"
    assert restored.applicants[1].related_to_person_id is None
    assert restored.warnings == []


def test_review_form_schema_and_payload_are_limited_to_v8_fields() -> None:
    source = extraction().model_dump(mode="json")
    source["note"].update(
        recipient_organization="legacy recipient",
        request_type="legacy free text",
    )
    source["applicants"][0].update(
        occupation="legacy occupation",
        organization="legacy organization",
        port_of_entry="legacy airport",
    )

    normalized = normalize_review_output_for_student_v8(
        source,
        case_id="case-locked",
        document_id="document-locked",
    )
    descriptor = review_form_schema_descriptor()

    assert list(normalized["note"]) == list(STUDENT_NOTE_FIELDS)
    assert normalized["note"]["request_type_code"] == "CAP_MOI"
    assert "recipient_organization" not in normalized["note"]
    assert "request_type" not in normalized["note"]
    assert set(normalized["applicants"][0]) == {
        "person_id",
        "sequence_number",
        *STUDENT_APPLICANT_FIELDS,
    }
    assert "occupation" not in normalized["applicants"][0]
    assert "organization" not in normalized["applicants"][0]
    assert "port_of_entry" not in normalized["applicants"][0]
    assert normalized["case_id"] == "case-locked"
    assert normalized["document_id"] == "document-locked"
    assert descriptor["note_fields"] == list(STUDENT_NOTE_FIELDS)
    assert descriptor["applicant_fields"] == list(STUDENT_APPLICANT_FIELDS)
    assert [item["value"] for item in descriptor["request_type_options"]] == [
        "CAP_MOI",
        "GIA_HAN",
        "CHUYEN_THI_THUC",
        "KHAC",
    ]


def test_student_target_rejects_invalid_root_and_extra_fields() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        student_target_to_compact_document(
            ["not", "an object"], case_id="case-1", document_id="document-1"
        )

    target = compact_document_to_student_target(extraction())
    target["a"][0]["extra"] = "not allowed"
    with pytest.raises(ValueError, match=r"a\[0\] must contain exactly the keys"):
        student_target_to_compact_document(
            target, case_id="case-1", document_id="document-1"
        )

    target = compact_document_to_student_target(extraction())
    del target["a"][0]["dob"]
    with pytest.raises(ValueError, match=r"a\[0\] must contain exactly the keys"):
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
    assert "compact JSON object" in prompt
    assert "root keys n and a" in prompt
    assert "no=note_number" in prompt
    assert "rt=request_type_code" in prompt
    assert "CHUYEN_THI_THUC" in prompt
    assert "dt=issue_date" not in prompt
    assert "pl=issue_place" not in prompt
    assert "nm=full_name" in prompt
    assert "pt=passport_type" in prompt
    assert "All listed keys are mandatory" in prompt


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
