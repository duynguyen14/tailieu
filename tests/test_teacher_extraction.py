from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.teacher_extraction import CompactDocumentExtraction
from tools.generate_teacher_extractions import (
    ReconstructionCandidate,
    build_messages,
    call_teacher,
    extract_json_object,
    teacher_input_hash,
    validate_teacher_output,
)


def candidate() -> ReconstructionCandidate:
    return ReconstructionCandidate(
        reconstruction_id="reconstruction-1",
        case_id="case-1",
        document_id="document-1",
        original_filename="000001.pdf",
        reconstruction_version="layout-v1",
        source_fingerprint="a" * 64,
        reconstructed_text="No: 850-2026\nName: Nicholas John Hogarth",
        structure_json='{"pages":[]}',
        reconstruction_status="REVIEWED",
    )


def settings(**overrides):
    values = {
        "teacher_schema_version": "compact-document-v1",
        "teacher_prompt_version": "teacher-v1",
        "teacher_model": "teacher-model",
        "teacher_chat_completions_url": "http://teacher.local/v1/chat/completions",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def valid_payload() -> dict:
    return {
        "schema_version": "compact-document-v1",
        "case_id": "case-1",
        "document_id": "document-1",
        "note": {"note_number": "850-2026"},
        "applicants": [
            {
                "person_id": "person_1",
                "full_name": "Nicholas John Hogarth",
            }
        ],
        "warnings": [],
    }


def valid_business_payload() -> dict:
    return {
        "note": {
            "note_number": "850-2026",
            "sender_organization": "Embassy of Australia",
            "request_type_code": "CAP_MOI",
        },
        "applicants": [
            {
                "full_name": "Nicholas John Hogarth",
                "date_of_birth": "1976-10-14",
                "gender": "Male",
                "nationality": "Australian",
                "passport_number": "RF0210530",
                "passport_type": "Ordinary",
                "passport_issue_date": None,
                "passport_expiration": "2030-11-13",
                "visa_type": "NG4",
                "number_of_entries": "MULTIPLE",
                "entry_date": None,
                "exit_date": None,
            }
        ],
    }


def test_extract_json_object_accepts_plain_and_fenced_json() -> None:
    assert extract_json_object('{"value":1}') == {"value": 1}
    assert extract_json_object('```json\n{"value":2}\n```') == {"value": 2}


def test_extract_json_object_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="root must be a JSON object"):
        extract_json_object("[1, 2]")


def test_compact_schema_rejects_duplicate_person_ids_and_invalid_links() -> None:
    payload = valid_payload()
    payload["applicants"].append(
        {"person_id": "person_1", "related_to_person_id": "missing"}
    )
    with pytest.raises(ValidationError):
        CompactDocumentExtraction.model_validate(payload)


def test_validate_teacher_output_checks_source_identifiers() -> None:
    result = json.loads(
        validate_teacher_output(
            json.dumps(valid_business_payload()), candidate=candidate()
        )
    )
    assert result["case_id"] == "case-1"
    assert result["document_id"] == "document-1"
    assert result["note"] == {
        "note_number": "850-2026",
        "sender_organization": "Embassy of Australia",
        "request_type_code": "CAP_MOI",
    }
    assert result["applicants"][0]["person_id"] == "person_1"
    assert result["applicants"][0]["sequence_number"] == 1
    assert result["applicants"][0]["passport_type"] == "Ordinary"
    assert "occupation" not in result["applicants"][0]
    assert result["warnings"] == []


def test_validate_teacher_output_accepts_passport_type() -> None:
    payload = valid_business_payload()

    result = json.loads(
        validate_teacher_output(json.dumps(payload), candidate=candidate())
    )

    assert result["applicants"][0]["passport_type"] == "Ordinary"


def test_validate_teacher_output_rejects_legacy_business_fields() -> None:
    payload = valid_business_payload()
    payload["applicants"][0]["occupation"] = "legacy field"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_teacher_output(json.dumps(payload), candidate=candidate())


def test_teacher_input_hash_is_deterministic_and_versioned() -> None:
    first = teacher_input_hash(candidate(), settings())
    assert first == teacher_input_hash(candidate(), settings())
    assert first != teacher_input_hash(candidate(), settings(teacher_prompt_version="teacher-v2"))
    assert first != teacher_input_hash(
        candidate(),
        settings(teacher_chat_completions_url="http://teacher-2/v1/chat/completions"),
    )


def test_build_messages_contains_schema_ids_and_reconstructed_text() -> None:
    messages = build_messages(candidate(), settings())
    assert messages[0]["role"] == "system"
    assert "Nicholas John Hogarth" in messages[1]["content"]
    assert "OUTPUT_JSON_SCHEMA" in messages[1]["content"]
    assert "business-v8" in messages[1]["content"]
    assert "occupation" not in messages[1]["content"]
    assert "recipient_organization" not in messages[1]["content"]


def test_call_teacher_validates_openai_compatible_response(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(valid_business_payload())
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }

    captured = {}

    def fake_post(url, *, content, headers, timeout):
        captured.update(
            url=url,
            payload=json.loads(content),
            headers=headers,
            timeout=timeout,
        )
        return FakeResponse()

    monkeypatch.setattr("tools.generate_teacher_extractions.httpx.post", fake_post)
    runtime_settings = settings(
        teacher_temperature=0.0,
        teacher_max_tokens=2048,
        teacher_use_json_mode=True,
        teacher_api_key=None,
        teacher_timeout_seconds=300.0,
    )
    result = call_teacher(candidate(), runtime_settings)

    assert result.valid is True
    assert result.total_tokens == 150
    assert result.finish_reason == "stop"
    assert captured["url"] == "http://teacher.local/v1/chat/completions"
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_call_teacher_serializes_validation_errors_without_crashing(monkeypatch) -> None:
    invalid_payload = valid_business_payload()
    invalid_payload["applicants"][0]["unexpected_field"] = "invalid"

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": json.dumps(invalid_payload)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    monkeypatch.setattr(
        "tools.generate_teacher_extractions.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    runtime_settings = settings(
        teacher_temperature=0.0,
        teacher_max_tokens=2048,
        teacher_use_json_mode=True,
        teacher_api_key=None,
        teacher_timeout_seconds=300.0,
    )

    result = call_teacher(candidate(), runtime_settings)

    assert result.validation_status == "INVALID"
    assert result.valid is False
    errors = json.loads(result.validation_errors_json)
    assert errors[0]["type"] == "extra_forbidden"
