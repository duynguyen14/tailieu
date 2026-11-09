import json
from datetime import UTC, datetime
from types import SimpleNamespace

from app.schemas.teacher_extraction import CompactDocumentExtraction
from tools.augment_student_extraction_data import (
    _replace_iso_date,
    augment_review,
)
from tools.export_student_extraction_dataset import build_synthetic_example


def source_review() -> SimpleNamespace:
    payload = {
        "schema_version": "compact-document-v1",
        "case_id": "case-1",
        "document_id": "document-1",
        "note": {
            "note_number": "2026-1630",
            "reference_number": None,
            "issue_date": "2026-03-01",
            "issue_place": None,
            "sender_organization": "Embassy of Example",
            "recipient_organization": None,
            "request_type": None,
            "request_type_code": "CAP_MOI",
        },
        "applicants": [
            {
                "person_id": "person_1",
                "sequence_number": 1,
                "full_name": "John Smith",
                "date_of_birth": "1980-01-02",
                "gender": "Male",
                "nationality": "American",
                "passport_number": "X00176766",
                "passport_type": "Diplomatic",
                "passport_issue_date": "2026-06-09",
                "passport_expiration": "2031-06-09",
                "passport_issuing_authority": None,
                "visa_type": "NG4",
                "number_of_entries": "MULTIPLE",
                "entry_date": "2026-10-30",
                "exit_date": "2026-11-30",
                "proposed_place_of_visa_collection": None,
                "port_of_entry": None,
                "occupation": None,
                "job_title": None,
                "organization": None,
                "purpose_of_entry": None,
                "relationship": None,
                "related_to_person_id": None,
            }
        ],
        "warnings": [],
    }
    text = """Note No. 2026-1630
Applicant: JOHN SMITH
Nationality: American
Passport No: X00176766
Date of issue: 9/6/2026
Date of expiry: 9/6/2031
Arrival date: 30 October 2026
Departure date: 30 November 2026
"""
    job = SimpleNamespace(
        input_text=text,
        case_id="case-1",
        document_id="document-1",
    )
    return SimpleNamespace(
        id="review-1",
        job_id="job-1",
        reviewed_output_json=json.dumps(payload),
        job=job,
    )


def test_replaces_numeric_and_textual_date_without_changing_format_style() -> None:
    text = "Issued 9/6/2026; arrival 30 October 2026."
    text, first_count = _replace_iso_date(text, "2026-06-09", "2026-07-11")
    text, second_count = _replace_iso_date(text, "2026-10-30", "2026-12-01")

    assert first_count == 1
    assert second_count == 1
    assert "11/7/2026" in text
    assert "01 December 2026" in text


def test_augmentation_changes_text_and_json_synchronously() -> None:
    review = source_review()

    example = augment_review(
        review,
        variation_index=1,
        seed=42,
        min_changes=2,
    )

    assert example.input_text != review.job.input_text
    assert example.target["note"]["request_type_code"] == "CAP_MOI"
    assert example.target["applicants"][0]["full_name"] != "John Smith"
    assert example.target["applicants"][0]["passport_number"] != "X00176766"
    assert "John Smith" not in example.input_text.title()
    assert "X00176766" not in example.input_text
    assert len(example.changes) >= 2
    CompactDocumentExtraction.model_validate(example.target)


def test_augmentation_is_deterministic_for_seed_and_variation() -> None:
    review = source_review()

    first = augment_review(review, variation_index=2, seed=99, min_changes=2)
    second = augment_review(review, variation_index=2, seed=99, min_changes=2)

    assert first.input_text == second.input_text
    assert first.target == second.target
    assert first.record_hash == second.record_hash


def test_exporter_marks_synthetic_example_and_keeps_it_in_train() -> None:
    review = source_review()
    generated = augment_review(review, variation_index=1, seed=42, min_changes=2)
    row = SimpleNamespace(
        id="synthetic-1",
        source_review_id=review.id,
        source_job_id=review.job_id,
        source_case_id="case-1",
        source_document_id="document-1",
        source_document=SimpleNamespace(original_filename="source.pdf"),
        input_text=generated.input_text,
        target_json=json.dumps(generated.target),
        generated_by="system:local-synthetic-v2",
        created_at=datetime.now(UTC),
        augmentation_version="local-synthetic-v2",
    )

    exported = build_synthetic_example(row)

    assert exported.is_synthetic is True
    assert exported.source_review_id == review.id
    assert exported.split == "train"
