from __future__ import annotations

import pytest

from tools.export_kie_ser_dataset import (
    EntitySpan,
    ExportToken,
    assign_bio_labels,
    normalized_bbox,
    segment_ocr_token,
    split_document_ids,
    validate_entity_identity,
)


def test_segment_long_ocr_line_at_precise_entity_span() -> None:
    text = "approval number (NG4) at the International Airports"
    token = ExportToken("token-1", 1, text, (100, 200, 1100, 250))
    start = text.index("NG4")
    spans = [
        EntitySpan(
            entity_id="entity-1",
            label="VISA_TYPE",
            field_path="applicants[0].visa_type",
            applicant_index=0,
            token_id=token.id,
            char_start=start,
            char_end=start + 3,
        )
    ]

    segments = segment_ocr_token(token, spans)
    assign_bio_labels(segments, {"entity-1": "VISA_TYPE"})

    assert [segment.text for segment in segments] == [
        "approval",
        "number",
        "(",
        "NG4",
        ")",
        "at",
        "the",
        "International",
        "Airports",
    ]
    assert [segment.label for segment in segments].count("B-VISA_TYPE") == 1
    assert next(segment for segment in segments if segment.text == "NG4").label == "B-VISA_TYPE"


def test_multiword_entity_gets_bio_labels() -> None:
    text = "Name Nicholas John Hogarth"
    token = ExportToken("token-1", 1, text, (0, 0, 520, 40))
    start = text.index("Nicholas")
    spans = [
        EntitySpan(
            entity_id="entity-1",
            label="FULL_NAME",
            field_path="applicants[0].full_name",
            applicant_index=0,
            token_id=token.id,
            char_start=start,
            char_end=len(text),
        )
    ]

    segments = segment_ocr_token(token, spans)
    assign_bio_labels(segments, {"entity-1": "FULL_NAME"})

    assert [segment.label for segment in segments] == [
        "O",
        "B-FULL_NAME",
        "I-FULL_NAME",
        "I-FULL_NAME",
    ]


def test_overlapping_entities_are_rejected() -> None:
    token = ExportToken("token-1", 1, "ABCDEF", (0, 0, 60, 20))
    spans = [
        EntitySpan("e1", "FULL_NAME", "applicants[0].full_name", 0, token.id, 0, 4),
        EntitySpan("e2", "NOTE_NUMBER", "note.note_number", None, token.id, 2, 6),
    ]

    with pytest.raises(ValueError, match="overlap"):
        segment_ocr_token(token, spans)


def test_document_split_has_no_leakage_and_keeps_validation_and_test() -> None:
    document_ids = [f"document-{index}" for index in range(10)]
    result = split_document_ids(
        document_ids,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    assert set(result) == set(document_ids)
    assert list(result.values()).count("train") == 8
    assert list(result.values()).count("val") == 1
    assert list(result.values()).count("test") == 1


def test_normalized_bbox_is_layoutlm_compatible() -> None:
    assert normalized_bbox((100, 200, 500, 600), 1000, 2000) == [100, 100, 500, 300]


def test_entity_identity_requires_applicant_index_to_match_field_path() -> None:
    with pytest.raises(ValueError, match="does not match"):
        validate_entity_identity(
            entity_id="entity-1",
            label="FULL_NAME",
            field_path="applicants[2].full_name",
            applicant_index=1,
        )


def test_entity_identity_requires_label_to_match_field_name() -> None:
    with pytest.raises(ValueError, match="expected NOTE_NUMBER"):
        validate_entity_identity(
            entity_id="entity-1",
            label="FULL_NAME",
            field_path="note.note_number",
            applicant_index=None,
        )
