from __future__ import annotations

from tools.audit_kie_alignment import (
    AuditToken,
    FieldTarget,
    _date_variants,
    _normalized_bbox,
    choose_status,
    find_candidates,
)


def token(order: int, text: str, y1: int = 100) -> AuditToken:
    return AuditToken(
        review_id="review-1",
        token_id=f"token-{order}",
        token_order=order,
        text=text,
        page_id="page-1",
        page_number=2,
        page_order=1,
        page_width=1654,
        page_height=2339,
        source_width=1654,
        source_height=2339,
        image_relative_path="ocr_preprocessed/document/page/run.png",
        source_image_relative_path="documents/document/page.png",
        is_preprocessed=True,
        orientation_angle=90,
        preprocess_metadata={"rotation": 90, "scale": 1.2},
        x1=100,
        y1=y1,
        x2=600,
        y2=y1 + 40,
    )


def test_date_value_matches_human_readable_ocr_date() -> None:
    candidates = find_candidates(
        FieldTarget("applicant", 0, "date_of_birth", "1976-10-14"),
        [token(1, "Date of birth: 14/10/1976")],
        minimum_score=0.88,
        max_span_tokens=4,
    )

    status, chosen = choose_status(
        candidates, minimum_score=0.88, ambiguity_margin=0.035
    )

    assert status == "MATCHED"
    assert chosen is not None
    assert chosen.page_number == 2
    assert chosen.token_ids == ("token-1",)


def test_repeated_value_is_ambiguous() -> None:
    candidates = find_candidates(
        FieldTarget("applicant", 0, "nationality", "Australia"),
        [token(1, "Australia", 100), token(2, "Australia", 800)],
        minimum_score=0.88,
        max_span_tokens=2,
    )

    status, chosen = choose_status(
        candidates, minimum_score=0.88, ambiguity_margin=0.035
    )

    assert status == "AMBIGUOUS"
    assert chosen is None


def test_split_name_matches_across_adjacent_ocr_tokens() -> None:
    candidates = find_candidates(
        FieldTarget("applicant", 0, "full_name", "Nicholas John Hogarth"),
        [
            token(1, "NICHOLAS", 100),
            token(2, "JOHN", 145),
            token(3, "HOGARTH", 190),
        ],
        minimum_score=0.88,
        max_span_tokens=4,
    )

    status, chosen = choose_status(
        candidates, minimum_score=0.88, ambiguity_margin=0.035
    )

    assert status == "MATCHED"
    assert chosen is not None
    assert chosen.token_ids == ("token-1", "token-2", "token-3")


def test_date_variants_cover_english_and_vietnamese_documents() -> None:
    variants = _date_variants("2026-10-14")

    assert "14 October 2026" in variants
    assert "October 14, 2026" in variants
    assert "14 Oct 26" in variants
    assert "ngày 14 tháng 10 năm 2026" in variants
    assert "ngày 14 tháng mười năm 2026" in variants


def test_iso_teacher_date_matches_english_month_ocr_text() -> None:
    candidates = find_candidates(
        FieldTarget("applicant", 0, "entry_date", "2026-10-14"),
        [token(1, "Proposed date of entry: 14th October 2026")],
        minimum_score=0.88,
        max_span_tokens=4,
    )

    status, chosen = choose_status(
        candidates, minimum_score=0.88, ambiguity_margin=0.035
    )

    assert status == "MATCHED"
    assert chosen is not None
    assert "date_context" in chosen.method


def test_bbox_is_normalized_against_processed_image_dimensions() -> None:
    assert _normalized_bbox((100, 200, 500, 600), 1000, 2000) == [100, 100, 500, 300]
