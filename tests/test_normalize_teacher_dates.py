import pytest

from tools.normalize_teacher_dates import (
    DateNormalizationError,
    normalize_date_value,
    normalize_review_dates,
)


def test_normalizes_audited_ambiguous_dmy_date() -> None:
    assert (
        normalize_date_value(
            "9/6/2026",
            case_id="4fe00b6a-ea1b-4495-86af-5fa8480b7291",
        )
        == "2026-06-09"
    )


def test_rejects_unknown_ambiguous_numeric_date() -> None:
    with pytest.raises(DateNormalizationError, match="ambiguous numeric"):
        normalize_date_value("9/6/2026", case_id="unknown-case")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("28/09/2022", "2022-09-28"),
        ("2025/07/22", "2025-07-22"),
        ("15 February 2023", "2023-02-15"),
        ("September 09, 2027", "2027-09-09"),
        ("01st August 2026", "2026-08-01"),
        ("27April 2026", "2026-04-27"),
    ],
)
def test_normalizes_supported_exact_dates(source: str, expected: str) -> None:
    assert normalize_date_value(source, case_id="case") == expected


def test_applies_visually_verified_manual_override() -> None:
    raw = {
        "applicants": [
            {
                "passport_issue_date": None,
                "passport_expiration": "2028-09-42",
                "entry_date": None,
                "exit_date": None,
            }
        ]
    }
    normalized, changes = normalize_review_dates(
        raw,
        case_id="f28f6ad0-1ad0-4766-a3b3-a0ec7b474ca0",
    )
    assert normalized["applicants"][0]["passport_expiration"] == "2028-09-28"
    assert len(changes) == 1


def test_normalizes_audited_ambiguous_birth_date() -> None:
    assert (
        normalize_date_value(
            "05/06/1982",
            case_id="2d7c724e-b96e-4786-b835-8808cef51326",
            field="date_of_birth",
        )
        == "1982-06-05"
    )


def test_normalizes_year_first_date_with_single_digit_components() -> None:
    assert (
        normalize_date_value(
            "1977-1-10",
            case_id="case",
            field="date_of_birth",
        )
        == "1977-01-10"
    )
