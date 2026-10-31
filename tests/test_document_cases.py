from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.case_extraction import CaseExtractionTarget
from tools.build_document_cases import group_document_pages
from tools.build_grounded_sft_dataset import collect_evidence
from tools.export_llm_annotation_seed import source_token_id


def page(number: int, label: str | None, status: str = "REVIEWED") -> SimpleNamespace:
    return SimpleNamespace(
        id=f"page-{number}",
        page_number=number,
        label=label,
        review_status=status,
    )


def test_groups_note_and_following_applicant_pages_into_one_case() -> None:
    cases = group_document_pages(
        [
            page(1, "DIPLOMATIC_NOTE"),
            page(2, "APPLICANT_TABLE"),
            page(3, "APPLICANT_LIST"),
            page(4, "OTHER"),
            page(5, "DIPLOMATIC_NOTE"),
        ]
    )

    assert len(cases) == 2
    assert [item.page_number for item in cases[0].pages] == [1, 2, 3]
    assert [item.page_role for item in cases[0].pages] == [
        "NOTE",
        "APPLICANT_DATA",
        "APPLICANT_DATA",
    ]
    assert [item.page_number for item in cases[1].pages] == [5]


def test_unreviewed_page_is_a_hard_case_boundary() -> None:
    cases = group_document_pages(
        [
            page(1, "DIPLOMATIC_NOTE"),
            page(2, "APPLICANT_TABLE", status="UNLABELED"),
            page(3, "APPLICANT_LIST"),
        ]
    )

    assert len(cases) == 2
    assert [item.page_number for item in cases[0].pages] == [1]
    assert [item.page_number for item in cases[1].pages] == [3]
    assert cases[1].status == "NEEDS_REVIEW"


def test_orphan_applicant_pages_are_kept_for_review() -> None:
    cases = group_document_pages(
        [page(2, "APPLICANT_TABLE"), page(3, "APPLICANT_LIST")]
    )

    assert len(cases) == 1
    assert cases[0].status == "NEEDS_REVIEW"
    assert [item.page_number for item in cases[0].pages] == [2, 3]


def test_case_evidence_uses_page_prefixed_token_ids() -> None:
    assert source_token_id(2, "t_0014") == "p2_t_0014"
    target = {
        "applicants": [
            {
                "passport_number": {
                    "value": "RF0210530",
                    "evidence": [{"page_number": 2, "token_ids": ["p2_t_0014"]}],
                }
            }
        ]
    }

    assert collect_evidence(target) == ["p2_t_0014"]


def test_case_target_requires_grounded_non_null_values() -> None:
    with pytest.raises(ValidationError):
        CaseExtractionTarget.model_validate(
            {
                "case_id": "case-1",
                "document_id": "document-1",
                "note": {"note_number": {"value": "850-2026"}},
            }
        )


def test_case_target_validates_page_prefix_on_evidence() -> None:
    with pytest.raises(ValidationError):
        CaseExtractionTarget.model_validate(
            {
                "case_id": "case-1",
                "document_id": "document-1",
                "applicants": [
                    {
                        "person_id": "person-1",
                        "passport_number": {
                            "value": "RF0210530",
                            "evidence": [
                                {
                                    "page_id": "page-2",
                                    "page_number": 2,
                                    "token_ids": ["p1_t_0014"],
                                }
                            ],
                        },
                    }
                ],
            }
        )
