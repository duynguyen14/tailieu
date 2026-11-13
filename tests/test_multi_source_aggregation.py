from pathlib import Path

from app.schemas.production_extraction import PassportMrzPageResult
from app.services.business_catalog import BusinessCatalogMapper
from app.services.multi_source_aggregation import build_multi_source_output


def test_passport_values_merge_by_exact_passport_and_are_catalog_codes() -> None:
    mapper = BusinessCatalogMapper.from_path(
        Path("config/business_catalog.v1.json")
    )
    target = {
        "n": {"no": "HANOI-6582", "snd": "Embassy of Canada", "rt": "CAP_MOI"},
        "a": [
            {
                "nm": "THERESSSA JOANNE DE HAAN",
                "dob": "1971-03-05",
                "sex": "F",
                "nat": "CAN",
                "pp": "DG411813",
                "pt": None,
                "pid": "2022-12-15",
                "pex": "2027-12-15",
                "vt": None,
                "ne": "MULTIPLE",
                "in": "2026-10-05",
                "out": "2026-10-09",
            }
        ],
    }
    result = PassportMrzPageResult(
        page_number=2,
        found=True,
        fields={
            "document_code": "P<",
            "surname": "DE HAAN",
            "given_names": "THERESSA JOANNE",
            "document_number": "DG411813",
            "nationality": "CAN",
            "birth_date": "1971-03-05",
            "sex": "Female",
            "expiry_date": "2027-12-16",
        },
    )

    output = build_multi_source_output(target, [result], mapper)

    assert output.note.note_number == ["HANOI-6582"]
    assert len(output.applicants) == 1
    applicant = output.applicants[0]
    assert applicant.full_name == [
        "THERESSSA JOANNE DE HAAN",
        "DE HAAN THERESSA JOANNE",
    ]
    assert applicant.gender == ["F"]
    assert applicant.nationality == ["CAN"]
    assert applicant.passport_type == ["P"]
    assert applicant.number_of_entries == ["MULTIPLE"]
    assert applicant.passport_expiration == ["2027-12-15", "2027-12-16"]


def test_unmatched_passport_is_returned_as_another_applicant_option() -> None:
    mapper = BusinessCatalogMapper.from_path(
        Path("config/business_catalog.v1.json")
    )
    target = {"n": {}, "a": []}
    result = PassportMrzPageResult(
        page_number=1,
        found=True,
        fields={
            "document_code": "P<",
            "surname": "DOE",
            "given_names": "JANE",
            "document_number": "AB1234567",
            "nationality": "USA",
            "birth_date": "1990-01-02",
            "sex": "F",
            "expiry_date": "2030-01-02",
        },
    )

    output = build_multi_source_output(target, [result], mapper)

    assert len(output.applicants) == 1
    assert output.applicants[0].passport_number == ["AB1234567"]
    assert output.applicants[0].nationality == ["USA"]
