from __future__ import annotations

from app.core.config import Settings
from app.services.business_catalog import BusinessCatalogMapper


def test_business_catalog_maps_closed_fields_and_preserves_open_fields() -> None:
    mapper = BusinessCatalogMapper.from_settings(Settings(_env_file=None))
    source = {
        "n": {"no": "123", "snd": "Embassy of Test", "rt": "CAP_MOI"},
        "a": [
            {
                "nm": "TEST PERSON",
                "dob": None,
                "sex": "M",
                "nat": "U.S.A",
                "pp": "A123",
                "pt": "công vụ",
                "pid": None,
                "pex": None,
                "vt": "Thị thực NG4",
                "ne": "nhiều lần",
                "in": None,
                "out": None,
            }
        ],
    }

    mapped, decisions = mapper.map_student_target(source)

    assert source["a"][0]["sex"] == "M"
    assert mapped["a"][0]["sex"] == "Male"
    assert mapped["a"][0]["nat"] == "U.S.A"
    assert mapped["a"][0]["pt"] == "Official"
    assert mapped["a"][0]["vt"] == "NG4"
    assert mapped["a"][0]["ne"] == "MULTIPLE"
    assert any(item.field == "nationality" and item.status == "UNMAPPED" for item in decisions)


def test_business_catalog_nulls_invalid_strict_visa_type() -> None:
    mapper = BusinessCatalogMapper.from_settings(Settings(_env_file=None))
    source = {
        "n": {"no": None, "snd": None, "rt": None},
        "a": [
            {
                "nm": "TEST",
                "dob": None,
                "sex": None,
                "nat": None,
                "pp": None,
                "pt": None,
                "pid": None,
                "pex": None,
                "vt": "diplomatic visa",
                "ne": None,
                "in": None,
                "out": None,
            }
        ],
    }

    mapped, decisions = mapper.map_student_target(source)

    assert mapped["a"][0]["vt"] is None
    visa_decision = next(item for item in decisions if item.field == "visa_type")
    assert visa_decision.status == "UNMAPPED"
    assert visa_decision.raw_value == "diplomatic visa"
