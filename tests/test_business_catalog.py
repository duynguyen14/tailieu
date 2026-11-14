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
    assert mapped["a"][0]["sex"] == "M"
    assert mapped["a"][0]["nat"] == "USA"
    assert mapped["a"][0]["pt"] == "O"
    assert mapped["a"][0]["vt"] == "NG4"
    assert mapped["a"][0]["ne"] == "MULTIPLE"
    assert any(item.field == "nationality" and item.status == "ALIAS" for item in decisions)


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


def test_business_catalog_maps_multilingual_country_aliases() -> None:
    mapper = BusinessCatalogMapper.from_settings(Settings(_env_file=None))

    assert mapper.map_value("nationality", "Française") == ("FRA", "ALIAS")
    assert mapper.map_value("nationality", "Deutschland") == ("DEU", "ALIAS")
    assert mapper.map_value("nationality", "日本") == ("JPN", "ALIAS")
    assert mapper.map_value("nationality", "Francheniste") == ("FRA", "ALIAS")


def test_business_catalog_fuzzy_match_is_scored_and_conservative() -> None:
    mapper = BusinessCatalogMapper.from_settings(Settings(_env_file=None))
    source = {
        "n": {"no": None, "snd": None, "rt": None},
        "a": [
            {
                "sex": None,
                "nat": "not a known nationality",
                "pt": "công vụu",
                "vt": None,
                "ne": None,
            }
        ],
    }

    mapped, decisions = mapper.map_student_target(source)

    assert mapped["a"][0]["pt"] == "O"
    passport_decision = next(item for item in decisions if item.field == "passport_type")
    assert passport_decision.status == "FUZZY"
    assert passport_decision.matched_alias == "Công vụ"
    assert passport_decision.match_score == 93.33
    assert passport_decision.second_best_score is not None
    assert mapped["a"][0]["nat"] is None
    nationality_decision = next(item for item in decisions if item.field == "nationality")
    assert nationality_decision.status == "UNMAPPED"
    assert nationality_decision.match_score is not None


def test_business_catalog_disambiguates_iso_code_from_localized_alias() -> None:
    mapper = BusinessCatalogMapper.from_settings(Settings(_env_file=None))

    assert mapper.map_value("nationality", "NGA") == ("NGA", "EXACT")
    assert mapper.map_value("nationality", "Nga") == ("RUS", "ALIAS")
    assert mapper.map_value("nationality", "ZZZ") == (None, "UNMAPPED")
