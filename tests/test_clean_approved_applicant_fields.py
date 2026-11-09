from tools.clean_approved_applicant_fields import (
    canonical_visa_type,
    clean_output,
    strip_name_titles,
)


def test_strips_common_and_ocr_honorifics() -> None:
    assert strip_name_titles("Mr. John Smith") == ("John Smith", "Male")
    assert strip_name_titles("Mrs Jane Smith") == ("Jane Smith", "Female")
    assert strip_name_titles("Miss. Jane Smith") == ("Jane Smith", "Female")
    assert strip_name_titles("Mr: John Smith") == ("John Smith", "Male")
    assert strip_name_titles("MIr. John Smith") == ("John Smith", "Male")
    assert strip_name_titles("H.E. Dr. Jane Smith") == ("Jane Smith", None)


def test_does_not_treat_name_prefix_as_honorific() -> None:
    assert strip_name_titles("Herwig Neuper") == ("Herwig Neuper", None)
    assert strip_name_titles("HELMUT LEOPOLD") == ("HELMUT LEOPOLD", None)
    assert strip_name_titles("HEND YOUSEF") == ("HEND YOUSEF", None)
    assert strip_name_titles("DRAPEAU Vincent") == ("DRAPEAU Vincent", None)


def test_canonicalizes_only_explicit_ng_code() -> None:
    assert canonical_visa_type("NG4") == "NG4"
    assert canonical_visa_type("Thị thực NG4") == "NG4"
    assert canonical_visa_type("NG4 single-entry visa on arrival") == "NG4"
    assert canonical_visa_type("diplomatic visa") is None
    assert canonical_visa_type("multiple") is None


def test_cleanup_infers_missing_gender_but_preserves_existing_gender() -> None:
    raw = {
        "applicants": [
            {"full_name": "Ms. Jane Smith", "gender": None, "visa_type": "Thị thực NG4"},
            {"full_name": "Mr. John Smith", "gender": "Male", "visa_type": "multiple"},
        ]
    }
    cleaned, changes = clean_output(raw)

    assert cleaned["applicants"][0] == {
        "full_name": "Jane Smith",
        "gender": "Female",
        "visa_type": "NG4",
    }
    assert cleaned["applicants"][1] == {
        "full_name": "John Smith",
        "gender": "Male",
        "visa_type": None,
    }
    assert len(changes) == 5
