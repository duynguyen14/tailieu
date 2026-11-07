from tools.backfill_request_type_codes import classify_request_type


def test_classifies_new_visa_without_mistaking_closing_renewal() -> None:
    source = """
    The Embassy requests that the Department issue a Vietnamese visa.
    The Embassy avails itself of this opportunity to renew to the Department
    the assurances of its highest consideration.
    """
    assert classify_request_type("Request for visa issuance", source) == "CAP_MOI"


def test_classifies_mixed_visa_and_residence_as_extension() -> None:
    source = (
        "Kindly issue a multiple entry visa and extend her residence permit "
        "to 08 August 2026."
    )
    assert (
        classify_request_type(
            "request the latter's kind assistance in the following matter",
            source,
        )
        == "GIA_HAN"
    )


def test_new_passport_has_priority_over_extension() -> None:
    source = "Please extend the NG4 visa on the applicant's new passport."
    assert classify_request_type("extension of visa", source) == "CHUYEN_THI_THUC"


def test_non_visa_permit_and_id_request_is_other() -> None:
    assert (
        classify_request_type(
            "issuance of Temporary Residence Permits and Diplomatic IDs",
            "Request for Temporary Residence Permits and Diplomatic IDs.",
        )
        == "KHAC"
    )
