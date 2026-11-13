from app.services.passport_mrz.parser import parse_mrz


def test_parse_standard_td3_passport() -> None:
    result = parse_mrz(
        [
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        ]
    )

    assert result is not None
    assert result.document_type == "TD3"
    assert result.checksum_pass is True
    assert result.fields["surname"] == "ERIKSSON"
    assert result.fields["given_names"] == "ANNA MARIA"
    assert result.fields["document_number"] == "L898902C3"
    assert result.fields["nationality"] == "UTO"
    assert result.fields["birth_date"] == "1974-08-12"
    assert result.fields["sex"] == "F"
    assert result.fields["expiry_date"] == "2012-04-15"


def test_invalid_mrz_returns_none() -> None:
    assert parse_mrz(["not an mrz", "short"]) is None
