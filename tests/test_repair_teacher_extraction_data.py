from tools.repair_teacher_extraction_data import (
    classify_request_type_strict,
    repair_output,
)


def test_diplomatic_closing_does_not_become_extension() -> None:
    text = (
        "Please issue a Vietnamese visa. The Embassy avails itself of this "
        "opportunity to renew the assurances of its highest consideration."
    )
    assert classify_request_type_strict(text).code == "CAP_MOI"


def test_explicit_visa_extension_wins_over_issuance() -> None:
    text = "Please issue a visa and extend her residence permit until August 2027."
    assert classify_request_type_strict(text).code == "GIA_HAN"


def test_plural_visa_extension_is_detected() -> None:
    assert (
        classify_request_type_strict("Please renew multiple visas for the officers.").code
        == "GIA_HAN"
    )


def test_request_for_visas_is_new_issuance() -> None:
    decision = classify_request_type_strict(
        "The Embassy requests visas for two officers."
    )
    assert decision.code == "CAP_MOI"


def test_new_passport_visa_is_transfer() -> None:
    text = "Please transfer the valid visa to her new passport."
    assert classify_request_type_strict(text).code == "CHUYEN_THI_THUC"


def test_trip_word_does_not_become_visa_transfer() -> None:
    text = "Chuyến đi bắt đầu ngày 28/11/2026. Thị thực xin được cấp tại Berlin."
    assert classify_request_type_strict(text).code is None


def test_reissue_without_new_passport_is_new_issuance() -> None:
    decision = classify_request_type_strict("Please reissue a multiple-entry visa.")
    assert decision.code == "CAP_MOI"


def test_vietnamese_d_stroke_is_normalized_for_transfer() -> None:
    text = "Đề nghị hỗ trợ chuyển đổi loại thị thực từ NG4 sang NG3."
    assert classify_request_type_strict(text).code == "CHUYEN_THI_THUC"


def test_minor_ocr_duplication_in_thi_thuc_is_tolerated() -> None:
    text = "Đề nghị hỗ trợ cấp thịi thực NG3 sử dụng nhiều lần."
    assert classify_request_type_strict(text).code == "CAP_MOI"


def test_repairs_known_wrong_field_and_invalid_link() -> None:
    raw = {
        "schema_version": "compact-document-v1",
        "case_id": "case-1",
        "document_id": "document-1",
        "note": {"request_type_code": "CHUYEN_THI_THUC"},
        "applicants": [
            {
                "person_id": "person_1",
                "visa_type": "entrée multiples",
                "number_of_entries": None,
                "related_to_person_id": "person_2",
            }
        ],
        "warnings": [],
    }
    repaired, changes, _ = repair_output(raw, "We kindly request issuance of a visa.")
    assert repaired["note"]["request_type_code"] == "CAP_MOI"
    assert repaired["applicants"][0]["visa_type"] is None
    assert repaired["applicants"][0]["number_of_entries"] == "MULTIPLE"
    assert repaired["applicants"][0]["related_to_person_id"] is None
    assert len(changes) == 4
