from __future__ import annotations

import pytest

from app.services.visa_form_anchor_extractor import (
    FIELD_CODES_ORDERED,
    VisaFormAnchorExtractor,
    extract_visa_form_fields,
    group_tokens_into_lines,
    normalize_tokens,
    strip_accents_and_lower,
)


def test_strip_accents_and_lower():
    assert strip_accents_and_lower("Tên đầy đủ") == "ten day du"
    assert strip_accents_and_lower("Họ và tên") == "ho va ten"
    assert strip_accents_and_lower("Ngày sinh (DD/MM/YYYY)") == "ngay sinh (dd/mm/yyyy)"
    assert strip_accents_and_lower("Nơi sinh") == "noi sinh"
    assert strip_accents_and_lower("Giới tính") == "gioi tinh"
    assert strip_accents_and_lower("Quốc tịch") == "quoc tich"
    assert strip_accents_and_lower("Số hộ chiếu") == "so ho chieu"
    assert strip_accents_and_lower("Ngày hết hạn hộ chiếu") == "ngay het han ho chieu"
    assert strip_accents_and_lower("Nơi nhận thị thực") == "noi nhan thi thuc"


def test_normalize_tokens():
    raw = [
        {"text": "  Hello  ", "x1": 10, "y1": 20, "x2": 50, "y2": 40},
        {"text": "", "x1": 0, "y1": 0, "x2": 1, "y2": 1},  # empty string ignored
        {"text": "World", "bbox": (60, 20, 100, 40)},
    ]
    tokens = normalize_tokens(raw, width=1000, height=1000)
    assert len(tokens) == 2
    assert tokens[0].text == "Hello"
    assert tokens[0].clean_text == "hello"
    assert tokens[0].x1 == 10
    assert tokens[1].text == "World"
    assert tokens[1].x1 == 60


def test_group_tokens_into_lines():
    raw = [
        {"text": "Line1_A", "x1": 10, "y1": 20, "x2": 50, "y2": 40},
        {"text": "Line2_A", "x1": 10, "y1": 60, "x2": 50, "y2": 80},
        {"text": "Line1_B", "x1": 60, "y1": 22, "x2": 100, "y2": 42},
    ]
    tokens = normalize_tokens(raw, width=1000, height=1000)
    lines = group_tokens_into_lines(tokens)
    assert len(lines) == 2
    assert [t.text for t in lines[0].tokens] == ["Line1_A", "Line1_B"]
    assert [t.text for t in lines[1].tokens] == ["Line2_A"]


def test_fallback_on_empty_tokens():
    results = extract_visa_form_fields([], 1000, 1000)
    assert len(results) == 19
    assert [r.field_code for r in results] == list(FIELD_CODES_ORDERED)
    assert [r.field_order for r in results] == list(range(1, 20))
    for r in results:
        assert not r.anchor_found
        assert r.value_text is None
        x1, y1, x2, y2 = r.bbox
        assert 0 <= x1 < x2 <= 1000
        assert 0 <= y1 < y2 <= 1000
        nx1, ny1, nx2, ny2 = r.normalized_bbox
        assert 0 <= nx1 < nx2 <= 1000
        assert 0 <= ny1 < ny2 <= 1000


def test_full_visa_form_anchor_and_value_extraction():
    tokens = [
        # 1. Full name (bilingual two lines)
        {"text": "1.", "x1": 60, "y1": 170, "x2": 80, "y2": 190},
        {"text": "Full", "x1": 85, "y1": 170, "x2": 120, "y2": 190},
        {"text": "name", "x1": 125, "y1": 170, "x2": 165, "y2": 190},
        {"text": "JUSTIN", "x1": 180, "y1": 170, "x2": 250, "y2": 190},
        {"text": "YOUNG", "x1": 255, "y1": 170, "x2": 320, "y2": 190},
        {"text": "Tên", "x1": 60, "y1": 192, "x2": 90, "y2": 208},
        {"text": "đầy", "x1": 95, "y1": 192, "x2": 125, "y2": 208},
        {"text": "đủ:", "x1": 130, "y1": 192, "x2": 155, "y2": 208},
        # 2. Date of birth
        {"text": "2.", "x1": 60, "y1": 210, "x2": 80, "y2": 230},
        {"text": "Date", "x1": 85, "y1": 210, "x2": 120, "y2": 230},
        {"text": "of", "x1": 125, "y1": 210, "x2": 140, "y2": 230},
        {"text": "birth", "x1": 145, "y1": 210, "x2": 185, "y2": 230},
        {"text": "(DD/MM/YYYY):", "x1": 190, "y1": 210, "x2": 280, "y2": 230},
        {"text": "27/09/1982", "x1": 290, "y1": 210, "x2": 370, "y2": 230},
        # 3. Place of birth
        {"text": "3.", "x1": 60, "y1": 245, "x2": 80, "y2": 265},
        {"text": "Place", "x1": 85, "y1": 245, "x2": 125, "y2": 265},
        {"text": "of", "x1": 130, "y1": 245, "x2": 145, "y2": 265},
        {"text": "birth:", "x1": 150, "y1": 245, "x2": 190, "y2": 265},
        {"text": "USA", "x1": 200, "y1": 245, "x2": 240, "y2": 265},
        # 4. Gender
        {"text": "4.", "x1": 60, "y1": 280, "x2": 80, "y2": 300},
        {"text": "Gender:", "x1": 85, "y1": 280, "x2": 140, "y2": 300},
        {"text": "Male", "x1": 150, "y1": 280, "x2": 190, "y2": 300},
        # 5. Nationality
        {"text": "5.", "x1": 60, "y1": 315, "x2": 80, "y2": 335},
        {"text": "Nationality:", "x1": 85, "y1": 315, "x2": 160, "y2": 335},
        {"text": "USA", "x1": 170, "y1": 315, "x2": 210, "y2": 335},
        # 6. Contact information header
        {"text": "6.", "x1": 60, "y1": 350, "x2": 80, "y2": 370},
        {"text": "Contact", "x1": 85, "y1": 350, "x2": 140, "y2": 370},
        {"text": "information:", "x1": 145, "y1": 350, "x2": 225, "y2": 370},
        # Address
        {"text": "Address:", "x1": 70, "y1": 375, "x2": 135, "y2": 395},
        {"text": "850", "x1": 145, "y1": 375, "x2": 175, "y2": 395},
        {"text": "NORTH", "x1": 180, "y1": 375, "x2": 235, "y2": 395},
        {"text": "RANDOLPH", "x1": 240, "y1": 375, "x2": 320, "y2": 395},
        {"text": "STREET,", "x1": 145, "y1": 398, "x2": 215, "y2": 415},
        {"text": "ARLINGTON", "x1": 220, "y1": 398, "x2": 310, "y2": 415},
        # Telephone
        {"text": "Telephone:", "x1": 70, "y1": 420, "x2": 150, "y2": 440},
        {"text": "3152728501", "x1": 160, "y1": 420, "x2": 250, "y2": 440},
        # Email
        {"text": "Email:", "x1": 70, "y1": 455, "x2": 120, "y2": 475},
        {"text": "YOUNGJPC@STATE.GOV", "x1": 130, "y1": 455, "x2": 300, "y2": 475},
        # 7. Sponsor information header (blank subfields)
        {"text": "7.", "x1": 60, "y1": 490, "x2": 80, "y2": 510},
        {"text": "Sponsor", "x1": 85, "y1": 490, "x2": 140, "y2": 510},
        {"text": "information:", "x1": 145, "y1": 490, "x2": 225, "y2": 510},
        {"text": "Name:", "x1": 70, "y1": 515, "x2": 120, "y2": 535},
        {"text": "Telephone:", "x1": 70, "y1": 545, "x2": 150, "y2": 565},
        # Right column
        # 8. Passport number
        {"text": "8.", "x1": 510, "y1": 210, "x2": 530, "y2": 230},
        {"text": "Passport", "x1": 535, "y1": 210, "x2": 600, "y2": 230},
        {"text": "number:", "x1": 605, "y1": 210, "x2": 660, "y2": 230},
        {"text": "X00165196", "x1": 670, "y1": 210, "x2": 770, "y2": 230},
        # Date of expiry
        {"text": "Date", "x1": 510, "y1": 245, "x2": 545, "y2": 265},
        {"text": "of", "x1": 550, "y1": 245, "x2": 565, "y2": 265},
        {"text": "expiry:", "x1": 570, "y1": 245, "x2": 620, "y2": 265},
        {"text": "17/02/2031", "x1": 630, "y1": 245, "x2": 720, "y2": 265},
        # 9. Number of entries
        {"text": "9.", "x1": 510, "y1": 280, "x2": 530, "y2": 300},
        {"text": "Number", "x1": 535, "y1": 280, "x2": 590, "y2": 300},
        {"text": "of", "x1": 595, "y1": 280, "x2": 610, "y2": 300},
        {"text": "entries:", "x1": 615, "y1": 280, "x2": 670, "y2": 300},
        {"text": "Multiple", "x1": 680, "y1": 280, "x2": 740, "y2": 300},
        # 10. Proposed date of entry
        {"text": "10.", "x1": 510, "y1": 315, "x2": 535, "y2": 335},
        {"text": "Proposed", "x1": 540, "y1": 315, "x2": 610, "y2": 335},
        {"text": "date", "x1": 615, "y1": 315, "x2": 645, "y2": 335},
        {"text": "of", "x1": 650, "y1": 315, "x2": 665, "y2": 335},
        {"text": "entry:", "x1": 670, "y1": 315, "x2": 715, "y2": 335},
        {"text": "30/07/2026", "x1": 725, "y1": 315, "x2": 815, "y2": 335},
        # 11. Proposed date of exit
        {"text": "11.", "x1": 510, "y1": 350, "x2": 535, "y2": 370},
        {"text": "Proposed", "x1": 540, "y1": 350, "x2": 610, "y2": 370},
        {"text": "date", "x1": 615, "y1": 350, "x2": 645, "y2": 370},
        {"text": "of", "x1": 650, "y1": 350, "x2": 665, "y2": 370},
        {"text": "exit:", "x1": 670, "y1": 350, "x2": 710, "y2": 370},
        {"text": "30/07/2027", "x1": 720, "y1": 350, "x2": 810, "y2": 370},
        # 12. Place of visa submission
        {"text": "12.", "x1": 510, "y1": 390, "x2": 535, "y2": 410},
        {"text": "Place", "x1": 540, "y1": 390, "x2": 580, "y2": 410},
        {"text": "of", "x1": 585, "y1": 390, "x2": 600, "y2": 410},
        {"text": "visa", "x1": 605, "y1": 390, "x2": 635, "y2": 410},
        {"text": "submission:", "x1": 640, "y1": 390, "x2": 730, "y2": 410},
        {"text": "Consular", "x1": 550, "y1": 420, "x2": 620, "y2": 438},
        {"text": "Department", "x1": 625, "y1": 420, "x2": 710, "y2": 438},
        # 13. Place of visa collection
        {"text": "13.", "x1": 510, "y1": 455, "x2": 535, "y2": 475},
        {"text": "Place", "x1": 540, "y1": 455, "x2": 580, "y2": 475},
        {"text": "of", "x1": 585, "y1": 455, "x2": 600, "y2": 475},
        {"text": "visa", "x1": 605, "y1": 455, "x2": 635, "y2": 475},
        {"text": "collection:", "x1": 640, "y1": 455, "x2": 720, "y2": 475},
        {"text": "Viet", "x1": 550, "y1": 480, "x2": 580, "y2": 498},
        {"text": "Nam", "x1": 585, "y1": 480, "x2": 615, "y2": 498},
        {"text": "Embassy", "x1": 620, "y1": 480, "x2": 680, "y2": 498},
        # 14. Purpose of entry
        {"text": "14.", "x1": 510, "y1": 515, "x2": 535, "y2": 535},
        {"text": "Purpose", "x1": 540, "y1": 515, "x2": 600, "y2": 535},
        {"text": "of", "x1": 605, "y1": 515, "x2": 620, "y2": 535},
        {"text": "entry:", "x1": 625, "y1": 515, "x2": 670, "y2": 535},
        {"text": "Official", "x1": 680, "y1": 515, "x2": 735, "y2": 535},
        {"text": "visit", "x1": 740, "y1": 515, "x2": 770, "y2": 535},
        # 19. Application date
        {"text": "Date/", "x1": 600, "y1": 700, "x2": 640, "y2": 720},
        {"text": "Ngày:", "x1": 645, "y1": 700, "x2": 685, "y2": 720},
        {"text": "28/02/2026", "x1": 695, "y1": 700, "x2": 785, "y2": 720},
    ]

    results = extract_visa_form_fields(tokens, 1000, 1000)
    assert len(results) == 19

    by_code = {r.field_code: r for r in results}

    assert by_code["full_name"].value_text == "JUSTIN YOUNG"
    assert by_code["full_name"].anchor_found is True

    assert by_code["date_of_birth"].value_text == "27/09/1982"
    assert by_code["place_of_birth"].value_text == "USA"
    assert by_code["gender"].value_text == "Male"
    assert by_code["nationality"].value_text == "USA"

    assert by_code["contact_address"].value_text == "850 NORTH RANDOLPH STREET, ARLINGTON"
    assert by_code["contact_phone"].value_text == "3152728501"
    assert by_code["contact_email"].value_text == "YOUNGJPC@STATE.GOV"

    # Sponsor fields are blank
    assert by_code["sponsor_name"].value_text is None
    assert by_code["sponsor_name"].anchor_found is True
    assert by_code["sponsor_phone"].value_text is None
    assert by_code["sponsor_phone"].anchor_found is True

    assert by_code["passport_number"].value_text == "X00165196"
    assert by_code["passport_expiration"].value_text == "17/02/2031"
    assert by_code["number_of_entries"].value_text == "Multiple"
    assert by_code["entry_date"].value_text == "30/07/2026"
    assert by_code["exit_date"].value_text == "30/07/2027"

    assert by_code["submission_place"].value_text == "Consular Department"
    assert by_code["collection_place"].value_text == "Viet Nam Embassy"
    assert by_code["purpose_of_entry"].value_text == "Official visit"
    assert by_code["application_date"].value_text == "28/02/2026"

    # All bounding boxes must satisfy DB check constraints
    for r in results:
        x1, y1, x2, y2 = r.bbox
        assert 0 <= x1 < x2 <= 1000, f"Failed x1 < x2 for {r.field_code}: {r.bbox}"
        assert 0 <= y1 < y2 <= 1000, f"Failed y1 < y2 for {r.field_code}: {r.bbox}"
        nx1, ny1, nx2, ny2 = r.normalized_bbox
        assert 0 <= nx1 < nx2 <= 1000, f"Failed nx1 < nx2 for {r.field_code}: {r.normalized_bbox}"
        assert 0 <= ny1 < ny2 <= 1000, f"Failed ny1 < ny2 for {r.field_code}: {r.normalized_bbox}"


def test_shifted_page_coordinate_tracking():
    # Simulate a page where content is shifted down by 40 units
    shift_tokens = [
        {"text": "1.", "x1": 60, "y1": 210, "x2": 80, "y2": 230},
        {"text": "Full", "x1": 85, "y1": 210, "x2": 120, "y2": 230},
        {"text": "name:", "x1": 125, "y1": 210, "x2": 165, "y2": 230},
        {"text": "JUSTIN", "x1": 180, "y1": 210, "x2": 250, "y2": 230},
        {"text": "8.", "x1": 510, "y1": 250, "x2": 530, "y2": 270},
        {"text": "Passport", "x1": 535, "y1": 250, "x2": 600, "y2": 270},
        {"text": "number:", "x1": 605, "y1": 250, "x2": 660, "y2": 270},
        {"text": "X00165196", "x1": 670, "y1": 250, "x2": 770, "y2": 270},
    ]

    results = extract_visa_form_fields(shift_tokens, 1000, 1000)
    by_code = {r.field_code: r for r in results}

    assert by_code["full_name"].anchor_found is True
    assert by_code["full_name"].value_text == "JUSTIN"
    # full_name Y is centered around 210-230
    assert by_code["full_name"].bbox[1] >= 200

    # date_of_birth was NOT present in tokens, but its synthesized bbox must track the shift
    assert by_code["date_of_birth"].anchor_found is False
    assert by_code["date_of_birth"].bbox[1] > 230


def test_service_create_template_with_ocr_tokens():
    from app.core.config import get_settings
    from app.db.database import Database
    from app.db.models import AnnotationDocument, AnnotationPage, VisaFormAnnotation
    from app.services.storage import AnnotationStorage
    from app.services.visa_form_annotation import VisaFormAnnotationService
    from sqlalchemy import select
    from uuid import uuid4

    settings = get_settings()
    db = Database(settings)
    storage = AnnotationStorage(settings.annotation_storage_path)
    service = VisaFormAnnotationService(database=db, storage=storage)

    # Create dummy doc & page in session
    test_page_id = str(uuid4())
    test_doc_id = str(uuid4())
    storage.write_bytes("tmp/test_vf_p1.png", b"\x89PNG\r\n\x1a\nfake_image_bytes")
    with db.session() as session:
        doc = AnnotationDocument(
            id=test_doc_id,
            original_filename="test_vf.pdf",
            pdf_relative_path="tmp/test_vf.pdf",
            sha256="test_hash_vf",
            page_count=1,
            review_status="REVIEWED",
        )
        page = AnnotationPage(
            id=test_page_id,
            document_id=test_doc_id,
            page_number=1,
            page_index=0,
            label="VISA_APPLICATION_FORM",
            review_status="REVIEWED",
            image_relative_path="tmp/test_vf_p1.png",
            mime_type="image/png",
            width=1000,
            height=1000,
        )
        session.add(doc)
        session.add(page)
        session.commit()

    sample_tokens = [
        {"text": "1.", "x1": 60, "y1": 170, "x2": 80, "y2": 190},
        {"text": "Full", "x1": 85, "y1": 170, "x2": 120, "y2": 190},
        {"text": "name:", "x1": 125, "y1": 170, "x2": 165, "y2": 190},
        {"text": "CARDALEEN", "x1": 180, "y1": 170, "x2": 260, "y2": 190},
        {"text": "YOUNG", "x1": 265, "y1": 170, "x2": 330, "y2": 190},
        {"text": "8.", "x1": 510, "y1": 210, "x2": 530, "y2": 230},
        {"text": "Passport", "x1": 535, "y1": 210, "x2": 600, "y2": 230},
        {"text": "number:", "x1": 605, "y1": 210, "x2": 660, "y2": 230},
        {"text": "X00165196", "x1": 670, "y1": 210, "x2": 770, "y2": 230},
    ]

    try:
        ann_id = service.create_template_annotation(
            page_id=test_page_id,
            ocr_tokens=sample_tokens,
            template_version="visa-form-test-v1",
        )
        detail = service.get_detail(ann_id)
        assert len(detail.fields) == 19
        by_code = {f.field_code: f for f in detail.fields}
        assert by_code["full_name"].value_text == "CARDALEEN YOUNG"
        assert by_code["passport_number"].value_text == "X00165196"
        assert by_code["full_name"].bbox[0] >= 170
        assert by_code["passport_number"].bbox[0] >= 660
    finally:
        # Cleanup
        try:
            storage_path = storage.resolve("tmp/test_vf_p1.png")
            if storage_path.exists():
                storage_path.unlink()
        except Exception:
            pass
        with db.session() as session:
            test_ann = session.scalar(
                select(VisaFormAnnotation).where(VisaFormAnnotation.page_id == test_page_id)
            )
            if test_ann:
                session.delete(test_ann)
            p = session.get(AnnotationPage, test_page_id)
            if p:
                session.delete(p)
            d = session.get(AnnotationDocument, test_doc_id)
            if d:
                session.delete(d)
            session.commit()

