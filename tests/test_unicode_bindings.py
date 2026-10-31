from sqlalchemy import Unicode

from app.db.models import OcrReview, OcrReviewToken, OcrToken


def test_multilingual_ocr_columns_use_unicode_bind_parameters() -> None:
    """Prevent pyodbc from binding Vietnamese OCR text as VARCHAR/TEXT."""

    assert isinstance(OcrToken.__table__.c.text.type, Unicode)
    assert isinstance(OcrReviewToken.__table__.c.reviewed_text.type, Unicode)
    assert isinstance(OcrReview.__table__.c.review_note.type, Unicode)
