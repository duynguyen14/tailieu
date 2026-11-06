from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, Settings
from app.db.models import (
    AnnotationDocument,
    KieAnnotationRun,
    KiePageAnnotation,
    OcrReviewToken,
)
from app.main import create_app


@contextmanager
def _client():
    (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kie-test-", dir=PROJECT_ROOT / "tmp") as folder:
        storage = Path(folder).relative_to(PROJECT_ROOT) / "storage"
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            device="cpu",
            database_url="sqlite:///:memory:",
            auto_create_db=True,
            annotation_storage_dir=storage,
        )
        with TestClient(create_app(settings)) as client:
            yield client


def test_kie_ser_then_re_review_creates_snapshots() -> None:
    with _client() as client:
        client.app.state.annotation_storage.write_bytes("page.png", b"image")
        with client.app.state.database.session() as session:
            session.add(
                AnnotationDocument(
                    id="doc", original_filename="case.pdf", pdf_relative_path="case.pdf",
                    sha256="a" * 64, page_count=1, review_status="REVIEWED",
                )
            )
            session.add(
                KieAnnotationRun(
                    id="run", teacher_review_id="teacher", case_id="case",
                    document_id="doc", algorithm_version="test", source_hash="b" * 64,
                    status="GENERATED", is_active=True,
                )
            )
            session.add(
                KiePageAnnotation(
                    id="kie-page", run_id="run", page_id="page", ocr_review_id="ocr",
                    page_order=1, page_number=1, image_relative_path="page.png",
                    image_width=100, image_height=100, orientation_angle=0,
                    ser_status="PENDING", re_status="BLOCKED",
                )
            )
            for token_id, order, text, x1, x2 in (
                ("token-1", 1, "NGUYEN VAN A", 1, 30),
                ("token-2", 2, "P123456", 40, 70),
            ):
                session.add(
                    OcrReviewToken(
                        id=token_id, review_id="ocr", token_order=order,
                        reviewed_text=text, x1=x1, y1=2, x2=x2, y2=12,
                        include_in_extraction=True,
                    )
                )

        queue = client.get("/api/v1/kie-review/next?mode=SER&from_index=0&to_index=50")
        assert queue.status_code == 200
        assert queue.json()["current"]["page_annotation_id"] == "kie-page"

        reviewed = client.put(
            "/api/v1/kie-review/pages/kie-page/ser",
            json={
                "reviewed_by": "thai", "review_note": "SER checked",
                "entities": [
                    {"label": "FULL_NAME", "value": "NGUYEN VAN A", "field_path": "applicants[0].full_name", "applicant_index": 0, "token_ids": ["token-1"], "spans": [{"token_id": "token-1", "char_start": 0, "char_end": 12}], "bbox": [1, 2, 30, 12]},
                    {"label": "PASSPORT_NUMBER", "value": "P123456", "field_path": "applicants[0].passport_number", "applicant_index": 0, "token_ids": ["token-2"], "spans": [{"token_id": "token-2", "char_start": 0, "char_end": 7}], "bbox": [40, 2, 70, 12]},
                ],
            },
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["re_status"] == "PENDING"

        detail = client.get("/api/v1/kie-review/pages/kie-page").json()
        head, tail = [entity["id"] for entity in detail["entities"]]
        relation = client.put(
            "/api/v1/kie-review/pages/kie-page/re",
            json={
                "reviewed_by": "thai", "review_note": None,
                "relations": [{"head_entity_id": head, "tail_entity_id": tail, "relation_type": "BELONGS_TO_APPLICANT", "applicant_index": 0}],
            },
        )
        assert relation.status_code == 200
        assert relation.json()["re_status"] == "APPROVED"

        stats = client.get("/api/v1/kie-review/statistics").json()
        assert stats["ser_approved"] == 1
        assert stats["re_approved"] == 1
