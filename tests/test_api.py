from __future__ import annotations

import base64
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import PROJECT_ROOT, Settings
from app.db.models import AnnotationDocument, AnnotationPage
from app.main import create_app


def _two_page_pdf() -> bytes:
    document = pymupdf.open()
    for text in ("Diplomatic note", "Applicant table"):
        page = document.new_page(width=300, height=400)
        page.insert_text((40, 60), text)
    result = document.tobytes()
    document.close()
    return result


@contextmanager
def _client(**overrides):
    temp_parent = PROJECT_ROOT / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="annotation-test-", dir=temp_parent) as temp_dir:
        relative_storage = Path(temp_dir).relative_to(PROJECT_ROOT) / "storage"
        values = {
            "model_backend": "mock",
            "device": "cpu",
            "mock_page_types": "DIPLOMATIC_NOTE,APPLICANT_TABLE",
            "pdf_dpi": 100,
            "max_render_pixels": 5_000_000,
            "database_url": "sqlite:///:memory:",
            "auto_create_db": True,
            "annotation_storage_dir": relative_storage,
        }
        values.update(overrides)
        settings = Settings(_env_file=None, **values)
        with TestClient(create_app(settings)) as client:
            yield client


def test_health_reports_synchronous_mode() -> None:
    with _client() as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["processing_mode"] == "synchronous"


def test_model_status_reports_mock_components_loaded() -> None:
    with _client() as client:
        response = client.get("/api/v1/models/status")

    payload = response.json()
    assert response.status_code == 200
    assert payload["backend"] == "mock"
    assert payload["effective_device"] == "cpu"
    assert all(component["loaded"] for component in payload["components"])


def test_extract_returns_two_typed_pages() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/documents/extract",
            files={"file": ("sample.pdf", _two_page_pdf(), "application/pdf")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "NEEDS_REVIEW"
    assert payload["runtime"]["backend"] == "mock"
    assert [page["page_type"] for page in payload["pages"]] == [
        "DIPLOMATIC_NOTE",
        "APPLICANT_TABLE",
    ]
    assert payload["pages"][0]["page_number"] == 1
    assert payload["pages"][1]["page_number"] == 2
    assert payload["pages"][0]["note"] == {"fields": {}, "applicants": []}
    assert payload["pages"][1]["table"] == {
        "table_bbox": None,
        "headers": [],
        "cells": [],
        "rows": [],
    }


def test_extract_rejects_non_pdf_extension() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/documents/extract",
            files={"file": ("sample.txt", b"not a pdf", "text/plain")},
        )

    assert response.status_code == 415


def test_extract_rejects_invalid_pdf_signature() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/documents/extract",
            files={"file": ("sample.pdf", b"not a pdf", "application/pdf")},
        )

    assert response.status_code == 400


def test_annotation_import_view_and_review_flow() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("review.pdf", _two_page_pdf(), "application/pdf")},
        )
        assert imported.status_code == 201
        import_payload = imported.json()
        document_id = import_payload["document"]["id"]
        assert import_payload["created"] is True
        assert import_payload["document"]["page_count"] == 2
        assert import_payload["document"]["review_status"] == "UNLABELED"

        viewed = client.get(f"/api/v1/annotation/documents/{document_id}/pages/1")
        assert viewed.status_code == 200
        image_bytes = base64.b64decode(viewed.json()["image_base64"])
        assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")

        first_review = client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/1/review",
            json={
                "label": "DIPLOMATIC_NOTE",
                "reviewed_by": "tester",
                "review_note": "checked",
            },
        )
        assert first_review.status_code == 200
        assert first_review.json()["document_review_status"] == "IN_PROGRESS"

        second_review = client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/2/review",
            json={"label": "APPLICANT_TABLE", "reviewed_by": "tester"},
        )
        assert second_review.status_code == 200
        assert second_review.json()["document_review_status"] == "REVIEWED"

        document = client.get(f"/api/v1/annotation/documents/{document_id}").json()
        assert document["reviewed_pages"] == 2
        assert [page["label"] for page in document["pages"]] == [
            "DIPLOMATIC_NOTE",
            "APPLICANT_TABLE",
        ]

        with client.app.state.database.session() as session:
            stored_document = session.scalar(
                select(AnnotationDocument).where(AnnotationDocument.id == document_id)
            )
            stored_pages = list(
                session.scalars(
                    select(AnnotationPage).where(AnnotationPage.document_id == document_id)
                ).all()
            )
            assert stored_document is not None
            stored_paths = [stored_document.pdf_relative_path] + [
                page.image_relative_path for page in stored_pages
            ]
            assert all(not Path(path).is_absolute() for path in stored_paths)
            assert all(".." not in Path(path).parts for path in stored_paths)


def test_annotation_import_is_idempotent_by_sha256() -> None:
    pdf_bytes = _two_page_pdf()
    with _client() as client:
        first = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("first.pdf", pdf_bytes, "application/pdf")},
        )
        second = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("renamed.pdf", pdf_bytes, "application/pdf")},
        )

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert first.json()["document"]["id"] == second.json()["document"]["id"]


def test_annotation_review_queue_returns_pending_pages_in_order() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("queue.pdf", _two_page_pdf(), "application/pdf")},
        )
        document_id = imported.json()["document"]["id"]

        first = client.get("/api/v1/annotation/review/next")
        assert first.status_code == 200
        assert first.json()["current"]["page"]["page_number"] == 1
        assert first.json()["current"]["document_page_count"] == 2
        assert first.json()["stats"] == {"pending": 2, "reviewed": 0, "total": 2}

        client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/1/review",
            json={"label": "DIPLOMATIC_NOTE", "reviewed_by": "tester"},
        )
        second = client.get("/api/v1/annotation/review/next").json()
        assert second["current"]["page"]["page_number"] == 2
        assert second["stats"] == {"pending": 1, "reviewed": 1, "total": 2}

        client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/2/review",
            json={"label": "APPLICANT_TABLE", "reviewed_by": "tester"},
        )
        complete = client.get("/api/v1/annotation/review/next").json()
        assert complete["current"] is None
        assert complete["stats"] == {"pending": 0, "reviewed": 2, "total": 2}


def test_annotation_labels_are_exactly_the_four_review_classes() -> None:
    with _client() as client:
        response = client.get("/api/v1/annotation/labels")

    assert response.status_code == 200
    assert [item["code"] for item in response.json()] == [
        "DIPLOMATIC_NOTE",
        "APPLICANT_TABLE",
        "APPLICANT_LIST",
        "OTHER",
    ]
