from __future__ import annotations

import base64
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import PROJECT_ROOT, Settings
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    PageLabelAudit,
    TableAnnotation,
    utc_now,
)
from app.main import create_app
from app.schemas.table_annotation import (
    TableElementPayload,
    TableElementType,
    TableRegionPayload,
)


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


def test_cors_wildcard_allows_any_origin_without_cookie_credentials() -> None:
    with _client(cors_origins="*") as client:
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://frontend.example.test",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


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


def test_annotation_review_accepts_visa_request_official_letter_label() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("visa.pdf", _two_page_pdf(), "application/pdf")},
        )
        document_id = imported.json()["document"]["id"]

        reviewed = client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/1/review",
            json={"label": "VISA_REQUEST_OFFICIAL_LETTER", "reviewed_by": "tester"},
        )

    assert reviewed.status_code == 200
    assert reviewed.json()["page"]["label"] == "VISA_REQUEST_OFFICIAL_LETTER"


def test_model_prediction_is_suggested_but_remains_unreviewed_until_human_save() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("predicted.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        document_id = imported["document"]["id"]
        page_id = imported["document"]["pages"][0]["id"]
        with client.app.state.database.session() as session:
            page = session.get(AnnotationPage, page_id)
            assert page is not None
            page.label = "APPLICANT_TABLE"
            page.model_predicted_label = "APPLICANT_TABLE"
            page.model_prediction_confidence = 0.91
            page.model_prediction_topk_json = json.dumps(
                [
                    {"label": "APPLICANT_TABLE", "confidence": 0.91},
                    {"label": "APPLICANT_LIST", "confidence": 0.06},
                    {"label": "OTHER", "confidence": 0.03},
                ]
            )
            page.model_prediction_version = "test-phase1"
            page.model_predicted_at = utc_now()

        queued = client.get("/api/v1/annotation/review/next")
        assert queued.status_code == 200
        current = queued.json()["current"]
        assert current["page"]["review_status"] == "UNLABELED"
        assert current["page"]["label"] == "APPLICANT_TABLE"
        assert current["prediction"]["label"] == "APPLICANT_TABLE"
        assert current["prediction"]["confidence"] == 0.91
        assert len(current["prediction"]["top_predictions"]) == 3

        reviewed = client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/1/review",
            json={"label": "DIPLOMATIC_NOTE", "reviewed_by": "thai"},
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["page"]["review_status"] == "REVIEWED"
        assert reviewed.json()["page"]["label"] == "DIPLOMATIC_NOTE"

        with client.app.state.database.session() as session:
            stored = session.get(AnnotationPage, page_id)
            assert stored is not None
            assert stored.label == "DIPLOMATIC_NOTE"
            assert stored.model_predicted_label == "APPLICANT_TABLE"
            assert stored.model_prediction_version == "test-phase1"


def test_annotation_labels_are_exactly_the_supported_review_classes() -> None:
    with _client() as client:
        response = client.get("/api/v1/annotation/labels")

    assert response.status_code == 200
    assert [item["code"] for item in response.json()] == [
        "DIPLOMATIC_NOTE",
        "APPLICANT_TABLE",
        "APPLICANT_LIST",
        "VISA_APPLICATION_FORM",
        "TEMPORARY_RESIDENCE_EXTENSION_FORM",
        "PASSPORT_FRONT",
        "VISA_STICKER",
        "VISA_REQUEST_OFFICIAL_LETTER",
        "OTHER",
    ]


def test_second_pass_label_audit_uses_only_first_pass_reviewed_target_pages() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("audit.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        document_id = imported["document"]["id"]
        page_1_id = imported["document"]["pages"][0]["id"]
        page_2_id = imported["document"]["pages"][1]["id"]

        client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/1/review",
            json={"label": "DIPLOMATIC_NOTE", "reviewed_by": "first-reviewer"},
        )
        with client.app.state.database.session() as session:
            predicted_only = session.get(AnnotationPage, page_2_id)
            assert predicted_only is not None
            predicted_only.label = "APPLICANT_TABLE"
            predicted_only.model_predicted_label = "APPLICANT_TABLE"
            predicted_only.model_prediction_confidence = 0.99
            predicted_only.model_prediction_version = "model-only"
            predicted_only.model_predicted_at = utc_now()

        queue = client.get("/api/v1/annotation/label-audit/review/next")
        assert queue.status_code == 200
        payload = queue.json()
        assert payload["current"]["document_id"] == document_id
        assert [item["page"]["id"] for item in payload["current"]["candidate_pages"]] == [
            page_1_id
        ]
        assert payload["stats"] == {
            "pending_pages": 1,
            "reviewed_pages": 0,
            "total_pages": 1,
            "pending_documents": 1,
            "total_documents": 1,
        }


def test_second_pass_label_audit_updates_canonical_labels_and_preserves_original() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("audit-update.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        document_id = imported["document"]["id"]
        page_1_id = imported["document"]["pages"][0]["id"]
        page_2_id = imported["document"]["pages"][1]["id"]
        client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/1/review",
            json={"label": "DIPLOMATIC_NOTE", "reviewed_by": "first-reviewer"},
        )
        client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/2/review",
            json={"label": "APPLICANT_TABLE", "reviewed_by": "first-reviewer"},
        )

        queued = client.get("/api/v1/annotation/label-audit/review/next").json()
        assert len(queued["current"]["candidate_pages"]) == 2
        reviewed = client.put(
            f"/api/v1/annotation/label-audit/documents/{document_id}/review",
            json={
                "reviewed_by": "thai",
                "pages": [
                    {"page_id": page_1_id, "label": "DIPLOMATIC_NOTE"},
                    {
                        "page_id": page_2_id,
                        "label": "OTHER",
                        "audit_note": "First pass was incorrect",
                    },
                ],
            },
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["reviewed_pages"] == 2

        complete = client.get("/api/v1/annotation/label-audit/review/next").json()
        assert complete["current"] is None
        assert complete["stats"]["pending_pages"] == 0
        assert complete["stats"]["reviewed_pages"] == 2

        with client.app.state.database.session() as session:
            page_2 = session.get(AnnotationPage, page_2_id)
            audits = list(
                session.scalars(
                    select(PageLabelAudit).order_by(PageLabelAudit.original_label.asc())
                ).all()
            )
            assert page_2 is not None
            assert page_2.label == "OTHER"
            assert len(audits) == 2
            corrected = next(item for item in audits if item.page_id == page_2_id)
            assert corrected.original_label == "APPLICANT_TABLE"
            assert corrected.audited_label == "OTHER"
            assert corrected.audited_by == "thai"


def test_table_annotation_prediction_and_human_review_flow() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("table-review.pdf", _two_page_pdf(), "application/pdf")},
        )
        document_id = imported.json()["document"]["id"]
        client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/2/review",
            json={"label": "APPLICANT_TABLE", "reviewed_by": "classifier-reviewer"},
        )
        page_id = imported.json()["document"]["pages"][1]["id"]
        table_service = client.app.state.table_annotation_service
        prediction_id = table_service.start_model_annotation(
            page_id=page_id,
            model_name="test-layout+test-structure",
            model_version="test-v1",
            force=False,
        )
        table_service.complete_model_annotation(
            annotation_id=prediction_id,
            processing_time_ms=123,
            tables=[
                TableRegionPayload(
                    table_index=0,
                    bbox=(20, 100, 280, 350),
                    confidence=0.9,
                    elements=[
                        TableElementPayload(
                            element_type=TableElementType.HEADER_ROW,
                            row_index=0,
                            bbox=(20, 100, 280, 150),
                            confidence=0.8,
                        ),
                        TableElementPayload(
                            element_type=TableElementType.DATA_ROW,
                            row_index=1,
                            bbox=(20, 150, 280, 220),
                            confidence=0.8,
                        ),
                        TableElementPayload(
                            element_type=TableElementType.COLUMN,
                            column_index=0,
                            bbox=(20, 100, 150, 350),
                            confidence=0.8,
                        ),
                        TableElementPayload(
                            element_type=TableElementType.CELL,
                            row_index=1,
                            column_index=0,
                            bbox=(20, 150, 150, 220),
                            confidence=0.8,
                        ),
                    ],
                )
            ],
        )

        queued = client.get("/api/v1/table-annotation/review/next")
        assert queued.status_code == 200
        queue_payload = queued.json()
        assert queue_payload["current"]["annotation"]["id"] == prediction_id
        assert queue_payload["current"]["page"]["page_number"] == 2
        assert queue_payload["current"]["tables"][0]["bbox"] == [20, 100, 280, 350]
        assert queue_payload["stats"] == {
            "pending": 1,
            "reviewed": 0,
            "failed": 0,
            "total": 1,
        }
        assert base64.b64decode(queue_payload["current"]["image_base64"]).startswith(
            b"\x89PNG\r\n\x1a\n"
        )

        reviewed = client.put(
            f"/api/v1/table-annotation/{prediction_id}/review",
            json={
                "reviewed_by": "thai",
                "review_note": "adjusted table edge",
                "tables": [
                    {
                        "table_index": 0,
                        "bbox": [15, 95, 285, 355],
                        "elements": [
                            {
                                "element_type": "HEADER_ROW",
                                "row_index": 0,
                                "bbox": [15, 95, 285, 150],
                            },
                            {
                                "element_type": "DATA_ROW",
                                "row_index": 1,
                                "bbox": [15, 150, 285, 225],
                            },
                        ],
                    }
                ],
            },
        )
        assert reviewed.status_code == 200
        reviewed_payload = reviewed.json()
        assert reviewed_payload["annotation"]["source"] == "HUMAN"
        assert reviewed_payload["annotation"]["status"] == "REVIEWED"
        assert reviewed_payload["annotation"]["parent_annotation_id"] == prediction_id
        assert reviewed_payload["tables"][0]["bbox"] == [15, 95, 285, 355]

        complete = client.get("/api/v1/table-annotation/review/next").json()
        assert complete["current"] is None
        assert complete["stats"] == {
            "pending": 0,
            "reviewed": 1,
            "failed": 0,
            "total": 1,
        }

        with client.app.state.database.session() as session:
            versions = list(
                session.scalars(
                    select(TableAnnotation)
                    .where(TableAnnotation.page_id == page_id)
                    .order_by(TableAnnotation.created_at)
                ).all()
            )
            assert len(versions) == 2
            assert versions[0].source == "MODEL"
            assert versions[0].is_active is False
            assert versions[1].source == "HUMAN"
            assert versions[1].is_active is True
