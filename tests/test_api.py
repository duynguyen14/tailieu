from __future__ import annotations

import argparse
import base64
import json
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import PROJECT_ROOT, Settings
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    CaseTextReconstruction,
    DocumentCase,
    DocumentCasePage,
    OcrLayoutRegion,
    OcrReview,
    OcrReviewRegion,
    OcrReviewToken,
    OcrRun,
    OcrToken,
    PageLabelAudit,
    TableAnnotation,
    TeacherExtractionAttempt,
    TeacherExtractionJob,
    TeacherExtractionReview,
    utc_now,
)
from app.main import create_app
from app.schemas.table_annotation import (
    TableElementPayload,
    TableElementType,
    TableRegionPayload,
)
from tools.reconstruct_case_text import _select_case_ids
from tools.run_grounded_ocr import eligible_pages


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


def test_text_reconstruction_review_creates_version_and_preserves_unicode() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("reconstruction.pdf", _two_page_pdf(), "application/pdf")},
        ).json()["document"]
        case_id = str(uuid4())
        reconstruction_id = str(uuid4())
        with client.app.state.database.session() as session:
            session.add(
                DocumentCase(
                    id=case_id,
                    document_id=imported["id"],
                    case_index=1,
                    status="AUTO_GROUPED",
                    is_active=True,
                    start_page_number=1,
                    end_page_number=2,
                )
            )
            session.flush()
            for index, page in enumerate(imported["pages"], start=1):
                session.add(
                    DocumentCasePage(
                        id=str(uuid4()),
                        case_id=case_id,
                        page_id=page["id"],
                        page_order=index,
                        page_role="NOTE" if index == 1 else "APPLICANT_DATA",
                    )
                )
            session.add(
                CaseTextReconstruction(
                    id=reconstruction_id,
                    case_id=case_id,
                    algorithm_version="test-v1",
                    source_fingerprint="a" * 64,
                    status="GENERATED",
                    is_active=True,
                    reconstructed_text="[PAGE 1]\nBản dịch tiếng Việt",
                    structure_json=json.dumps({"pages": []}),
                    page_count=2,
                    token_count=3,
                )
            )

        queued = client.get("/api/v1/text-reconstruction/review/next")
        assert queued.status_code == 200
        queued_payload = queued.json()
        assert queued_payload["stats"] == {"pending": 1, "reviewed": 0, "total": 1}
        assert len(queued_payload["current"]["pages"]) == 2
        assert queued_payload["current"]["reconstructed_text"].endswith(
            "Bản dịch tiếng Việt"
        )

        corrected = "[PAGE 1]\nBản dịch tiếng Việt không chính thức"
        reviewed = client.put(
            f"/api/v1/text-reconstruction/{reconstruction_id}/review",
            json={
                "reviewed_by": "thai",
                "review_note": "đã sửa thứ tự đọc",
                "reconstructed_text": corrected,
            },
        )
        assert reviewed.status_code == 200
        reviewed_payload = reviewed.json()
        new_id = reviewed_payload["reconstruction"]["id"]
        assert new_id != reconstruction_id
        assert reviewed_payload["reconstruction"]["parent_reconstruction_id"] == reconstruction_id
        assert reviewed_payload["reconstruction"]["status"] == "REVIEWED"
        assert reviewed_payload["reconstructed_text"] == corrected

        empty_queue = client.get("/api/v1/text-reconstruction/review/next").json()
        assert empty_queue["current"] is None
        assert empty_queue["stats"] == {"pending": 0, "reviewed": 1, "total": 1}

        with client.app.state.database.session() as session:
            versions = list(
                session.scalars(
                    select(CaseTextReconstruction)
                    .where(CaseTextReconstruction.case_id == case_id)
                    .order_by(CaseTextReconstruction.created_at.asc())
                ).all()
            )
            assert len(versions) == 2
            assert versions[0].is_active is False
            assert versions[1].is_active is True
            assert versions[1].reconstructed_text == corrected


def test_teacher_extraction_review_versions_json_and_updates_queue() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("teacher.pdf", _two_page_pdf(), "application/pdf")},
        ).json()["document"]
        case_id = str(uuid4())
        reconstruction_id = str(uuid4())
        job_id = str(uuid4())
        attempt_id = str(uuid4())
        review_id = str(uuid4())
        output = {
            "schema_version": "compact-document-v1",
            "case_id": case_id,
            "document_id": imported["id"],
            "note": {"note_number": "850-2026"},
            "applicants": [],
            "warnings": [],
        }
        with client.app.state.database.session() as session:
            session.add(
                DocumentCase(
                    id=case_id,
                    document_id=imported["id"],
                    case_index=1,
                    status="AUTO_GROUPED",
                    is_active=True,
                    start_page_number=1,
                    end_page_number=2,
                )
            )
            session.flush()
            session.add(
                CaseTextReconstruction(
                    id=reconstruction_id,
                    case_id=case_id,
                    algorithm_version="test-v1",
                    source_fingerprint="b" * 64,
                    status="REVIEWED",
                    is_active=True,
                    reconstructed_text="No: 850-2026",
                    structure_json=json.dumps({"pages": []}),
                    page_count=1,
                    token_count=2,
                )
            )
            session.flush()
            session.add(
                TeacherExtractionJob(
                    id=job_id,
                    reconstruction_id=reconstruction_id,
                    case_id=case_id,
                    document_id=imported["id"],
                    input_hash="c" * 64,
                    input_text="No: 850-2026",
                    source_structure_json=json.dumps({"pages": []}),
                    reconstruction_version="test-v1",
                    schema_version="compact-document-v1",
                    prompt_version="teacher-v1",
                    teacher_model="teacher-model",
                    teacher_endpoint="http://teacher/v1/chat/completions",
                    status="COMPLETED",
                    is_active=True,
                    attempt_count=1,
                    completed_at=utc_now(),
                )
            )
            session.flush()
            session.add(
                TeacherExtractionAttempt(
                    id=attempt_id,
                    job_id=job_id,
                    attempt_number=1,
                    request_json=json.dumps({"model": "teacher-model"}),
                    raw_response_json=json.dumps({"choices": []}),
                    assistant_content=json.dumps(output),
                    parsed_output_json=json.dumps(output),
                    validation_status="VALID",
                    http_status=200,
                    latency_ms=1200,
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    finish_reason="stop",
                )
            )
            session.flush()
            session.add(
                TeacherExtractionReview(
                    id=review_id,
                    job_id=job_id,
                    source_attempt_id=attempt_id,
                    reviewed_output_json=json.dumps(output),
                    review_status="PENDING",
                    is_active=True,
                )
            )

        queued = client.get("/api/v1/teacher-extraction/review/next")
        assert queued.status_code == 200
        queued_payload = queued.json()
        assert queued_payload["stats"] == {
            "pending": 1,
            "approved": 0,
            "rejected": 0,
            "total": 1,
        }
        assert queued_payload["current"]["review"]["id"] == review_id
        assert queued_payload["current"]["case"]["original_filename"] == "teacher.pdf"
        assert queued_payload["current"]["attempt"]["latency_ms"] == 1200
        assert queued_payload["current_index"] is None

        ranged = client.get(
            "/api/v1/teacher-extraction/review/next",
            params={"from_index": 0, "to_index": 0},
        )
        assert ranged.status_code == 200
        assert ranged.json()["current"]["review"]["id"] == review_id
        assert ranged.json()["current_index"] == 0
        assert client.get(
            "/api/v1/teacher-extraction/review/next",
            params={"from_index": 1, "to_index": 1},
        ).json()["current"] is None
        assert client.get(
            "/api/v1/teacher-extraction/review/next",
            params={"from_index": 0},
        ).status_code == 400
        assert client.get(
            "/api/v1/teacher-extraction/review/next",
            params={"from_index": 2, "to_index": 1},
        ).status_code == 400

        with client.app.state.database.session() as session:
            job = session.get(TeacherExtractionJob, job_id)
            assert job is not None
            job.is_active = False
        hidden_queue = client.get("/api/v1/teacher-extraction/review/next").json()
        assert hidden_queue["current"] is None
        assert hidden_queue["stats"]["total"] == 0
        with client.app.state.database.session() as session:
            job = session.get(TeacherExtractionJob, job_id)
            assert job is not None
            job.is_active = True

        edited_output = {
            **output,
            "note": {"note_number": "850-2026-A"},
            "applicants": [
                {
                    "person_id": "person_1",
                    "sequence_number": 1,
                    "full_name": "NICHOLAS JOHN HOGARTH",
                    "number_of_entries": "MULTIPLE",
                }
            ],
        }
        saved = client.put(
            f"/api/v1/teacher-extraction/{review_id}/review",
            json={
                "reviewed_by": "thai",
                "decision": "PENDING",
                "reviewed_output": edited_output,
                "review_note": "đã sửa số công hàm",
            },
        )
        assert saved.status_code == 200
        saved_payload = saved.json()
        saved_id = saved_payload["review"]["id"]
        assert saved_id != review_id
        assert saved_payload["review"]["parent_review_id"] == review_id
        assert saved_payload["reviewed_output"]["note"]["note_number"] == "850-2026-A"
        assert (
            saved_payload["reviewed_output"]["applicants"][0]["number_of_entries"]
            == "MULTIPLE"
        )
        ranged_saved = client.get(
            "/api/v1/teacher-extraction/review/next",
            params={"from_index": 0, "to_index": 0},
        ).json()
        assert ranged_saved["current"]["review"]["id"] == saved_id
        assert ranged_saved["current_index"] == 0

        approved = client.put(
            f"/api/v1/teacher-extraction/{saved_id}/review",
            json={
                "reviewed_by": "thai",
                "decision": "APPROVED",
                "reviewed_output": edited_output,
                "review_note": "đã kiểm tra",
            },
        )
        assert approved.status_code == 200
        assert approved.json()["review"]["review_status"] == "APPROVED"

        empty_queue = client.get("/api/v1/teacher-extraction/review/next").json()
        assert empty_queue["current"] is None
        assert empty_queue["stats"] == {
            "pending": 0,
            "approved": 1,
            "rejected": 0,
            "total": 1,
        }

        with client.app.state.database.session() as session:
            versions = list(
                session.scalars(
                    select(TeacherExtractionReview)
                    .where(TeacherExtractionReview.job_id == job_id)
                    .order_by(TeacherExtractionReview.created_at.asc())
                ).all()
            )
            assert len(versions) == 3
            assert [version.is_active for version in versions] == [False, False, True]


def test_text_reconstruction_tool_selects_ready_pdf_by_filename_or_document() -> None:
    with _client() as client:
        unready_document = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("unready.pdf", _two_page_pdf(), "application/pdf")},
        ).json()["document"]
        ready_document = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("ready.pdf", _two_page_pdf(), "application/pdf")},
        ).json()["document"]
        unready_case_id = str(uuid4())
        ready_case_id = str(uuid4())
        ready_run_id = str(uuid4())
        with client.app.state.database.session() as session:
            session.add_all(
                [
                    DocumentCase(
                        id=unready_case_id,
                        document_id=unready_document["id"],
                        case_index=1,
                        status="AUTO_GROUPED",
                        is_active=True,
                        start_page_number=1,
                        end_page_number=1,
                    ),
                    DocumentCase(
                        id=ready_case_id,
                        document_id=ready_document["id"],
                        case_index=1,
                        status="AUTO_GROUPED",
                        is_active=True,
                        start_page_number=1,
                        end_page_number=1,
                    ),
                ]
            )
            session.flush()
            session.add_all(
                [
                    DocumentCasePage(
                        id=str(uuid4()),
                        case_id=unready_case_id,
                        page_id=unready_document["pages"][0]["id"],
                        page_order=1,
                        page_role="NOTE",
                    ),
                    DocumentCasePage(
                        id=str(uuid4()),
                        case_id=ready_case_id,
                        page_id=ready_document["pages"][0]["id"],
                        page_order=1,
                        page_role="NOTE",
                    ),
                    OcrRun(
                        id=ready_run_id,
                        page_id=ready_document["pages"][0]["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="test-v1",
                        device="gpu:0",
                    ),
                ]
            )
            session.flush()
            session.add(
                OcrReview(
                    id=str(uuid4()),
                    ocr_run_id=ready_run_id,
                    status="REVIEWED",
                    is_active=True,
                    include_in_llm=True,
                    reviewed_by="thai",
                )
            )

        defaults = {
            "case_id": None,
            "document_id": None,
            "filename": None,
            "limit": 0,
            "ready_only": False,
            "include_incomplete": False,
        }
        with client.app.state.database.session() as session:
            ready_args = argparse.Namespace(
                **{**defaults, "ready_only": True, "limit": 1}
            )
            assert _select_case_ids(session, ready_args) == [ready_case_id]

            filename_args = argparse.Namespace(
                **{**defaults, "filename": "READY.PDF"}
            )
            assert _select_case_ids(session, filename_args) == [ready_case_id]

            document_args = argparse.Namespace(
                **{**defaults, "document_id": unready_document["id"]}
            )
            assert _select_case_ids(session, document_args) == [unready_case_id]


def test_text_reconstruction_treats_intentional_page_exclusion_as_decided() -> None:
    with _client() as client:
        document = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("excluded-page.pdf", _two_page_pdf(), "application/pdf")},
        ).json()["document"]
        case_id = str(uuid4())
        run_id = str(uuid4())
        with client.app.state.database.session() as session:
            session.add(
                DocumentCase(
                    id=case_id,
                    document_id=document["id"],
                    case_index=1,
                    status="AUTO_GROUPED",
                    is_active=True,
                    start_page_number=1,
                    end_page_number=1,
                )
            )
            session.flush()
            session.add_all(
                [
                    DocumentCasePage(
                        id=str(uuid4()),
                        case_id=case_id,
                        page_id=document["pages"][0]["id"],
                        page_order=1,
                        page_role="NOTE",
                    ),
                    OcrRun(
                        id=run_id,
                        page_id=document["pages"][0]["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="test-v1",
                        device="gpu:0",
                    ),
                ]
            )
            session.flush()
            session.add(
                OcrReview(
                    id=str(uuid4()),
                    ocr_run_id=run_id,
                    status="REVIEWED",
                    is_active=True,
                    include_in_llm=False,
                    reviewed_by="thai",
                )
            )

        args = argparse.Namespace(
            case_id=None,
            document_id=document["id"],
            filename=None,
            limit=0,
            ready_only=True,
            include_incomplete=False,
        )
        with client.app.state.database.session() as session:
            assert _select_case_ids(session, args) == [case_id]


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


def test_grounded_ocr_review_preserves_raw_and_saves_corrected_snapshot() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("ocr-review.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        page = imported["document"]["pages"][0]
        run_id = str(uuid4())
        first_token_id = str(uuid4())
        second_token_id = str(uuid4())
        with client.app.state.database.session() as session:
            session.add(
                OcrRun(
                    id=run_id,
                    page_id=page["id"],
                    status="COMPLETED",
                    is_active=True,
                    model_name="det+rec",
                    model_version="test-v1",
                    device="gpu:0",
                    processing_time_ms=123,
                )
            )
            session.add_all(
                [
                    OcrToken(
                        id=first_token_id,
                        ocr_run_id=run_id,
                        token_index=1,
                        token_id="t_0001",
                        text="RF02I053O",
                        confidence=Decimal("0.81"),
                        x1=10,
                        y1=20,
                        x2=120,
                        y2=40,
                    ),
                    OcrToken(
                        id=second_token_id,
                        ocr_run_id=run_id,
                        token_index=2,
                        token_id="t_0002",
                        text="false positive",
                        confidence=Decimal("0.55"),
                        x1=10,
                        y1=50,
                        x2=130,
                        y2=70,
                        include_in_extraction=False,
                        ignored_reason="INSIDE_SEAL",
                    ),
                    OcrLayoutRegion(
                        id=str(uuid4()),
                        ocr_run_id=run_id,
                        region_index=1,
                        region_type="SEAL",
                        confidence=Decimal("0.97"),
                        x1=5,
                        y1=45,
                        x2=140,
                        y2=75,
                    ),
                ]
            )

        queue = client.get("/api/v1/ocr-review/review/next")
        assert queue.status_code == 200
        current = queue.json()["current"]
        assert current["run"]["id"] == run_id
        assert current["run"]["device"] == "gpu:0"
        assert current["tokens"][0]["raw_text"] == "RF02I053O"
        assert current["tokens"][0]["reviewed_text"] == "RF02I053O"
        assert current["tokens"][1]["include_in_extraction"] is False
        assert current["tokens"][1]["ignored_reason"] == "INSIDE_SEAL"
        assert current["exclusion_regions"][0]["region_type"] == "SEAL"
        source_region_id = current["exclusion_regions"][0]["source_region_id"]

        reviewed = client.put(
            f"/api/v1/ocr-review/{run_id}/review",
            json={
                "reviewed_by": "thai",
                "review_note": "correct OCR and remove false positive",
                "tokens": [
                    {
                        "source_token_id": first_token_id,
                        "reviewed_text": "RF0210530",
                        "reviewed_bbox": [8, 18, 122, 42],
                    },
                    {
                        "source_token_id": second_token_id,
                        "reviewed_text": "false positive",
                        "reviewed_bbox": [10, 50, 130, 70],
                        "include_in_extraction": False,
                        "ignored_reason": "INSIDE_SEAL",
                    },
                    {
                        "source_token_id": None,
                        "reviewed_text": "Added missing line",
                        "reviewed_bbox": [10, 80, 150, 100],
                    },
                ],
                "exclusion_regions": [
                    {
                        "source_region_id": source_region_id,
                        "region_type": "SEAL",
                        "reviewed_bbox": [5, 45, 140, 75],
                    }
                ],
            },
        )
        assert reviewed.status_code == 200
        result = reviewed.json()
        assert result["run"]["reviewed_by"] == "thai"
        assert result["tokens"][0]["raw_text"] == "RF02I053O"
        assert result["tokens"][0]["reviewed_text"] == "RF0210530"
        assert result["tokens"][0]["is_changed"] is True
        assert result["tokens"][1]["is_human_added"] is False
        assert result["tokens"][1]["include_in_extraction"] is False
        assert result["tokens"][2]["is_human_added"] is True
        assert len(result["exclusion_regions"]) == 1

        history = client.get(
            "/api/v1/ocr-review/history",
            params={"page": 1, "page_size": 1, "decision": "APPROVED"},
        )
        assert history.status_code == 200
        history_payload = history.json()
        assert history_payload["page"] == 1
        assert history_payload["page_size"] == 1
        assert history_payload["total_items"] == 1
        assert history_payload["total_pages"] == 1
        assert history_payload["items"][0]["run_id"] == run_id
        assert history_payload["items"][0]["decision"] == "APPROVED"
        assert history_payload["items"][0]["token_count"] == 3
        assert history_payload["items"][0]["included_token_count"] == 2
        assert history_payload["items"][0]["ignored_token_count"] == 1
        assert history_payload["items"][0]["human_added_token_count"] == 1

        completed_queue = client.get("/api/v1/ocr-review/review/next").json()
        assert completed_queue["current"] is None
        assert completed_queue["stats"]["pending"] == 0
        assert completed_queue["stats"]["reviewed"] == 1

        with client.app.state.database.session() as session:
            raw_first = session.get(OcrToken, first_token_id)
            raw_second = session.get(OcrToken, second_token_id)
            active_review = session.scalar(
                select(OcrReview).where(OcrReview.is_active == True)  # noqa: E712
            )
            review_tokens = list(
                session.scalars(
                    select(OcrReviewToken).order_by(OcrReviewToken.token_order)
                ).all()
            )
            review_regions = list(session.scalars(select(OcrReviewRegion)).all())
            assert raw_first is not None and raw_first.text == "RF02I053O"
            assert raw_second is not None and raw_second.text == "false positive"
            assert active_review is not None
            assert [token.reviewed_text for token in review_tokens] == [
                "RF0210530",
                "false positive",
                "Added missing line",
            ]
            assert review_tokens[1].include_in_extraction is False
            assert review_regions[0].region_type == "SEAL"


def test_ocr_review_queue_searches_run_or_page_id_and_sorts_by_created_date() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("ocr-queue-sort.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        pages = imported["document"]["pages"]
        older_run_id = str(uuid4())
        newer_run_id = str(uuid4())
        now = utc_now()
        with client.app.state.database.session() as session:
            session.add_all(
                [
                    OcrRun(
                        id=older_run_id,
                        page_id=pages[0]["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="test-v1",
                        device="gpu:0",
                        created_at=now - timedelta(minutes=5),
                    ),
                    OcrRun(
                        id=newer_run_id,
                        page_id=pages[1]["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="test-v1",
                        device="gpu:0",
                        created_at=now,
                    ),
                ]
            )

        newest = client.get(
            "/api/v1/ocr-review/review/next",
            params={"sort_order": "NEWEST"},
        )
        assert newest.status_code == 200
        assert newest.json()["current"]["run"]["id"] == newer_run_id

        oldest = client.get(
            "/api/v1/ocr-review/review/next",
            params={"sort_order": "OLDEST"},
        )
        assert oldest.status_code == 200
        assert oldest.json()["current"]["run"]["id"] == older_run_id

        by_page_id = client.get(
            "/api/v1/ocr-review/review/next",
            params={"search_id": pages[0]["id"], "sort_order": "NEWEST"},
        )
        assert by_page_id.status_code == 200
        assert by_page_id.json()["current"]["run"]["id"] == older_run_id
        assert by_page_id.json()["current_index"] == 1

        by_run_id = client.get(
            "/api/v1/ocr-review/review/next",
            params={"search_id": newer_run_id, "sort_order": "OLDEST"},
        )
        assert by_run_id.status_code == 200
        assert by_run_id.json()["current"]["page"]["id"] == pages[1]["id"]
        assert by_run_id.json()["current_index"] == 1


def test_ocr_review_queue_keeps_manual_index_range_stable_after_review() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("ocr-range.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        pages = imported["document"]["pages"]
        older_run_id = str(uuid4())
        newer_run_id = str(uuid4())
        now = utc_now()
        with client.app.state.database.session() as session:
            session.add_all(
                [
                    OcrRun(
                        id=older_run_id,
                        page_id=pages[0]["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="test-v1",
                        device="gpu:0",
                        created_at=now - timedelta(minutes=5),
                    ),
                    OcrRun(
                        id=newer_run_id,
                        page_id=pages[1]["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="test-v1",
                        device="gpu:0",
                        created_at=now,
                    ),
                ]
            )

        first_range = client.get(
            "/api/v1/ocr-review/review/next",
            params={"from_index": 0, "to_index": 0, "sort_order": "NEWEST"},
        )
        assert first_range.status_code == 200
        assert first_range.json()["current"]["run"]["id"] == newer_run_id
        assert first_range.json()["current_index"] == 0

        second_range = client.get(
            "/api/v1/ocr-review/review/next",
            params={"from_index": 1, "to_index": 1, "sort_order": "NEWEST"},
        )
        assert second_range.status_code == 200
        assert second_range.json()["current"]["run"]["id"] == older_run_id
        assert second_range.json()["current_index"] == 1

        reviewed = client.put(
            f"/api/v1/ocr-review/{newer_run_id}/review",
            json={
                "reviewed_by": "thai",
                "include_in_llm": False,
                "tokens": [],
                "exclusion_regions": [],
            },
        )
        assert reviewed.status_code == 200

        completed_first_range = client.get(
            "/api/v1/ocr-review/review/next",
            params={"from_index": 0, "to_index": 0, "sort_order": "NEWEST"},
        )
        assert completed_first_range.status_code == 200
        assert completed_first_range.json()["current"] is None
        assert completed_first_range.json()["current_index"] is None

        unchanged_second_range = client.get(
            "/api/v1/ocr-review/review/next",
            params={"from_index": 1, "to_index": 1, "sort_order": "NEWEST"},
        )
        assert unchanged_second_range.status_code == 200
        assert unchanged_second_range.json()["current"]["run"]["id"] == older_run_id
        assert unchanged_second_range.json()["current_index"] == 1

        incomplete_range = client.get(
            "/api/v1/ocr-review/review/next",
            params={"from_index": 0},
        )
        assert incomplete_range.status_code == 400
        reversed_range = client.get(
            "/api/v1/ocr-review/review/next",
            params={"from_index": 5, "to_index": 4},
        )
        assert reversed_range.status_code == 400


def test_grounded_ocr_review_can_skip_page_without_deleting_ocr_run() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("skip-ocr.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        page = imported["document"]["pages"][0]
        run_id = str(uuid4())
        with client.app.state.database.session() as session:
            session.add(
                OcrRun(
                    id=run_id,
                    page_id=page["id"],
                    status="COMPLETED",
                    is_active=True,
                    model_name="det+rec",
                    model_version="test-v1",
                    device="gpu:0",
                )
            )

        response = client.put(
            f"/api/v1/ocr-review/{run_id}/review",
            json={
                "reviewed_by": "thai",
                "review_note": "Vietnamese OCR quality is not acceptable",
                "include_in_llm": False,
                "tokens": [],
                "exclusion_regions": [],
            },
        )

        assert response.status_code == 200
        assert response.json()["run"]["include_in_llm"] is False
        skipped_history = client.get(
            "/api/v1/ocr-review/history",
            params={"decision": "SKIPPED", "filename": "skip-ocr"},
        )
        assert skipped_history.status_code == 200
        skipped_payload = skipped_history.json()
        assert skipped_payload["total_items"] == 1
        assert skipped_payload["items"][0]["decision"] == "SKIPPED"
        assert skipped_payload["items"][0]["reviewed_by"] == "thai"
        queue = client.get("/api/v1/ocr-review/review/next").json()
        assert queue["current"] is None
        assert queue["stats"]["skipped"] == 1
        with client.app.state.database.session() as session:
            run = session.get(OcrRun, run_id)
            review = session.scalar(
                select(OcrReview).where(OcrReview.ocr_run_id == run_id)
            )
            assert run is not None and run.is_active is True
            assert review is not None and review.include_in_llm is False

        documents = client.get(
            "/api/v1/ocr-review/documents",
            params={"filename": "skip-ocr.pdf"},
        )
        assert documents.status_code == 200
        document_payload = documents.json()
        assert document_payload["total_items"] == 1
        assert document_payload["items"][0]["included_page_count"] == 0
        assert document_payload["items"][0]["excluded_page_count"] == 1

        detail = client.get(
            f"/api/v1/ocr-review/documents/{imported['document']['id']}"
        )
        assert detail.status_code == 200
        assert len(detail.json()["pages"]) == 1
        assert detail.json()["pages"][0]["run"]["include_in_llm"] is False

        restored = client.put(
            f"/api/v1/ocr-review/{run_id}/inclusion",
            json={
                "include_in_llm": True,
                "reviewed_by": "thai",
                "review_note": "restore after PDF-level verification",
            },
        )
        assert restored.status_code == 200
        restored_payload = restored.json()
        assert restored_payload["run"]["include_in_llm"] is True
        with client.app.state.database.session() as session:
            versions = list(
                session.scalars(
                    select(OcrReview)
                    .where(OcrReview.ocr_run_id == run_id)
                    .order_by(OcrReview.created_at.asc())
                ).all()
            )
            assert len(versions) == 2
            assert versions[0].is_active is False
            assert versions[1].is_active is True
            assert versions[1].parent_review_id == versions[0].id


def test_grounded_ocr_review_preserves_vietnamese_unicode_end_to_end() -> None:
    vietnamese_text = (
        "Đại Sứ Quán Hoa Kỳ tại Hà Nội kính chào Cục Lãnh Sự "
        "Bộ Ngoại giao nước Cộng hòa Xã hội Chủ nghĩa Việt Nam"
    )
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("unicode-review.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        page = imported["document"]["pages"][0]
        run_id = str(uuid4())
        token_id = str(uuid4())
        with client.app.state.database.session() as session:
            session.add(
                OcrRun(
                    id=run_id,
                    page_id=page["id"],
                    status="COMPLETED",
                    is_active=True,
                    model_name="det+rec",
                    model_version="unicode-test",
                    device="cpu",
                )
            )
            session.add(
                OcrToken(
                    id=token_id,
                    ocr_run_id=run_id,
                    token_index=1,
                    token_id="t_0001",
                    text="raw OCR",
                    confidence=Decimal("0.99"),
                    x1=10,
                    y1=20,
                    x2=250,
                    y2=60,
                )
            )

        response = client.put(
            f"/api/v1/ocr-review/{run_id}/review",
            json={
                "reviewed_by": "thai",
                "include_in_llm": True,
                "tokens": [
                    {
                        "source_token_id": token_id,
                        "reviewed_text": vietnamese_text,
                        "reviewed_bbox": [10, 20, 250, 60],
                    }
                ],
                "exclusion_regions": [],
            },
        )

        assert response.status_code == 200
        assert response.json()["tokens"][0]["reviewed_text"] == vietnamese_text
        first_review_id = response.json()["run"]["review_id"]
        corrected_text = f"{vietnamese_text}."
        second_response = client.put(
            f"/api/v1/ocr-review/{run_id}/review",
            json={
                "reviewed_by": "thai",
                "review_note": "corrected snapshot version",
                "include_in_llm": True,
                "tokens": [
                    {
                        "source_token_id": token_id,
                        "reviewed_text": corrected_text,
                        "reviewed_bbox": [12, 22, 252, 62],
                    }
                ],
                "exclusion_regions": [],
            },
        )
        assert second_response.status_code == 200
        second_payload = second_response.json()
        second_review_id = second_payload["run"]["review_id"]
        assert second_review_id != first_review_id
        assert second_payload["tokens"][0]["reviewed_text"] == corrected_text
        assert second_payload["tokens"][0]["reviewed_bbox"] == [12, 22, 252, 62]
        detail = client.get(f"/api/v1/ocr-review/{run_id}")
        assert detail.status_code == 200
        assert detail.json()["tokens"][0]["reviewed_text"] == corrected_text
        with client.app.state.database.session() as session:
            reviews = list(
                session.scalars(
                    select(OcrReview)
                    .where(OcrReview.ocr_run_id == run_id)
                    .order_by(OcrReview.created_at.asc())
                ).all()
            )
            assert len(reviews) == 2
            assert reviews[0].id == first_review_id
            assert reviews[0].is_active is False
            assert reviews[1].id == second_review_id
            assert reviews[1].parent_review_id == first_review_id
            assert reviews[1].is_active is True
            stored = session.scalar(
                select(OcrReviewToken).where(
                    OcrReviewToken.review_id == second_review_id,
                    OcrReviewToken.source_token_id == token_id,
                )
            )
            assert stored is not None
            assert stored.reviewed_text == corrected_text


def test_pending_review_ocr_excludes_reviewed_and_skipped_pages() -> None:
    source = pymupdf.open()
    for text in ("Reviewed note", "Skipped table", "Pending list"):
        source.new_page(width=300, height=400).insert_text((40, 60), text)
    pdf_bytes = source.tobytes()
    source.close()

    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("pending-ocr.pdf", pdf_bytes, "application/pdf")},
        ).json()
        document = imported["document"]
        reviewed_page = document["pages"][0]
        skipped_page = document["pages"][1]
        pending_page = document["pages"][2]
        reviewed_run_id = str(uuid4())
        skipped_run_id = str(uuid4())
        pending_run_id = str(uuid4())
        case_id = str(uuid4())

        with client.app.state.database.session() as session:
            for page, label in (
                (reviewed_page, "DIPLOMATIC_NOTE"),
                (skipped_page, "APPLICANT_TABLE"),
                (pending_page, "APPLICANT_LIST"),
            ):
                stored_page = session.get(AnnotationPage, page["id"])
                assert stored_page is not None
                stored_page.label = label
                stored_page.review_status = "REVIEWED"

            case = DocumentCase(
                id=case_id,
                document_id=document["id"],
                case_index=1,
                status="REVIEWED",
                is_active=True,
                start_page_number=1,
                end_page_number=3,
            )
            case.pages.extend(
                [
                    DocumentCasePage(
                        id=str(uuid4()),
                        page_id=reviewed_page["id"],
                        page_order=1,
                        page_role="NOTE",
                    ),
                    DocumentCasePage(
                        id=str(uuid4()),
                        page_id=skipped_page["id"],
                        page_order=2,
                        page_role="APPLICANT_DATA",
                    ),
                    DocumentCasePage(
                        id=str(uuid4()),
                        page_id=pending_page["id"],
                        page_order=3,
                        page_role="APPLICANT_DATA",
                    ),
                ]
            )
            session.add(case)
            session.add_all(
                [
                    OcrRun(
                        id=reviewed_run_id,
                        page_id=reviewed_page["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="old-v1",
                        device="gpu:0",
                    ),
                    OcrRun(
                        id=skipped_run_id,
                        page_id=skipped_page["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="old-v1",
                        device="gpu:0",
                    ),
                    OcrRun(
                        id=pending_run_id,
                        page_id=pending_page["id"],
                        status="COMPLETED",
                        is_active=True,
                        model_name="det+rec",
                        model_version="old-v1",
                        device="gpu:0",
                    ),
                    OcrReview(
                        id=str(uuid4()),
                        ocr_run_id=reviewed_run_id,
                        is_active=True,
                        include_in_llm=True,
                        reviewed_by="thai",
                    ),
                    OcrReview(
                        id=str(uuid4()),
                        ocr_run_id=skipped_run_id,
                        is_active=True,
                        include_in_llm=False,
                        reviewed_by="thai",
                    ),
                ]
            )

        history_page_1 = client.get(
            "/api/v1/ocr-review/history",
            params={"page": 1, "page_size": 1},
        ).json()
        history_page_2 = client.get(
            "/api/v1/ocr-review/history",
            params={"page": 2, "page_size": 1},
        ).json()
        assert history_page_1["page"] == 1
        assert history_page_2["page"] == 2
        assert history_page_1["page_size"] == 1
        assert history_page_1["total_items"] == 2
        assert history_page_1["total_pages"] == 2
        assert len(history_page_1["items"]) == 1
        assert len(history_page_2["items"]) == 1
        assert history_page_1["items"][0]["review_id"] != history_page_2["items"][0]["review_id"]

        pages = eligible_pages(
            client.app.state.database,
            page_types=["DIPLOMATIC_NOTE", "APPLICANT_TABLE", "APPLICANT_LIST"],
            page_id=None,
            case_id=None,
            limit=0,
            force=False,
            pending_review_only=True,
            preprocess_version="orientation-v1",
        )

        assert [page.id for page in pages] == [pending_page["id"]]

        with client.app.state.database.session() as session:
            pending_run = session.get(OcrRun, pending_run_id)
            assert pending_run is not None
            pending_run.preprocess_version = "orientation-v1"

        already_processed = eligible_pages(
            client.app.state.database,
            page_types=["DIPLOMATIC_NOTE", "APPLICANT_TABLE", "APPLICANT_LIST"],
            page_id=None,
            case_id=None,
            limit=0,
            force=False,
            pending_review_only=True,
            preprocess_version="orientation-v1",
        )

        assert already_processed == []

        rerunnable_unreviewed = eligible_pages(
            client.app.state.database,
            page_types=["DIPLOMATIC_NOTE", "APPLICANT_TABLE", "APPLICANT_LIST"],
            page_id=None,
            case_id=None,
            limit=0,
            force=False,
            pending_review_only=False,
            preprocess_version="orientation-v1",
            rerun_unreviewed=True,
        )

        assert [page.id for page in rerunnable_unreviewed] == [pending_page["id"]]


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
        overview = client.get(
            f"/api/v1/annotation/label-audit/documents/{document_id}/overview"
        )
        assert overview.status_code == 200
        overview_pages = overview.json()["pages"]
        assert len(overview_pages) == 2
        assert overview_pages[0]["audit_status"] == "PENDING"
        assert overview_pages[1]["audit_status"] is None
        assert overview_pages[1]["page"]["review_status"] == "UNLABELED"
        rejected = client.put(
            f"/api/v1/annotation/label-audit/documents/{document_id}/pages/2",
            json={"label": "APPLICANT_LIST", "reviewed_by": "thai"},
        )
        assert rejected.status_code == 400


def test_full_pdf_overview_can_relabel_an_old_non_target_human_label() -> None:
    with _client() as client:
        imported = client.post(
            "/api/v1/annotation/documents",
            files={"file": ("overview-relabel.pdf", _two_page_pdf(), "application/pdf")},
        ).json()
        document_id = imported["document"]["id"]
        page_2_id = imported["document"]["pages"][1]["id"]
        client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/1/review",
            json={"label": "DIPLOMATIC_NOTE", "reviewed_by": "first-reviewer"},
        )
        client.put(
            f"/api/v1/annotation/documents/{document_id}/pages/2/review",
            json={"label": "OTHER", "reviewed_by": "first-reviewer"},
        )

        # Seed the document audit with the target page. The old OTHER page is
        # intentionally absent until a human corrects it from the overview.
        queued = client.get("/api/v1/annotation/label-audit/review/next").json()
        assert len(queued["current"]["candidate_pages"]) == 1

        corrected = client.put(
            f"/api/v1/annotation/label-audit/documents/{document_id}/pages/2",
            json={
                "label": "APPLICANT_LIST",
                "reviewed_by": "thai",
                "audit_note": "Nhãn cũ bị gán nhầm",
            },
        )
        assert corrected.status_code == 200
        payload = corrected.json()
        assert payload["original_label"] == "OTHER"
        assert payload["audited_label"] == "APPLICANT_LIST"
        assert payload["audit_status"] == "REVIEWED"
        assert payload["page"]["label"] == "APPLICANT_LIST"

        reopened = client.get(
            f"/api/v1/annotation/label-audit/documents/{document_id}"
        ).json()
        assert len(reopened["candidate_pages"]) == 2
        corrected_item = next(
            item for item in reopened["candidate_pages"] if item["page"]["id"] == page_2_id
        )
        assert corrected_item["original_label"] == "OTHER"
        assert corrected_item["audited_label"] == "APPLICANT_LIST"
        assert corrected_item["audit_note"] == "Nhãn cũ bị gán nhầm"


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

        reopened = client.get(
            f"/api/v1/annotation/label-audit/documents/{document_id}"
        )
        assert reopened.status_code == 200
        reopened_pages = reopened.json()["candidate_pages"]
        assert [item["audited_label"] for item in reopened_pages] == [
            "DIPLOMATIC_NOTE",
            "OTHER",
        ]
        previous = client.get("/api/v1/annotation/label-audit/review/previous")
        assert previous.status_code == 200
        assert previous.json()["document_id"] == document_id
        no_earlier = client.get(
            "/api/v1/annotation/label-audit/review/previous",
            params={"before_document_id": document_id},
        )
        assert no_earlier.status_code == 200
        assert no_earlier.json() is None

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
