from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, Settings
from app.main import create_app


@contextmanager
def _client():
    temp_parent = PROJECT_ROOT / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="production-api-test-", dir=temp_parent) as temp_dir:
        relative_storage = Path(temp_dir).relative_to(PROJECT_ROOT) / "storage"
        settings = Settings(
            _env_file=None,
            model_backend="mock",
            device="cpu",
            database_url="sqlite:///:memory:",
            auto_create_db=True,
            annotation_storage_dir=relative_storage,
            production_extraction_enabled=False,
        )
        with TestClient(create_app(settings)) as client:
            yield client


def test_production_readiness_explicitly_reports_seal_detection_disabled() -> None:
    with _client() as client:
        response = client.get("/api/v1/document-extractions/readiness")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "ready": False,
        "loading": False,
        "seal_detection_enabled": False,
        "models": [],
        "error": None,
    }


def test_production_upload_rejects_missing_or_invalid_api_key_with_empty_401() -> None:
    with _client() as client:
        missing_key = client.post(
            "/api/v1/document-extractions",
            files={"file": ("scan.pdf", b"%PDF-test", "application/pdf")},
        )
        wrong_key = client.post(
            "/api/v1/document-extractions",
            headers={"X-API-Key": "wrong-key"},
            files={"file": ("scan.pdf", b"%PDF-test", "application/pdf")},
        )

    assert missing_key.status_code == 401
    assert missing_key.content == b""
    assert wrong_key.status_code == 401
    assert wrong_key.content == b""


def test_production_upload_is_rejected_until_preloaded_runtime_is_ready() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/document-extractions",
            headers={"X-API-Key": "9148fca3-187c-46d1-95d5-5c8c4b8ea1ad"},
            files={"file": ("scan.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Production extraction is disabled by configuration."


def test_page_classification_preview_is_rejected_when_production_is_disabled() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/page-classification-preview",
            files={"file": ("scan.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Production extraction is disabled by configuration."


def test_production_endpoint_accepts_file_and_base64_and_returns_clean_business_json() -> None:
    import base64
    from app.production_main import create_production_app
    from app.schemas.production_extraction import (
        ProductionExtractionResponse,
        ProductionModelStatus,
        ProductionReadinessResponse,
        ProductionTiming,
    )
    from app.schemas.student_extraction import student_target_to_compact_document

    settings = Settings(
        _env_file=None,
        production_extraction_enabled=False,
        production_api_key="9148fca3-187c-46d1-95d5-5c8c4b8ea1ad",
        database_url="sqlite:///:memory:",
    )
    app = create_production_app(settings)

    compact_output = {
        "n": {"no": "479/2025", "snd": "Embassy of Test", "rt": "CAP_MOI"},
        "a": [
            {
                "nm": "JOHN DOE",
                "dob": "1967-11-20",
                "sex": "M",
                "nat": "USA",
                "pp": "A1234567",
                "pt": "ORDINARY",
                "pid": "2020-01-01",
                "pex": "2030-01-01",
                "vt": "DN1",
                "ne": "MULTIPLE",
                "in": "2026-11-10",
                "out": "2026-12-10",
            }
        ],
    }

    class FakeService:
        def readiness(self) -> ProductionReadinessResponse:
            return ProductionReadinessResponse(
                enabled=True,
                ready=True,
                loading=False,
                seal_detection_enabled=False,
                models=[
                    ProductionModelStatus(
                        name="student_extractor",
                        loaded=True,
                        device="cuda:2",
                        model="qwen",
                    )
                ],
                error=None,
            )

        def extract(self, *, pdf_path: Path, filename: str, sha256: str):
            assert pdf_path.read_bytes().startswith(b"%PDF-")
            return ProductionExtractionResponse(
                request_id="req-1",
                filename=filename,
                sha256=sha256,
                page_count=1,
                compact_model_output=compact_output,
                extraction=student_target_to_compact_document(
                    compact_output,
                    case_id="req-1",
                    document_id=sha256,
                ),
                reconstructed_text="[PAGE 1]",
                pages=[],
                timings=ProductionTiming(
                    render_ms=1.0,
                    orientation_ms=1.0,
                    classification_ms=1.0,
                    ocr_ms=1.0,
                    reconstruction_ms=1.0,
                    extraction_ms=1.0,
                    total_ms=6.0,
                ),
                warnings=[],
            )

        def close(self) -> None:
            pass

    with TestClient(app) as client:
        client.app.state.production_extraction_service = FakeService()
        headers = {"X-API-Key": "9148fca3-187c-46d1-95d5-5c8c4b8ea1ad"}

        file_resp = client.post(
            "/api/v1/document-extractions",
            headers=headers,
            files={"file": ("scan.pdf", b"%PDF-1.4 file content", "application/pdf")},
        )
        b64_payload = base64.b64encode(b"%PDF-1.4 base64 content").decode("ascii")
        json_resp = client.post(
            "/api/v1/document-extractions",
            headers=headers,
            json={
                "filename": "base64_scan.pdf",
                "file_base64": f"data:application/pdf;base64,{b64_payload}",
            },
        )

        debug_resp = client.post(
            "/api/v1/document-extractions/debug",
            headers=headers,
            files={"file": ("scan.pdf", b"%PDF-1.4 file content", "application/pdf")},
        )

    expected_clean_json = {
        "note": {
            "note_number": ["479/2025"],
            "sender_organization": ["Embassy of Test"],
            "request_type_code": ["CAP_MOI"],
        },
        "applicants": [
            {
                "full_name": ["JOHN DOE"],
                "date_of_birth": ["1967-11-20"],
                "gender": ["M"],
                "nationality": ["USA"],
                "passport_number": ["A1234567"],
                "passport_type": ["ORDINARY"],
                "passport_issue_date": ["2020-01-01"],
                "passport_expiration": ["2030-01-01"],
                "visa_type": ["DN1"],
                "number_of_entries": ["MULTIPLE"],
                "entry_date": ["2026-11-10"],
                "exit_date": ["2026-12-10"],
            }
        ],
    }
    assert file_resp.status_code == 200
    assert file_resp.json() == expected_clean_json
    assert json_resp.status_code == 200
    assert json_resp.json() == expected_clean_json

    assert debug_resp.status_code == 200
    debug_json = debug_resp.json()
    assert debug_json["request_id"] == "req-1"
    assert "timings" in debug_json
    assert "pages" in debug_json
    assert debug_json["timings"]["render_ms"] == 1.0
    assert debug_json["extraction"]["note"]["note_number"] == "479/2025"
    assert len(debug_json["extraction"]["applicants"]) == 1


def test_student_runtime_normalizes_non_iso_dates_and_repairs_truncated_json() -> None:
    from app.schemas.student_extraction import student_target_to_compact_document
    from app.services.production_extraction import _StudentRuntime

    truncated_raw = (
        '{"n":{"no":"018/2026","snd":"Embassy","rt":"CAP_MOI"},'
        '"a":[{"nm":"TRAN VAN A","dob":"20 November 1967","sex":"M","nat":"VNM",'
        '"pp":"B12345","pt":"DIPLOMATIC","pid":"05/11/2020","pex":"unterminated-date-string'
    )
    repaired = _StudentRuntime._parse_or_repair_student_output(truncated_raw)
    normalized = _StudentRuntime._normalize_target(repaired)
    compact_doc = student_target_to_compact_document(
        normalized,
        case_id="case-1",
        document_id="doc-1",
    )

    assert compact_doc.note.note_number == "018/2026"
    assert len(compact_doc.applicants) == 1
    assert compact_doc.applicants[0].date_of_birth == "1967-11-20"
    assert compact_doc.applicants[0].passport_issue_date == "2020-11-05"
    assert compact_doc.applicants[0].passport_expiration is None


