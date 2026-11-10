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


def test_production_upload_is_rejected_until_preloaded_runtime_is_ready() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/document-extractions",
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

