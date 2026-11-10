from __future__ import annotations

import hashlib
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import httpx
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, Settings
from app.production_main import create_production_app
from app.production_ocr_main import create_ocr_app
from app.schemas.production_extraction import ProductionModelStatus
from app.services.pdf_renderer import PdfRenderer
from app.services.production_extraction import ProductionExtractionService


@contextmanager
def _disabled_ocr_client():
    settings = Settings(
        _env_file=None,
        production_ocr_service_enabled=False,
        database_url="sqlite:///:memory:",
    )
    with TestClient(create_ocr_app(settings)) as client:
        yield client


def test_private_ocr_readiness_reports_disabled_without_loading_paddle() -> None:
    with _disabled_ocr_client() as client:
        response = client.get("/internal/v1/ocr/readiness")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "ready": False,
        "loading": False,
        "seal_detection_enabled": False,
        "models": [],
        "error": None,
    }


def test_dedicated_public_app_keeps_existing_endpoint_contract() -> None:
    settings = Settings(
        _env_file=None,
        production_extraction_enabled=False,
        database_url="sqlite:///:memory:",
    )
    with TestClient(create_production_app(settings)) as client:
        response = client.get("/api/v1/document-extractions/readiness")

    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert response.json()["seal_detection_enabled"] is False


def _split_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "production_extraction_enabled": True,
        "production_split_runtime": True,
        "production_ocr_service_url": "http://ocr.test/internal/v1/ocr",
        "production_include_reconstructed_text": True,
    }
    values.update(overrides)
    return Settings(**values)


def _service(settings: Settings) -> ProductionExtractionService:
    renderer = cast(PdfRenderer, object())
    service = ProductionExtractionService(settings, renderer)
    service._ready = True
    service._models = [
        ProductionModelStatus(
            name="student_extractor",
            loaded=True,
            device="cuda:2",
            model="qwen",
        )
    ]
    return service


def test_public_readiness_combines_remote_ocr_and_local_student_models() -> None:
    settings = _split_settings()
    service = _service(settings)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://ocr.test/internal/v1/ocr/readiness"
        return httpx.Response(
            200,
            json={
                "enabled": True,
                "ready": True,
                "loading": False,
                "seal_detection_enabled": False,
                "models": [
                    {
                        "name": "ocr_detection_recognition",
                        "loaded": True,
                        "device": "cuda:3",
                        "model": "paddle",
                    }
                ],
                "error": None,
            },
        )

    service._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    readiness = service.readiness()
    service.close()

    assert readiness.ready is True
    assert [model.name for model in readiness.models] == [
        "ocr_detection_recognition",
        "student_extractor",
    ]


def test_split_runtime_proxies_pdf_text_to_student_and_preserves_diagnostics() -> None:
    settings = _split_settings()
    service = _service(settings)
    pdf_bytes = b"%PDF-split-runtime-test"
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    class FakeStudent:
        def generate(self, text: str):
            assert text == "[PAGE 1]\nEmbassy requests a new visa."
            return (
                {
                    "n": {"no": "123", "snd": "Embassy", "rt": "CAP_MOI"},
                    "a": [],
                },
                12,
                34.5,
            )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "http://ocr.test/internal/v1/ocr/reconstruct"
        return httpx.Response(
            200,
            json={
                "filename": "scan.pdf",
                "sha256": sha256,
                "page_count": 1,
                "reconstructed_text": "[PAGE 1]\nEmbassy requests a new visa.",
                "pages": [
                    {
                        "page_number": 1,
                        "rotation_angle": 0,
                        "orientation_confidence": 0.99,
                        "page_type": "DIPLOMATIC_NOTE",
                        "classification_confidence": 0.98,
                        "used_for_extraction": True,
                        "ocr_token_count": 7,
                        "ocr_processing_time_ms": 12.0,
                    }
                ],
                "timings": {
                    "render_ms": 1.0,
                    "orientation_ms": 2.0,
                    "classification_ms": 3.0,
                    "ocr_ms": 4.0,
                    "reconstruction_ms": 5.0,
                    "total_ms": 15.0,
                },
                "warnings": ["example warning"],
            },
        )

    service._student = cast(Any, FakeStudent())
    service._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    temp_parent = PROJECT_ROOT / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="split-runtime-", dir=temp_parent) as temp_dir:
        path = Path(temp_dir) / "scan.pdf"
        path.write_bytes(pdf_bytes)
        result = service.extract(pdf_path=path, filename="scan.pdf", sha256=sha256)
    service.close()

    assert result.extraction.note.note_number == "123"
    assert result.extraction.note.request_type_code == "CAP_MOI"
    assert result.reconstructed_text == "[PAGE 1]\nEmbassy requests a new visa."
    assert result.pages[0].ocr_token_count == 7
    assert result.timings.extraction_ms == 34.5
    assert result.warnings == ["example warning"]
