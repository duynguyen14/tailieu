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


def test_private_classification_preview_rejects_request_when_ocr_is_disabled() -> None:
    with _disabled_ocr_client() as client:
        response = client.post(
            "/internal/v1/ocr/page-classifications",
            files={"file": ("scan.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Production OCR service is disabled by configuration."
    )


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


def test_split_runtime_proxies_page_classification_preview() -> None:
    settings = _split_settings()
    service = _service(settings)
    pdf_bytes = b"%PDF-classification-preview-test"
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == (
            "http://ocr.test/internal/v1/ocr/page-classifications"
        )
        return httpx.Response(
            200,
            json={
                "filename": "scan.pdf",
                "sha256": sha256,
                "page_count": 1,
                "target_page_types": ["DIPLOMATIC_NOTE"],
                "pages": [
                    {
                        "page_number": 1,
                        "image_content_type": "image/png",
                        "image_base64": "aW1hZ2U=",
                        "width": 1200,
                        "height": 1600,
                        "rotation_angle": 0,
                        "orientation_confidence": 0.99,
                        "predicted_type": "OTHER",
                        "confidence": 0.71,
                        "top_predictions": [
                            {"label": "OTHER", "confidence": 0.71},
                            {"label": "DIPLOMATIC_NOTE", "confidence": 0.29},
                        ],
                        "is_extraction_target": False,
                    }
                ],
                "timings": {
                    "render_ms": 1.0,
                    "orientation_ms": 2.0,
                    "classification_ms": 3.0,
                    "total_ms": 6.0,
                },
            },
        )

    service._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    temp_parent = PROJECT_ROOT / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="classification-preview-", dir=temp_parent) as temp_dir:
        path = Path(temp_dir) / "scan.pdf"
        path.write_bytes(pdf_bytes)
        result = service.preview_page_classifications(
            pdf_path=path,
            filename="scan.pdf",
            sha256=sha256,
        )
    service.close()

    assert result.pages[0].predicted_type == "OTHER"
    assert result.pages[0].is_extraction_target is False
    assert result.pages[0].top_predictions[1].label == "DIPLOMATIC_NOTE"


def test_ocr_reconstruct_falls_back_to_all_pages_when_no_page_matches_target_types() -> None:
    from types import SimpleNamespace
    from PIL import Image
    from app.domain.types import RenderedPage
    from app.models.paddle_ocr_engine import OcrTokenPrediction
    from app.services.production_ocr import ProductionOcrService

    settings = Settings(
        _env_file=None,
        production_ocr_service_enabled=True,
        production_ocr_enable_content_crop=False,
        production_ocr_enable_small_text_upscale=False,
        production_ocr_enable_refinement=False,
        ocr_enable_orientation=False,
    )

    class FakeRenderer:
        def render(self, pdf_path: Path, *, include_png_bytes: bool = True):
            return [
                RenderedPage(
                    page_index=0,
                    page_number=1,
                    width=400,
                    height=300,
                    image_bytes=b"",
                    pil_image=Image.new("RGB", (400, 300), "white"),
                ),
                RenderedPage(
                    page_index=1,
                    page_number=2,
                    width=400,
                    height=300,
                    image_bytes=b"",
                    pil_image=Image.new("RGB", (400, 300), "white"),
                ),
            ]

    class FakeClassifier:
        def predict_images(self, images, *, top_k: int = 1):
            return [
                SimpleNamespace(label=SimpleNamespace(value="OTHER"), confidence=0.91),
                SimpleNamespace(label=SimpleNamespace(value="PASSPORT_FRONT"), confidence=0.88),
            ]

    class FakeOrientation:
        def predict_image(self, image):
            return SimpleNamespace(angle=0, confidence=0.99)

    class FakeOcr:
        def predict_image(self, image):
            return [
                OcrTokenPrediction(
                    token_id="t1",
                    token_index=0,
                    text="FALLBACK TEXT",
                    confidence=0.99,
                    bbox=(10, 10, 120, 30),
                    polygon=((10, 10), (120, 10), (120, 30), (10, 30)),
                )
            ]

    ocr_service = ProductionOcrService(settings, cast(PdfRenderer, FakeRenderer()))
    ocr_service._ready = True
    ocr_service._classifier = cast(Any, FakeClassifier())
    ocr_service._orientation = cast(Any, FakeOrientation())
    ocr_service._ocr = cast(Any, FakeOcr())

    result = ocr_service.reconstruct(
        pdf_path=Path("dummy.pdf"),
        filename="dummy.pdf",
        sha256="abc1234567890",
    )

    assert result.page_count == 2
    assert [p.used_for_extraction for p in result.pages] == [True, True]
    assert [p.ocr_token_count for p in result.pages] == [1, 1]
    assert "FALLBACK TEXT" in result.reconstructed_text
    assert any("falling back to OCR on all 2 page(s)" in w for w in result.warnings)
