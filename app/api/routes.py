from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.core.config import ModelBackend, Settings
from app.core.runtime import RuntimeSelection
from app.models.factory import ModelBundle
from app.schemas.document import (
    DocumentExtractionResponse,
    HealthResponse,
    ModelComponentStatus,
    ModelsStatusResponse,
)
from app.services.exceptions import DocumentValidationError, PipelineUnavailableError
from app.services.pipeline import DocumentPipeline

router = APIRouter()


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _runtime(request: Request) -> RuntimeSelection:
    return request.app.state.runtime


def _models(request: Request) -> ModelBundle:
    return request.app.state.models


def _pipeline(request: Request) -> DocumentPipeline:
    return request.app.state.pipeline


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = _settings(request)
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.app_env,
        processing_mode="synchronous",
    )


@router.get("/models/status", response_model=ModelsStatusResponse)
def models_status(request: Request) -> ModelsStatusResponse:
    settings = _settings(request)
    runtime = _runtime(request)
    models = _models(request)
    components: list[ModelComponentStatus] = []
    for name, (enabled, configured_path) in settings.configured_model_paths().items():
        path_exists = bool(configured_path and configured_path.exists())
        if not enabled:
            message = "Component disabled by environment configuration."
        elif settings.model_backend == ModelBackend.MOCK:
            message = "Mock adapter loaded; configured model path is not used."
        elif not path_exists:
            message = "Configured model path does not exist."
        elif not models.adapters_loaded:
            message = "Model path exists, but the real backend adapter is not implemented yet."
        else:
            message = "Ready."
        components.append(
            ModelComponentStatus(
                name=name,
                enabled=enabled,
                configured_path=str(configured_path) if configured_path else None,
                path_exists=path_exists,
                loaded=enabled and models.adapters_loaded,
                message=message,
            )
        )
    return ModelsStatusResponse(
        backend=runtime.backend,
        requested_device=runtime.requested_device,
        effective_device=runtime.effective_device,
        gpu_available=runtime.gpu_available,
        runtime_reason=runtime.reason,
        components=components,
    )


@router.post(
    "/documents/extract",
    response_model=DocumentExtractionResponse,
    status_code=status.HTTP_200_OK,
)
def extract_document(
    request: Request,
    file: Annotated[UploadFile, File(description="Scanned PDF document")],
) -> DocumentExtractionResponse:
    """Process one PDF fully before returning; no queue or background execution."""

    settings = _settings(request)
    filename = Path(file.filename or "document.pdf").name
    suffix = Path(filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with tempfile.TemporaryDirectory(prefix="document-extraction-") as temp_dir:
            pdf_path = Path(temp_dir) / "input.pdf"
            with pdf_path.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > settings.max_file_size_bytes:
                        raise DocumentValidationError(
                            f"File exceeds the {settings.max_file_size_mb} MB limit."
                        )
                    digest.update(chunk)
                    output.write(chunk)

            if total_bytes == 0:
                raise DocumentValidationError("The uploaded file is empty.")
            with pdf_path.open("rb") as source:
                if source.read(5) != b"%PDF-":
                    raise DocumentValidationError(
                        "The uploaded file does not have a PDF signature."
                    )

            return _pipeline(request).extract(
                pdf_path=pdf_path,
                filename=filename,
                sha256=digest.hexdigest(),
            )
    except DocumentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PipelineUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    finally:
        file.file.close()
