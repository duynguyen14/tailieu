from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.production_extraction import ProductionPageClassificationResponse
from app.services.exceptions import DocumentValidationError, ProductionExtractionError
from app.services.production_extraction import ProductionExtractionService

router = APIRouter(
    prefix="/page-classification-preview",
    tags=["production-page-classification"],
)


def _service(request: Request) -> ProductionExtractionService:
    return request.app.state.production_extraction_service


@router.post("", response_model=ProductionPageClassificationResponse)
def preview_page_classifications(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF rendered to classified PNG pages")],
) -> ProductionPageClassificationResponse:
    service = _service(request)
    readiness = service.readiness()
    if not readiness.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Production extraction is disabled by configuration.",
        )
    if not readiness.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=readiness.error or "Production models are not ready.",
        )

    settings = request.app.state.settings
    filename = Path(file.filename or "document.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with tempfile.TemporaryDirectory(prefix="classification-preview-upload-") as temp_dir:
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
            return service.preview_page_classifications(
                pdf_path=pdf_path,
                filename=filename,
                sha256=digest.hexdigest(),
            )
    except DocumentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ProductionExtractionError as exc:
        code = (
            status.HTTP_429_TOO_MANY_REQUESTS
            if "worker is busy" in str(exc)
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    finally:
        file.file.close()
