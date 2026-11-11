from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from starlette.datastructures import UploadFile

from app.schemas.production_extraction import (
    ProductionExtractionResponse,
    ProductionReadinessResponse,
)
from app.schemas.teacher_extraction import (
    TeacherBusinessApplicantV8,
    TeacherBusinessNoteV8,
    TeacherBusinessOutputV8,
)
from app.services.exceptions import DocumentValidationError, ProductionExtractionError
from app.services.production_extraction import ProductionExtractionService

router = APIRouter(prefix="/document-extractions", tags=["production-extraction"])


def _service(request: Request) -> ProductionExtractionService:
    return request.app.state.production_extraction_service


def _verify_api_key(request: Request) -> bool:
    settings = request.app.state.settings
    expected_key = (getattr(settings, "production_api_key", None) or "").strip()
    if not expected_key:
        return True

    configured_header = (getattr(settings, "production_api_key_header", None) or "X-API-Key").strip()
    candidate_headers = [
        configured_header,
        "X-API-Key",
        "X-Request-Key",
        "api-key",
        "key",
        "Authorization",
    ]
    for header_name in candidate_headers:
        raw_val = request.headers.get(header_name)
        if not raw_val:
            continue
        token = raw_val.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if hmac.compare_digest(token, expected_key):
            return True
    return False


def _decode_base64_pdf(raw_b64: str) -> bytes:
    cleaned = raw_b64.strip()
    if cleaned.lower().startswith("data:") and "," in cleaned[:128]:
        cleaned = cleaned.split(",", maxsplit=1)[1].strip()
    cleaned = "".join(cleaned.split())
    if not cleaned:
        raise DocumentValidationError("The provided base64 content is empty.")
    padding = len(cleaned) % 4
    if padding:
        cleaned += "=" * (4 - padding)
    try:
        return base64.b64decode(cleaned, validate=False)
    except (binascii.Error, ValueError) as exc:
        try:
            return base64.urlsafe_b64decode(cleaned)
        except Exception:
            raise DocumentValidationError("Invalid base64-encoded PDF content.") from exc


async def _extract_pdf_bytes_and_filename(request: Request, max_bytes: int, max_mb: int) -> tuple[bytes, str]:
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            body: Any = await request.json()
        except Exception as exc:
            raise DocumentValidationError("Invalid JSON request body.") from exc

        filename = "document.pdf"
        b64_value: str | None = None
        if isinstance(body, str):
            b64_value = body
        elif isinstance(body, dict):
            raw_name = body.get("filename") or body.get("file_name") or body.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                filename = Path(raw_name.strip()).name
                if not filename.lower().endswith(".pdf"):
                    filename = f"{filename}.pdf"
            for key in ("file_base64", "base64", "pdf_base64", "file", "data", "content"):
                val = body.get(key)
                if isinstance(val, str) and val.strip():
                    b64_value = val
                    break
        if not b64_value:
            raise DocumentValidationError(
                "Missing PDF base64 data (expected 'file_base64' or 'base64' field)."
            )
        pdf_bytes = _decode_base64_pdf(b64_value)
        if len(pdf_bytes) > max_bytes:
            raise DocumentValidationError(f"File exceeds the {max_mb} MB limit.")
        return pdf_bytes, filename

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        try:
            filename = "document.pdf"
            raw_name = form.get("filename") or form.get("file_name")
            if isinstance(raw_name, str) and raw_name.strip():
                filename = Path(raw_name.strip()).name

            # 1. Check uploaded file field
            for key in ("file", "document", "pdf"):
                item = form.get(key)
                if isinstance(item, UploadFile) or hasattr(item, "file"):
                    upload = item  # type: ignore[assignment]
                    if getattr(upload, "filename", None):
                        filename = Path(upload.filename).name
                    if Path(filename).suffix.lower() != ".pdf":
                        raise HTTPException(
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Only PDF files are supported.",
                        )
                    chunks: list[bytes] = []
                    total = 0
                    while chunk := upload.file.read(1024 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise DocumentValidationError(
                                f"File exceeds the {max_mb} MB limit."
                            )
                        chunks.append(chunk)
                    raw_bytes = b"".join(chunks)
                    if not raw_bytes:
                        raise DocumentValidationError("The uploaded file is empty.")
                    if not raw_bytes.startswith(b"%PDF-"):
                        try:
                            raw_bytes = _decode_base64_pdf(raw_bytes.decode("utf-8", errors="ignore"))
                        except Exception:
                            pass
                    return raw_bytes, filename

            # 2. Check base64 string inside form fields
            for key in ("file_base64", "base64", "pdf_base64", "file", "data"):
                val = form.get(key)
                if isinstance(val, str) and val.strip():
                    if not filename.lower().endswith(".pdf"):
                        filename = f"{filename}.pdf"
                    pdf_bytes = _decode_base64_pdf(val)
                    if len(pdf_bytes) > max_bytes:
                        raise DocumentValidationError(f"File exceeds the {max_mb} MB limit.")
                    return pdf_bytes, filename

            raise DocumentValidationError(
                "Provide either a PDF file upload ('file') or base64 string ('file_base64' / 'base64')."
            )
        finally:
            await form.close()

    # Fallback: raw body as base64 or PDF bytes
    raw_body = await request.body()
    if not raw_body:
        raise DocumentValidationError("Request body is empty.")
    if raw_body.startswith(b"%PDF-"):
        if len(raw_body) > max_bytes:
            raise DocumentValidationError(f"File exceeds the {max_mb} MB limit.")
        return raw_body, "document.pdf"
    pdf_bytes = _decode_base64_pdf(raw_body.decode("utf-8", errors="ignore"))
    if len(pdf_bytes) > max_bytes:
        raise DocumentValidationError(f"File exceeds the {max_mb} MB limit.")
    return pdf_bytes, "document.pdf"


@router.get("/readiness", response_model=ProductionReadinessResponse)
def production_readiness(request: Request) -> ProductionReadinessResponse:
    return _service(request).readiness()


async def _run_extraction_pipeline(request: Request) -> ProductionExtractionResponse:
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
    try:
        pdf_bytes, filename = await _extract_pdf_bytes_and_filename(
            request,
            max_bytes=settings.max_file_size_bytes,
            max_mb=settings.max_file_size_mb,
        )
        if not pdf_bytes:
            raise DocumentValidationError("The uploaded file is empty.")
        if not pdf_bytes.startswith(b"%PDF-"):
            raise DocumentValidationError(
                "The uploaded file does not have a PDF signature."
            )
        sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        with tempfile.TemporaryDirectory(prefix="production-upload-") as temp_dir:
            pdf_path = Path(temp_dir) / "input.pdf"
            pdf_path.write_bytes(pdf_bytes)
            return service.extract(
                pdf_path=pdf_path,
                filename=filename,
                sha256=sha256,
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


@router.post("", response_model=TeacherBusinessOutputV8)
async def create_document_extraction(
    request: Request,
) -> TeacherBusinessOutputV8 | Response:
    if not _verify_api_key(request):
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)
    result = await _run_extraction_pipeline(request)
    return TeacherBusinessOutputV8(
        note=TeacherBusinessNoteV8(
            note_number=result.extraction.note.note_number,
            sender_organization=result.extraction.note.sender_organization,
            request_type_code=result.extraction.note.request_type_code,
        ),
        applicants=[
            TeacherBusinessApplicantV8(
                full_name=item.full_name,
                date_of_birth=item.date_of_birth,
                gender=item.gender,
                nationality=item.nationality,
                passport_number=item.passport_number,
                passport_type=item.passport_type,
                passport_issue_date=item.passport_issue_date,
                passport_expiration=item.passport_expiration,
                visa_type=item.visa_type,
                number_of_entries=item.number_of_entries,
                entry_date=item.entry_date,
                exit_date=item.exit_date,
            )
            for item in result.extraction.applicants
        ],
    )


@router.post("/debug", response_model=ProductionExtractionResponse)
async def create_document_extraction_debug(
    request: Request,
) -> ProductionExtractionResponse | Response:
    if not _verify_api_key(request):
        return Response(status_code=status.HTTP_401_UNAUTHORIZED)
    return await _run_extraction_pipeline(request)


