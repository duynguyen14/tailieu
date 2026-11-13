from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.production_extraction import (
    PassportMrzBatchRequest,
    PassportMrzBatchResponse,
    PassportMrzReadinessResponse,
)
from app.services.passport_mrz.service import PassportMrzService

router = APIRouter(prefix="/internal/v1/passports", tags=["passport-mrz-internal"])


def _service(request: Request) -> PassportMrzService:
    return request.app.state.passport_mrz_service


@router.get("/readiness", response_model=PassportMrzReadinessResponse)
def readiness(request: Request) -> PassportMrzReadinessResponse:
    return _service(request).readiness()


@router.post("/read", response_model=PassportMrzBatchResponse)
def read_passports(
    payload: PassportMrzBatchRequest, request: Request
) -> PassportMrzBatchResponse:
    service = _service(request)
    state = service.readiness()
    if not state.enabled or not state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.error or "Passport MRZ service is not ready",
        )
    try:
        return service.read(payload)
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

