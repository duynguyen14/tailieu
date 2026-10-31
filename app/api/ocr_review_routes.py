from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.ocr_review import (
    OcrReviewDetail,
    OcrReviewQueueResponse,
    OcrReviewRequest,
    OcrReviewStats,
)
from app.services.ocr_review import (
    OcrReviewConflictError,
    OcrReviewNotFoundError,
    OcrReviewService,
    OcrReviewValidationError,
)


router = APIRouter(prefix="/ocr-review", tags=["ocr-review"])


def _service(request: Request) -> OcrReviewService:
    return request.app.state.ocr_review_service


@router.get("/statistics", response_model=OcrReviewStats)
def ocr_review_statistics(request: Request) -> OcrReviewStats:
    _, stats = _service(request).get_next_review()
    return stats


@router.get("/review/next", response_model=OcrReviewQueueResponse)
def get_next_ocr_review(request: Request) -> OcrReviewQueueResponse:
    current, stats = _service(request).get_next_review()
    return OcrReviewQueueResponse(current=current, stats=stats)


@router.get("/{run_id}", response_model=OcrReviewDetail)
def get_ocr_review(request: Request, run_id: str) -> OcrReviewDetail:
    try:
        return _service(request).get_detail(run_id)
    except OcrReviewNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put("/{run_id}/review", response_model=OcrReviewDetail)
def review_ocr(
    request: Request, run_id: str, payload: OcrReviewRequest
) -> OcrReviewDetail:
    try:
        return _service(request).review(run_id=run_id, payload=payload)
    except OcrReviewNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OcrReviewConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except OcrReviewValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
