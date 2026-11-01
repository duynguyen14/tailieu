from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.schemas.ocr_review import (
    OcrReviewDetail,
    OcrReviewHistoryResponse,
    OcrReviewQueueResponse,
    OcrReviewRequest,
    OcrReviewStats,
)
from app.services.ocr_review import (
    OcrReviewConflictError,
    OcrReviewNotFoundError,
    OcrReviewPersistenceError,
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
def get_next_ocr_review(
    request: Request,
    search_id: str | None = Query(default=None, max_length=36),
    sort_order: Literal["NEWEST", "OLDEST"] = Query(default="NEWEST"),
) -> OcrReviewQueueResponse:
    current, stats = _service(request).get_next_review(
        search_id=search_id,
        sort_order=sort_order,
    )
    return OcrReviewQueueResponse(current=current, stats=stats)


@router.get("/history", response_model=OcrReviewHistoryResponse)
def get_ocr_review_history(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    decision: Literal["ALL", "APPROVED", "SKIPPED"] = Query(default="ALL"),
    filename: str | None = Query(default=None, max_length=255),
    page_type: str | None = Query(default=None, max_length=64),
    reviewed_by: str | None = Query(default=None, max_length=100),
) -> OcrReviewHistoryResponse:
    return _service(request).get_history(
        page=page,
        page_size=page_size,
        decision=decision,
        filename=filename,
        page_type=page_type,
        reviewed_by=reviewed_by,
    )


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
    except OcrReviewPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
