from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.schemas.teacher_extraction import (
    TeacherExtractionReviewDetail,
    TeacherExtractionReviewQueueResponse,
    TeacherExtractionReviewRequest,
    TeacherExtractionReviewSaveResponse,
    TeacherExtractionReviewStats,
)
from app.services.teacher_extraction_review import (
    TeacherExtractionReviewConflictError,
    TeacherExtractionReviewNotFoundError,
    TeacherExtractionReviewService,
    TeacherExtractionReviewValidationError,
)


router = APIRouter(prefix="/teacher-extraction", tags=["teacher-extraction"])


def _service(request: Request) -> TeacherExtractionReviewService:
    return request.app.state.teacher_extraction_review_service


@router.get("/statistics", response_model=TeacherExtractionReviewStats)
def teacher_extraction_statistics(request: Request) -> TeacherExtractionReviewStats:
    return _service(request).get_statistics()


@router.get(
    "/review/next",
    response_model=TeacherExtractionReviewQueueResponse,
)
def get_next_teacher_extraction_review(
    request: Request,
    from_index: int | None = Query(default=None, ge=0),
    to_index: int | None = Query(default=None, ge=0),
) -> TeacherExtractionReviewQueueResponse:
    if (from_index is None) != (to_index is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="from_index and to_index must be supplied together.",
        )
    if from_index is not None and to_index is not None and to_index < from_index:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="to_index must be greater than or equal to from_index.",
        )
    current, stats, current_index = _service(request).get_next_review(
        from_index=from_index,
        to_index=to_index,
    )
    return TeacherExtractionReviewQueueResponse(
        current=current,
        current_index=current_index,
        stats=stats,
    )


@router.get("/{review_id}", response_model=TeacherExtractionReviewDetail)
def get_teacher_extraction_review(
    request: Request,
    review_id: str,
) -> TeacherExtractionReviewDetail:
    try:
        return _service(request).get_detail(review_id)
    except TeacherExtractionReviewNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except TeacherExtractionReviewConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.put("/{review_id}/review", response_model=TeacherExtractionReviewSaveResponse)
def review_teacher_extraction(
    request: Request,
    review_id: str,
    payload: TeacherExtractionReviewRequest,
) -> TeacherExtractionReviewSaveResponse:
    try:
        return _service(request).review(review_id=review_id, payload=payload)
    except TeacherExtractionReviewNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except TeacherExtractionReviewConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except TeacherExtractionReviewValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
