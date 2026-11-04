from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.teacher_extraction import (
    TeacherExtractionReviewDetail,
    TeacherExtractionReviewQueueResponse,
    TeacherExtractionReviewRequest,
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
) -> TeacherExtractionReviewQueueResponse:
    current, stats = _service(request).get_next_review()
    return TeacherExtractionReviewQueueResponse(current=current, stats=stats)


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


@router.put("/{review_id}/review", response_model=TeacherExtractionReviewDetail)
def review_teacher_extraction(
    request: Request,
    review_id: str,
    payload: TeacherExtractionReviewRequest,
) -> TeacherExtractionReviewDetail:
    try:
        return _service(request).review(review_id=review_id, payload=payload)
    except TeacherExtractionReviewNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except TeacherExtractionReviewConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except TeacherExtractionReviewValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
