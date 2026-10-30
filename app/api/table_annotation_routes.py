from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.table_annotation import (
    TableAnnotationDetail,
    TableReviewQueueResponse,
    TableReviewRequest,
    TableReviewStats,
)
from app.services.table_annotation import (
    TableAnnotationConflictError,
    TableAnnotationNotFoundError,
    TableAnnotationService,
    TableAnnotationValidationError,
)


router = APIRouter(prefix="/table-annotation", tags=["table-annotation"])


def _service(request: Request) -> TableAnnotationService:
    return request.app.state.table_annotation_service


@router.get("/statistics", response_model=TableReviewStats)
def table_review_statistics(request: Request) -> TableReviewStats:
    _, stats = _service(request).get_next_review()
    return stats


@router.get("/review/next", response_model=TableReviewQueueResponse)
def get_next_table_review(request: Request) -> TableReviewQueueResponse:
    current, stats = _service(request).get_next_review()
    return TableReviewQueueResponse(current=current, stats=stats)


@router.get("/{annotation_id}", response_model=TableAnnotationDetail)
def get_table_annotation(request: Request, annotation_id: str) -> TableAnnotationDetail:
    try:
        return _service(request).get_detail(annotation_id)
    except TableAnnotationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put("/{annotation_id}/review", response_model=TableAnnotationDetail)
def review_table_annotation(
    request: Request,
    annotation_id: str,
    payload: TableReviewRequest,
) -> TableAnnotationDetail:
    try:
        return _service(request).review(annotation_id=annotation_id, payload=payload)
    except TableAnnotationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except TableAnnotationConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except TableAnnotationValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
