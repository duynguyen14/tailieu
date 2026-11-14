from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.schemas.visa_form_annotation import (
    VisaFormAnnotationDetail,
    VisaFormFieldDefinition,
    VisaFormReviewQueueResponse,
    VisaFormReviewRequest,
    VisaFormReviewStats,
)
from app.services.visa_form_annotation import (
    VisaFormAnnotationConflictError,
    VisaFormAnnotationNotFoundError,
    VisaFormAnnotationService,
    VisaFormAnnotationValidationError,
)

router = APIRouter(prefix="/visa-form-annotation", tags=["visa-form-annotation"])


def _service(request: Request) -> VisaFormAnnotationService:
    return request.app.state.visa_form_annotation_service


@router.get("/labels", response_model=list[VisaFormFieldDefinition])
def visa_form_labels(request: Request) -> list[VisaFormFieldDefinition]:
    return _service(request).field_definitions()


@router.get("/statistics", response_model=VisaFormReviewStats)
def visa_form_statistics(request: Request) -> VisaFormReviewStats:
    _, stats, _ = _service(request).get_next_review()
    return stats


@router.get("/review/next", response_model=VisaFormReviewQueueResponse)
def get_next_visa_form_review(
    request: Request,
    search_id: str | None = Query(default=None),
    sort_order: str = Query(default="NEWEST", pattern="^(NEWEST|OLDEST)$"),
    from_index: int | None = Query(default=None, ge=0),
    to_index: int | None = Query(default=None, ge=0),
) -> VisaFormReviewQueueResponse:
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
        search_id=search_id,
        sort_order=sort_order,
        from_index=from_index,
        to_index=to_index,
    )
    return VisaFormReviewQueueResponse(
        current=current, stats=stats, current_index=current_index
    )


@router.get("/{annotation_id}", response_model=VisaFormAnnotationDetail)
def get_visa_form_annotation(
    request: Request, annotation_id: str
) -> VisaFormAnnotationDetail:
    try:
        return _service(request).get_detail(annotation_id)
    except VisaFormAnnotationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put("/{annotation_id}/review", response_model=VisaFormAnnotationDetail)
def review_visa_form_annotation(
    request: Request,
    annotation_id: str,
    payload: VisaFormReviewRequest,
) -> VisaFormAnnotationDetail:
    try:
        return _service(request).review(annotation_id=annotation_id, payload=payload)
    except VisaFormAnnotationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except VisaFormAnnotationConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except VisaFormAnnotationValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
