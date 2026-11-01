from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.text_reconstruction import (
    TextReconstructionDetail,
    TextReconstructionQueueResponse,
    TextReconstructionReviewRequest,
    TextReconstructionStats,
)
from app.services.text_reconstruction_review import (
    TextReconstructionConflictError,
    TextReconstructionNotFoundError,
    TextReconstructionReviewService,
    TextReconstructionValidationError,
)


router = APIRouter(prefix="/text-reconstruction", tags=["text-reconstruction"])


def _service(request: Request) -> TextReconstructionReviewService:
    return request.app.state.text_reconstruction_review_service


@router.get("/statistics", response_model=TextReconstructionStats)
def text_reconstruction_statistics(request: Request) -> TextReconstructionStats:
    return _service(request).get_statistics()


@router.get("/review/next", response_model=TextReconstructionQueueResponse)
def get_next_text_reconstruction(
    request: Request,
) -> TextReconstructionQueueResponse:
    current, stats = _service(request).get_next_review()
    return TextReconstructionQueueResponse(current=current, stats=stats)


@router.get("/{reconstruction_id}", response_model=TextReconstructionDetail)
def get_text_reconstruction(
    request: Request, reconstruction_id: str
) -> TextReconstructionDetail:
    try:
        return _service(request).get_detail(reconstruction_id)
    except TextReconstructionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put("/{reconstruction_id}/review", response_model=TextReconstructionDetail)
def review_text_reconstruction(
    request: Request,
    reconstruction_id: str,
    payload: TextReconstructionReviewRequest,
) -> TextReconstructionDetail:
    try:
        return _service(request).review(
            reconstruction_id=reconstruction_id,
            payload=payload,
        )
    except TextReconstructionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except TextReconstructionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except TextReconstructionValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
