from fastapi import APIRouter, HTTPException, Query, Request, status

from app.schemas.kie_review import (
    KieCaseView,
    KieQueueResponse,
    KieReReviewRequest,
    KieReviewSaveResponse,
    KieReviewStats,
    KieSerReviewRequest,
)
from app.services.kie_review import (
    KieReviewConflictError,
    KieReviewNotFoundError,
    KieReviewService,
    KieReviewValidationError,
)

router = APIRouter(prefix="/kie-review", tags=["kie-review"])


def _service(request: Request) -> KieReviewService:
    return request.app.state.kie_review_service


def _handle(call):
    try:
        return call()
    except KieReviewNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except KieReviewConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except KieReviewValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/statistics", response_model=KieReviewStats)
def statistics(request: Request):
    return _service(request).get_statistics()


@router.get("/next", response_model=KieQueueResponse)
def next_item(
    request: Request,
    mode: str = Query(pattern="^(SER|RE)$"),
    from_index: int | None = Query(default=None, ge=0),
    to_index: int | None = Query(default=None, ge=0),
):
    if (from_index is None) != (to_index is None) or (from_index is not None and to_index < from_index):
        raise HTTPException(status_code=400, detail="Supply a valid from_index/to_index pair.")
    return _handle(lambda: _service(request).get_next(mode=mode, from_index=from_index, to_index=to_index))


@router.get("/pages/{page_annotation_id}", response_model=KieCaseView)
def detail(request: Request, page_annotation_id: str):
    return _handle(lambda: _service(request).get_detail(page_annotation_id))


@router.put("/pages/{page_annotation_id}/ser", response_model=KieReviewSaveResponse)
def review_ser(request: Request, page_annotation_id: str, payload: KieSerReviewRequest):
    return _handle(lambda: _service(request).review_ser(page_annotation_id, payload))


@router.put("/pages/{page_annotation_id}/re", response_model=KieReviewSaveResponse)
def review_re(request: Request, page_annotation_id: str, payload: KieReReviewRequest):
    return _handle(lambda: _service(request).review_re(page_annotation_id, payload))
