from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.annotation import AnnotationPageSummary
from app.schemas.page_label_audit import (
    PageLabelAuditDocument,
    PageLabelAuditItem,
    PageLabelAuditQueueResponse,
    PageLabelAuditReviewRequest,
    PageLabelAuditReviewResponse,
)
from app.services.page_label_audit import (
    PageLabelAuditNotFoundError,
    PageLabelAuditService,
    PageLabelAuditValidationError,
)


router = APIRouter(prefix="/annotation/label-audit", tags=["page-label-audit"])


def _service(request: Request) -> PageLabelAuditService:
    return request.app.state.page_label_audit_service


@router.get("/review/next", response_model=PageLabelAuditQueueResponse)
def get_next_page_label_audit(request: Request) -> PageLabelAuditQueueResponse:
    service = _service(request)
    document, records, stats = service.get_next_document()
    current = None
    if document is not None:
        current = PageLabelAuditDocument(
            document_id=document.id,
            original_filename=document.original_filename,
            document_page_count=document.page_count,
            candidate_pages=[
                PageLabelAuditItem(
                    audit_id=record.audit.id,
                    original_label=record.audit.original_label,
                    audited_label=record.audit.audited_label,
                    audit_status=record.audit.audit_status,
                    page=AnnotationPageSummary.model_validate(record.page),
                    image_base64=base64.b64encode(
                        service.read_page_image(record.page)
                    ).decode("ascii"),
                )
                for record in records
            ],
        )
    return PageLabelAuditQueueResponse(current=current, stats=stats)


@router.put(
    "/documents/{document_id}/review",
    response_model=PageLabelAuditReviewResponse,
)
def review_page_labels(
    request: Request,
    document_id: str,
    payload: PageLabelAuditReviewRequest,
) -> PageLabelAuditReviewResponse:
    try:
        reviewed_pages = _service(request).review_document(
            document_id=document_id,
            reviewed_by=payload.reviewed_by,
            decisions=payload.pages,
        )
        return PageLabelAuditReviewResponse(
            document_id=document_id,
            reviewed_pages=reviewed_pages,
        )
    except PageLabelAuditNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PageLabelAuditValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
