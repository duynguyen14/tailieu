from __future__ import annotations

import base64
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.schemas.annotation import AnnotationPageSummary
from app.schemas.page_label_audit import (
    PageLabelAuditDocument,
    PageLabelAuditDocumentOverview,
    PageLabelAuditItem,
    PageLabelAuditOverviewItem,
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


def _document_response(
    service: PageLabelAuditService,
    document,
    records,
) -> PageLabelAuditDocument:
    return PageLabelAuditDocument(
        document_id=document.id,
        original_filename=document.original_filename,
        document_page_count=document.page_count,
        candidate_pages=[
            PageLabelAuditItem(
                audit_id=record.audit.id,
                original_label=record.audit.original_label,
                audited_label=record.audit.audited_label,
                audit_status=record.audit.audit_status,
                audit_note=record.audit.audit_note,
                page=AnnotationPageSummary.model_validate(record.page),
                image_base64=base64.b64encode(service.read_page_image(record.page)).decode(
                    "ascii"
                ),
            )
            for record in records
        ],
    )


@router.get("/review/next", response_model=PageLabelAuditQueueResponse)
def get_next_page_label_audit(request: Request) -> PageLabelAuditQueueResponse:
    service = _service(request)
    document, records, stats = service.get_next_document()
    current = None
    if document is not None:
        current = _document_response(service, document, records)
    return PageLabelAuditQueueResponse(current=current, stats=stats)


@router.get(
    "/review/previous",
    response_model=PageLabelAuditDocument | None,
)
def get_previous_page_label_audit(
    request: Request,
    before_document_id: Annotated[str | None, Query()] = None,
) -> PageLabelAuditDocument | None:
    service = _service(request)
    result = service.get_previous_document(before_document_id)
    if result is None:
        return None
    document, records = result
    return _document_response(service, document, records)


@router.get(
    "/documents/{document_id}",
    response_model=PageLabelAuditDocument,
)
def get_page_label_audit_document(
    request: Request, document_id: str
) -> PageLabelAuditDocument:
    service = _service(request)
    try:
        document, records = service.get_document(document_id)
        return _document_response(service, document, records)
    except PageLabelAuditNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/documents/{document_id}/overview",
    response_model=PageLabelAuditDocumentOverview,
)
def get_page_label_audit_document_overview(
    request: Request, document_id: str
) -> PageLabelAuditDocumentOverview:
    service = _service(request)
    try:
        document, audit_by_page = service.get_document_overview(document_id)
        return PageLabelAuditDocumentOverview(
            document_id=document.id,
            original_filename=document.original_filename,
            document_page_count=document.page_count,
            pages=[
                PageLabelAuditOverviewItem(
                    page=AnnotationPageSummary.model_validate(page),
                    image_base64=base64.b64encode(service.read_page_image(page)).decode(
                        "ascii"
                    ),
                    original_label=(
                        audit_by_page[page.id].original_label
                        if page.id in audit_by_page
                        else None
                    ),
                    audited_label=(
                        audit_by_page[page.id].audited_label
                        if page.id in audit_by_page
                        else None
                    ),
                    audit_status=(
                        audit_by_page[page.id].audit_status
                        if page.id in audit_by_page
                        else None
                    ),
                )
                for page in document.pages
            ],
        )
    except PageLabelAuditNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


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
