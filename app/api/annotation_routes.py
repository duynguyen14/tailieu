from __future__ import annotations

import base64
import json
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status

from app.db.models import AnnotationDocument, AnnotationPage
from app.schemas.annotation import (
    AnnotationDocumentDetail,
    AnnotationDocumentList,
    AnnotationDocumentSummary,
    AnnotationImportResponse,
    AnnotationPageImageResponse,
    AnnotationPageSummary,
    AnnotationReviewQueueResponse,
    AnnotationReviewStats,
    LabelDefinition,
    PageReviewRequest,
    PageReviewResponse,
    PagePrediction,
    PagePredictionItem,
    ReviewStatus,
)
from app.services.annotation import AnnotationNotFoundError, AnnotationService
from app.services.exceptions import DocumentValidationError

router = APIRouter(prefix="/annotation", tags=["annotation"])


LABELS = [
    LabelDefinition(
        code="DIPLOMATIC_NOTE",
        display_name="Công hàm",
        description="Trang công hàm ngoại giao hoặc công hàm lãnh sự.",
    ),
    LabelDefinition(
        code="APPLICANT_TABLE",
        display_name="Bảng người nộp",
        description="Thông tin người nộp được trình bày theo hàng và cột.",
    ),
    LabelDefinition(
        code="APPLICANT_LIST",
        display_name="Danh sách người nộp",
        description="Thông tin người nộp được trình bày dạng danh sách.",
    ),
    LabelDefinition(
        code="VISA_APPLICATION_FORM",
        display_name="Tờ khai đề nghị cấp thị thực",
        description="Visa application form hoặc tờ khai đề nghị cấp thị thực.",
    ),
    LabelDefinition(
        code="TEMPORARY_RESIDENCE_EXTENSION_FORM",
        display_name="Phiếu trình gia hạn tạm trú",
        description="Phiếu trình, đơn hoặc biểu mẫu đề nghị gia hạn tạm trú.",
    ),
    LabelDefinition(
        code="PASSPORT_FRONT",
        display_name="Hộ chiếu mặt trước",
        description="Trang hoặc mặt trước của hộ chiếu.",
    ),
    LabelDefinition(
        code="VISA_STICKER",
        display_name="Thị thực dán trên hộ chiếu",
        description="Trang hộ chiếu có tem hoặc nhãn thị thực được dán.",
    ),
    LabelDefinition(
        code="VISA_REQUEST_OFFICIAL_LETTER",
        display_name="Công văn đề nghị cấp thị thực",
        description="Công văn của cơ quan nhà nước đề nghị cấp thị thực hoặc hỗ trợ nhập cảnh.",
    ),
    LabelDefinition(
        code="OTHER",
        display_name="Tài liệu khác",
        description="Trang không thuộc các nhóm tài liệu đã định nghĩa.",
    ),
]


def _service(request: Request) -> AnnotationService:
    return request.app.state.annotation_service


def _page_summary(page: AnnotationPage) -> AnnotationPageSummary:
    return AnnotationPageSummary.model_validate(page)


def _document_summary(document: AnnotationDocument) -> AnnotationDocumentSummary:
    reviewed_pages = sum(
        page.review_status == ReviewStatus.REVIEWED.value for page in document.pages
    )
    return AnnotationDocumentSummary(
        id=document.id,
        original_filename=document.original_filename,
        sha256=document.sha256,
        page_count=document.page_count,
        reviewed_pages=reviewed_pages,
        review_status=document.review_status,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _document_detail(document: AnnotationDocument) -> AnnotationDocumentDetail:
    summary = _document_summary(document)
    return AnnotationDocumentDetail(
        **summary.model_dump(),
        pages=[_page_summary(page) for page in document.pages],
    )


def _page_image_response(
    service: AnnotationService,
    document: AnnotationDocument,
    page: AnnotationPage,
) -> AnnotationPageImageResponse:
    image_base64 = base64.b64encode(service.read_page_image(page)).decode("ascii")
    return AnnotationPageImageResponse(
        document_id=document.id,
        original_filename=document.original_filename,
        document_page_count=document.page_count,
        page=_page_summary(page),
        image_base64=image_base64,
        prediction=_page_prediction(page),
    )


def _page_prediction(page: AnnotationPage) -> PagePrediction | None:
    if not (
        page.model_predicted_label
        and page.model_prediction_confidence is not None
        and page.model_prediction_version
        and page.model_predicted_at
    ):
        return None
    try:
        raw_top_predictions = json.loads(page.model_prediction_topk_json or "[]")
        top_predictions = [
            PagePredictionItem(
                label=item["label"],
                confidence=float(item["confidence"]),
            )
            for item in raw_top_predictions
            if isinstance(item, dict) and "label" in item and "confidence" in item
        ]
        return PagePrediction(
            label=page.model_predicted_label,
            confidence=float(page.model_prediction_confidence),
            model_version=page.model_prediction_version,
            predicted_at=page.model_predicted_at,
            top_predictions=top_predictions,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


@router.get("/labels", response_model=list[LabelDefinition])
def list_annotation_labels() -> list[LabelDefinition]:
    return LABELS


@router.post(
    "/documents",
    response_model=AnnotationImportResponse,
    status_code=status.HTTP_201_CREATED,
)
def import_annotation_document(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF to split into review pages")],
) -> AnnotationImportResponse:
    try:
        created, document = _service(request).import_pdf(
            source=file.file,
            original_filename=file.filename or "document.pdf",
        )
        if not created:
            response.status_code = status.HTTP_200_OK
        return AnnotationImportResponse(
            created=created,
            document=_document_detail(document),
        )
    except DocumentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    finally:
        file.file.close()


@router.get("/documents", response_model=AnnotationDocumentList)
def list_annotation_documents(
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    review_status: ReviewStatus | None = None,
) -> AnnotationDocumentList:
    documents, total = _service(request).list_documents(
        offset=offset,
        limit=limit,
        review_status=review_status,
    )
    return AnnotationDocumentList(
        items=[_document_summary(document) for document in documents],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/documents/{document_id}", response_model=AnnotationDocumentDetail)
def get_annotation_document(request: Request, document_id: str) -> AnnotationDocumentDetail:
    try:
        document = _service(request).get_document(document_id)
        return _document_detail(document)
    except AnnotationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/documents/{document_id}/pages/{page_number}",
    response_model=AnnotationPageImageResponse,
)
def get_annotation_page(
    request: Request,
    document_id: str,
    page_number: int,
) -> AnnotationPageImageResponse:
    try:
        service = _service(request)
        document, page = service.get_page(document_id, page_number)
        return _page_image_response(service, document, page)
    except AnnotationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/review/next", response_model=AnnotationReviewQueueResponse)
def get_next_annotation_review_page(request: Request) -> AnnotationReviewQueueResponse:
    service = _service(request)
    document, page, pending, reviewed, total = service.get_next_unreviewed_page()
    current = None
    if document is not None and page is not None:
        current = _page_image_response(service, document, page)
    return AnnotationReviewQueueResponse(
        current=current,
        stats=AnnotationReviewStats(
            pending=pending,
            reviewed=reviewed,
            total=total,
        ),
    )


@router.put(
    "/documents/{document_id}/pages/{page_number}/review",
    response_model=PageReviewResponse,
)
def review_annotation_page(
    request: Request,
    document_id: str,
    page_number: int,
    payload: PageReviewRequest,
) -> PageReviewResponse:
    try:
        document_status, page = _service(request).review_page(
            document_id=document_id,
            page_number=page_number,
            label=payload.label,
            reviewed_by=payload.reviewed_by,
            review_note=payload.review_note,
        )
        return PageReviewResponse(
            document_id=document_id,
            document_review_status=document_status,
            page=_page_summary(page),
        )
    except AnnotationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
