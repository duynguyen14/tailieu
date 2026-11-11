from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, select, true, update
from sqlalchemy.orm import selectinload

from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    VisaFormAnnotation,
    VisaFormFieldAnnotation,
    utc_now,
)
from app.schemas.annotation import AnnotationLabel, ReviewStatus
from app.schemas.visa_form_annotation import (
    VisaFormAnnotationDetail,
    VisaFormAnnotationInfo,
    VisaFormAnnotationSource,
    VisaFormAnnotationStatus,
    VisaFormFieldDefinition,
    VisaFormFieldPayload,
    VisaFormPageInfo,
    VisaFormReviewRequest,
    VisaFormReviewStats,
)
from app.services.storage import AnnotationStorage

DEFAULT_TEMPLATE_VERSION = "visa-application-form-v1"


@dataclass(frozen=True)
class TemplateField:
    code: str
    label: str
    normalized_bbox: tuple[int, int, int, int]


# Coordinates are in a 0..1000 space and describe value areas, not captions.
# They intentionally start slightly wide; the human review screen is the source
# of truth used later to export a YOLO training set.
VISA_FORM_TEMPLATE_FIELDS: tuple[TemplateField, ...] = (
    TemplateField("full_name", "Họ và tên", (145, 176, 390, 200)),
    TemplateField("date_of_birth", "Ngày sinh", (160, 211, 350, 235)),
    TemplateField("place_of_birth", "Nơi sinh", (160, 247, 390, 271)),
    TemplateField("gender", "Giới tính", (135, 283, 300, 307)),
    TemplateField("nationality", "Quốc tịch", (160, 318, 350, 343)),
    TemplateField("contact_address", "Địa chỉ liên hệ", (75, 375, 440, 420)),
    TemplateField("contact_phone", "Điện thoại liên hệ", (145, 420, 355, 445)),
    TemplateField("contact_email", "Email liên hệ", (95, 453, 410, 480)),
    TemplateField("sponsor_name", "Tên người/cơ quan mời", (125, 500, 420, 532)),
    TemplateField("sponsor_phone", "Điện thoại người/cơ quan mời", (135, 538, 365, 566)),
    TemplateField("passport_number", "Số hộ chiếu", (625, 211, 870, 236)),
    TemplateField("passport_expiration", "Hạn hộ chiếu", (635, 247, 875, 272)),
    TemplateField("number_of_entries", "Số lần nhập cảnh", (635, 283, 875, 308)),
    TemplateField("entry_date", "Ngày dự kiến nhập cảnh", (650, 319, 885, 344)),
    TemplateField("exit_date", "Ngày dự kiến xuất cảnh", (650, 354, 885, 379)),
    TemplateField("submission_place", "Nơi nộp hồ sơ", (515, 405, 890, 440)),
    TemplateField("collection_place", "Nơi nhận thị thực", (515, 474, 890, 508)),
    TemplateField("purpose_of_entry", "Mục đích nhập cảnh", (625, 520, 975, 565)),
    TemplateField("application_date", "Ngày làm đơn", (650, 690, 900, 725)),
)

FIELD_CODES = tuple(item.code for item in VISA_FORM_TEMPLATE_FIELDS)


class VisaFormAnnotationNotFoundError(LookupError):
    pass


class VisaFormAnnotationConflictError(RuntimeError):
    pass


class VisaFormAnnotationValidationError(ValueError):
    pass


def normalized_to_pixel(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, round(x1 * width / 1000))),
        max(0, min(height - 1, round(y1 * height / 1000))),
        max(1, min(width, round(x2 * width / 1000))),
        max(1, min(height, round(y2 * height / 1000))),
    )


def pixel_to_normalized(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        round(x1 * 1000 / width),
        round(y1 * 1000 / height),
        round(x2 * 1000 / width),
        round(y2 * 1000 / height),
    )


class VisaFormAnnotationService:
    def __init__(self, *, database: Database, storage: AnnotationStorage) -> None:
        self._database = database
        self._storage = storage

    @staticmethod
    def field_definitions() -> list[VisaFormFieldDefinition]:
        return [
            VisaFormFieldDefinition(code=item.code, label=item.label, order=index)
            for index, item in enumerate(VISA_FORM_TEMPLATE_FIELDS, start=1)
        ]

    def eligible_pages(
        self,
        *,
        limit: int | None = None,
        include_unreviewed: bool = False,
        page_id: str | None = None,
        replace_existing: bool = False,
    ) -> list[AnnotationPage]:
        with self._database.session() as session:
            statement = select(AnnotationPage).where(
                AnnotationPage.label == AnnotationLabel.VISA_APPLICATION_FORM.value
            )
            if not include_unreviewed:
                statement = statement.where(
                    AnnotationPage.review_status == ReviewStatus.REVIEWED.value
                )
            if page_id:
                statement = statement.where(AnnotationPage.id == page_id)
            if not replace_existing:
                active_exists = (
                    select(VisaFormAnnotation.id)
                    .where(
                        VisaFormAnnotation.page_id == AnnotationPage.id,
                        VisaFormAnnotation.is_active == true(),
                    )
                    .exists()
                )
                statement = statement.where(~active_exists)
            statement = statement.order_by(
                AnnotationPage.created_at.desc(), AnnotationPage.page_index.asc()
            )
            if limit is not None:
                statement = statement.limit(limit)
            return list(session.scalars(statement).all())

    def create_template_annotation(
        self,
        *,
        page_id: str,
        template_version: str = DEFAULT_TEMPLATE_VERSION,
        replace_existing: bool = False,
    ) -> str:
        with self._database.session() as session:
            page = session.get(AnnotationPage, page_id)
            if page is None:
                raise VisaFormAnnotationNotFoundError("Annotation page was not found.")
            if page.label != AnnotationLabel.VISA_APPLICATION_FORM.value:
                raise VisaFormAnnotationConflictError(
                    "Only VISA_APPLICATION_FORM pages can receive this template."
                )
            active = session.scalar(
                select(VisaFormAnnotation).where(
                    VisaFormAnnotation.page_id == page.id,
                    VisaFormAnnotation.is_active == true(),
                )
            )
            if active is not None and not replace_existing:
                raise VisaFormAnnotationConflictError(
                    "An active visa-form annotation already exists for this page."
                )
            if active is not None:
                active.is_active = False
                active.updated_at = utc_now()

            annotation = VisaFormAnnotation(
                id=str(uuid4()),
                page_id=page.id,
                parent_annotation_id=active.id if active else None,
                source=VisaFormAnnotationSource.TEMPLATE.value,
                status=VisaFormAnnotationStatus.PENDING.value,
                template_version=template_version,
                is_active=True,
                image_width=page.width,
                image_height=page.height,
            )
            session.add(annotation)
            session.flush()
            for order, item in enumerate(VISA_FORM_TEMPLATE_FIELDS, start=1):
                x1, y1, x2, y2 = normalized_to_pixel(
                    item.normalized_bbox, page.width, page.height
                )
                annotation.fields.append(
                    VisaFormFieldAnnotation(
                        id=str(uuid4()),
                        field_code=item.code,
                        field_order=order,
                        value_text=None,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        nx1=item.normalized_bbox[0],
                        ny1=item.normalized_bbox[1],
                        nx2=item.normalized_bbox[2],
                        ny2=item.normalized_bbox[3],
                    )
                )
            session.flush()
            return annotation.id

    def get_detail(self, annotation_id: str) -> VisaFormAnnotationDetail:
        with self._database.session() as session:
            annotation = self._load_annotation(session, annotation_id)
            return self._detail_from_model(session, annotation)

    def get_next_review(
        self,
    ) -> tuple[VisaFormAnnotationDetail | None, VisaFormReviewStats]:
        with self._database.session() as session:
            annotation = session.scalar(
                select(VisaFormAnnotation)
                .options(
                    selectinload(VisaFormAnnotation.page),
                    selectinload(VisaFormAnnotation.fields),
                )
                .where(
                    VisaFormAnnotation.is_active == true(),
                    VisaFormAnnotation.status == VisaFormAnnotationStatus.PENDING.value,
                )
                .order_by(
                    VisaFormAnnotation.created_at.desc(), VisaFormAnnotation.id.desc()
                )
                .limit(1)
            )
            stats = self._stats(session)
            return (
                self._detail_from_model(session, annotation) if annotation else None,
                stats,
            )

    def review(
        self, *, annotation_id: str, payload: VisaFormReviewRequest
    ) -> VisaFormAnnotationDetail:
        with self._database.session() as session:
            current = self._load_annotation(session, annotation_id, lock=True)
            if not current.is_active or current.status != VisaFormAnnotationStatus.PENDING.value:
                raise VisaFormAnnotationConflictError(
                    "Only an active PENDING annotation can be reviewed."
                )
            page = session.get(AnnotationPage, current.page_id)
            if page is None:
                raise VisaFormAnnotationNotFoundError("Annotation page was not found.")
            self._validate_fields(payload.fields, page.width, page.height)

            session.execute(
                update(VisaFormAnnotation)
                .where(
                    VisaFormAnnotation.page_id == current.page_id,
                    VisaFormAnnotation.is_active == true(),
                )
                .values(is_active=False, updated_at=utc_now())
            )
            reviewed = VisaFormAnnotation(
                id=str(uuid4()),
                page_id=current.page_id,
                parent_annotation_id=current.id,
                source=VisaFormAnnotationSource.HUMAN.value,
                status=payload.decision.value,
                template_version=current.template_version,
                is_active=True,
                image_width=page.width,
                image_height=page.height,
                reviewed_by=payload.reviewed_by.strip(),
                review_note=payload.review_note.strip() if payload.review_note else None,
                reviewed_at=utc_now(),
            )
            session.add(reviewed)
            session.flush()
            for item in sorted(payload.fields, key=lambda field: field.field_order):
                nx1, ny1, nx2, ny2 = pixel_to_normalized(
                    item.bbox, page.width, page.height
                )
                reviewed.fields.append(
                    VisaFormFieldAnnotation(
                        id=str(uuid4()),
                        field_code=item.field_code,
                        field_order=item.field_order,
                        value_text=item.value_text.strip() if item.value_text else None,
                        x1=item.bbox[0],
                        y1=item.bbox[1],
                        x2=item.bbox[2],
                        y2=item.bbox[3],
                        nx1=nx1,
                        ny1=ny1,
                        nx2=nx2,
                        ny2=ny2,
                    )
                )
            session.flush()
            return self._detail_from_model(session, reviewed)

    def _load_annotation(self, session, annotation_id: str, *, lock: bool = False):
        statement = (
            select(VisaFormAnnotation)
            .options(
                selectinload(VisaFormAnnotation.page),
                selectinload(VisaFormAnnotation.fields),
            )
            .where(VisaFormAnnotation.id == annotation_id)
        )
        if lock:
            statement = statement.with_for_update()
        annotation = session.scalar(statement)
        if annotation is None:
            raise VisaFormAnnotationNotFoundError("Visa-form annotation was not found.")
        return annotation

    def _detail_from_model(
        self, session, annotation: VisaFormAnnotation
    ) -> VisaFormAnnotationDetail:
        page = annotation.page
        document = session.get(AnnotationDocument, page.document_id)
        if document is None:
            raise VisaFormAnnotationNotFoundError("Annotation document was not found.")
        return VisaFormAnnotationDetail(
            annotation=VisaFormAnnotationInfo(
                id=annotation.id,
                page_id=annotation.page_id,
                parent_annotation_id=annotation.parent_annotation_id,
                source=annotation.source,
                status=annotation.status,
                template_version=annotation.template_version,
                is_active=annotation.is_active,
                reviewed_by=annotation.reviewed_by,
                review_note=annotation.review_note,
                reviewed_at=annotation.reviewed_at,
                created_at=annotation.created_at,
                updated_at=annotation.updated_at,
            ),
            page=VisaFormPageInfo(
                id=page.id,
                document_id=page.document_id,
                original_filename=document.original_filename,
                document_page_count=document.page_count,
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                mime_type=page.mime_type,
            ),
            image_base64=base64.b64encode(
                self._storage.read_bytes(page.image_relative_path)
            ).decode("ascii"),
            fields=[self._field_payload(field) for field in annotation.fields],
        )

    @staticmethod
    def _field_payload(field: VisaFormFieldAnnotation) -> VisaFormFieldPayload:
        return VisaFormFieldPayload(
            field_code=field.field_code,
            field_order=field.field_order,
            value_text=field.value_text,
            bbox=(field.x1, field.y1, field.x2, field.y2),
            normalized_bbox=(field.nx1, field.ny1, field.nx2, field.ny2),
        )

    @staticmethod
    def _validate_fields(
        fields: Sequence[VisaFormFieldPayload], width: int, height: int
    ) -> None:
        codes = [field.field_code for field in fields]
        orders = [field.field_order for field in fields]
        if len(fields) != len(FIELD_CODES) or set(codes) != set(FIELD_CODES):
            missing = sorted(set(FIELD_CODES) - set(codes))
            extra = sorted(set(codes) - set(FIELD_CODES))
            raise VisaFormAnnotationValidationError(
                f"Exactly 19 field codes are required; missing={missing}, extra={extra}."
            )
        if len(codes) != len(set(codes)) or len(orders) != len(set(orders)):
            raise VisaFormAnnotationValidationError(
                "field_code and field_order values must be unique."
            )
        expected_order = {code: index for index, code in enumerate(FIELD_CODES, start=1)}
        for field in fields:
            if field.field_order != expected_order[field.field_code]:
                raise VisaFormAnnotationValidationError(
                    f"Invalid field_order for {field.field_code}."
                )
            x1, y1, x2, y2 = field.bbox
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
                raise VisaFormAnnotationValidationError(
                    f"BBox for {field.field_code} exceeds page dimensions."
                )

    @staticmethod
    def _stats(session) -> VisaFormReviewStats:
        rows = dict(
            session.execute(
                select(VisaFormAnnotation.status, func.count(VisaFormAnnotation.id))
                .where(VisaFormAnnotation.is_active == true())
                .group_by(VisaFormAnnotation.status)
            ).all()
        )
        pending = int(rows.get(VisaFormAnnotationStatus.PENDING.value, 0))
        reviewed = int(rows.get(VisaFormAnnotationStatus.REVIEWED.value, 0))
        rejected = int(rows.get(VisaFormAnnotationStatus.REJECTED.value, 0))
        return VisaFormReviewStats(
            pending=pending,
            reviewed=reviewed,
            rejected=rejected,
            total=pending + reviewed + rejected,
        )
