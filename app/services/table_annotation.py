from __future__ import annotations

import base64
from collections.abc import Sequence
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import func, select, true, update
from sqlalchemy.orm import selectinload

from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    TableAnnotation,
    TableRegion,
    TableStructureElement,
    utc_now,
)
from app.schemas.annotation import AnnotationLabel, ReviewStatus
from app.schemas.table_annotation import (
    TableAnnotationDetail,
    TableAnnotationInfo,
    TableAnnotationSource,
    TableAnnotationStatus,
    TableElementPayload,
    TablePageInfo,
    TableRegionPayload,
    TableReviewRequest,
    TableReviewStats,
)
from app.services.storage import AnnotationStorage


class TableAnnotationNotFoundError(LookupError):
    pass


class TableAnnotationConflictError(RuntimeError):
    pass


class TableAnnotationValidationError(ValueError):
    pass


class TableAnnotationService:
    def __init__(self, *, database: Database, storage: AnnotationStorage) -> None:
        self._database = database
        self._storage = storage

    def eligible_pages(
        self,
        *,
        limit: int,
        page_id: str | None = None,
        force: bool = False,
    ) -> list[AnnotationPage]:
        with self._database.session() as session:
            statement = (
                select(AnnotationPage)
                .where(
                    AnnotationPage.review_status == ReviewStatus.REVIEWED.value,
                    AnnotationPage.label == AnnotationLabel.APPLICANT_TABLE.value,
                )
                .order_by(AnnotationPage.created_at.asc(), AnnotationPage.page_index.asc())
                .limit(limit)
            )
            if page_id:
                statement = statement.where(AnnotationPage.id == page_id)
            active_human_exists = (
                select(TableAnnotation.id)
                .where(
                    TableAnnotation.page_id == AnnotationPage.id,
                    TableAnnotation.source == TableAnnotationSource.HUMAN.value,
                    TableAnnotation.is_active == true(),
                )
                .exists()
            )
            statement = statement.where(~active_human_exists)
            if not force:
                active_annotation_exists = (
                    select(TableAnnotation.id)
                    .where(
                        TableAnnotation.page_id == AnnotationPage.id,
                        TableAnnotation.is_active == true(),
                    )
                    .exists()
                )
                statement = statement.where(~active_annotation_exists)
            return list(session.scalars(statement).all())

    def read_page_image(self, page: AnnotationPage) -> bytes:
        return self._storage.read_bytes(page.image_relative_path)

    def start_model_annotation(
        self,
        *,
        page_id: str,
        model_name: str,
        model_version: str,
        force: bool,
    ) -> str:
        annotation_id = str(uuid4())
        with self._database.session() as session:
            page = session.get(AnnotationPage, page_id)
            if page is None:
                raise TableAnnotationNotFoundError("Annotation page was not found.")
            if (
                page.review_status != ReviewStatus.REVIEWED.value
                or page.label != AnnotationLabel.APPLICANT_TABLE.value
            ):
                raise TableAnnotationConflictError(
                    "Only reviewed APPLICANT_TABLE pages can be processed."
                )

            active = session.scalar(
                select(TableAnnotation).where(
                    TableAnnotation.page_id == page_id,
                    TableAnnotation.is_active == true(),
                )
            )
            if active is not None:
                if active.source == TableAnnotationSource.HUMAN.value:
                    raise TableAnnotationConflictError(
                        "A human-reviewed table annotation already exists for this page."
                    )
                if not force:
                    raise TableAnnotationConflictError(
                        "An active model prediction already exists for this page."
                    )
                active.is_active = False
                active.updated_at = utc_now()
                session.flush()

            session.add(
                TableAnnotation(
                    id=annotation_id,
                    page_id=page_id,
                    source=TableAnnotationSource.MODEL.value,
                    status=TableAnnotationStatus.PROCESSING.value,
                    is_active=True,
                    model_name=model_name,
                    model_version=model_version,
                )
            )
        return annotation_id

    def complete_model_annotation(
        self,
        *,
        annotation_id: str,
        tables: Sequence[TableRegionPayload],
        processing_time_ms: int,
    ) -> None:
        with self._database.session() as session:
            annotation = session.get(TableAnnotation, annotation_id)
            if annotation is None:
                raise TableAnnotationNotFoundError("Table annotation was not found.")
            page = session.get(AnnotationPage, annotation.page_id)
            if page is None:
                raise TableAnnotationNotFoundError("Annotation page was not found.")
            self._validate_tables(tables, page.width, page.height)
            self._replace_regions(session, annotation, tables)
            annotation.status = TableAnnotationStatus.PREDICTED.value
            annotation.processing_time_ms = processing_time_ms
            annotation.error_message = None
            annotation.updated_at = utc_now()

    def fail_model_annotation(
        self, *, annotation_id: str, error_message: str, processing_time_ms: int
    ) -> None:
        with self._database.session() as session:
            annotation = session.get(TableAnnotation, annotation_id)
            if annotation is None:
                return
            annotation.status = TableAnnotationStatus.FAILED.value
            annotation.processing_time_ms = processing_time_ms
            annotation.error_message = error_message[:4000]
            annotation.updated_at = utc_now()

    def get_detail(self, annotation_id: str) -> TableAnnotationDetail:
        with self._database.session() as session:
            annotation = self._load_annotation(session, annotation_id)
            return self._detail_from_model(session, annotation)

    def get_next_review(self) -> tuple[TableAnnotationDetail | None, TableReviewStats]:
        with self._database.session() as session:
            pending_filter = (
                TableAnnotation.source == TableAnnotationSource.MODEL.value,
                TableAnnotation.status == TableAnnotationStatus.PREDICTED.value,
                TableAnnotation.is_active == true(),
            )
            pending = session.scalar(
                select(func.count(TableAnnotation.id)).where(*pending_filter)
            ) or 0
            reviewed = session.scalar(
                select(func.count(TableAnnotation.id)).where(
                    TableAnnotation.source == TableAnnotationSource.HUMAN.value,
                    TableAnnotation.status == TableAnnotationStatus.REVIEWED.value,
                    TableAnnotation.is_active == true(),
                )
            ) or 0
            failed = session.scalar(
                select(func.count(TableAnnotation.id)).where(
                    TableAnnotation.status == TableAnnotationStatus.FAILED.value,
                    TableAnnotation.is_active == true(),
                )
            ) or 0
            annotation = session.scalar(
                select(TableAnnotation)
                .options(
                    selectinload(TableAnnotation.page),
                    selectinload(TableAnnotation.regions).selectinload(TableRegion.elements),
                )
                .where(*pending_filter)
                .order_by(TableAnnotation.created_at.asc())
                .limit(1)
            )
            stats = TableReviewStats(
                pending=pending,
                reviewed=reviewed,
                failed=failed,
                total=pending + reviewed + failed,
            )
            if annotation is None:
                return None, stats
            return self._detail_from_model(session, annotation), stats

    def review(
        self, *, annotation_id: str, payload: TableReviewRequest
    ) -> TableAnnotationDetail:
        with self._database.session() as session:
            current = self._load_annotation(session, annotation_id, lock=True)
            if not current.is_active:
                raise TableAnnotationConflictError("This annotation is no longer active.")
            if current.status not in {
                TableAnnotationStatus.PREDICTED.value,
                TableAnnotationStatus.REVIEWED.value,
            }:
                raise TableAnnotationConflictError(
                    "Only predicted or reviewed annotations can be reviewed."
                )
            page = session.get(AnnotationPage, current.page_id)
            if page is None:
                raise TableAnnotationNotFoundError("Annotation page was not found.")
            self._validate_tables(payload.tables, page.width, page.height)

            session.execute(
                update(TableAnnotation)
                .where(
                    TableAnnotation.page_id == current.page_id,
                    TableAnnotation.is_active == true(),
                )
                .values(is_active=False, updated_at=utc_now())
            )
            session.flush()

            reviewed = TableAnnotation(
                id=str(uuid4()),
                page_id=current.page_id,
                parent_annotation_id=current.id,
                source=TableAnnotationSource.HUMAN.value,
                status=TableAnnotationStatus.REVIEWED.value,
                is_active=True,
                model_name=current.model_name,
                model_version=current.model_version,
                reviewed_by=payload.reviewed_by.strip(),
                review_note=payload.review_note.strip() if payload.review_note else None,
                reviewed_at=utc_now(),
            )
            session.add(reviewed)
            session.flush()
            self._replace_regions(session, reviewed, payload.tables)
            session.flush()
            return self._detail_from_model(session, reviewed)

    def _load_annotation(self, session, annotation_id: str, *, lock: bool = False):
        statement = (
            select(TableAnnotation)
            .options(
                selectinload(TableAnnotation.page),
                selectinload(TableAnnotation.regions).selectinload(TableRegion.elements),
            )
            .where(TableAnnotation.id == annotation_id)
        )
        if lock:
            statement = statement.with_for_update()
        annotation = session.scalar(statement)
        if annotation is None:
            raise TableAnnotationNotFoundError("Table annotation was not found.")
        return annotation

    def _detail_from_model(self, session, annotation: TableAnnotation) -> TableAnnotationDetail:
        page = annotation.page
        document = session.get(AnnotationDocument, page.document_id)
        if document is None:
            raise TableAnnotationNotFoundError("Annotation document was not found.")
        image_base64 = base64.b64encode(self.read_page_image(page)).decode("ascii")
        tables = [self._region_payload(region) for region in annotation.regions]
        return TableAnnotationDetail(
            annotation=TableAnnotationInfo(
                id=annotation.id,
                page_id=annotation.page_id,
                parent_annotation_id=annotation.parent_annotation_id,
                source=annotation.source,
                status=annotation.status,
                is_active=annotation.is_active,
                model_name=annotation.model_name,
                model_version=annotation.model_version,
                processing_time_ms=annotation.processing_time_ms,
                error_message=annotation.error_message,
                reviewed_by=annotation.reviewed_by,
                review_note=annotation.review_note,
                reviewed_at=annotation.reviewed_at,
                created_at=annotation.created_at,
                updated_at=annotation.updated_at,
            ),
            page=TablePageInfo(
                id=page.id,
                document_id=page.document_id,
                original_filename=document.original_filename,
                document_page_count=document.page_count,
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                mime_type=page.mime_type,
            ),
            image_base64=image_base64,
            tables=tables,
        )

    @staticmethod
    def _region_payload(region: TableRegion) -> TableRegionPayload:
        return TableRegionPayload(
            table_index=region.table_index,
            bbox=(region.x1, region.y1, region.x2, region.y2),
            confidence=float(region.confidence) if region.confidence is not None else None,
            elements=[
                TableElementPayload(
                    element_type=element.element_type,
                    row_index=element.row_index,
                    column_index=element.column_index,
                    row_span=element.row_span,
                    column_span=element.column_span,
                    bbox=(element.x1, element.y1, element.x2, element.y2),
                    confidence=(
                        float(element.confidence) if element.confidence is not None else None
                    ),
                )
                for element in region.elements
            ],
        )

    @staticmethod
    def _replace_regions(session, annotation, tables: Sequence[TableRegionPayload]) -> None:
        annotation.regions.clear()
        session.flush()
        for table in tables:
            region = TableRegion(
                id=str(uuid4()),
                annotation_id=annotation.id,
                table_index=table.table_index,
                x1=table.bbox[0],
                y1=table.bbox[1],
                x2=table.bbox[2],
                y2=table.bbox[3],
                confidence=(Decimal(str(table.confidence)) if table.confidence is not None else None),
            )
            for item in table.elements:
                region.elements.append(
                    TableStructureElement(
                        id=str(uuid4()),
                        element_type=item.element_type.value,
                        row_index=item.row_index,
                        column_index=item.column_index,
                        row_span=item.row_span,
                        column_span=item.column_span,
                        x1=item.bbox[0],
                        y1=item.bbox[1],
                        x2=item.bbox[2],
                        y2=item.bbox[3],
                        confidence=(
                            Decimal(str(item.confidence))
                            if item.confidence is not None
                            else None
                        ),
                    )
                )
            annotation.regions.append(region)

    @staticmethod
    def _validate_tables(
        tables: Sequence[TableRegionPayload], page_width: int, page_height: int
    ) -> None:
        indexes = [table.table_index for table in tables]
        if len(indexes) != len(set(indexes)):
            raise TableAnnotationValidationError("table_index values must be unique.")
        for table in tables:
            tx1, ty1, tx2, ty2 = table.bbox
            if tx2 > page_width or ty2 > page_height:
                raise TableAnnotationValidationError("Table bbox exceeds the page dimensions.")
            for element in table.elements:
                x1, y1, x2, y2 = element.bbox
                if x1 < tx1 or y1 < ty1 or x2 > tx2 or y2 > ty2:
                    raise TableAnnotationValidationError(
                        "A structure element bbox lies outside its table bbox."
                    )
