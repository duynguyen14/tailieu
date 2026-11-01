from __future__ import annotations

import base64
import json
import math
from uuid import uuid4

from sqlalchemy import case, false, func, select, true

from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    OcrLayoutRegion,
    OcrReview,
    OcrReviewRegion,
    OcrReviewToken,
    OcrRun,
    OcrToken,
    utc_now,
)
from app.schemas.ocr_review import (
    OcrReviewDetail,
    OcrReviewHistoryItem,
    OcrReviewHistoryResponse,
    OcrReviewPageInfo,
    OcrReviewRegionView,
    OcrReviewRequest,
    OcrReviewRunInfo,
    OcrReviewStats,
    OcrReviewTokenView,
)
from app.services.storage import AnnotationStorage


class OcrReviewNotFoundError(LookupError):
    pass


class OcrReviewConflictError(RuntimeError):
    pass


class OcrReviewValidationError(ValueError):
    pass


class OcrReviewPersistenceError(RuntimeError):
    """Raised when SQL Server does not persist the submitted review verbatim."""


class OcrReviewService:
    def __init__(self, *, database: Database, storage: AnnotationStorage) -> None:
        self._database = database
        self._storage = storage

    def get_next_review(
        self,
        *,
        search_id: str | None = None,
        sort_order: str = "NEWEST",
    ) -> tuple[OcrReviewDetail | None, OcrReviewStats]:
        with self._database.session() as session:
            stats = self._statistics(session)
            active_review_exists = (
                select(OcrReview.id)
                .where(
                    OcrReview.ocr_run_id == OcrRun.id,
                    OcrReview.is_active == true(),
                )
                .exists()
            )
            statement = select(OcrRun.id).where(
                OcrRun.is_active == true(),
                OcrRun.status == "COMPLETED",
                ~active_review_exists,
            )
            normalized_id = search_id.strip() if search_id else ""
            if normalized_id:
                statement = statement.where(
                    (OcrRun.id == normalized_id) | (OcrRun.page_id == normalized_id)
                )
            order = (
                OcrRun.created_at.desc()
                if sort_order == "NEWEST"
                else OcrRun.created_at.asc()
            )
            run_id = session.scalar(statement.order_by(order, OcrRun.id.asc()).limit(1))
        if run_id is None:
            return None, stats
        return self.get_detail(run_id), stats

    def get_detail(self, run_id: str) -> OcrReviewDetail:
        with self._database.session() as session:
            run = session.get(OcrRun, run_id)
            if run is None:
                raise OcrReviewNotFoundError("OCR run was not found.")
            return self._detail(session, run)

    def get_history(
        self,
        *,
        page: int,
        page_size: int,
        decision: str,
        filename: str | None,
        page_type: str | None,
        reviewed_by: str | None,
    ) -> OcrReviewHistoryResponse:
        filters = [
            OcrReview.is_active == true(),
            OcrRun.is_active == true(),
        ]
        if decision == "APPROVED":
            filters.append(OcrReview.include_in_llm == true())
        elif decision == "SKIPPED":
            filters.append(OcrReview.include_in_llm == false())
        if filename:
            filters.append(
                func.lower(AnnotationDocument.original_filename).like(
                    f"%{filename.strip().lower()}%"
                )
            )
        if page_type:
            filters.append(AnnotationPage.label == page_type)
        if reviewed_by:
            filters.append(
                func.lower(OcrReview.reviewed_by).like(
                    f"%{reviewed_by.strip().lower()}%"
                )
            )

        joins = (
            (OcrRun, OcrRun.id == OcrReview.ocr_run_id),
            (AnnotationPage, AnnotationPage.id == OcrRun.page_id),
            (
                AnnotationDocument,
                AnnotationDocument.id == AnnotationPage.document_id,
            ),
        )
        with self._database.session() as session:
            count_statement = select(func.count(OcrReview.id)).select_from(OcrReview)
            for model, condition in joins:
                count_statement = count_statement.join(model, condition)
            total_items = session.scalar(count_statement.where(*filters)) or 0

            statement = select(
                OcrReview,
                OcrRun,
                AnnotationPage,
                AnnotationDocument,
            ).select_from(OcrReview)
            for model, condition in joins:
                statement = statement.join(model, condition)
            rows = session.execute(
                statement
                .where(*filters)
                .order_by(OcrReview.reviewed_at.desc(), OcrReview.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()

            review_ids = [review.id for review, _, _, _ in rows]
            token_counts: dict[str, tuple[int, int, int]] = {}
            if review_ids:
                count_rows = session.execute(
                    select(
                        OcrReviewToken.review_id,
                        func.count(OcrReviewToken.id),
                        func.sum(
                            case(
                                (OcrReviewToken.include_in_extraction == true(), 1),
                                else_=0,
                            )
                        ),
                        func.sum(
                            case(
                                (OcrReviewToken.source_token_id.is_(None), 1),
                                else_=0,
                            )
                        ),
                    )
                    .where(OcrReviewToken.review_id.in_(review_ids))
                    .group_by(OcrReviewToken.review_id)
                ).all()
                token_counts = {
                    review_id: (int(total), int(included or 0), int(added or 0))
                    for review_id, total, included, added in count_rows
                }

            items = []
            for review, run, annotation_page, document in rows:
                total, included, added = token_counts.get(review.id, (0, 0, 0))
                items.append(
                    OcrReviewHistoryItem(
                        review_id=review.id,
                        run_id=run.id,
                        page_id=annotation_page.id,
                        document_id=document.id,
                        original_filename=document.original_filename,
                        document_page_count=document.page_count,
                        page_number=annotation_page.page_number,
                        page_type=annotation_page.label,
                        decision=(
                            "APPROVED" if review.include_in_llm else "SKIPPED"
                        ),
                        reviewed_by=review.reviewed_by,
                        review_note=review.review_note,
                        reviewed_at=review.reviewed_at,
                        token_count=total,
                        included_token_count=included,
                        ignored_token_count=total - included,
                        human_added_token_count=added,
                    )
                )

        return OcrReviewHistoryResponse(
            items=items,
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=math.ceil(total_items / page_size) if total_items else 0,
        )

    def review(self, *, run_id: str, payload: OcrReviewRequest) -> OcrReviewDetail:
        reviewer = payload.reviewed_by.strip()
        if not reviewer:
            raise OcrReviewValidationError("Reviewer name cannot be empty.")

        with self._database.session() as session:
            run = session.scalar(
                select(OcrRun).where(OcrRun.id == run_id).with_for_update()
            )
            if run is None:
                raise OcrReviewNotFoundError("OCR run was not found.")
            if not run.is_active or run.status != "COMPLETED":
                raise OcrReviewConflictError(
                    "Only an active COMPLETED OCR run can be reviewed."
                )
            page = session.get(AnnotationPage, run.page_id)
            if page is None:
                raise OcrReviewNotFoundError("OCR page was not found.")
            review_width = run.processed_width or page.width
            review_height = run.processed_height or page.height

            source_tokens = {
                token.id: token
                for token in session.scalars(
                    select(OcrToken).where(OcrToken.ocr_run_id == run.id)
                ).all()
            }
            submitted_sources = [
                token.source_token_id
                for token in payload.tokens
                if token.source_token_id is not None
            ]
            if len(submitted_sources) != len(set(submitted_sources)):
                raise OcrReviewValidationError(
                    "A source OCR token can appear only once in a review."
                )
            unknown = sorted(set(submitted_sources) - set(source_tokens))
            if unknown:
                raise OcrReviewValidationError(
                    f"Review contains tokens from another OCR run: {unknown}."
                )
            source_regions = {
                region.id: region
                for region in session.scalars(
                    select(OcrLayoutRegion).where(OcrLayoutRegion.ocr_run_id == run.id)
                ).all()
            }
            submitted_region_sources = [
                region.source_region_id
                for region in payload.exclusion_regions
                if region.source_region_id is not None
            ]
            if len(submitted_region_sources) != len(set(submitted_region_sources)):
                raise OcrReviewValidationError(
                    "A source exclusion region can appear only once in a review."
                )
            unknown_regions = sorted(set(submitted_region_sources) - set(source_regions))
            if unknown_regions:
                raise OcrReviewValidationError(
                    f"Review contains regions from another OCR run: {unknown_regions}."
                )
            for token in payload.tokens:
                text = token.reviewed_text.strip()
                if not text:
                    raise OcrReviewValidationError("Reviewed token text cannot be blank.")
                x1, y1, x2, y2 = token.reviewed_bbox
                if x2 > review_width or y2 > review_height:
                    raise OcrReviewValidationError(
                        f"Token bbox {token.reviewed_bbox} exceeds page dimensions "
                        f"{review_width}x{review_height}."
                    )
            for region in payload.exclusion_regions:
                x1, y1, x2, y2 = region.reviewed_bbox
                if x2 > review_width or y2 > review_height:
                    raise OcrReviewValidationError(
                        f"Region bbox {region.reviewed_bbox} exceeds page dimensions "
                        f"{review_width}x{review_height}."
                    )

            previous = session.scalar(
                select(OcrReview).where(
                    OcrReview.ocr_run_id == run.id,
                    OcrReview.is_active == true(),
                )
            )
            now = utc_now()
            if previous is not None:
                previous.is_active = False
                previous.updated_at = now

            review = OcrReview(
                id=str(uuid4()),
                ocr_run_id=run.id,
                parent_review_id=previous.id if previous else None,
                status="REVIEWED",
                is_active=True,
                include_in_llm=payload.include_in_llm,
                reviewed_by=reviewer,
                review_note=payload.review_note.strip() if payload.review_note else None,
                reviewed_at=now,
            )
            session.add(review)
            session.flush()
            for order, token in enumerate(payload.tokens, start=1):
                x1, y1, x2, y2 = token.reviewed_bbox
                session.add(
                    OcrReviewToken(
                        id=str(uuid4()),
                        review_id=review.id,
                        source_token_id=token.source_token_id,
                        token_order=order,
                        reviewed_text=token.reviewed_text.strip(),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        include_in_extraction=token.include_in_extraction,
                        ignored_reason=token.ignored_reason,
                    )
                )
            for order, region in enumerate(payload.exclusion_regions, start=1):
                x1, y1, x2, y2 = region.reviewed_bbox
                session.add(
                    OcrReviewRegion(
                        id=str(uuid4()),
                        review_id=review.id,
                        source_region_id=region.source_region_id,
                        region_order=order,
                        region_type=region.region_type,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
            session.flush()

            # Do not trust the in-memory ORM values after INSERT.  Expire them
            # and read the rows back through SQL Server so an encoding/binding
            # problem cannot be reported to the frontend as a successful save.
            session.expire_all()
            stored_tokens = list(
                session.scalars(
                    select(OcrReviewToken)
                    .where(OcrReviewToken.review_id == review.id)
                    .order_by(OcrReviewToken.token_order.asc())
                ).all()
            )
            expected_tokens = [
                (
                    token.source_token_id,
                    token.reviewed_text.strip(),
                    tuple(token.reviewed_bbox),
                    token.include_in_extraction,
                    token.ignored_reason,
                )
                for token in payload.tokens
            ]
            actual_tokens = [
                (
                    token.source_token_id,
                    token.reviewed_text,
                    (token.x1, token.y1, token.x2, token.y2),
                    token.include_in_extraction,
                    token.ignored_reason,
                )
                for token in stored_tokens
            ]
            if actual_tokens != expected_tokens:
                raise OcrReviewPersistenceError(
                    "SQL Server returned review data different from the submitted "
                    "payload. The transaction was rolled back to protect Unicode text."
                )
            return self._detail(session, run)

    def _detail(self, session, run: OcrRun) -> OcrReviewDetail:
        page = session.get(AnnotationPage, run.page_id)
        if page is None:
            raise OcrReviewNotFoundError("OCR page was not found.")
        document = session.get(AnnotationDocument, page.document_id)
        if document is None:
            raise OcrReviewNotFoundError("OCR document was not found.")
        raw_tokens = list(
            session.scalars(
                select(OcrToken)
                .where(OcrToken.ocr_run_id == run.id)
                .order_by(OcrToken.token_index.asc())
            ).all()
        )
        raw_by_id = {token.id: token for token in raw_tokens}
        raw_regions = list(
            session.scalars(
                select(OcrLayoutRegion)
                .where(OcrLayoutRegion.ocr_run_id == run.id)
                .order_by(OcrLayoutRegion.region_index.asc())
            ).all()
        )
        raw_region_by_id = {region.id: region for region in raw_regions}
        review = session.scalar(
            select(OcrReview).where(
                OcrReview.ocr_run_id == run.id,
                OcrReview.is_active == true(),
            )
        )

        if review is None:
            tokens = [self._raw_token_view(token) for token in raw_tokens]
            exclusion_regions = [self._raw_region_view(region) for region in raw_regions]
        else:
            reviewed_tokens = list(
                session.scalars(
                    select(OcrReviewToken)
                    .where(OcrReviewToken.review_id == review.id)
                    .order_by(OcrReviewToken.token_order.asc())
                ).all()
            )
            tokens = [
                self._reviewed_token_view(token, raw_by_id.get(token.source_token_id))
                for token in reviewed_tokens
            ]
            reviewed_regions = list(
                session.scalars(
                    select(OcrReviewRegion)
                    .where(OcrReviewRegion.review_id == review.id)
                    .order_by(OcrReviewRegion.region_order.asc())
                ).all()
            )
            exclusion_regions = [
                self._reviewed_region_view(
                    region, raw_region_by_id.get(region.source_region_id)
                )
                for region in reviewed_regions
            ]

        return OcrReviewDetail(
            run=OcrReviewRunInfo(
                id=run.id,
                page_id=run.page_id,
                status=run.status,
                model_name=run.model_name,
                model_version=run.model_version,
                device=run.device,
                processing_time_ms=run.processing_time_ms,
                orientation_angle=run.orientation_angle,
                orientation_confidence=(
                    float(run.orientation_confidence)
                    if run.orientation_confidence is not None
                    else None
                ),
                preprocess_version=run.preprocess_version,
                preprocess_metadata=(
                    json.loads(run.preprocess_metadata_json)
                    if run.preprocess_metadata_json
                    else None
                ),
                created_at=run.created_at,
                review_id=review.id if review else None,
                include_in_llm=review.include_in_llm if review else None,
                reviewed_by=review.reviewed_by if review else None,
                review_note=review.review_note if review else None,
                reviewed_at=review.reviewed_at if review else None,
            ),
            page=OcrReviewPageInfo(
                id=page.id,
                document_id=page.document_id,
                original_filename=document.original_filename,
                document_page_count=document.page_count,
                page_number=page.page_number,
                width=run.processed_width or page.width,
                height=run.processed_height or page.height,
                source_width=page.width,
                source_height=page.height,
                is_preprocessed=run.processed_image_relative_path is not None,
                mime_type=(
                    "image/png" if run.processed_image_relative_path else page.mime_type
                ),
                page_type=page.label,
            ),
            image_base64=base64.b64encode(
                self._storage.read_bytes(
                    run.processed_image_relative_path or page.image_relative_path
                )
            ).decode("ascii"),
            tokens=tokens,
            exclusion_regions=exclusion_regions,
        )

    @staticmethod
    def _raw_token_view(token: OcrToken) -> OcrReviewTokenView:
        bbox = (token.x1, token.y1, token.x2, token.y2)
        return OcrReviewTokenView(
            source_token_id=token.id,
            raw_token_code=token.token_id,
            raw_text=token.text,
            raw_bbox=bbox,
            confidence=float(token.confidence),
            reviewed_text=token.text,
            reviewed_bbox=bbox,
            raw_include_in_extraction=token.include_in_extraction,
            include_in_extraction=token.include_in_extraction,
            ignored_reason=token.ignored_reason,
            is_human_added=False,
            is_changed=False,
        )

    @staticmethod
    def _reviewed_token_view(
        token: OcrReviewToken, source: OcrToken | None
    ) -> OcrReviewTokenView:
        reviewed_bbox = (token.x1, token.y1, token.x2, token.y2)
        raw_bbox = (
            (source.x1, source.y1, source.x2, source.y2) if source is not None else None
        )
        return OcrReviewTokenView(
            source_token_id=token.source_token_id,
            raw_token_code=source.token_id if source else None,
            raw_text=source.text if source else None,
            raw_bbox=raw_bbox,
            confidence=float(source.confidence) if source else None,
            reviewed_text=token.reviewed_text,
            reviewed_bbox=reviewed_bbox,
            raw_include_in_extraction=(source.include_in_extraction if source else None),
            include_in_extraction=token.include_in_extraction,
            ignored_reason=token.ignored_reason,
            is_human_added=source is None,
            is_changed=(
                source is None
                or token.reviewed_text != source.text
                or reviewed_bbox != raw_bbox
                or token.include_in_extraction != source.include_in_extraction
            ),
        )

    @staticmethod
    def _raw_region_view(region: OcrLayoutRegion) -> OcrReviewRegionView:
        bbox = (region.x1, region.y1, region.x2, region.y2)
        return OcrReviewRegionView(
            source_region_id=region.id,
            region_type=region.region_type,
            source="MODEL",
            confidence=float(region.confidence),
            raw_bbox=bbox,
            reviewed_bbox=bbox,
            is_human_added=False,
            is_changed=False,
        )

    @staticmethod
    def _reviewed_region_view(
        region: OcrReviewRegion, source: OcrLayoutRegion | None
    ) -> OcrReviewRegionView:
        reviewed_bbox = (region.x1, region.y1, region.x2, region.y2)
        raw_bbox = (
            (source.x1, source.y1, source.x2, source.y2) if source is not None else None
        )
        return OcrReviewRegionView(
            source_region_id=region.source_region_id,
            region_type=region.region_type,
            source="MODEL" if source is not None else "HUMAN",
            confidence=float(source.confidence) if source is not None else None,
            raw_bbox=raw_bbox,
            reviewed_bbox=reviewed_bbox,
            is_human_added=source is None,
            is_changed=(
                source is None
                or region.region_type != source.region_type
                or reviewed_bbox != raw_bbox
            ),
        )

    @staticmethod
    def _statistics(session) -> OcrReviewStats:
        active_review_exists = (
            select(OcrReview.id)
            .where(
                OcrReview.ocr_run_id == OcrRun.id,
                OcrReview.is_active == true(),
            )
            .exists()
        )
        pending = session.scalar(
            select(func.count(OcrRun.id)).where(
                OcrRun.is_active == true(),
                OcrRun.status == "COMPLETED",
                ~active_review_exists,
            )
        ) or 0
        reviewed = session.scalar(
            select(func.count(OcrReview.id))
            .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
            .where(
                OcrReview.is_active == true(),
                OcrReview.include_in_llm == true(),
                OcrRun.is_active == true(),
            )
        ) or 0
        skipped = session.scalar(
            select(func.count(OcrReview.id))
            .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
            .where(
                OcrReview.is_active == true(),
                OcrReview.include_in_llm == false(),
                OcrRun.is_active == true(),
            )
        ) or 0
        failed = session.scalar(
            select(func.count(OcrRun.id)).where(
                OcrRun.is_active == true(), OcrRun.status == "FAILED"
            )
        ) or 0
        return OcrReviewStats(
            pending=pending,
            reviewed=reviewed,
            skipped=skipped,
            failed=failed,
            total=pending + reviewed + skipped + failed,
        )
