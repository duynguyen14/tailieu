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
    OcrBboxTextPreview,
    OcrBboxTextPreviewLine,
    OcrBboxTextPreviewToken,
    OcrReviewDetail,
    OcrReviewDocumentDetail,
    OcrReviewDocumentListResponse,
    OcrReviewDocumentSummary,
    OcrReviewHistoryItem,
    OcrReviewHistoryResponse,
    OcrReviewInclusionRequest,
    OcrReviewPageInfo,
    OcrReviewRegionView,
    OcrReviewRequest,
    OcrReviewRunInfo,
    OcrReviewStats,
    OcrReviewTokenView,
)
from app.services.storage import AnnotationStorage
from app.services.text_reconstruction import ReconstructionToken, reconstruct_bbox_lines


class OcrReviewNotFoundError(LookupError):
    pass


class OcrReviewConflictError(RuntimeError):
    pass


class OcrReviewValidationError(ValueError):
    pass


class OcrReviewPersistenceError(RuntimeError):
    """Raised when SQL Server does not persist the submitted review verbatim."""


class OcrReviewService:
    def __init__(
        self,
        *,
        database: Database,
        storage: AnnotationStorage,
        line_tolerance_ratio: float = 0.55,
    ) -> None:
        self._database = database
        self._storage = storage
        self._line_tolerance_ratio = line_tolerance_ratio

    def get_next_review(
        self,
        *,
        search_id: str | None = None,
        sort_order: str = "NEWEST",
        from_index: int | None = None,
        to_index: int | None = None,
    ) -> tuple[OcrReviewDetail | None, OcrReviewStats, int | None]:
        with self._database.session() as session:
            stats = self._statistics(session)
            current_index: int | None = None
            active_review_exists = (
                select(OcrReview.id)
                .where(
                    OcrReview.ocr_run_id == OcrRun.id,
                    OcrReview.is_active == true(),
                )
                .exists()
            )
            completed_statement = select(OcrRun.id).where(
                OcrRun.is_active == true(),
                OcrRun.status == "COMPLETED",
            )
            normalized_id = search_id.strip() if search_id else ""
            order = (
                OcrRun.created_at.desc()
                if sort_order == "NEWEST"
                else OcrRun.created_at.asc()
            )
            if normalized_id:
                # Exact ID lookup intentionally ignores the numeric range so a
                # reviewer can inspect a pending run without changing their
                # current work interval.
                statement = completed_statement.where(
                    ~active_review_exists,
                    (OcrRun.id == normalized_id) | (OcrRun.page_id == normalized_id),
                )
                run_id = session.scalar(
                    statement.order_by(order, OcrRun.id.asc()).limit(1)
                )
                if run_id is not None:
                    ordered_run_ids = list(
                        session.scalars(
                            completed_statement.order_by(order, OcrRun.id.asc())
                        ).all()
                    )
                    current_index = ordered_run_ids.index(run_id)
            elif from_index is not None and to_index is not None:
                # Rank the complete active OCR population before filtering out
                # reviewed rows. Otherwise every successful review would shift
                # all later offsets and two manually assigned ranges would
                # overlap or skip work.
                range_run_ids = list(
                    session.scalars(
                        completed_statement
                        .order_by(order, OcrRun.id.asc())
                        .offset(from_index)
                        .limit(to_index - from_index + 1)
                    ).all()
                )
                reviewed_run_ids: set[str] = set()
                if range_run_ids:
                    reviewed_run_ids = set(
                        session.scalars(
                            select(OcrReview.ocr_run_id).where(
                                OcrReview.is_active == true(),
                                OcrReview.ocr_run_id.in_(range_run_ids),
                            )
                        ).all()
                    )
                run_id = None
                for local_index, candidate_id in enumerate(range_run_ids):
                    if candidate_id not in reviewed_run_ids:
                        run_id = candidate_id
                        current_index = from_index + local_index
                        break
            else:
                run_id = session.scalar(
                    completed_statement
                    .where(~active_review_exists)
                    .order_by(order, OcrRun.id.asc())
                    .limit(1)
                )
        if run_id is None:
            return None, stats, None
        return self.get_detail(run_id), stats, current_index

    def get_detail(self, run_id: str) -> OcrReviewDetail:
        with self._database.session() as session:
            run = session.get(OcrRun, run_id)
            if run is None:
                raise OcrReviewNotFoundError("OCR run was not found.")
            return self._detail(session, run)

    def get_bbox_text_preview(self, run_id: str) -> OcrBboxTextPreview:
        """Build a read-only preview from the active human OCR snapshot only."""

        detail = self.get_detail(run_id)
        if detail.run.review_id is None:
            raise OcrReviewConflictError(
                "BBox text preview requires an active human OCR review."
            )
        if detail.run.include_in_llm is not True:
            raise OcrReviewConflictError(
                "BBox text preview is unavailable because this OCR page was skipped."
            )

        included_views = [
            token for token in detail.tokens if token.include_in_extraction
        ]
        reconstruction_tokens = [
            ReconstructionToken(
                id=str(token_index),
                text=token.reviewed_text,
                bbox=token.reviewed_bbox,
                order=token_index,
            )
            for token_index, token in enumerate(included_views, start=1)
        ]
        line_results = reconstruct_bbox_lines(
            reconstruction_tokens,
            tolerance_ratio=self._line_tolerance_ratio,
        )
        line_by_token_id = {
            token_id: line.line_index
            for line in line_results
            for token_id in line.token_ids
        }
        preview_tokens = [
            OcrBboxTextPreviewToken(
                token_index=token_index,
                source_token_id=token.source_token_id,
                text=token.reviewed_text,
                bbox=token.reviewed_bbox,
                line_index=line_by_token_id[str(token_index)],
            )
            for token_index, token in enumerate(included_views, start=1)
        ]
        preview_lines = [
            OcrBboxTextPreviewLine(
                line_index=line.line_index,
                bbox=line.bbox,
                text=line.text,
                token_indexes=[int(token_id) for token_id in line.token_ids],
                segments=list(line.segments),
                quality_score=line.quality_score,
                warnings=list(line.warnings),
            )
            for line in line_results
        ]
        return OcrBboxTextPreview(
            algorithm_version="bbox-reading-order-v2",
            line_tolerance_ratio=self._line_tolerance_ratio,
            run=detail.run,
            page=detail.page,
            image_base64=detail.image_base64,
            tokens=preview_tokens,
            lines=preview_lines,
            reconstructed_text="\n".join(line.text for line in line_results),
            included_token_count=len(included_views),
            ignored_token_count=len(detail.tokens) - len(included_views),
        )

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

    def get_documents(
        self,
        *,
        page: int,
        page_size: int,
        filename: str | None,
        sort_order: str,
    ) -> OcrReviewDocumentListResponse:
        filters = [
            OcrReview.is_active == true(),
            OcrRun.is_active == true(),
            OcrRun.status == "COMPLETED",
        ]
        if filename and filename.strip():
            filters.append(
                func.lower(AnnotationDocument.original_filename).like(
                    f"%{filename.strip().lower()}%"
                )
            )

        latest_reviewed_at = func.max(OcrReview.reviewed_at).label(
            "latest_reviewed_at"
        )
        statement = (
            select(
                AnnotationDocument.id,
                AnnotationDocument.original_filename,
                AnnotationDocument.page_count,
                func.count(OcrReview.id).label("reviewed_ocr_page_count"),
                func.sum(
                    case((OcrReview.include_in_llm == true(), 1), else_=0)
                ).label("included_page_count"),
                func.sum(
                    case((OcrReview.include_in_llm == false(), 1), else_=0)
                ).label("excluded_page_count"),
                latest_reviewed_at,
            )
            .select_from(OcrReview)
            .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
            .join(AnnotationPage, AnnotationPage.id == OcrRun.page_id)
            .join(
                AnnotationDocument,
                AnnotationDocument.id == AnnotationPage.document_id,
            )
            .where(*filters)
            .group_by(
                AnnotationDocument.id,
                AnnotationDocument.original_filename,
                AnnotationDocument.page_count,
            )
        )
        order = (
            latest_reviewed_at.desc()
            if sort_order == "NEWEST"
            else latest_reviewed_at.asc()
        )
        with self._database.session() as session:
            total_items = session.scalar(
                select(func.count(func.distinct(AnnotationDocument.id)))
                .select_from(OcrReview)
                .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
                .join(AnnotationPage, AnnotationPage.id == OcrRun.page_id)
                .join(
                    AnnotationDocument,
                    AnnotationDocument.id == AnnotationPage.document_id,
                )
                .where(*filters)
            ) or 0
            rows = session.execute(
                statement
                .order_by(order, AnnotationDocument.id.asc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()

        return OcrReviewDocumentListResponse(
            items=[self._document_summary(row) for row in rows],
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=math.ceil(total_items / page_size) if total_items else 0,
        )

    def get_document_detail(self, document_id: str) -> OcrReviewDocumentDetail:
        with self._database.session() as session:
            document = session.get(AnnotationDocument, document_id)
            if document is None:
                raise OcrReviewNotFoundError("OCR document was not found.")
            rows = session.execute(
                select(OcrReview, OcrRun, AnnotationPage)
                .select_from(OcrReview)
                .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
                .join(AnnotationPage, AnnotationPage.id == OcrRun.page_id)
                .where(
                    AnnotationPage.document_id == document_id,
                    OcrReview.is_active == true(),
                    OcrRun.is_active == true(),
                    OcrRun.status == "COMPLETED",
                )
                .order_by(AnnotationPage.page_number.asc(), OcrRun.id.asc())
            ).all()
            if not rows:
                raise OcrReviewNotFoundError(
                    "This PDF has no active human-reviewed OCR pages."
                )
            summary = OcrReviewDocumentSummary(
                document_id=document.id,
                original_filename=document.original_filename,
                document_page_count=document.page_count,
                reviewed_ocr_page_count=len(rows),
                included_page_count=sum(1 for review, _, _ in rows if review.include_in_llm),
                excluded_page_count=sum(
                    1 for review, _, _ in rows if not review.include_in_llm
                ),
                latest_reviewed_at=max(review.reviewed_at for review, _, _ in rows),
            )
            details = [self._detail(session, run) for _, run, _ in rows]
        return OcrReviewDocumentDetail(document=summary, pages=details)

    def get_document_pages_in_session(
        self,
        session,
        *,
        document_id: str,
        included_only: bool = False,
    ) -> list[OcrReviewDetail]:
        """Build reviewed OCR page views without opening a nested transaction."""
        conditions = [
            AnnotationPage.document_id == document_id,
            OcrReview.is_active == true(),
            OcrReview.status == "REVIEWED",
            OcrRun.is_active == true(),
            OcrRun.status == "COMPLETED",
        ]
        if included_only:
            conditions.append(OcrReview.include_in_llm == true())
        rows = session.execute(
            select(OcrRun)
            .select_from(OcrReview)
            .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
            .join(AnnotationPage, AnnotationPage.id == OcrRun.page_id)
            .where(*conditions)
            .order_by(AnnotationPage.page_number.asc(), OcrRun.id.asc())
        ).scalars().all()
        return [self._detail(session, run) for run in rows]

    def set_inclusion(
        self,
        *,
        run_id: str,
        payload: OcrReviewInclusionRequest,
    ) -> OcrReviewDetail:
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
                    "Only an active COMPLETED OCR run can be changed."
                )
            previous = session.scalar(
                select(OcrReview)
                .where(
                    OcrReview.ocr_run_id == run.id,
                    OcrReview.is_active == true(),
                )
                .with_for_update()
            )
            if previous is None:
                raise OcrReviewConflictError(
                    "The OCR page must have an active human review first."
                )

            previous_tokens = list(
                session.scalars(
                    select(OcrReviewToken)
                    .where(OcrReviewToken.review_id == previous.id)
                    .order_by(OcrReviewToken.token_order.asc())
                ).all()
            )
            previous_regions = list(
                session.scalars(
                    select(OcrReviewRegion)
                    .where(OcrReviewRegion.review_id == previous.id)
                    .order_by(OcrReviewRegion.region_order.asc())
                ).all()
            )
            now = utc_now()
            previous.is_active = False
            previous.updated_at = now
            review = OcrReview(
                id=str(uuid4()),
                ocr_run_id=run.id,
                parent_review_id=previous.id,
                status="REVIEWED",
                is_active=True,
                include_in_llm=payload.include_in_llm,
                reviewed_by=reviewer,
                review_note=(
                    payload.review_note.strip()
                    if payload.review_note and payload.review_note.strip()
                    else previous.review_note
                ),
                reviewed_at=now,
            )
            session.add(review)
            session.flush()
            for token in previous_tokens:
                session.add(
                    OcrReviewToken(
                        id=str(uuid4()),
                        review_id=review.id,
                        source_token_id=token.source_token_id,
                        token_order=token.token_order,
                        reviewed_text=token.reviewed_text,
                        x1=token.x1,
                        y1=token.y1,
                        x2=token.x2,
                        y2=token.y2,
                        include_in_extraction=token.include_in_extraction,
                        ignored_reason=token.ignored_reason,
                    )
                )
            for region in previous_regions:
                session.add(
                    OcrReviewRegion(
                        id=str(uuid4()),
                        review_id=review.id,
                        source_region_id=region.source_region_id,
                        region_order=region.region_order,
                        region_type=region.region_type,
                        x1=region.x1,
                        y1=region.y1,
                        x2=region.x2,
                        y2=region.y2,
                    )
                )
            session.flush()
            return self._detail(session, run)

    @staticmethod
    def _document_summary(row) -> OcrReviewDocumentSummary:
        return OcrReviewDocumentSummary(
            document_id=row.id,
            original_filename=row.original_filename,
            document_page_count=row.page_count,
            reviewed_ocr_page_count=int(row.reviewed_ocr_page_count),
            included_page_count=int(row.included_page_count or 0),
            excluded_page_count=int(row.excluded_page_count or 0),
            latest_reviewed_at=row.latest_reviewed_at,
        )

    def review(
        self,
        *,
        run_id: str,
        payload: OcrReviewRequest,
        load_detail: bool = True,
    ) -> OcrReviewDetail | None:
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
            # Batch maintenance tools may operate against a shared database
            # whose image storage is mounted only on the API server.  The
            # review rows can still be validated and verified entirely from
            # SQL Server; only the response image needs storage access.
            if not load_detail:
                return None
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
