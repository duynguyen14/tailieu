from __future__ import annotations

import base64
from uuid import uuid4

from sqlalchemy import func, select, true

from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    OcrReview,
    OcrReviewToken,
    OcrRun,
    OcrToken,
    utc_now,
)
from app.schemas.ocr_review import (
    OcrReviewDetail,
    OcrReviewPageInfo,
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


class OcrReviewService:
    def __init__(self, *, database: Database, storage: AnnotationStorage) -> None:
        self._database = database
        self._storage = storage

    def get_next_review(self) -> tuple[OcrReviewDetail | None, OcrReviewStats]:
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
            run_id = session.scalar(
                select(OcrRun.id)
                .where(
                    OcrRun.is_active == true(),
                    OcrRun.status == "COMPLETED",
                    ~active_review_exists,
                )
                .order_by(OcrRun.created_at.asc())
                .limit(1)
            )
        if run_id is None:
            return None, stats
        return self.get_detail(run_id), stats

    def get_detail(self, run_id: str) -> OcrReviewDetail:
        with self._database.session() as session:
            run = session.get(OcrRun, run_id)
            if run is None:
                raise OcrReviewNotFoundError("OCR run was not found.")
            return self._detail(session, run)

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
            for token in payload.tokens:
                text = token.reviewed_text.strip()
                if not text:
                    raise OcrReviewValidationError("Reviewed token text cannot be blank.")
                x1, y1, x2, y2 = token.reviewed_bbox
                if x2 > page.width or y2 > page.height:
                    raise OcrReviewValidationError(
                        f"Token bbox {token.reviewed_bbox} exceeds page dimensions "
                        f"{page.width}x{page.height}."
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
                    )
                )
            session.flush()
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
        review = session.scalar(
            select(OcrReview).where(
                OcrReview.ocr_run_id == run.id,
                OcrReview.is_active == true(),
            )
        )

        if review is None:
            tokens = [self._raw_token_view(token) for token in raw_tokens]
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

        return OcrReviewDetail(
            run=OcrReviewRunInfo(
                id=run.id,
                page_id=run.page_id,
                status=run.status,
                model_name=run.model_name,
                model_version=run.model_version,
                device=run.device,
                processing_time_ms=run.processing_time_ms,
                review_id=review.id if review else None,
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
                width=page.width,
                height=page.height,
                mime_type=page.mime_type,
                page_type=page.label,
            ),
            image_base64=base64.b64encode(
                self._storage.read_bytes(page.image_relative_path)
            ).decode("ascii"),
            tokens=tokens,
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
            is_human_added=source is None,
            is_changed=(
                source is None
                or token.reviewed_text != source.text
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
            failed=failed,
            total=pending + reviewed + failed,
        )
