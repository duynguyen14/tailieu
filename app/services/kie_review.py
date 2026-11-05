from __future__ import annotations

import base64
import json
import mimetypes
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import func, select, true, update

from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    KieAnnotationReview,
    KieAnnotationRun,
    KieEntityAnnotation,
    KieFieldProposal,
    KiePageAnnotation,
    KieRelationAnnotation,
    OcrReviewToken,
    utc_now,
)
from app.schemas.kie_review import (
    KieCaseView,
    KieEntityView,
    KieFieldProposalView,
    KieQueueResponse,
    KieRelationView,
    KieReReviewRequest,
    KieReviewSaveResponse,
    KieReviewStats,
    KieSerReviewRequest,
    KieTokenView,
)
from app.services.storage import AnnotationStorage


class KieReviewNotFoundError(LookupError):
    pass


class KieReviewConflictError(RuntimeError):
    pass


class KieReviewValidationError(ValueError):
    pass


class KieReviewService:
    def __init__(self, *, database: Database, storage: AnnotationStorage) -> None:
        self._database = database
        self._storage = storage

    def get_statistics(self) -> KieReviewStats:
        with self._database.session() as session:
            return self._statistics(session)

    def get_next(
        self,
        *,
        mode: str,
        from_index: int | None = None,
        to_index: int | None = None,
    ) -> KieQueueResponse:
        mode = mode.upper()
        if mode not in {"SER", "RE"}:
            raise KieReviewValidationError("mode must be SER or RE")
        with self._database.session() as session:
            stats = self._statistics(session)
            base = (
                select(KiePageAnnotation.id)
                .join(KieAnnotationRun, KieAnnotationRun.id == KiePageAnnotation.run_id)
                .where(KieAnnotationRun.is_active == true())
                .order_by(KieAnnotationRun.created_at, KiePageAnnotation.page_order, KiePageAnnotation.id)
            )
            if from_index is not None and to_index is not None:
                ids = list(session.scalars(base.offset(from_index).limit(to_index - from_index + 1)))
            else:
                ids = list(session.scalars(base))
            status_column = KiePageAnnotation.ser_status if mode == "SER" else KiePageAnnotation.re_status
            pending = "PENDING"
            page_id = session.scalar(
                select(KiePageAnnotation.id)
                .where(KiePageAnnotation.id.in_(ids), status_column == pending)
                .order_by(KiePageAnnotation.created_at, KiePageAnnotation.page_order)
                .limit(1)
            ) if ids else None
            current = self._detail(session, page_id) if page_id else None
            if current and from_index is not None:
                current.current_index = from_index + ids.index(page_id)
            return KieQueueResponse(current=current, stats=stats)

    def get_detail(self, page_annotation_id: str) -> KieCaseView:
        with self._database.session() as session:
            detail = self._detail(session, page_annotation_id)
            if detail is None:
                raise KieReviewNotFoundError("KIE page annotation was not found.")
            return detail

    def review_ser(self, page_annotation_id: str, payload: KieSerReviewRequest) -> KieReviewSaveResponse:
        reviewer = payload.reviewed_by.strip()
        if not reviewer:
            raise KieReviewValidationError("Reviewer name cannot be empty.")
        with self._database.session() as session:
            page = session.scalar(select(KiePageAnnotation).where(KiePageAnnotation.id == page_annotation_id).with_for_update())
            if page is None:
                raise KieReviewNotFoundError("KIE page annotation was not found.")
            valid_tokens = set(session.scalars(select(OcrReviewToken.id).where(OcrReviewToken.review_id == page.ocr_review_id)))
            old_entities = {item.id: item for item in session.scalars(select(KieEntityAnnotation).where(KieEntityAnnotation.page_annotation_id == page.id, KieEntityAnnotation.is_active == true()))}
            for item in payload.entities:
                if any(token_id not in valid_tokens for token_id in item.token_ids):
                    raise KieReviewValidationError("An entity references a token outside the approved OCR snapshot.")
                x1, y1, x2, y2 = item.bbox
                if x2 > page.image_width or y2 > page.image_height:
                    raise KieReviewValidationError("An entity bbox is outside the OCR image.")
            now = utc_now()
            session.execute(update(KieEntityAnnotation).where(KieEntityAnnotation.page_annotation_id == page.id, KieEntityAnnotation.is_active == true()).values(is_active=False, updated_at=now))
            session.execute(update(KieRelationAnnotation).where(KieRelationAnnotation.page_annotation_id == page.id, KieRelationAnnotation.is_active == true()).values(is_active=False, updated_at=now))
            for index, item in enumerate(payload.entities, start=1):
                x1, y1, x2, y2 = item.bbox
                nx1, ny1, nx2, ny2 = self._normalized(item.bbox, page.image_width, page.image_height)
                session.add(KieEntityAnnotation(
                    id=str(uuid4()), page_annotation_id=page.id,
                    parent_entity_id=item.id if item.id in old_entities else None,
                    entity_key=f"human_{index}", label=item.label.strip(), value=item.value.strip(),
                    field_path=item.field_path, applicant_index=item.applicant_index,
                    token_ids_json=json.dumps(item.token_ids), x1=x1, y1=y1, x2=x2, y2=y2,
                    nx1=nx1, ny1=ny1, nx2=nx2, ny2=ny2, confidence=Decimal("1"),
                    source="HUMAN", is_active=True,
                ))
            page.ser_status = "APPROVED"
            page.re_status = "PENDING"
            page.ser_reviewed_by = reviewer
            page.ser_reviewed_at = now
            page.updated_at = now
            review = self._new_snapshot(session, page, "SER", reviewer, payload.review_note, {"entities": [item.model_dump() for item in payload.entities]})
            session.flush()
            return self._save_response(page, review)

    def review_re(self, page_annotation_id: str, payload: KieReReviewRequest) -> KieReviewSaveResponse:
        reviewer = payload.reviewed_by.strip()
        with self._database.session() as session:
            page = session.scalar(select(KiePageAnnotation).where(KiePageAnnotation.id == page_annotation_id).with_for_update())
            if page is None:
                raise KieReviewNotFoundError("KIE page annotation was not found.")
            if page.ser_status != "APPROVED":
                raise KieReviewConflictError("SER must be approved before RE review.")
            entities = {item.id for item in session.scalars(select(KieEntityAnnotation).where(KieEntityAnnotation.page_annotation_id == page.id, KieEntityAnnotation.is_active == true()))}
            old_relations = {item.id: item for item in session.scalars(select(KieRelationAnnotation).where(KieRelationAnnotation.page_annotation_id == page.id, KieRelationAnnotation.is_active == true()))}
            for item in payload.relations:
                if item.head_entity_id == item.tail_entity_id or item.head_entity_id not in entities or item.tail_entity_id not in entities:
                    raise KieReviewValidationError("Relation endpoints must be two active entities on this page.")
            now = utc_now()
            session.execute(update(KieRelationAnnotation).where(KieRelationAnnotation.page_annotation_id == page.id, KieRelationAnnotation.is_active == true()).values(is_active=False, updated_at=now))
            for item in payload.relations:
                session.add(KieRelationAnnotation(
                    id=str(uuid4()), page_annotation_id=page.id,
                    parent_relation_id=item.id if item.id in old_relations else None,
                    head_entity_id=item.head_entity_id, tail_entity_id=item.tail_entity_id,
                    relation_type=item.relation_type.strip(), applicant_index=item.applicant_index,
                    confidence=Decimal("1"), source="HUMAN", is_active=True,
                ))
            page.re_status = "APPROVED"
            page.re_reviewed_by = reviewer
            page.re_reviewed_at = now
            page.updated_at = now
            review = self._new_snapshot(session, page, "RE", reviewer, payload.review_note, {"relations": [item.model_dump() for item in payload.relations]})
            session.flush()
            return self._save_response(page, review)

    def _new_snapshot(self, session, page, mode, reviewer, note, body):
        previous = session.scalar(select(KieAnnotationReview).where(KieAnnotationReview.page_annotation_id == page.id, KieAnnotationReview.mode == mode, KieAnnotationReview.is_active == true()).order_by(KieAnnotationReview.created_at.desc()).limit(1))
        if previous:
            previous.is_active = False
        review = KieAnnotationReview(id=str(uuid4()), page_annotation_id=page.id, parent_review_id=previous.id if previous else None, mode=mode, payload_json=json.dumps(body, ensure_ascii=False), reviewed_by=reviewer, review_note=note, is_active=True)
        session.add(review)
        return review

    @staticmethod
    def _normalized(bbox, width, height):
        x1, y1, x2, y2 = bbox
        nx1 = max(0, min(999, round(x1 * 1000 / width)))
        ny1 = max(0, min(999, round(y1 * 1000 / height)))
        nx2 = max(nx1 + 1, min(1000, round(x2 * 1000 / width)))
        ny2 = max(ny1 + 1, min(1000, round(y2 * 1000 / height)))
        return [nx1, ny1, nx2, ny2]

    def _detail(self, session, page_annotation_id):
        if not page_annotation_id:
            return None
        row = session.execute(
            select(KiePageAnnotation, KieAnnotationRun, AnnotationDocument)
            .join(KieAnnotationRun, KieAnnotationRun.id == KiePageAnnotation.run_id)
            .join(AnnotationDocument, AnnotationDocument.id == KieAnnotationRun.document_id)
            .where(KiePageAnnotation.id == page_annotation_id, KieAnnotationRun.is_active == true())
        ).first()
        if not row:
            return None
        page, run, document = row
        tokens = list(session.scalars(select(OcrReviewToken).where(OcrReviewToken.review_id == page.ocr_review_id, OcrReviewToken.include_in_extraction == true()).order_by(OcrReviewToken.token_order)))
        proposals = list(session.scalars(select(KieFieldProposal).where(KieFieldProposal.run_id == run.id).order_by(KieFieldProposal.field_path)))
        entities = list(session.scalars(select(KieEntityAnnotation).where(KieEntityAnnotation.page_annotation_id == page.id, KieEntityAnnotation.is_active == true()).order_by(KieEntityAnnotation.entity_key)))
        relations = list(session.scalars(select(KieRelationAnnotation).where(KieRelationAnnotation.page_annotation_id == page.id, KieRelationAnnotation.is_active == true()).order_by(KieRelationAnnotation.created_at)))
        image_bytes = self._storage.read_bytes(page.image_relative_path)
        mime = mimetypes.guess_type(page.image_relative_path)[0] or "image/png"
        return KieCaseView(
            run_id=run.id, case_id=run.case_id, document_id=run.document_id,
            original_filename=document.original_filename, page_annotation_id=page.id,
            page_id=page.page_id, page_number=page.page_number, page_order=page.page_order,
            image_width=page.image_width, image_height=page.image_height,
            image_mime_type=mime, image_base64=base64.b64encode(image_bytes).decode("ascii"),
            ser_status=page.ser_status, re_status=page.re_status,
            tokens=[KieTokenView(id=t.id, order=t.token_order, text=t.reviewed_text, bbox=[t.x1,t.y1,t.x2,t.y2]) for t in tokens],
            proposals=[KieFieldProposalView(id=p.id, field_path=p.field_path, entity_type=p.entity_type, entity_index=p.entity_index, field_name=p.field_name, expected_value=p.expected_value, match_status=p.match_status, chosen=json.loads(p.chosen_json) if p.chosen_json else None, candidates=json.loads(p.candidates_json)) for p in proposals],
            entities=[KieEntityView(id=e.id, label=e.label, value=e.value, field_path=e.field_path, applicant_index=e.applicant_index, token_ids=json.loads(e.token_ids_json), bbox=[e.x1,e.y1,e.x2,e.y2], normalized_bbox=[e.nx1,e.ny1,e.nx2,e.ny2], confidence=float(e.confidence) if e.confidence is not None else None, source=e.source, candidate_rank=e.candidate_rank) for e in entities],
            relations=[KieRelationView(id=r.id, head_entity_id=r.head_entity_id, tail_entity_id=r.tail_entity_id, relation_type=r.relation_type, applicant_index=r.applicant_index, confidence=float(r.confidence) if r.confidence is not None else None, source=r.source) for r in relations],
        )

    def _statistics(self, session):
        rows = session.execute(select(KiePageAnnotation.ser_status, KiePageAnnotation.re_status, func.count()).join(KieAnnotationRun, KieAnnotationRun.id == KiePageAnnotation.run_id).where(KieAnnotationRun.is_active == true()).group_by(KiePageAnnotation.ser_status, KiePageAnnotation.re_status)).all()
        stats = KieReviewStats()
        for ser, re_status, count in rows:
            stats.total_pages += count
            setattr(stats, f"ser_{ser.lower()}", getattr(stats, f"ser_{ser.lower()}") + count)
            setattr(stats, f"re_{re_status.lower()}", getattr(stats, f"re_{re_status.lower()}") + count)
        return stats

    @staticmethod
    def _save_response(page, review):
        return KieReviewSaveResponse(page_annotation_id=page.id, mode=review.mode, review_id=review.id, parent_review_id=review.parent_review_id, reviewed_at=review.reviewed_at, ser_status=page.ser_status, re_status=page.re_status)
