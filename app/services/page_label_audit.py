from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, select, true, update

from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    DocumentCase,
    PageLabelAudit,
    utc_now,
)
from app.schemas.annotation import AnnotationLabel, ReviewStatus
from app.schemas.page_label_audit import (
    PageLabelAuditDecision,
    PageLabelAuditStats,
)
from app.services.storage import AnnotationStorage


TARGET_LABELS = (
    AnnotationLabel.DIPLOMATIC_NOTE.value,
    AnnotationLabel.APPLICANT_TABLE.value,
    AnnotationLabel.APPLICANT_LIST.value,
)


class PageLabelAuditNotFoundError(LookupError):
    pass


class PageLabelAuditValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PageLabelAuditRecord:
    audit: PageLabelAudit
    page: AnnotationPage


class PageLabelAuditService:
    def __init__(self, *, database: Database, storage: AnnotationStorage) -> None:
        self._database = database
        self._storage = storage

    def get_next_document(
        self,
    ) -> tuple[
        AnnotationDocument | None,
        list[PageLabelAuditRecord],
        PageLabelAuditStats,
    ]:
        with self._database.session() as session:
            self._seed_pending_candidates(session)
            stats = self._statistics(session)
            document_id = session.scalar(
                select(AnnotationPage.document_id)
                .join(PageLabelAudit, PageLabelAudit.page_id == AnnotationPage.id)
                .join(
                    AnnotationDocument,
                    AnnotationDocument.id == AnnotationPage.document_id,
                )
                .where(
                    PageLabelAudit.is_active == true(),
                    PageLabelAudit.audit_status == "PENDING",
                )
                .order_by(
                    AnnotationDocument.created_at.asc(),
                    AnnotationPage.page_index.asc(),
                )
                .limit(1)
            )
            if document_id is None:
                return None, [], stats

            document = session.get(AnnotationDocument, document_id)
            if document is None:
                raise PageLabelAuditNotFoundError("Audit document was not found.")
            rows = session.execute(
                select(PageLabelAudit, AnnotationPage)
                .join(AnnotationPage, AnnotationPage.id == PageLabelAudit.page_id)
                .where(
                    AnnotationPage.document_id == document_id,
                    PageLabelAudit.is_active == true(),
                )
                .order_by(AnnotationPage.page_index.asc())
            ).all()
            records = [PageLabelAuditRecord(audit=row[0], page=row[1]) for row in rows]
            return document, records, stats

    def read_page_image(self, page: AnnotationPage) -> bytes:
        return self._storage.read_bytes(page.image_relative_path)

    def review_document(
        self,
        *,
        document_id: str,
        reviewed_by: str,
        decisions: list[PageLabelAuditDecision],
    ) -> int:
        reviewer = reviewed_by.strip()
        if not reviewer:
            raise PageLabelAuditValidationError("Reviewer name cannot be empty.")
        decision_by_page = {decision.page_id: decision for decision in decisions}
        if len(decision_by_page) != len(decisions):
            raise PageLabelAuditValidationError("Duplicate page_id in audit payload.")

        with self._database.session() as session:
            document = session.get(AnnotationDocument, document_id)
            if document is None:
                raise PageLabelAuditNotFoundError("Audit document was not found.")
            rows = session.execute(
                select(PageLabelAudit, AnnotationPage)
                .join(AnnotationPage, AnnotationPage.id == PageLabelAudit.page_id)
                .where(
                    AnnotationPage.document_id == document_id,
                    PageLabelAudit.is_active == true(),
                )
                .order_by(AnnotationPage.page_index.asc())
            ).all()
            expected_ids = {row[1].id for row in rows}
            submitted_ids = set(decision_by_page)
            if expected_ids != submitted_ids:
                missing = sorted(expected_ids - submitted_ids)
                unexpected = sorted(submitted_ids - expected_ids)
                raise PageLabelAuditValidationError(
                    f"Audit payload must contain every candidate page exactly once; "
                    f"missing={missing}, unexpected={unexpected}."
                )

            now = utc_now()
            for audit, page in rows:
                if page.review_status != ReviewStatus.REVIEWED.value:
                    raise PageLabelAuditValidationError(
                        f"Page {page.id} is no longer first-pass REVIEWED."
                    )
                decision = decision_by_page[page.id]
                audit.audited_label = decision.label.value
                audit.audit_status = "REVIEWED"
                audit.audited_by = reviewer
                audit.audit_note = (
                    decision.audit_note.strip() if decision.audit_note else None
                )
                audit.audited_at = now
                audit.updated_at = now

                # The canonical page label becomes the second-pass human label.
                # First-pass reviewer metadata remains untouched in annotation_pages.
                page.label = decision.label.value
                page.updated_at = now

            # Any case created before this second pass is stale. Preserve it as
            # history and require a fresh build from audited labels.
            session.execute(
                update(DocumentCase)
                .where(
                    DocumentCase.document_id == document_id,
                    DocumentCase.is_active == true(),
                )
                .values(is_active=False, updated_at=now)
            )
            return len(rows)

    def _seed_pending_candidates(self, session) -> int:
        active_audit_exists = (
            select(PageLabelAudit.id)
            .where(
                PageLabelAudit.page_id == AnnotationPage.id,
                PageLabelAudit.is_active == true(),
            )
            .exists()
        )
        pages = list(
            session.scalars(
                select(AnnotationPage)
                .where(
                    AnnotationPage.review_status == ReviewStatus.REVIEWED.value,
                    AnnotationPage.label.in_(TARGET_LABELS),
                    ~active_audit_exists,
                )
                .order_by(AnnotationPage.created_at.asc(), AnnotationPage.page_index.asc())
            ).all()
        )
        for page in pages:
            session.add(
                PageLabelAudit(
                    id=str(uuid4()),
                    page_id=page.id,
                    original_label=page.label,
                    audited_label=None,
                    audit_status="PENDING",
                    is_active=True,
                    source_reviewed_at=page.reviewed_at,
                )
            )
        session.flush()
        return len(pages)

    @staticmethod
    def _statistics(session) -> PageLabelAuditStats:
        base = (PageLabelAudit.is_active == true(),)
        total_pages = session.scalar(
            select(func.count(PageLabelAudit.id)).where(*base)
        ) or 0
        pending_pages = session.scalar(
            select(func.count(PageLabelAudit.id)).where(
                *base, PageLabelAudit.audit_status == "PENDING"
            )
        ) or 0
        reviewed_pages = session.scalar(
            select(func.count(PageLabelAudit.id)).where(
                *base, PageLabelAudit.audit_status == "REVIEWED"
            )
        ) or 0
        total_documents = session.scalar(
            select(func.count(func.distinct(AnnotationPage.document_id)))
            .join(PageLabelAudit, PageLabelAudit.page_id == AnnotationPage.id)
            .where(*base)
        ) or 0
        pending_documents = session.scalar(
            select(func.count(func.distinct(AnnotationPage.document_id)))
            .join(PageLabelAudit, PageLabelAudit.page_id == AnnotationPage.id)
            .where(*base, PageLabelAudit.audit_status == "PENDING")
        ) or 0
        return PageLabelAuditStats(
            pending_pages=pending_pages,
            reviewed_pages=reviewed_pages,
            total_pages=total_pages,
            pending_documents=pending_documents,
            total_documents=total_documents,
        )
