from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from sqlalchemy import select, true, update
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    CaseExtractionAnnotation,
    DocumentCase,
    DocumentCasePage,
    PageLabelAudit,
    utc_now,
)
from app.schemas.annotation import ReviewStatus
from app.schemas.case_extraction import CASE_SCHEMA_VERSION


NOTE_LABEL = "DIPLOMATIC_NOTE"
APPLICANT_LABELS = {"APPLICANT_TABLE", "APPLICANT_LIST"}


class ClassifiedPage(Protocol):
    id: str
    page_number: int
    label: str | None
    review_status: str


@dataclass(frozen=True)
class CasePageDraft:
    page_id: str
    page_number: int
    page_role: str


@dataclass
class CaseDraft:
    pages: list[CasePageDraft] = field(default_factory=list)
    status: str = "AUTO_GROUPED"

    @property
    def start_page_number(self) -> int:
        return min(page.page_number for page in self.pages)

    @property
    def end_page_number(self) -> int:
        return max(page.page_number for page in self.pages)


@dataclass(frozen=True)
class AuditedClassifiedPage:
    id: str
    page_number: int
    label: str | None
    review_status: str


def group_document_pages(pages: list[ClassifiedPage]) -> list[CaseDraft]:
    """Create exactly one logical case for one source PDF.

    The business invariant is one imported PDF equals one dossier. Therefore
    target pages do not need to be contiguous and unrelated pages do not split
    the dossier. Only second-pass reviewed NOTE/TABLE/LIST pages are attached.
    A dossier missing either the note side or applicant-data side is retained
    as NEEDS_REVIEW instead of being silently treated as complete.
    """

    case_pages: list[CasePageDraft] = []
    has_note = False
    has_applicant_data = False
    for page in sorted(pages, key=lambda item: item.page_number):
        if page.review_status != ReviewStatus.REVIEWED.value:
            continue
        label = (page.label or "").upper()
        if label == NOTE_LABEL:
            case_pages.append(CasePageDraft(page.id, page.page_number, "NOTE"))
            has_note = True
        elif label in APPLICANT_LABELS:
            case_pages.append(
                CasePageDraft(page.id, page.page_number, "APPLICANT_DATA")
            )
            has_applicant_data = True

    if not case_pages:
        return []
    status = "AUTO_GROUPED" if has_note and has_applicant_data else "NEEDS_REVIEW"
    return [CaseDraft(pages=case_pages, status=status)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one logical dossier case per reviewed source PDF."
    )
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--limit-documents", type=int, default=0)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Deactivate old active cases and create a new grouping; never deletes history.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_documents < 0:
        raise ValueError("--limit-documents must be zero or greater")

    settings = get_settings()
    if settings.llm_schema_version != CASE_SCHEMA_VERSION:
        raise RuntimeError(
            f"LLM_SCHEMA_VERSION must be {CASE_SCHEMA_VERSION!r}; "
            f"current value is {settings.llm_schema_version!r}."
        )
    database = Database(settings)
    created_cases = 0
    created_pages = 0
    planned_cases = 0
    planned_pages = 0
    skipped_documents = 0
    try:
        with database.session() as session:
            audited_target_exists = (
                select(PageLabelAudit.id)
                .join(AnnotationPage, AnnotationPage.id == PageLabelAudit.page_id)
                .where(
                    AnnotationPage.document_id == AnnotationDocument.id,
                    PageLabelAudit.is_active == true(),
                    PageLabelAudit.audit_status == "REVIEWED",
                    PageLabelAudit.audited_label.in_(
                        [NOTE_LABEL, *sorted(APPLICANT_LABELS)]
                    ),
                )
                .exists()
            )
            statement = (
                select(AnnotationDocument)
                .options(selectinload(AnnotationDocument.pages))
                .where(audited_target_exists)
                .order_by(AnnotationDocument.created_at.asc())
            )
            if args.document_id:
                statement = statement.where(AnnotationDocument.id == args.document_id)
            if args.limit_documents > 0:
                statement = statement.limit(args.limit_documents)
            documents = list(session.scalars(statement).unique().all())

            for document in documents:
                audit_rows = list(
                    session.scalars(
                        select(PageLabelAudit)
                        .join(AnnotationPage, AnnotationPage.id == PageLabelAudit.page_id)
                        .where(
                            AnnotationPage.document_id == document.id,
                            PageLabelAudit.is_active == true(),
                        )
                    ).all()
                )
                if any(audit.audit_status != "REVIEWED" for audit in audit_rows):
                    skipped_documents += 1
                    print(
                        f"SKIP document={document.id}: second-pass label audit is incomplete"
                    )
                    continue
                audit_by_page = {audit.page_id: audit for audit in audit_rows}
                audited_pages = [
                    AuditedClassifiedPage(
                        id=page.id,
                        page_number=page.page_number,
                        label=(
                            audit_by_page[page.id].audited_label
                            if page.id in audit_by_page
                            else None
                        ),
                        review_status=(
                            ReviewStatus.REVIEWED.value
                            if page.id in audit_by_page
                            else ReviewStatus.UNLABELED.value
                        ),
                    )
                    for page in document.pages
                ]
                active_ids = list(
                    session.scalars(
                        select(DocumentCase.id).where(
                            DocumentCase.document_id == document.id,
                            DocumentCase.is_active == true(),
                        )
                    ).all()
                )
                if active_ids and not args.rebuild:
                    skipped_documents += 1
                    print(
                        f"SKIP document={document.id}: {len(active_ids)} active case(s); "
                        "use --rebuild to version the grouping"
                    )
                    continue

                drafts = group_document_pages(audited_pages)
                planned_cases += len(drafts)
                planned_pages += sum(len(draft.pages) for draft in drafts)
                print(
                    f"document={document.id} file={document.original_filename!r} "
                    f"cases={len(drafts)}"
                )
                if args.dry_run:
                    for index, draft in enumerate(drafts, start=1):
                        page_numbers = [page.page_number for page in draft.pages]
                        print(f"  case={index} status={draft.status} pages={page_numbers}")
                    continue

                if active_ids:
                    session.execute(
                        update(DocumentCase)
                        .where(DocumentCase.id.in_(active_ids))
                        .values(is_active=False, updated_at=utc_now())
                    )

                for case_index, draft in enumerate(drafts, start=1):
                    case = DocumentCase(
                        id=str(uuid4()),
                        document_id=document.id,
                        case_index=case_index,
                        status=draft.status,
                        is_active=True,
                        start_page_number=draft.start_page_number,
                        end_page_number=draft.end_page_number,
                    )
                    for page_order, page in enumerate(draft.pages, start=1):
                        case.pages.append(
                            DocumentCasePage(
                                id=str(uuid4()),
                                page_id=page.page_id,
                                page_order=page_order,
                                page_role=page.page_role,
                            )
                        )
                    case.extraction_annotations.append(
                        CaseExtractionAnnotation(
                            id=str(uuid4()),
                            schema_version=settings.llm_schema_version,
                            target_json=None,
                            review_status="NEEDS_LABEL",
                        )
                    )
                    session.add(case)
                    created_cases += 1
                    created_pages += len(draft.pages)
    finally:
        database.close()

    print(
        "Document case build finished: "
        f"planned_cases={planned_cases}, planned_case_pages={planned_pages}, "
        f"created_cases={created_cases}, created_case_pages={created_pages}, "
        f"skipped_documents={skipped_documents}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
