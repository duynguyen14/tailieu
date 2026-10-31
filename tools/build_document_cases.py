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


def group_document_pages(pages: list[ClassifiedPage]) -> list[CaseDraft]:
    """Create contiguous cases from reviewed page classifications.

    Any unreviewed or unrelated page is a hard boundary. An applicant page
    without a preceding note is retained as an orphan case for human review.
    """

    cases: list[CaseDraft] = []
    current: CaseDraft | None = None

    def finish() -> None:
        nonlocal current
        if current is not None and current.pages:
            cases.append(current)
        current = None

    for page in sorted(pages, key=lambda item: item.page_number):
        if page.review_status != ReviewStatus.REVIEWED.value:
            finish()
            continue

        label = (page.label or "").upper()
        if label == NOTE_LABEL:
            finish()
            current = CaseDraft(
                pages=[CasePageDraft(page.id, page.page_number, "NOTE")]
            )
            continue

        if label in APPLICANT_LABELS:
            if current is None:
                current = CaseDraft(status="NEEDS_REVIEW")
            current.pages.append(
                CasePageDraft(page.id, page.page_number, "APPLICANT_DATA")
            )
            continue

        finish()

    finish()
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group reviewed page classifications into logical document cases."
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
    skipped_documents = 0
    try:
        with database.session() as session:
            statement = (
                select(AnnotationDocument)
                .options(selectinload(AnnotationDocument.pages))
                .where(
                    AnnotationDocument.pages.any(
                        AnnotationPage.label.in_([NOTE_LABEL, *sorted(APPLICANT_LABELS)])
                    )
                )
                .order_by(AnnotationDocument.created_at.asc())
            )
            if args.document_id:
                statement = statement.where(AnnotationDocument.id == args.document_id)
            if args.limit_documents > 0:
                statement = statement.limit(args.limit_documents)
            documents = list(session.scalars(statement).unique().all())

            for document in documents:
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

                drafts = group_document_pages(list(document.pages))
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
        f"cases={created_cases}, case_pages={created_pages}, "
        f"skipped_documents={skipped_documents}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
