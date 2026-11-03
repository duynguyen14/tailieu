from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, select, true
from sqlalchemy.orm import joinedload, selectinload

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    CaseTextReconstruction,
    DocumentCase,
    DocumentCasePage,
    OcrReview,
    OcrReviewToken,
    OcrRun,
    TableAnnotation,
    TableRegion,
    utc_now,
)
from app.services.text_reconstruction import (
    PageReconstructionInput,
    ReconstructionElement,
    ReconstructionTable,
    ReconstructionToken,
    reconstruct_case,
)


@dataclass
class ReconstructionSummary:
    discovered: int = 0
    created: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0


class CaseNotReadyError(RuntimeError):
    """The case is valid but does not yet have all approved reconstruction inputs."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build versioned structured text from approved OCR snapshots without "
            "modifying the source OCR tokens."
        )
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--case-id",
        default=None,
        help="Technical case identifier (mainly for debugging).",
    )
    selector.add_argument(
        "--document-id",
        default=None,
        help="Reconstruct the active case belonging to this imported PDF ID.",
    )
    selector.add_argument(
        "--filename",
        default=None,
        help="Reconstruct active cases whose source PDF filename matches exactly.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help=(
            "Select only cases whose every case page has an active approved OCR "
            "review. With --limit, skipped/unready cases do not consume the limit."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-text", action="store_true")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also process cases currently marked NEEDS_REVIEW.",
    )
    parser.add_argument(
        "--allow-predicted-tables",
        action="store_true",
        help="Use active PREDICTED table structure when no human-reviewed version exists.",
    )
    return parser.parse_args()


def _approved_ocr_review_exists():
    """Correlated predicate for an approved active OCR review of one case page."""

    return (
        select(OcrReview.id)
        .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
        .where(
            OcrRun.page_id == DocumentCasePage.page_id,
            OcrRun.status == "COMPLETED",
            OcrRun.is_active == true(),
            OcrReview.is_active == true(),
            OcrReview.include_in_llm == true(),
        )
        .correlate(DocumentCasePage)
        .exists()
    )


def _select_case_ids(session, args: argparse.Namespace) -> list[str]:
    statement = select(DocumentCase.id).where(DocumentCase.is_active == true())
    if not args.include_incomplete:
        statement = statement.where(DocumentCase.status != "NEEDS_REVIEW")
    if args.case_id:
        statement = statement.where(DocumentCase.id == args.case_id)
    elif args.document_id:
        statement = statement.where(DocumentCase.document_id == args.document_id)
    elif args.filename:
        normalized_filename = args.filename.strip().lower()
        if not normalized_filename:
            raise ValueError("--filename cannot be blank")
        statement = statement.join(
            AnnotationDocument,
            AnnotationDocument.id == DocumentCase.document_id,
        ).where(func.lower(AnnotationDocument.original_filename) == normalized_filename)
    if args.ready_only:
        case_page_exists = (
            select(DocumentCasePage.id)
            .where(DocumentCasePage.case_id == DocumentCase.id)
            .correlate(DocumentCase)
            .exists()
        )
        unready_page_exists = (
            select(DocumentCasePage.id)
            .where(
                DocumentCasePage.case_id == DocumentCase.id,
                ~_approved_ocr_review_exists(),
            )
            .correlate(DocumentCase)
            .exists()
        )
        statement = statement.where(case_page_exists, ~unready_page_exists)
    statement = statement.order_by(
        DocumentCase.document_id.asc(), DocumentCase.case_index.asc()
    )
    if args.limit:
        statement = statement.limit(args.limit)
    return list(session.scalars(statement).all())


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_table(
    session,
    page_id: str,
    *,
    allow_predicted: bool,
) -> TableAnnotation | None:
    statuses = ["REVIEWED"]
    if allow_predicted:
        statuses.append("PREDICTED")
    return session.scalar(
        select(TableAnnotation)
        .options(
            selectinload(TableAnnotation.regions).selectinload(TableRegion.elements)
        )
        .where(
            TableAnnotation.page_id == page_id,
            TableAnnotation.is_active == true(),
            TableAnnotation.status.in_(statuses),
        )
    )


def _page_input(session, case_page: DocumentCasePage, *, allow_predicted_tables: bool):
    page = case_page.page
    run = session.scalar(
        select(OcrRun).where(
            OcrRun.page_id == page.id,
            OcrRun.status == "COMPLETED",
            OcrRun.is_active == true(),
        )
    )
    if run is None:
        raise CaseNotReadyError(
            f"Page {page.page_number} has no active completed OCR run."
        )
    review = session.scalar(
        select(OcrReview).where(
            OcrReview.ocr_run_id == run.id,
            OcrReview.is_active == true(),
            OcrReview.include_in_llm == true(),
        )
    )
    if review is None:
        raise CaseNotReadyError(
            f"Page {page.page_number} has no active approved OCR review."
        )
    reviewed_tokens = list(
        session.scalars(
            select(OcrReviewToken)
            .where(
                OcrReviewToken.review_id == review.id,
                OcrReviewToken.include_in_extraction == true(),
            )
            .order_by(OcrReviewToken.token_order.asc())
        ).all()
    )
    tokens = tuple(
        ReconstructionToken(
            id=token.id,
            text=token.reviewed_text,
            bbox=(token.x1, token.y1, token.x2, token.y2),
            order=token.token_order,
        )
        for token in reviewed_tokens
    )
    table_annotation = _load_table(
        session, page.id, allow_predicted=allow_predicted_tables
    )
    tables: tuple[ReconstructionTable, ...] = ()
    if table_annotation is not None:
        tables = tuple(
            ReconstructionTable(
                table_index=region.table_index,
                bbox=(region.x1, region.y1, region.x2, region.y2),
                elements=tuple(
                    ReconstructionElement(
                        element_type=element.element_type,
                        bbox=(element.x1, element.y1, element.x2, element.y2),
                        row_index=element.row_index,
                        column_index=element.column_index,
                        row_span=element.row_span,
                        column_span=element.column_span,
                    )
                    for element in region.elements
                ),
            )
            for region in table_annotation.regions
        )
    source = {
        "page_id": page.id,
        "page_number": page.page_number,
        "page_label": page.label,
        "page_role": case_page.page_role,
        "ocr_run_id": run.id,
        "ocr_review_id": review.id,
        "table_annotation_id": table_annotation.id if table_annotation else None,
        "token_count": len(tokens),
    }
    return (
        PageReconstructionInput(
            page_id=page.id,
            page_number=page.page_number,
            page_type=page.label or "UNKNOWN",
            page_role=case_page.page_role,
            tokens=tokens,
            tables=tables,
        ),
        source,
    )


def run(args: argparse.Namespace) -> ReconstructionSummary:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    settings = get_settings()
    database = Database(settings)
    summary = ReconstructionSummary()
    try:
        with database.session() as session:
            case_ids = _select_case_ids(session, args)
        summary.discovered = len(case_ids)

        for position, case_id in enumerate(case_ids, start=1):
            case_filename: str | None = None
            try:
                with database.session() as session:
                    case = session.scalar(
                        select(DocumentCase)
                        .options(
                            joinedload(DocumentCase.document),
                            selectinload(DocumentCase.pages).joinedload(
                                DocumentCasePage.page
                            ),
                        )
                        .where(DocumentCase.id == case_id)
                        .with_for_update()
                    )
                    if case is None:
                        raise RuntimeError("Document case was not found.")
                    case_filename = case.document.original_filename
                    log_context = f"file={case_filename!r} case={case.id}"
                    page_inputs: list[PageReconstructionInput] = []
                    source_pages: list[dict] = []
                    for case_page in case.pages:
                        page_input, source = _page_input(
                            session,
                            case_page,
                            allow_predicted_tables=args.allow_predicted_tables,
                        )
                        page_inputs.append(page_input)
                        source_pages.append(source)
                    if not page_inputs:
                        raise CaseNotReadyError("Case has no eligible pages.")

                    source_payload = {
                        "case_id": case.id,
                        "document_id": case.document_id,
                        "algorithm_version": settings.text_reconstruction_version,
                        "pages": source_pages,
                    }
                    source_fingerprint = _fingerprint(source_payload)
                    active = session.scalar(
                        select(CaseTextReconstruction).where(
                            CaseTextReconstruction.case_id == case.id,
                            CaseTextReconstruction.is_active == true(),
                        )
                    )
                    if (
                        active is not None
                        and active.source_fingerprint == source_fingerprint
                        and active.algorithm_version
                        == settings.text_reconstruction_version
                        and not args.force
                    ):
                        summary.unchanged += 1
                        print(f"[{position}/{len(case_ids)}] unchanged {log_context}")
                        continue

                    reconstructed_text, page_results = reconstruct_case(
                        page_inputs,
                        tolerance_ratio=settings.text_reconstruction_line_tolerance,
                        assume_first_row_header=settings.table_assume_first_row_header,
                    )
                    structure = {
                        **source_payload,
                        "source_fingerprint": source_fingerprint,
                        "original_filename": case.document.original_filename,
                        "pages": [
                            {**source, **result.metadata()}
                            for source, result in zip(source_pages, page_results, strict=True)
                        ],
                    }
                    if args.print_text:
                        print(f"\n===== CASE {case.id} =====\n{reconstructed_text}\n")
                    if args.dry_run:
                        summary.created += 1
                        print(
                            f"[{position}/{len(case_ids)}] DRY-RUN {log_context} "
                            f"pages={len(page_results)} chars={len(reconstructed_text)}"
                        )
                        continue

                    if active is not None:
                        active.is_active = False
                        active.updated_at = utc_now()
                        session.flush()
                    session.add(
                        CaseTextReconstruction(
                            id=str(uuid4()),
                            case_id=case.id,
                            parent_reconstruction_id=active.id if active else None,
                            algorithm_version=settings.text_reconstruction_version,
                            source_fingerprint=source_fingerprint,
                            status="GENERATED",
                            is_active=True,
                            reconstructed_text=reconstructed_text,
                            structure_json=json.dumps(structure, ensure_ascii=False),
                            page_count=len(page_results),
                            token_count=sum(result.token_count for result in page_results),
                        )
                    )
                    summary.created += 1
                    print(
                        f"[{position}/{len(case_ids)}] created {log_context} "
                        f"pages={len(page_results)} chars={len(reconstructed_text)}"
                    )
            except CaseNotReadyError as exc:
                summary.skipped += 1
                file_context = f"file={case_filename!r} " if case_filename else ""
                print(
                    f"[{position}/{len(case_ids)}] SKIPPED {file_context}"
                    f"case={case_id}: {exc}"
                )
            except Exception as exc:
                summary.failed += 1
                file_context = f"file={case_filename!r} " if case_filename else ""
                print(
                    f"[{position}/{len(case_ids)}] FAILED {file_context}"
                    f"case={case_id}: {exc}"
                )
        return summary
    finally:
        database.close()


def main() -> None:
    summary = run(parse_args())
    print(
        "Text reconstruction finished: "
        f"discovered={summary.discovered}, created={summary.created}, "
        f"unchanged={summary.unchanged}, skipped={summary.skipped}, "
        f"failed={summary.failed}"
    )
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
