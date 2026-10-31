from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select, true
from sqlalchemy.orm import joinedload, selectinload

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import DocumentCase, DocumentCasePage, OcrRun
from app.schemas.case_extraction import CASE_SCHEMA_VERSION


SYSTEM_PROMPT = """You extract one structured diplomatic application case.
The input can contain a diplomatic note and one or more applicant pages.
Use only supplied OCR tokens. Every non-null value must cite evidence token IDs.
Never invent coordinates or facts. Return valid JSON only."""


FIELD_TEMPLATE = {
    "value": None,
    "normalized_value": None,
    "evidence": [{"page_id": None, "page_number": None, "token_ids": []}],
}


CASE_SCHEMA = {
    "schema_version": "document-case-v1",
    "case_id": None,
    "document_id": None,
    "note": {
        "note_number": FIELD_TEMPLATE,
        "reference_number": FIELD_TEMPLATE,
        "issue_date": FIELD_TEMPLATE,
        "issue_place": FIELD_TEMPLATE,
        "sender_organization": FIELD_TEMPLATE,
        "recipient_organization": FIELD_TEMPLATE,
        "request_type": FIELD_TEMPLATE,
    },
    "applicants": [
        {
            "person_id": "person_1",
            "sequence_number": FIELD_TEMPLATE,
            "full_name": FIELD_TEMPLATE,
            "date_of_birth": FIELD_TEMPLATE,
            "gender": FIELD_TEMPLATE,
            "nationality": FIELD_TEMPLATE,
            "passport_number": FIELD_TEMPLATE,
            "visa_type": FIELD_TEMPLATE,
            "entry_date": FIELD_TEMPLATE,
            "exit_date": FIELD_TEMPLATE,
            "occupation": FIELD_TEMPLATE,
            "job_title": FIELD_TEMPLATE,
            "organization": FIELD_TEMPLATE,
            "purpose_of_entry": FIELD_TEMPLATE,
            "relationship": FIELD_TEMPLATE,
            "related_to_person_id": None,
        }
    ],
    "warnings": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export complete OCR-grounded document cases for LLM annotation."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def source_token_id(page_number: int, token_id: str) -> str:
    return f"p{page_number}_{token_id}"


def page_token_lines(case_page: DocumentCasePage, run: OcrRun) -> str:
    page = case_page.page
    lines = [
        f"[PAGE {page.page_number} | ROLE={case_page.page_role} | LABEL={page.label}]"
    ]
    lines.extend(
        f"{source_token_id(page.page_number, token.token_id)} | "
        f"bbox=[{token.x1},{token.y1},{token.x2},{token.y2}] | "
        f"conf={float(token.confidence):.4f} | {token.text}"
        for token in run.tokens
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater")

    settings = get_settings()
    if settings.llm_schema_version != CASE_SCHEMA_VERSION:
        raise RuntimeError(
            f"LLM_SCHEMA_VERSION must be {CASE_SCHEMA_VERSION!r}; "
            f"current value is {settings.llm_schema_version!r}."
        )
    output = args.output or (settings.llm_dataset_path / "annotation_seed.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    database = Database(settings)
    exported = 0
    skipped = 0
    try:
        statement = (
            select(DocumentCase)
            .options(
                joinedload(DocumentCase.document),
                selectinload(DocumentCase.pages).joinedload(DocumentCasePage.page),
            )
            .where(DocumentCase.is_active == true())
            .order_by(DocumentCase.document_id.asc(), DocumentCase.case_index.asc())
        )
        if args.case_id:
            statement = statement.where(DocumentCase.id == args.case_id)
        if args.limit > 0:
            statement = statement.limit(args.limit)

        with database.session() as session:
            cases = list(session.scalars(statement).unique().all())
            page_ids = [case_page.page_id for case in cases for case_page in case.pages]
            runs: list[OcrRun] = []
            if page_ids:
                runs = list(
                    session.scalars(
                        select(OcrRun)
                        .options(selectinload(OcrRun.tokens))
                        .where(
                            OcrRun.page_id.in_(page_ids),
                            OcrRun.status == "COMPLETED",
                            OcrRun.is_active == true(),
                        )
                    ).all()
                )
            run_by_page = {run.page_id: run for run in runs}

            with output.open("w", encoding="utf-8") as handle:
                for case in cases:
                    missing = [
                        case_page.page.page_number
                        for case_page in case.pages
                        if case_page.page_id not in run_by_page
                    ]
                    if missing:
                        message = (
                            f"Case {case.id} has no completed active OCR for pages {missing}"
                        )
                        if args.fail_on_incomplete:
                            raise RuntimeError(message)
                        print(f"SKIP: {message}")
                        skipped += 1
                        continue

                    sections: list[str] = []
                    allowed_token_ids: list[str] = []
                    page_records: list[dict] = []
                    for case_page in case.pages:
                        page = case_page.page
                        run = run_by_page[case_page.page_id]
                        sections.append(page_token_lines(case_page, run))
                        token_ids = [
                            source_token_id(page.page_number, token.token_id)
                            for token in run.tokens
                        ]
                        allowed_token_ids.extend(token_ids)
                        page_records.append(
                            {
                                "page_id": page.id,
                                "page_number": page.page_number,
                                "page_role": case_page.page_role,
                                "page_type": page.label,
                                "ocr_run_id": run.id,
                            }
                        )

                    schema = dict(CASE_SCHEMA)
                    schema["case_id"] = case.id
                    schema["document_id"] = case.document_id
                    prompt = (
                        f"CASE_ID: {case.id}\n"
                        f"DOCUMENT_ID: {case.document_id}\n"
                        f"OUTPUT_SCHEMA_TEMPLATE: {json.dumps(schema, ensure_ascii=False)}\n"
                        "OCR_PAGES:\n"
                        + "\n\n".join(sections)
                    )
                    record = {
                        "example_id": case.id,
                        "case_id": case.id,
                        "document_id": case.document_id,
                        "case_index": case.case_index,
                        "case_status": case.status,
                        "schema_version": settings.llm_schema_version,
                        "original_filename": case.document.original_filename,
                        "pages": page_records,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "allowed_token_ids": allowed_token_ids,
                        "assistant_target": None,
                        "review_status": "NEEDS_LABEL",
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    exported += 1
    finally:
        database.close()

    print(
        f"Exported {exported} complete OCR-grounded cases to {output}; "
        f"skipped_incomplete={skipped}"
    )


if __name__ == "__main__":
    main()
