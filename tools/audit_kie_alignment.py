from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import ValidationError
from sqlalchemy import select, true
from sqlalchemy.orm import joinedload

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationPage,
    DocumentCasePage,
    OcrReview,
    OcrReviewToken,
    OcrRun,
    TeacherExtractionJob,
    TeacherExtractionReview,
)
from app.schemas.student_extraction import APPLICANT_FIELDS, NOTE_FIELDS
from app.schemas.teacher_extraction import (
    TEACHER_SCHEMA_VERSION,
    CompactDocumentExtraction,
)


DEFAULT_OUTPUT_DIR = Path("output/kie_alignment_audit_v1")
MatchStatus = Literal[
    "MATCHED", "AMBIGUOUS", "NOT_FOUND", "SKIPPED_NULL", "SKIPPED_DERIVED"
]

# These values are business interpretations or summaries, not necessarily literal
# spans in the source.  They must not reduce the direct KIE alignment score.
DERIVED_FIELDS = {"note.request_type", "applicant.purpose_of_entry", "applicant.relationship"}


@dataclass(frozen=True)
class AuditToken:
    review_id: str
    token_id: str
    token_order: int
    text: str
    page_id: str
    page_number: int
    page_order: int
    page_width: int
    page_height: int
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Candidate:
    page_id: str
    page_number: int
    page_order: int
    page_width: int
    page_height: int
    token_ids: tuple[str, ...]
    token_orders: tuple[int, ...]
    text: str
    bbox: tuple[int, int, int, int]
    base_score: float
    score: float
    method: str

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass(frozen=True)
class FieldTarget:
    entity_type: str
    entity_index: int | None
    field: str
    value: str | int | None

    @property
    def path(self) -> str:
        if self.entity_type == "note":
            return f"note.{self.field}"
        return f"applicants[{self.entity_index}].{self.field}"

    @property
    def field_kind(self) -> str:
        return f"{self.entity_type}.{self.field}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether human-approved teacher fields can be grounded to "
            "human-approved OCR token/bbox snapshots for SER/RE training."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--minimum-score", type=float, default=0.88)
    parser.add_argument("--ambiguity-margin", type=float, default=0.035)
    parser.add_argument("--max-span-tokens", type=int, default=6)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and audit the DB but only print the summary; write no files.",
    )
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser.parse_args()


def _fold_text(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character) and character.isalnum()
    )


def _date_variants(value: str) -> set[str]:
    variants = {value}
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not match:
        return variants
    year, month, day = (int(part) for part in match.groups())
    try:
        parsed = date(year, month, day)
    except ValueError:
        return variants
    variants.update(
        {
            parsed.strftime("%d/%m/%Y"),
            parsed.strftime("%d-%m-%Y"),
            parsed.strftime("%d.%m.%Y"),
            f"{parsed.day}/{parsed.month}/{parsed.year}",
            f"{parsed.day}-{parsed.month}-{parsed.year}",
            parsed.strftime("%B %d, %Y"),
            parsed.strftime("%d %B %Y"),
        }
    )
    return variants


def _semantic_variants(field_kind: str, value: str) -> set[str]:
    variants = _date_variants(value)
    folded = _fold_text(value)
    aliases: dict[str, dict[str, tuple[str, ...]]] = {
        "applicant.gender": {
            "male": ("male", "nam"),
            "female": ("female", "nu", "nữ"),
        },
        "applicant.number_of_entries": {
            "single": ("single", "one entry", "one time", "mot lan", "một lần"),
            "multiple": ("multiple", "multiple entries", "nhieu lan", "nhiều lần"),
        },
        "applicant.passport_type": {
            "ordinary": ("ordinary", "pho thong", "phổ thông"),
            "diplomatic": ("diplomatic", "ngoai giao", "ngoại giao"),
            "official": ("official", "service", "cong vu", "công vụ"),
        },
    }
    for canonical, values in aliases.get(field_kind, {}).items():
        if folded in {_fold_text(canonical), *(_fold_text(item) for item in values)}:
            variants.update(values)
    return variants


def _score_text(target_variants: Iterable[str], candidate_text: str) -> tuple[float, str]:
    candidate = _fold_text(candidate_text)
    if not candidate:
        return 0.0, "empty"
    best_score = 0.0
    best_method = "fuzzy"
    for raw_variant in target_variants:
        variant = _fold_text(raw_variant)
        if not variant:
            continue
        if variant == candidate:
            score, method = 1.0, "exact"
        elif variant in candidate:
            coverage = len(variant) / len(candidate)
            score, method = 0.94 + min(0.055, 0.055 * coverage), "contained"
        elif candidate in variant and len(candidate) >= max(3, int(len(variant) * 0.65)):
            coverage = len(candidate) / len(variant)
            score, method = 0.86 + 0.08 * coverage, "partial"
        else:
            score = SequenceMatcher(None, variant, candidate, autojunk=False).ratio()
            method = "fuzzy"
        if score > best_score:
            best_score, best_method = score, method
    return min(best_score, 1.0), best_method


def _union_bbox(tokens: list[AuditToken]) -> tuple[int, int, int, int]:
    return (
        min(token.x1 for token in tokens),
        min(token.y1 for token in tokens),
        max(token.x2 for token in tokens),
        max(token.y2 for token in tokens),
    )


def _overlap(first: Candidate, second: Candidate) -> bool:
    return first.page_id == second.page_id and bool(
        set(first.token_ids).intersection(second.token_ids)
    )


def find_candidates(
    target: FieldTarget,
    tokens: list[AuditToken],
    *,
    minimum_score: float,
    max_span_tokens: int,
) -> list[Candidate]:
    if target.value is None:
        return []
    value = str(target.value).strip()
    if not value:
        return []
    variants = _semantic_variants(target.field_kind, value)
    pages: dict[str, list[AuditToken]] = defaultdict(list)
    for token in tokens:
        pages[token.page_id].append(token)

    raw_candidates: list[Candidate] = []
    candidate_floor = max(0.72, minimum_score - 0.12)
    for page_tokens in pages.values():
        page_tokens.sort(key=lambda item: item.token_order)
        for start in range(len(page_tokens)):
            for width in range(1, min(max_span_tokens, len(page_tokens) - start) + 1):
                span = page_tokens[start : start + width]
                text = " ".join(token.text.strip() for token in span if token.text.strip())
                score, method = _score_text(variants, text)
                if score < candidate_floor:
                    continue
                raw_candidates.append(
                    Candidate(
                        page_id=span[0].page_id,
                        page_number=span[0].page_number,
                        page_order=span[0].page_order,
                        page_width=span[0].page_width,
                        page_height=span[0].page_height,
                        token_ids=tuple(token.token_id for token in span),
                        token_orders=tuple(token.token_order for token in span),
                        text=text,
                        bbox=_union_bbox(span),
                        base_score=score,
                        score=score,
                        method=method,
                    )
                )

    # Spans around the same occurrence overlap heavily. Keep the strongest one so
    # an exact phrase is not incorrectly reported as ambiguous with its own neighbors.
    collapsed: list[Candidate] = []
    for candidate in sorted(
        raw_candidates,
        key=lambda item: (item.base_score, -len(item.token_ids)),
        reverse=True,
    ):
        if any(_overlap(candidate, kept) for kept in collapsed):
            continue
        collapsed.append(candidate)
    return sorted(collapsed, key=lambda item: item.score, reverse=True)


def _with_anchor(candidate: Candidate, anchor: Candidate | None) -> Candidate:
    if anchor is None or candidate.page_id != anchor.page_id:
        return candidate
    height = max(candidate.page_height, 1)
    relative_distance = abs(candidate.center_y - anchor.center_y) / height
    bonus = 0.045
    if relative_distance <= 0.035:
        bonus += 0.075
    elif relative_distance <= 0.08:
        bonus += 0.05
    elif relative_distance <= 0.18:
        bonus += 0.02
    return Candidate(
        **{
            **candidate.__dict__,
            "score": min(1.0, candidate.base_score + bonus),
            "method": f"{candidate.method}+applicant_anchor",
        }
    )


def choose_status(
    candidates: list[Candidate],
    *,
    minimum_score: float,
    ambiguity_margin: float,
) -> tuple[MatchStatus, Candidate | None]:
    if not candidates or candidates[0].score < minimum_score:
        return "NOT_FOUND", None
    best = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        if second.score >= minimum_score and best.score - second.score <= ambiguity_margin:
            return "AMBIGUOUS", None
    return "MATCHED", best


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "page_id": candidate.page_id,
        "page_number": candidate.page_number,
        "page_order": candidate.page_order,
        "token_ids": list(candidate.token_ids),
        "token_orders": list(candidate.token_orders),
        "text": candidate.text,
        "bbox": list(candidate.bbox),
        "base_score": round(candidate.base_score, 6),
        "selection_score": round(candidate.score, 6),
        "method": candidate.method,
    }


def _field_targets(extraction: CompactDocumentExtraction) -> list[FieldTarget]:
    targets = [
        FieldTarget("note", None, field, getattr(extraction.note, field))
        for field in NOTE_FIELDS
    ]
    for index, applicant in enumerate(extraction.applicants):
        targets.extend(
            FieldTarget("applicant", index, field, getattr(applicant, field))
            for field in APPLICANT_FIELDS
        )
    return targets


def _normalize_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    applicants = normalized.get("applicants")
    if isinstance(applicants, list):
        for index, applicant in enumerate(applicants, start=1):
            if isinstance(applicant, dict):
                applicant["person_id"] = f"person_{index}"
                applicant["sequence_number"] = index
                applicant["related_to_person_id"] = None
    return normalized


def _review_statement(limit: int):
    statement = (
        select(TeacherExtractionReview)
        .join(TeacherExtractionJob, TeacherExtractionJob.id == TeacherExtractionReview.job_id)
        .options(
            joinedload(TeacherExtractionReview.job).joinedload(
                TeacherExtractionJob.document
            )
        )
        .where(
            TeacherExtractionReview.is_active == true(),
            TeacherExtractionReview.review_status == "APPROVED",
            TeacherExtractionReview.reviewed_by.is_not(None),
            TeacherExtractionReview.reviewed_at.is_not(None),
            TeacherExtractionJob.is_active == true(),
            TeacherExtractionJob.status == "COMPLETED",
            TeacherExtractionJob.schema_version == TEACHER_SCHEMA_VERSION,
        )
        .order_by(
            TeacherExtractionJob.document_id.asc(),
            TeacherExtractionJob.case_id.asc(),
            TeacherExtractionReview.created_at.asc(),
        )
    )
    return statement.limit(limit) if limit > 0 else statement


def _load_case_tokens(session, case_ids: list[str]) -> dict[str, list[AuditToken]]:
    if not case_ids:
        return {}
    statement = (
        select(
            DocumentCasePage.case_id,
            DocumentCasePage.page_order,
            AnnotationPage.id.label("page_id"),
            AnnotationPage.page_number,
            AnnotationPage.width,
            AnnotationPage.height,
            OcrReview.id.label("review_id"),
            OcrReview.reviewed_at,
            OcrReviewToken.id.label("review_token_id"),
            OcrReviewToken.token_order,
            OcrReviewToken.reviewed_text,
            OcrReviewToken.x1,
            OcrReviewToken.y1,
            OcrReviewToken.x2,
            OcrReviewToken.y2,
        )
        .join(AnnotationPage, AnnotationPage.id == DocumentCasePage.page_id)
        .join(OcrRun, OcrRun.page_id == AnnotationPage.id)
        .join(OcrReview, OcrReview.ocr_run_id == OcrRun.id)
        .join(OcrReviewToken, OcrReviewToken.review_id == OcrReview.id)
        .where(
            DocumentCasePage.case_id.in_(case_ids),
            OcrRun.is_active == true(),
            OcrRun.status == "COMPLETED",
            OcrReview.is_active == true(),
            OcrReview.status == "REVIEWED",
            OcrReview.include_in_llm == true(),
            OcrReviewToken.include_in_extraction == true(),
        )
        .order_by(
            DocumentCasePage.case_id,
            DocumentCasePage.page_order,
            OcrReview.reviewed_at.desc(),
            OcrReviewToken.token_order,
        )
    )
    rows = session.execute(statement).all()

    # Defensive handling for historical data containing more than one active
    # snapshot: choose the newest review per page, never merge snapshots.
    newest_review: dict[tuple[str, str], tuple[datetime, str]] = {}
    for row in rows:
        key = (row.case_id, row.page_id)
        reviewed_at = row.reviewed_at
        current = newest_review.get(key)
        if current is None or reviewed_at > current[0]:
            newest_review[key] = (reviewed_at, row.review_id)

    result: dict[str, list[AuditToken]] = defaultdict(list)
    for row in rows:
        if newest_review[(row.case_id, row.page_id)][1] != row.review_id:
            continue
        text = row.reviewed_text.strip()
        if not text:
            continue
        result[row.case_id].append(
            AuditToken(
                review_id=row.review_id,
                token_id=row.review_token_id,
                token_order=row.token_order,
                text=text,
                page_id=row.page_id,
                page_number=row.page_number,
                page_order=row.page_order,
                page_width=row.width,
                page_height=row.height,
                x1=row.x1,
                y1=row.y1,
                x2=row.x2,
                y2=row.y2,
            )
        )
    return dict(result)


def _audit_document(
    review: TeacherExtractionReview,
    tokens: list[AuditToken],
    *,
    minimum_score: float,
    ambiguity_margin: float,
    max_span_tokens: int,
    top_candidates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_payload = json.loads(review.reviewed_output_json)
    extraction = CompactDocumentExtraction.model_validate(
        _normalize_review_payload(raw_payload)
    )
    targets = _field_targets(extraction)

    candidates_by_path = {
        target.path: find_candidates(
            target,
            tokens,
            minimum_score=minimum_score,
            max_span_tokens=max_span_tokens,
        )
        for target in targets
        if target.value is not None and target.field_kind not in DERIVED_FIELDS
    }

    # A passport number is the safest person anchor; fall back to a unique name.
    anchors: dict[int, Candidate] = {}
    for applicant_index in range(len(extraction.applicants)):
        for field in ("passport_number", "full_name"):
            path = f"applicants[{applicant_index}].{field}"
            status, chosen = choose_status(
                candidates_by_path.get(path, []),
                minimum_score=minimum_score,
                ambiguity_margin=ambiguity_margin,
            )
            if status == "MATCHED" and chosen is not None:
                anchors[applicant_index] = chosen
                break

    alignments: list[dict[str, Any]] = []
    for target in targets:
        value = target.value
        if value is None or (isinstance(value, str) and not value.strip()):
            status: MatchStatus = "SKIPPED_NULL"
            ranked: list[Candidate] = []
            chosen = None
        elif target.field_kind in DERIVED_FIELDS:
            status = "SKIPPED_DERIVED"
            ranked = []
            chosen = None
        else:
            ranked = candidates_by_path.get(target.path, [])
            if target.entity_type == "applicant" and target.entity_index is not None:
                anchor = anchors.get(target.entity_index)
                ranked = sorted(
                    (_with_anchor(candidate, anchor) for candidate in ranked),
                    key=lambda item: item.score,
                    reverse=True,
                )
            status, chosen = choose_status(
                ranked,
                minimum_score=minimum_score,
                ambiguity_margin=ambiguity_margin,
            )
        alignments.append(
            {
                "review_id": review.id,
                "job_id": review.job_id,
                "case_id": review.job.case_id,
                "document_id": review.job.document_id,
                "original_filename": review.job.document.original_filename,
                "entity_type": target.entity_type,
                "entity_index": target.entity_index,
                "field": target.field,
                "field_kind": target.field_kind,
                "field_path": target.path,
                "value": value,
                "status": status,
                "chosen": _candidate_payload(chosen) if chosen else None,
                "candidates": [
                    _candidate_payload(candidate)
                    for candidate in ranked[:top_candidates]
                ],
            }
        )

    status_counts = Counter(item["status"] for item in alignments)
    direct_total = sum(
        status_counts[status]
        for status in ("MATCHED", "AMBIGUOUS", "NOT_FOUND")
    )
    document_summary = {
        "review_id": review.id,
        "job_id": review.job_id,
        "case_id": review.job.case_id,
        "document_id": review.job.document_id,
        "original_filename": review.job.document.original_filename,
        "approved_ocr_pages": len({token.page_id for token in tokens}),
        "approved_ocr_tokens": len(tokens),
        "applicant_count": len(extraction.applicants),
        "status_counts": dict(status_counts),
        "direct_non_null_fields": direct_total,
        "direct_match_rate": (
            round(status_counts["MATCHED"] / direct_total, 6) if direct_total else None
        ),
    }
    return document_summary, alignments


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _recommendation(status_counts: Counter[str]) -> dict[str, Any]:
    direct_total = sum(
        status_counts[status] for status in ("MATCHED", "AMBIGUOUS", "NOT_FOUND")
    )
    matched = status_counts["MATCHED"]
    ambiguous = status_counts["AMBIGUOUS"]
    match_rate = matched / direct_total if direct_total else 0.0
    ambiguity_rate = ambiguous / direct_total if direct_total else 0.0
    if matched >= 500 and match_rate >= 0.75 and ambiguity_rate <= 0.15:
        verdict = "READY_FOR_SER_RE_PILOT"
        message = (
            "Enough high-confidence grounded fields for a pilot SER/RE dataset. "
            "Manually resolve AMBIGUOUS rows before final training export."
        )
    elif matched >= 250 and match_rate >= 0.5:
        verdict = "NEEDS_TARGETED_ALIGNMENT_REVIEW"
        message = (
            "Useful seed data exists, but ambiguous/not-found fields need targeted "
            "review or stronger row/person anchoring before training."
        )
    else:
        verdict = "NOT_READY_FOR_KIE_TRAINING"
        message = (
            "Too few direct fields are grounded reliably. Improve OCR reviews or "
            "add entity-span review before training."
        )
    return {
        "verdict": verdict,
        "message": message,
        "direct_non_null_fields": direct_total,
        "matched_fields": matched,
        "match_rate": round(match_rate, 6),
        "ambiguity_rate": round(ambiguity_rate, 6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater.")
    if not 0.5 <= args.minimum_score <= 1:
        raise ValueError("--minimum-score must be between 0.5 and 1.0.")
    if not 0 <= args.ambiguity_margin <= 0.25:
        raise ValueError("--ambiguity-margin must be between 0 and 0.25.")
    if args.max_span_tokens < 1 or args.top_candidates < 1:
        raise ValueError("--max-span-tokens and --top-candidates must be positive.")

    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("KIE alignment audit must run with DB_NAME=anotation_2.")
    database = Database(settings)
    invalid_items: list[dict[str, str]] = []
    document_summaries: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    try:
        with database.session() as session:
            reviews = list(
                session.scalars(_review_statement(args.limit)).unique().all()
            )
            case_tokens = _load_case_tokens(
                session, sorted({review.job.case_id for review in reviews})
            )
            for index, review in enumerate(reviews, start=1):
                try:
                    document_summary, document_alignments = _audit_document(
                        review,
                        case_tokens.get(review.job.case_id, []),
                        minimum_score=args.minimum_score,
                        ambiguity_margin=args.ambiguity_margin,
                        max_span_tokens=args.max_span_tokens,
                        top_candidates=args.top_candidates,
                    )
                except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                    invalid_items.append({"review_id": review.id, "error": str(exc)})
                    if args.fail_on_invalid:
                        raise RuntimeError(
                            f"Invalid approved teacher review {review.id}: {exc}"
                        ) from exc
                    print(f"[{index}/{len(reviews)}] INVALID review={review.id}: {exc}")
                    continue
                document_summaries.append(document_summary)
                alignments.extend(document_alignments)
                counts = document_summary["status_counts"]
                print(
                    f"[{index}/{len(reviews)}] file='{document_summary['original_filename']}' "
                    f"matched={counts.get('MATCHED', 0)} "
                    f"ambiguous={counts.get('AMBIGUOUS', 0)} "
                    f"not_found={counts.get('NOT_FOUND', 0)}"
                )
    finally:
        database.close()

    status_counts = Counter(item["status"] for item in alignments)
    field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in alignments:
        field_counts[item["field_kind"]][item["status"]] += 1
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database": settings.db_name,
        "settings": {
            "minimum_score": args.minimum_score,
            "ambiguity_margin": args.ambiguity_margin,
            "max_span_tokens": args.max_span_tokens,
            "top_candidates": args.top_candidates,
        },
        "documents_discovered": len(document_summaries) + len(invalid_items),
        "documents_audited": len(document_summaries),
        "documents_without_approved_ocr": sum(
            item["approved_ocr_pages"] == 0 for item in document_summaries
        ),
        "invalid_documents": len(invalid_items),
        "status_counts": dict(status_counts),
        "counts_by_field": {
            field: dict(counts) for field, counts in sorted(field_counts.items())
        },
        "recommendation": _recommendation(status_counts),
        "invalid_items": invalid_items,
    }

    if not args.dry_run:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir.is_absolute()
            else (PROJECT_ROOT / args.output_dir).resolve()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_jsonl(output_dir / "documents.jsonl", document_summaries)
        _write_jsonl(output_dir / "alignments.jsonl", alignments)
        _write_jsonl(
            output_dir / "matched.jsonl",
            (item for item in alignments if item["status"] == "MATCHED"),
        )
        _write_jsonl(
            output_dir / "ambiguous.jsonl",
            (item for item in alignments if item["status"] == "AMBIGUOUS"),
        )
        _write_jsonl(
            output_dir / "not_found.jsonl",
            (item for item in alignments if item["status"] == "NOT_FOUND"),
        )
        summary["output_dir"] = str(output_dir)

    print("KIE alignment audit finished:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    try:
        run(parse_args())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"KIE alignment audit could not start: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
