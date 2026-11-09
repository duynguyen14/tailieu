from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select, true, update
from sqlalchemy.orm import joinedload

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import TeacherExtractionJob, TeacherExtractionReview, utc_now
from app.schemas.teacher_extraction import (
    CompactDocumentExtraction,
    RequestTypeCode,
)

REPAIR_VERSION = "teacher-data-repair-v1"
SYSTEM_REVIEWER = f"system:{REPAIR_VERSION}"
QWEN_MODEL_MARKER = "qwen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair high-confidence teacher JSON errors using reconstructed source text. "
            "The default is a read-only preview."
        )
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create versioned review snapshots; omit for a read-only preview.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("output/teacher_data_repairs"),
    )
    parser.add_argument("--reviewed-by", default=SYSTEM_REVIEWER)
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Also inspect active local-model fallback reviews, not only Qwen.",
    )
    return parser.parse_args()


def _fold(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    ).casefold().replace("đ", "d")
    return re.sub(r"\s+", " ", normalized)


# Do not match diplomatic closing phrases such as "renew the assurances". Every
# extension expression below must explicitly target a visa, stay or residence.
_TRANSFER_PATTERNS = (
    re.compile(r"\b(?:transfer|transferring)\b.{0,90}\bvisas?\b"),
    re.compile(r"\bvisas?\b.{0,90}\b(?:transfer|transferred)\b"),
    # Require the actual phrase "chuyen [doi] thi thuc". A loose
    # "chuyen ... thi thuc" also matches ordinary text such as "chuyen di ...
    # thi thuc" (trip ... visa) and creates false positives.
    re.compile(r"\bchuyen(?: doi(?: loai)?)? thi thuc\b"),
    re.compile(r"\bthi thuc\b.{0,120}\b(?:ho chieu moi|doi ho chieu)\b"),
    re.compile(r"\b(?:new|replacement) passport\b.{0,160}\bvisas?\b"),
    re.compile(r"\bvisas?\b.{0,160}\b(?:new|replacement) passport\b"),
    re.compile(r"\bnouveau passeport\b.{0,160}\bvisas?\b"),
)

_EXTENSION_PATTERNS = (
    re.compile(r"\bgia han\b.{0,100}\b(?:thi thuc|tam tru|luu tru)\b"),
    re.compile(r"\b(?:thi thuc|tam tru|luu tru)\b.{0,100}\bgia han\b"),
    re.compile(r"\b(?:extend(?:ing|ed|s)?|extension|renew(?:ing|ed|s)?|renewal|prolong(?:ation|ement))\b.{0,100}\b(?:[vy]isas?|residence|stay|permit)\b"),
    re.compile(r"\b(?:[vy]isas?|residence|stay|permit)\b.{0,100}\b(?:extend(?:ing|ed|s)?|extension|renew(?:ing|ed|s)?|renewal|prolong(?:ation|ement))\b"),
    re.compile(r"\brenouvellement\b.{0,100}\b(?:visa|sejour|residence)\b"),
    re.compile(r"\bprorroga\b.{0,100}\b(?:visado|visa|residencia)\b"),
)

_ISSUANCE_PATTERNS = (
    re.compile(
        r"\b(?:issue|issued|issuance|issuing|reissue|re-issue|grant)\b"
        r".{0,120}\b(?:visas?|entry visas?)\b"
    ),
    re.compile(
        r"\b(?:visas?|entry visas?)\b.{0,100}"
        r"\b(?:issue|issued|issuance|issuing|grant|accorde)\b"
    ),
    re.compile(r"\brequest(?:s|ed|ing)?\b.{0,140}\b(?:visas?|visa clearance)\b"),
    re.compile(r"\b(?:receive|recevoir|recibir)\b.{0,100}\b(?:visas?|visados?)\b"),
    re.compile(r"\bfacilitat(?:e|es|ed|ing|ion)\b.{0,120}\bvisas?\b"),
    re.compile(
        r"\b(?:de nghi|xin|ho tro)\b.{0,160}\bcap\b"
        r".{0,80}\bth[iy]+\s*thuc\b"
    ),
    re.compile(r"\bcap\b.{0,80}\bth[iy]+\s*thuc\b"),
    re.compile(r"\b(?:delivrer|delivrance|accorder)\b.{0,120}\bvisas?\b"),
    re.compile(r"\bvisa on arrival\b"),
    re.compile(r"\bvisa approval(?: number)?\b"),
    re.compile(r"\b(?:solicita|solicitud)\b.{0,140}\b(?:visados?|visas?)\b"),
)

_DIPLOMATIC_CLOSING = re.compile(
    # OCR often drops "assurances of its highest consideration" or inserts a
    # long recipient name. Removing the conventional lead-in through the word
    # "renew" is sufficient and does not remove any substantive request.
    r"\bavails itself.{0,240}?\brenew(?:s|ed|ing)?\b",
    re.DOTALL,
)

@dataclass(frozen=True)
class RequestTypeDecision:
    code: RequestTypeCode | None
    evidence: str | None
    reason: str


def _evidence(patterns: tuple[re.Pattern[str], ...], source: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(source)
        if match:
            start = max(0, match.start() - 60)
            end = min(len(source), match.end() + 60)
            return source[start:end].strip()
    return None


def classify_request_type_strict(reconstructed_text: str) -> RequestTypeDecision:
    source = _fold(reconstructed_text)
    # Remove the conventional closing before looking for a genuine renewal.
    source = _DIPLOMATIC_CLOSING.sub(" ", source)
    evidence = _evidence(_TRANSFER_PATTERNS, source)
    if evidence:
        return RequestTypeDecision("CHUYEN_THI_THUC", evidence, "explicit-transfer")
    evidence = _evidence(_EXTENSION_PATTERNS, source)
    if evidence:
        return RequestTypeDecision("GIA_HAN", evidence, "explicit-extension")
    evidence = _evidence(_ISSUANCE_PATTERNS, source)
    if evidence:
        return RequestTypeDecision("CAP_MOI", evidence, "explicit-issuance")
    return RequestTypeDecision(None, None, "no-high-confidence-signal")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _repair_person_links(raw: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    applicants = raw.get("applicants")
    if not isinstance(applicants, list):
        return
    known = {
        applicant.get("person_id")
        for applicant in applicants
        if isinstance(applicant, dict) and applicant.get("person_id")
    }
    for index, applicant in enumerate(applicants):
        if not isinstance(applicant, dict):
            continue
        related = applicant.get("related_to_person_id")
        person_id = applicant.get("person_id")
        if related is not None and (related not in known or related == person_id):
            applicant["related_to_person_id"] = None
            changes.append(
                {
                    "path": f"applicants[{index}].related_to_person_id",
                    "old": related,
                    "new": None,
                    "reason": "invalid-person-reference",
                }
            )


def _repair_known_visa_type_errors(
    raw: dict[str, Any], changes: list[dict[str, Any]]
) -> None:
    applicants = raw.get("applicants")
    if not isinstance(applicants, list):
        return
    for index, applicant in enumerate(applicants):
        if not isinstance(applicant, dict):
            continue
        value = applicant.get("visa_type")
        folded = _fold(value)
        replacement_entries: str | None = None
        reason: str | None = None
        if folded == "assistant mechanical manager":
            reason = "occupation-stored-as-visa-type"
        elif "entree multiples" in folded:
            reason = "entry-count-stored-as-visa-type"
            replacement_entries = "MULTIPLE"
        elif folded.startswith("good for multiple entries"):
            reason = "visa-validity-stored-as-visa-type"
            replacement_entries = "MULTIPLE"
        if reason is None:
            continue
        applicant["visa_type"] = None
        changes.append(
            {
                "path": f"applicants[{index}].visa_type",
                "old": value,
                "new": None,
                "reason": reason,
            }
        )
        if replacement_entries and not applicant.get("number_of_entries"):
            applicant["number_of_entries"] = replacement_entries
            changes.append(
                {
                    "path": f"applicants[{index}].number_of_entries",
                    "old": None,
                    "new": replacement_entries,
                    "reason": reason,
                }
            )


def repair_output(
    raw: dict[str, Any], reconstructed_text: str
) -> tuple[dict[str, Any], list[dict[str, Any]], RequestTypeDecision]:
    repaired = json.loads(json.dumps(raw, ensure_ascii=False))
    changes: list[dict[str, Any]] = []
    decision = classify_request_type_strict(reconstructed_text)
    note = repaired.get("note")
    if isinstance(note, dict) and decision.code is not None:
        current = note.get("request_type_code")
        if current != decision.code:
            note["request_type_code"] = decision.code
            changes.append(
                {
                    "path": "note.request_type_code",
                    "old": current,
                    "new": decision.code,
                    "reason": decision.reason,
                    "evidence": decision.evidence,
                }
            )
    _repair_known_visa_type_errors(repaired, changes)
    _repair_person_links(repaired, changes)
    return repaired, changes, decision


@dataclass
class Summary:
    discovered: int = 0
    changed_reviews: int = 0
    unchanged_reviews: int = 0
    invalid_json: int = 0
    invalid_after_repair: int = 0
    ambiguous_request_type: int = 0
    no_request_type_signal: int = 0


def _backup_path(directory: Path) -> Path:
    resolved = directory if directory.is_absolute() else PROJECT_ROOT / directory
    resolved.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return resolved / f"{REPAIR_VERSION}-{stamp}.jsonl"


def _review_note(previous: str | None) -> str:
    line = f"High-confidence repair applied ({REPAIR_VERSION})."
    return f"{previous.rstrip()}\n{line}" if previous else line


def run(
    args: argparse.Namespace,
) -> tuple[Summary, Counter[str], Counter[str], Path | None]:
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("Repair must run with DB_NAME=anotation_2.")
    database = Database(settings)
    summary = Summary()
    change_counts: Counter[str] = Counter()
    request_type_counts: Counter[str] = Counter()
    backup_rows: list[dict[str, Any]] = []
    backup_path: Path | None = None
    try:
        with database.session() as session:
            statement = (
                select(TeacherExtractionReview)
                .join(
                    TeacherExtractionJob,
                    TeacherExtractionJob.id == TeacherExtractionReview.job_id,
                )
                .options(
                    joinedload(TeacherExtractionReview.job).joinedload(
                        TeacherExtractionJob.document
                    )
                )
                .where(
                    TeacherExtractionReview.is_active == true(),
                    TeacherExtractionJob.is_active == true(),
                    TeacherExtractionJob.status == "COMPLETED",
                )
                .order_by(TeacherExtractionReview.created_at, TeacherExtractionReview.id)
            )
            if not args.include_local:
                statement = statement.where(
                    TeacherExtractionJob.teacher_model.ilike(
                        f"%{QWEN_MODEL_MARKER}%"
                    )
                )
            if args.limit > 0:
                statement = statement.limit(args.limit)
            reviews = list(session.scalars(statement).unique())
            summary.discovered = len(reviews)
            for index, review in enumerate(reviews, start=1):
                try:
                    raw = json.loads(review.reviewed_output_json)
                    if not isinstance(raw, dict):
                        raise ValueError("JSON root is not an object")
                except (json.JSONDecodeError, ValueError) as exc:
                    summary.invalid_json += 1
                    print(
                        f"[{index}/{len(reviews)}] INVALID "
                        f"file={review.job.document.original_filename!r}: {exc}"
                    )
                    continue

                repaired, changes, decision = repair_output(raw, review.job.input_text)
                note_after = repaired.get("note")
                code_after = (
                    note_after.get("request_type_code")
                    if isinstance(note_after, dict)
                    else None
                )
                request_type_counts[str(code_after or "MISSING")] += 1
                if decision.reason == "no-high-confidence-signal":
                    summary.no_request_type_signal += 1
                if decision.code is None:
                    note = raw.get("note")
                    current_code = note.get("request_type_code") if isinstance(note, dict) else None
                    source_preview = re.sub(
                        r"\s+", " ", review.job.input_text
                    ).strip()[:4000]
                    request_type = (
                        note.get("request_type") if isinstance(note, dict) else None
                    )
                    print(
                        f"[{index}/{len(reviews)}] MANUAL-CHECK "
                        f"file={review.job.document.original_filename!r} "
                        f"current={current_code!r} reason={decision.reason} "
                        f"request_type={request_type!r}\n  SOURCE: {source_preview}"
                    )
                try:
                    CompactDocumentExtraction.model_validate(repaired)
                except Exception as exc:
                    summary.invalid_after_repair += 1
                    print(
                        f"[{index}/{len(reviews)}] INVALID-AFTER "
                        f"file={review.job.document.original_filename!r}: {exc}"
                    )
                    continue
                if not changes:
                    summary.unchanged_reviews += 1
                    continue

                summary.changed_reviews += 1
                for change in changes:
                    change_counts[change["path"]] += 1
                backup_rows.append(
                    {
                        "review_id": review.id,
                        "job_id": review.job_id,
                        "case_id": review.job.case_id,
                        "filename": review.job.document.original_filename,
                        "status": review.review_status,
                        "original_json": raw,
                        "repaired_json": repaired,
                        "changes": changes,
                    }
                )
                action = "DRY-RUN"
                if args.execute:
                    now = utc_now()
                    session.execute(
                        update(TeacherExtractionReview)
                        .where(
                            TeacherExtractionReview.job_id == review.job_id,
                            TeacherExtractionReview.is_active == true(),
                        )
                        .values(is_active=False, updated_at=now)
                    )
                    session.flush()
                    session.add(
                        TeacherExtractionReview(
                            id=str(uuid4()),
                            job_id=review.job_id,
                            source_attempt_id=review.source_attempt_id,
                            parent_review_id=review.id,
                            reviewed_output_json=_canonical_json(repaired),
                            review_status=review.review_status,
                            is_active=True,
                            reviewed_by=args.reviewed_by,
                            review_note=_review_note(review.review_note),
                            reviewed_at=now,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    action = "REPAIRED"
                summary_text = ", ".join(
                    f"{item['path']}:{item['old']!r}->{item['new']!r}" for item in changes
                )
                print(
                    f"[{index}/{len(reviews)}] {action} "
                    f"file={review.job.document.original_filename!r} {summary_text}"
                )

            if backup_rows:
                backup_path = _backup_path(args.backup_dir)
                with backup_path.open("w", encoding="utf-8") as handle:
                    for row in backup_rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not args.execute:
                session.rollback()
    finally:
        database.close()
    return summary, change_counts, request_type_counts, backup_path


def main() -> None:
    args = parse_args()
    try:
        summary, change_counts, request_type_counts, backup_path = run(args)
    except Exception as exc:
        raise SystemExit(f"Teacher data repair could not start: {exc}") from exc
    print("Teacher data repair:", "EXECUTED" if args.execute else "DRY RUN")
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    print("Changes:")
    for path, count in sorted(change_counts.items()):
        print(f"  {path}: {count}")
    print("Request types after repair:")
    for code in ("CAP_MOI", "GIA_HAN", "CHUYEN_THI_THUC", "KHAC", "MISSING"):
        print(f"  {code}: {request_type_counts[code]}")
    if backup_path:
        print(f"Audit backup: {backup_path}")
    if not args.execute:
        print("No database rows were changed. Add --execute after reviewing this report.")


if __name__ == "__main__":
    main()
