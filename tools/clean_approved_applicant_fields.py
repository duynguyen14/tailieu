from __future__ import annotations

import argparse
import json
import re
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
from app.schemas.teacher_extraction import CompactDocumentExtraction

CLEANUP_VERSION = "approved-applicant-cleanup-v1"
SYSTEM_REVIEWER = f"system:{CLEANUP_VERSION}"

_ONE_TITLE = re.compile(
    r"^\s*(?:h\.\s*e\.?|(?:dr|mrs|miss|ms|m(?:i|l)?r)\b)\s*[.:]?\s*",
    re.IGNORECASE,
)
_FEMALE_TITLE = re.compile(r"^\s*(?:mrs|miss|ms)\b\s*[.:]?\s*", re.IGNORECASE)
_MALE_TITLE = re.compile(r"^\s*m(?:i|l)?r\b\s*[.:]?\s*", re.IGNORECASE)
_NG_CODE = re.compile(r"\bNG\s*[- ]?\s*(\d+[A-Z]?)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean approved applicant names and visa codes. The default is a "
            "read-only preview."
        )
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("output/approved_applicant_cleanups"),
    )
    parser.add_argument("--reviewed-by", default=SYSTEM_REVIEWER)
    return parser.parse_args()


def strip_name_titles(value: Any) -> tuple[Any, str | None]:
    if not isinstance(value, str) or not value.strip():
        return value, None
    inferred_gender: str | None = None
    if _FEMALE_TITLE.match(value):
        inferred_gender = "Female"
    elif _MALE_TITLE.match(value):
        inferred_gender = "Male"
    cleaned = value
    while True:
        match = _ONE_TITLE.match(cleaned)
        if match is None or match.end() == 0:
            break
        cleaned = cleaned[match.end() :]
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return (cleaned or value.strip()), inferred_gender


def canonical_visa_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _NG_CODE.search(text)
    if match is None:
        return None
    return f"NG{match.group(1).upper()}"


def clean_output(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cleaned = json.loads(json.dumps(raw, ensure_ascii=False))
    changes: list[dict[str, Any]] = []
    applicants = cleaned.get("applicants")
    if not isinstance(applicants, list):
        raise ValueError("applicants must be an array")
    for index, applicant in enumerate(applicants):
        if not isinstance(applicant, dict):
            raise ValueError(f"applicants[{index}] must be an object")

        old_name = applicant.get("full_name")
        new_name, inferred_gender = strip_name_titles(old_name)
        if new_name != old_name:
            applicant["full_name"] = new_name
            changes.append(
                {
                    "path": f"applicants[{index}].full_name",
                    "old": old_name,
                    "new": new_name,
                    "reason": "remove-name-honorific",
                }
            )
        old_gender = applicant.get("gender")
        if inferred_gender is not None and not str(old_gender or "").strip():
            applicant["gender"] = inferred_gender
            changes.append(
                {
                    "path": f"applicants[{index}].gender",
                    "old": old_gender,
                    "new": inferred_gender,
                    "reason": "infer-gender-from-honorific",
                }
            )

        old_visa_type = applicant.get("visa_type")
        new_visa_type = canonical_visa_type(old_visa_type)
        if new_visa_type != old_visa_type:
            applicant["visa_type"] = new_visa_type
            changes.append(
                {
                    "path": f"applicants[{index}].visa_type",
                    "old": old_visa_type,
                    "new": new_visa_type,
                    "reason": (
                        "canonical-ng-code"
                        if new_visa_type is not None
                        else "non-ng-visa-description"
                    ),
                }
            )
    return cleaned, changes


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _backup_path(directory: Path) -> Path:
    resolved = directory if directory.is_absolute() else PROJECT_ROOT / directory
    resolved.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return resolved / f"{CLEANUP_VERSION}-{stamp}.jsonl"


def _review_note(previous: str | None) -> str:
    line = f"Cleaned applicant honorifics and visa codes ({CLEANUP_VERSION})."
    return f"{previous.rstrip()}\n{line}" if previous else line


@dataclass
class Summary:
    discovered: int = 0
    changed_reviews: int = 0
    changed_values: int = 0
    unchanged_reviews: int = 0
    invalid_json: int = 0
    invalid_after_cleanup: int = 0


def run(args: argparse.Namespace) -> tuple[Summary, Counter[str], Path | None]:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater.")
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("Cleanup must run with DB_NAME=anotation_2.")
    database = Database(settings)
    summary = Summary()
    field_counts: Counter[str] = Counter()
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
                    TeacherExtractionReview.review_status == "APPROVED",
                    TeacherExtractionJob.is_active == true(),
                    TeacherExtractionJob.status == "COMPLETED",
                )
                .order_by(TeacherExtractionReview.created_at, TeacherExtractionReview.id)
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
                cleaned, changes = clean_output(raw)
                try:
                    CompactDocumentExtraction.model_validate(cleaned)
                except Exception as exc:
                    summary.invalid_after_cleanup += 1
                    print(
                        f"[{index}/{len(reviews)}] INVALID-AFTER "
                        f"file={review.job.document.original_filename!r}: {exc}"
                    )
                    continue
                if not changes:
                    summary.unchanged_reviews += 1
                    continue
                summary.changed_reviews += 1
                summary.changed_values += len(changes)
                for change in changes:
                    field_counts[change["path"].rsplit(".", 1)[-1]] += 1
                backup_rows.append(
                    {
                        "review_id": review.id,
                        "job_id": review.job_id,
                        "case_id": review.job.case_id,
                        "filename": review.job.document.original_filename,
                        "status": review.review_status,
                        "original_json": raw,
                        "cleaned_json": cleaned,
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
                            reviewed_output_json=_canonical_json(cleaned),
                            review_status="APPROVED",
                            is_active=True,
                            reviewed_by=args.reviewed_by,
                            review_note=_review_note(review.review_note),
                            reviewed_at=now,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    action = "CLEANED"
                summary_text = ", ".join(
                    f"{item['path']}:{item['old']!r}->{item['new']!r}"
                    for item in changes
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
    return summary, field_counts, backup_path


def main() -> None:
    args = parse_args()
    try:
        summary, field_counts, backup_path = run(args)
    except Exception as exc:
        raise SystemExit(f"Approved applicant cleanup could not start: {exc}") from exc
    print("Approved applicant cleanup:", "EXECUTED" if args.execute else "DRY RUN")
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    print("Changed fields:")
    for field, count in sorted(field_counts.items()):
        print(f"  {field}: {count}")
    if backup_path:
        print(f"Audit backup: {backup_path}")
    if not args.execute:
        print("No database rows were changed. Add --execute after reviewing this report.")


if __name__ == "__main__":
    main()
