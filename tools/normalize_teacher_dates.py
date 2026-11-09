from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select, true, update
from sqlalchemy.orm import joinedload

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import TeacherExtractionJob, TeacherExtractionReview, utc_now
from app.schemas.teacher_extraction import CompactDocumentExtraction

REPAIR_VERSION = "teacher-date-iso-v1"
SYSTEM_REVIEWER = f"system:{REPAIR_VERSION}"
DATE_FIELDS = (
    "date_of_birth",
    "passport_issue_date",
    "passport_expiration",
    "entry_date",
    "exit_date",
)

# Numeric birth dates in these reviewed cases were checked against the source
# tables/text. Their convention is day/month/year.
AUDITED_DMY_DOB_CASE_IDS = frozenset(
    {
        "0015446b-1e1c-4f72-b0da-f21e1b2050e8",
        "178bd309-c5c1-404c-8e9a-9c5734fac054",
        "29fba394-3ff5-4155-acad-11c0bca90da4",
        "2d7c724e-b96e-4786-b835-8808cef51326",
        "2e2aeeb3-c893-48df-86c5-7a4ad60840f7",
        "368ba657-592e-4148-bae2-7ff2a5233a50",
        "3edfeb52-2440-4ae2-9873-7c8fc0f54d43",
        "44c989a2-9bc0-41d6-b493-c3b85c19a746",
        "459da979-dcff-4738-8ca6-dce276925390",
        "4a7ac665-084d-41a8-9351-1130d080bc71",
        "4b6d2529-aa8c-480e-a240-1269eb0c7af0",
        "4fe00b6a-ea1b-4495-86af-5fa8480b7291",
        "68469865-6393-4e67-bbc2-337a56001fcd",
        "68500d6c-163b-40a2-898f-5d28a9d6f310",
        "720525e2-c656-4b65-89ea-1d0eb983eb23",
        "74be21d4-7026-48de-bbf8-3189cbd249c8",
        "92d0ef8f-449e-41d5-96fb-164d0f82c503",
        "9c0676e7-9bf7-4697-848e-478b119a8305",
        "b7518664-4af7-4195-a512-2e0cb02f4fc8",
        "b95b7ac4-e5e9-4a85-8f4d-26a326f7eb57",
        "b9e6f7d4-90a9-4b47-98f4-b51b3760dc85",
        "babf9071-4f23-43e6-b65f-868a6ccaddac",
        "c72b85c4-b845-4208-b8e1-8f3069da6abb",
        "ccde3933-cfb0-4751-9c69-08af10728ed8",
        "dff476ec-a89e-4f33-874a-ca8bd284f797",
        "e28ee299-0640-4897-97f0-9de4b7ffe0ab",
        "e2fe5d7d-2a59-47d4-92d4-cd6ee073c254",
        "f98150eb-3a67-4d65-9d6c-e75949b50a95",
        "fc2ac090-4724-4186-a472-5696a9f28cfb",
    }
)

# These 39 cases contain numeric dates where both leading components are <= 12.
# Each was checked against its reconstructed source. Their documents use D/M/Y;
# keeping this as an explicit allowlist prevents future ambiguous dates from being
# guessed silently.
AUDITED_DMY_CASE_IDS = frozenset(
    {
        "0015446b-1e1c-4f72-b0da-f21e1b2050e8",
        "137216a3-ebdd-48e5-8325-69aca652f4e7",
        "178bd309-c5c1-404c-8e9a-9c5734fac054",
        "29fba394-3ff5-4155-acad-11c0bca90da4",
        "2d7c724e-b96e-4786-b835-8808cef51326",
        "368ba657-592e-4148-bae2-7ff2a5233a50",
        "3cc810f5-5577-44cf-b227-ba1511937a18",
        "3df7e372-022b-48cc-9ae4-c6a7f6a78856",
        "44c989a2-9bc0-41d6-b493-c3b85c19a746",
        "459da979-dcff-4738-8ca6-dce276925390",
        "4a7ac665-084d-41a8-9351-1130d080bc71",
        "4fe00b6a-ea1b-4495-86af-5fa8480b7291",
        "5d4589f7-0baa-4550-a2bb-ac68597d6b23",
        "70d12392-88ef-4b9a-8b8c-6eb3f46778a5",
        "720525e2-c656-4b65-89ea-1d0eb983eb23",
        "7e4017d5-619d-487d-bc15-5b09230d096f",
        "8000e62e-07e2-4648-a897-71cc6821cba4",
        "898213c0-c639-44e4-80b4-01b3cb7b356c",
        "92d0ef8f-449e-41d5-96fb-164d0f82c503",
        "955ca69d-2d55-48e0-a485-502edf37e4cc",
        "972b549a-f866-4ff1-8a6b-b2fce08d6bdd",
        "9c0676e7-9bf7-4697-848e-478b119a8305",
        "a3ae6c9b-3cc2-48ab-be9f-fd924fb9961e",
        "b7518664-4af7-4195-a512-2e0cb02f4fc8",
        "b95b7ac4-e5e9-4a85-8f4d-26a326f7eb57",
        "b9e6f7d4-90a9-4b47-98f4-b51b3760dc85",
        "babf9071-4f23-43e6-b65f-868a6ccaddac",
        "be44a498-afb5-4349-9d04-53457c5c65e3",
        "c6d0dae1-4de9-4945-a1a7-30c5ff6f8be3",
        "ccde3933-cfb0-4751-9c69-08af10728ed8",
        "dd705025-640e-4c53-8078-b2bc00b5aa9d",
        "dff476ec-a89e-4f33-874a-ca8bd284f797",
        "e28ee299-0640-4897-97f0-9de4b7ffe0ab",
        "e2fe5d7d-2a59-47d4-92d4-cd6ee073c254",
        "e74e7bae-a1da-4260-a55d-5e69c03b4e7d",
        "ea8ee1bc-b69d-45be-b48a-1d1c919194b1",
        "f3c837af-eeeb-4f08-942a-6e071f9b1bdf",
        "fc2ac090-4724-4186-a472-5696a9f28cfb",
        "fd9ad005-de28-4abf-8810-2fe73586cdba",
    }
)

# Values checked directly against source text/PDF pages. The three impossible
# ISO-looking dates were visually verified on the original scanned tables.
MANUAL_OVERRIDES: dict[tuple[str, int, str], str | None] = {
    (
        "dff476ec-a89e-4f33-874a-ca8bd284f797",
        0,
        "date_of_birth",
    ): "1990-12-07",
    (
        "e74e7bae-a1da-4260-a55d-5e69c03b4e7d",
        0,
        "date_of_birth",
    ): "1995-04-17",
    (
        "38f929e3-02e4-4e37-b48c-ce9e28b3d74e",
        0,
        "entry_date",
    ): "2026-07-27",
    (
        "f28f6ad0-1ad0-4766-a3b3-a0ec7b474ca0",
        0,
        "passport_expiration",
    ): "2028-09-28",
    (
        "d615c716-3a19-4658-96f4-e1cf22c71388",
        0,
        "passport_expiration",
    ): "2029-10-05",
    (
        "f34f6fbb-cff2-406f-a983-0c5355509704",
        0,
        "exit_date",
    ): "2027-09-03",
    (
        "e74e7bae-a1da-4260-a55d-5e69c03b4e7d",
        0,
        "passport_expiration",
    ): "2030-02-17",
    (
        "14c25c7d-4ddf-4fac-8d0f-2899ab056482",
        0,
        "exit_date",
    ): None,
    (
        "79fb03ac-0d13-480f-aa05-6f2024b2aa06",
        0,
        "entry_date",
    ): None,
    (
        "6d44ce06-96d8-4a5f-a1dc-c3f9567d931a",
        0,
        "passport_issue_date",
    ): None,
    # The source PDF itself says "From October 05, 2026 to April 05, 2026".
    # Because the printed end date precedes the start date, do not invent a
    # corrected year for training data; retain the reliable start date only.
    (
        "6136164d-0ee6-40dd-a9ba-2f95eb1bc9be",
        0,
        "exit_date",
    ): None,
}

_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_YEAR_FIRST = re.compile(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$")
_ORDINAL = re.compile(r"(?i)(\d{1,2})(?:st|nd|rd|th)")


class DateNormalizationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize five applicant date fields to valid YYYY-MM-DD. "
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
        default=Path("output/teacher_date_repairs"),
    )
    parser.add_argument("--reviewed-by", default=SYSTEM_REVIEWER)
    return parser.parse_args()


def _iso(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError as exc:
        raise DateNormalizationError(str(exc)) from exc


def normalize_date_value(
    value: Any, *, case_id: str, field: str | None = None
) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() == "null":
        return None

    match = _ISO_DATE.fullmatch(text)
    if match:
        return _iso(*map(int, match.groups()))

    match = _YEAR_FIRST.fullmatch(text)
    if match:
        return _iso(*map(int, match.groups()))

    match = _NUMERIC_DATE.fullmatch(text)
    if match:
        first, second, year = map(int, match.groups())
        if first > 12 and second <= 12:
            return _iso(year, second, first)
        if second > 12 and first <= 12:
            return _iso(year, first, second)
        if first <= 12 and second <= 12:
            is_audited = case_id in AUDITED_DMY_CASE_IDS or (
                field == "date_of_birth" and case_id in AUDITED_DMY_DOB_CASE_IDS
            )
            if not is_audited:
                raise DateNormalizationError(
                    f"ambiguous numeric date outside audited DMY cases: {text!r}"
                )
            return _iso(year, second, first)
        raise DateNormalizationError(f"invalid numeric date: {text!r}")

    cleaned = _ORDINAL.sub(r"\1", text)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for pattern in (
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(cleaned, pattern).date().isoformat()
        except ValueError:
            continue
    raise DateNormalizationError(f"unsupported date value: {text!r}")


def normalize_review_dates(
    raw: dict[str, Any], *, case_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = json.loads(json.dumps(raw, ensure_ascii=False))
    changes: list[dict[str, Any]] = []
    applicants = normalized.get("applicants")
    if not isinstance(applicants, list):
        raise DateNormalizationError("applicants must be an array")
    for index, applicant in enumerate(applicants):
        if not isinstance(applicant, dict):
            raise DateNormalizationError(f"applicants[{index}] must be an object")
        for field in DATE_FIELDS:
            old = applicant.get(field)
            override_key = (case_id, index, field)
            if override_key in MANUAL_OVERRIDES:
                new = MANUAL_OVERRIDES[override_key]
                reason = "manual-source-verification"
            else:
                new = normalize_date_value(old, case_id=case_id, field=field)
                reason = "iso-normalization"
            if old != new:
                applicant[field] = new
                changes.append(
                    {
                        "path": f"applicants[{index}].{field}",
                        "old": old,
                        "new": new,
                        "reason": reason,
                    }
                )
    return normalized, changes


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _backup_path(directory: Path) -> Path:
    resolved = directory if directory.is_absolute() else PROJECT_ROOT / directory
    resolved.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return resolved / f"{REPAIR_VERSION}-{stamp}.jsonl"


def _review_note(previous: str | None) -> str:
    line = f"Normalized applicant dates to ISO ({REPAIR_VERSION})."
    return f"{previous.rstrip()}\n{line}" if previous else line


@dataclass
class Summary:
    discovered: int = 0
    changed_reviews: int = 0
    changed_values: int = 0
    unchanged_reviews: int = 0
    invalid_json: int = 0
    unresolved: int = 0
    invalid_after_normalization: int = 0


def run(
    args: argparse.Namespace,
) -> tuple[Summary, Counter[str], Path | None]:
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("Date normalization must run with DB_NAME=anotation_2.")
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
                try:
                    normalized, changes = normalize_review_dates(
                        raw, case_id=review.job.case_id
                    )
                except DateNormalizationError as exc:
                    summary.unresolved += 1
                    print(
                        f"[{index}/{len(reviews)}] UNRESOLVED "
                        f"file={review.job.document.original_filename!r}: {exc}"
                    )
                    continue
                try:
                    CompactDocumentExtraction.model_validate(normalized)
                except Exception as exc:
                    summary.invalid_after_normalization += 1
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
                        "normalized_json": normalized,
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
                            reviewed_output_json=_canonical_json(normalized),
                            review_status=review.review_status,
                            is_active=True,
                            reviewed_by=args.reviewed_by,
                            review_note=_review_note(review.review_note),
                            reviewed_at=now,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    action = "NORMALIZED"
                change_text = ", ".join(
                    f"{item['path']}:{item['old']!r}->{item['new']!r}"
                    for item in changes
                )
                print(
                    f"[{index}/{len(reviews)}] {action} "
                    f"file={review.job.document.original_filename!r} {change_text}"
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
        raise SystemExit(f"Teacher date normalization could not start: {exc}") from exc
    print("Teacher date normalization:", "EXECUTED" if args.execute else "DRY RUN")
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    print("Changed fields:")
    for field in DATE_FIELDS:
        print(f"  {field}: {field_counts[field]}")
    if backup_path:
        print(f"Audit backup: {backup_path}")
    if not args.execute:
        print("No database rows were changed. Add --execute after reviewing this report.")


if __name__ == "__main__":
    main()
