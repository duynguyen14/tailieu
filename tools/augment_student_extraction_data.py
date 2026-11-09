from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import select, true, update
from sqlalchemy.orm import joinedload

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    StudentTrainingExample,
    TeacherExtractionJob,
    TeacherExtractionReview,
)
from app.schemas.student_extraction import (
    STUDENT_SCHEMA_VERSION,
    compact_document_to_student_target,
    student_target_to_compact_document,
)
from app.schemas.teacher_extraction import (
    TEACHER_SCHEMA_VERSION,
    CompactDocumentExtraction,
)
from tools.export_student_extraction_dataset import (
    canonical_json,
    normalize_teacher_payload_for_student,
    split_name,
)

AUGMENTATION_VERSION = "local-synthetic-v2"
GENERATED_BY = f"system:{AUGMENTATION_VERSION}"

NATIONALITIES = (
    "American",
    "Australian",
    "British",
    "Canadian",
    "French",
    "German",
    "Indian",
    "Indonesian",
    "Japanese",
    "Korean",
    "New Zealand",
    "Singaporean",
    "Thai",
)
MALE_GIVEN_NAMES = (
    "Adrian",
    "Daniel",
    "Ethan",
    "Gabriel",
    "Julian",
    "Marcus",
    "Nathan",
    "Oliver",
)
FEMALE_GIVEN_NAMES = (
    "Amelia",
    "Charlotte",
    "Elena",
    "Isabella",
    "Natalie",
    "Olivia",
    "Sophia",
    "Victoria",
)
NEUTRAL_GIVEN_NAMES = (
    "Alex",
    "Cameron",
    "Jordan",
    "Morgan",
    "Robin",
    "Taylor",
)
FAMILY_NAMES = (
    "Anderson",
    "Bennett",
    "Campbell",
    "Foster",
    "Harrison",
    "Mitchell",
    "Parker",
    "Sullivan",
    "Thompson",
    "Walker",
)
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTH_ABBREVIATIONS = tuple(month[:3] for month in MONTHS)


@dataclass(frozen=True)
class AugmentedExample:
    source_review_id: str
    source_job_id: str
    source_case_id: str
    source_document_id: str
    input_text: str
    target: dict[str, Any]
    variation_index: int
    seed: int
    changes: list[dict[str, Any]]
    record_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create local synthetic text/JSON pairs from approved teacher reviews. "
            "The default is a read-only preview."
        )
    )
    parser.add_argument("--multiplier", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-changes", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--replace-active",
        action="store_true",
        help=(
            "Deactivate existing active synthetic examples before inserting the "
            "new version. Requires --execute; history is preserved."
        ),
    )
    return parser.parse_args()


def _stable_rng(seed: int, source_review_id: str, variation_index: int) -> random.Random:
    material = f"{seed}:{source_review_id}:{variation_index}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)


def _case_like(replacement: str, source: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source.islower():
        return replacement.lower()
    if source.istitle():
        return replacement.title()
    return replacement


def _replace_literal(text: str, old: str, new: str) -> tuple[str, int]:
    if not old or old.casefold() == new.casefold():
        return text, 0
    pattern = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)", re.IGNORECASE)
    return pattern.subn(lambda match: _case_like(new, match.group(0)), text)


def _randomize_identifier(value: str, rng: random.Random, *, keep_year: bool) -> str:
    parts: list[str] = []
    cursor = 0
    for match in re.finditer(r"[A-Za-z]+|\d+", value):
        parts.append(value[cursor : match.start()])
        token = match.group(0)
        if keep_year and re.fullmatch(r"20\d{2}", token):
            replacement = token
        elif token.isdigit():
            replacement = "".join(rng.choice("0123456789") for _ in token)
            if len(token) > 1 and replacement[0] == "0":
                replacement = rng.choice("123456789") + replacement[1:]
        else:
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            replacement = "".join(rng.choice(alphabet) for _ in token)
            if token.islower():
                replacement = replacement.lower()
        parts.append(replacement)
        cursor = match.end()
    parts.append(value[cursor:])
    candidate = "".join(parts)
    if candidate == value:
        candidate = value + str(rng.randrange(1, 10))
    return candidate


def _new_name(gender: Any, rng: random.Random) -> str:
    normalized = str(gender or "").strip().casefold()
    if normalized in {"male", "m", "nam"}:
        given_names = MALE_GIVEN_NAMES
    elif normalized in {"female", "f", "nữ", "nu"}:
        given_names = FEMALE_GIVEN_NAMES
    else:
        given_names = NEUTRAL_GIVEN_NAMES
    return f"{rng.choice(given_names)} {rng.choice(FAMILY_NAMES)}"


def _different_choice(options: tuple[str, ...], old: str, rng: random.Random) -> str:
    candidates = [item for item in options if item.casefold() != old.casefold()]
    return rng.choice(candidates)


def _format_number(value: int, width: int) -> str:
    return str(value).zfill(width) if width > 1 else str(value)


def _month_like(month: int, source: str) -> str:
    replacement = MONTHS[month - 1] if len(source) > 3 else MONTH_ABBREVIATIONS[month - 1]
    return _case_like(replacement, source)


def _replace_iso_date(text: str, old_iso: str, new_iso: str) -> tuple[str, int]:
    old = date.fromisoformat(old_iso)
    new = date.fromisoformat(new_iso)
    total = 0

    day_token = rf"0?{old.day}" if old.day < 10 else str(old.day)
    month_token = rf"0?{old.month}" if old.month < 10 else str(old.month)

    day_first = re.compile(
        rf"(?<!\d)(?P<day>{day_token})(?P<sep>[./-])"
        rf"(?P<month>{month_token})(?P=sep)(?P<year>{old.year})(?!\d)"
    )

    def day_first_replacement(match: re.Match[str]) -> str:
        return (
            _format_number(new.day, len(match.group("day")))
            + match.group("sep")
            + _format_number(new.month, len(match.group("month")))
            + match.group("sep")
            + str(new.year)
        )

    text, count = day_first.subn(day_first_replacement, text)
    total += count

    year_first = re.compile(
        rf"(?<!\d)(?P<year>{old.year})(?P<sep>[./-])"
        rf"(?P<month>{month_token})(?P=sep)(?P<day>{day_token})(?!\d)"
    )

    def year_first_replacement(match: re.Match[str]) -> str:
        return (
            str(new.year)
            + match.group("sep")
            + _format_number(new.month, len(match.group("month")))
            + match.group("sep")
            + _format_number(new.day, len(match.group("day")))
        )

    text, count = year_first.subn(year_first_replacement, text)
    total += count

    month_names = rf"{MONTHS[old.month - 1]}|{MONTH_ABBREVIATIONS[old.month - 1]}"
    text_month_first = re.compile(
        rf"(?i)(?<![A-Za-z])(?P<month>{month_names})\s+"
        rf"(?P<day>{day_token})(?P<ordinal>st|nd|rd|th)?\s*,?\s*"
        rf"(?P<year>{old.year})(?!\d)"
    )

    def month_first_replacement(match: re.Match[str]) -> str:
        original = match.group(0)
        comma = "," if "," in original else ""
        ordinal = match.group("ordinal") or ""
        return (
            f"{_month_like(new.month, match.group('month'))} "
            f"{_format_number(new.day, len(match.group('day')))}{ordinal}{comma} "
            f"{new.year}"
        )

    text, count = text_month_first.subn(month_first_replacement, text)
    total += count

    text_day_first = re.compile(
        rf"(?i)(?<!\d)(?P<day>{day_token})(?P<ordinal>st|nd|rd|th)?\s*"
        rf"(?P<month>{month_names})\s*(?P<year>{old.year})(?!\d)"
    )

    def text_day_first_replacement(match: re.Match[str]) -> str:
        original = match.group(0)
        compact = " " not in original
        ordinal = match.group("ordinal") or ""
        separator = "" if compact else " "
        return (
            f"{_format_number(new.day, len(match.group('day')))}{ordinal}{separator}"
            f"{_month_like(new.month, match.group('month'))}{separator}{new.year}"
        )

    text, count = text_day_first.subn(text_day_first_replacement, text)
    total += count
    return text, total


def _set_change(
    changes: list[dict[str, Any]],
    container: dict[str, Any],
    field: str,
    new_value: Any,
    *,
    path: str,
    replacements: int,
) -> None:
    old_value = container.get(field)
    if old_value == new_value:
        return
    container[field] = new_value
    changes.append(
        {
            "path": path,
            "old": old_value,
            "new": new_value,
            "text_replacements": replacements,
        }
    )


def _validate_business_dates(payload: dict[str, Any]) -> None:
    for index, applicant in enumerate(payload.get("applicants", [])):
        issue = applicant.get("passport_issue_date")
        expiry = applicant.get("passport_expiration")
        entry = applicant.get("entry_date")
        exit_value = applicant.get("exit_date")
        if issue and expiry and date.fromisoformat(issue) > date.fromisoformat(expiry):
            raise ValueError(f"applicants[{index}] passport issue is after expiry")
        if entry and exit_value and date.fromisoformat(entry) > date.fromisoformat(exit_value):
            raise ValueError(f"applicants[{index}] entry is after exit")


def augment_review(
    review: TeacherExtractionReview,
    *,
    variation_index: int,
    seed: int,
    min_changes: int,
) -> AugmentedExample:
    rng = _stable_rng(seed, review.id, variation_index)
    source = json.loads(review.reviewed_output_json)
    target = copy.deepcopy(source)
    text = review.job.input_text
    changes: list[dict[str, Any]] = []

    note = target.get("note")
    if isinstance(note, dict):
        old_note_number = note.get("note_number")
        if isinstance(old_note_number, str) and old_note_number.strip():
            new_note_number = _randomize_identifier(
                old_note_number, rng, keep_year=True
            )
            replaced_text, replacements = _replace_literal(
                text, old_note_number, new_note_number
            )
            if replacements:
                text = replaced_text
                _set_change(
                    changes,
                    note,
                    "note_number",
                    new_note_number,
                    path="note.note_number",
                    replacements=replacements,
                )

    applicants = target.get("applicants")
    if not isinstance(applicants, list):
        raise ValueError("Approved review applicants must be an array.")

    name_map: dict[str, str] = {}
    nationality_map: dict[str, str] = {}
    passport_map: dict[str, str] = {}
    used_names: set[str] = set()
    for index, applicant in enumerate(applicants):
        if not isinstance(applicant, dict):
            raise ValueError(f"applicants[{index}] must be an object")

        old_name = applicant.get("full_name")
        if isinstance(old_name, str) and old_name.strip():
            cache_key = old_name.casefold()
            new_name = name_map.get(cache_key)
            replacements = 0
            if new_name is None:
                for _ in range(50):
                    candidate = _new_name(applicant.get("gender"), rng)
                    if candidate not in used_names:
                        new_name = candidate
                        used_names.add(candidate)
                        break
                if new_name is None:
                    raise ValueError("Could not generate a unique applicant name")
                replaced_text, replacements = _replace_literal(text, old_name, new_name)
                if replacements:
                    text = replaced_text
                    name_map[cache_key] = new_name
                else:
                    new_name = None
            if new_name is not None:
                _set_change(
                    changes,
                    applicant,
                    "full_name",
                    new_name,
                    path=f"applicants[{index}].full_name",
                    replacements=replacements,
                )

        old_passport = applicant.get("passport_number")
        if isinstance(old_passport, str) and old_passport.strip():
            cache_key = old_passport.casefold()
            new_passport = passport_map.get(cache_key)
            replacements = 0
            if new_passport is None:
                new_passport = _randomize_identifier(
                    old_passport, rng, keep_year=False
                )
                replaced_text, replacements = _replace_literal(
                    text, old_passport, new_passport
                )
                if replacements:
                    text = replaced_text
                    passport_map[cache_key] = new_passport
                else:
                    new_passport = None
            if new_passport is not None:
                _set_change(
                    changes,
                    applicant,
                    "passport_number",
                    new_passport,
                    path=f"applicants[{index}].passport_number",
                    replacements=replacements,
                )

        old_nationality = applicant.get("nationality")
        if isinstance(old_nationality, str) and old_nationality.strip():
            cache_key = old_nationality.casefold()
            new_nationality = nationality_map.get(cache_key)
            replacements = 0
            if new_nationality is None:
                new_nationality = _different_choice(
                    NATIONALITIES, old_nationality, rng
                )
                replaced_text, replacements = _replace_literal(
                    text, old_nationality, new_nationality
                )
                if replacements:
                    text = replaced_text
                    nationality_map[cache_key] = new_nationality
                else:
                    new_nationality = None
            if new_nationality is not None:
                _set_change(
                    changes,
                    applicant,
                    "nationality",
                    new_nationality,
                    path=f"applicants[{index}].nationality",
                    replacements=replacements,
                )

    text_before_dates = text
    target_before_dates = copy.deepcopy(target)
    changes_before_dates = len(changes)
    date_locations: dict[str, list[tuple[dict[str, Any], str, str]]] = {}
    if isinstance(note, dict):
        issue_date = note.get("issue_date")
        if isinstance(issue_date, str):
            try:
                date.fromisoformat(issue_date)
            except ValueError:
                pass
            else:
                date_locations.setdefault(issue_date, []).append(
                    (note, "issue_date", "note.issue_date")
                )
    for index, applicant in enumerate(applicants):
        for field in (
            "passport_issue_date",
            "passport_expiration",
            "entry_date",
            "exit_date",
        ):
            value = applicant.get(field)
            if not isinstance(value, str):
                continue
            try:
                date.fromisoformat(value)
            except ValueError:
                continue
            date_locations.setdefault(value, []).append(
                (applicant, field, f"applicants[{index}].{field}")
            )

    date_shift = timedelta(days=rng.randint(14, 180))
    for old_iso, locations in date_locations.items():
        new_iso = (date.fromisoformat(old_iso) + date_shift).isoformat()
        replaced_text, replacements = _replace_iso_date(text, old_iso, new_iso)
        if not replacements:
            continue
        text = replaced_text
        for container, field, path in locations:
            _set_change(
                changes,
                container,
                field,
                new_iso,
                path=path,
                replacements=replacements,
            )

    try:
        _validate_business_dates(target)
    except ValueError:
        text = text_before_dates
        target = target_before_dates
        changes = changes[:changes_before_dates]

    if len(changes) < min_changes:
        raise ValueError(
            f"Only {len(changes)} synchronized field changes; need {min_changes}."
        )
    if text == review.job.input_text:
        raise ValueError("Synthetic input text did not change.")
    _validate_business_dates(target)
    extraction = CompactDocumentExtraction.model_validate(
        normalize_teacher_payload_for_student(target)
    )
    student_target = compact_document_to_student_target(extraction)
    student_target_to_compact_document(
        student_target,
        case_id=review.job.case_id,
        document_id=review.job.document_id,
    )
    record_material = {
        # A synthetic row is a versioned training artifact. Include the
        # augmentation version so a corrected regeneration can coexist with
        # the inactive historical version while remaining deterministic.
        "augmentation_version": AUGMENTATION_VERSION,
        "schema_version": STUDENT_SCHEMA_VERSION,
        "source_review_id": review.id,
        "input_text": text,
        "target": target,
    }
    record_hash = hashlib.sha256(
        canonical_json(record_material).encode("utf-8")
    ).hexdigest()
    return AugmentedExample(
        source_review_id=review.id,
        source_job_id=review.job_id,
        source_case_id=review.job.case_id,
        source_document_id=review.job.document_id,
        input_text=text,
        target=target,
        variation_index=variation_index,
        seed=seed,
        changes=changes,
        record_hash=record_hash,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.multiplier < 1:
        raise ValueError("--multiplier must be at least 1.")
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater.")
    if args.min_changes < 1:
        raise ValueError("--min-changes must be at least 1.")
    if args.replace_active and not args.execute:
        raise ValueError("--replace-active requires --execute.")
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("Synthetic augmentation must run with DB_NAME=anotation_2.")

    database = Database(settings)
    if args.execute:
        database.create_tables()
    summary: Counter[str] = Counter()
    change_counts: Counter[str] = Counter()
    previews: list[dict[str, Any]] = []
    try:
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
                TeacherExtractionJob.schema_version == TEACHER_SCHEMA_VERSION,
            )
            .order_by(TeacherExtractionReview.created_at, TeacherExtractionReview.id)
        )
        with database.session() as session:
            if args.execute and args.replace_active:
                now = datetime.now(UTC)
                result = session.execute(
                    update(StudentTrainingExample)
                    .where(
                        StudentTrainingExample.is_active == true(),
                        StudentTrainingExample.is_synthetic == true(),
                    )
                    .values(is_active=False, updated_at=now)
                )
                summary["deactivated_previous"] = result.rowcount or 0
            reviews = [
                review
                for review in session.scalars(statement).unique().all()
                if split_name(review.job.document_id) == "train"
            ]
            if args.limit > 0:
                reviews = reviews[: args.limit]
            summary["source_reviews"] = len(reviews)

            existing_variants: set[tuple[str, int]] = set()
            if args.execute:
                existing_variants = {
                    (row.source_review_id, row.variation_index)
                    for row in session.scalars(
                        select(StudentTrainingExample).where(
                            StudentTrainingExample.augmentation_version
                            == AUGMENTATION_VERSION,
                            StudentTrainingExample.seed == args.seed,
                        )
                    )
                }

            for review in reviews:
                for variation_index in range(1, args.multiplier + 1):
                    summary["attempted"] += 1
                    if (review.id, variation_index) in existing_variants:
                        summary["existing_skipped"] += 1
                        continue
                    try:
                        example = augment_review(
                            review,
                            variation_index=variation_index,
                            seed=args.seed,
                            min_changes=args.min_changes,
                        )
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        summary["failed"] += 1
                        if len(previews) < 10:
                            previews.append(
                                {
                                    "source_review_id": review.id,
                                    "variation_index": variation_index,
                                    "error": str(exc),
                                }
                            )
                        continue
                    summary["generated"] += 1
                    for change in example.changes:
                        change_counts[change["path"].rsplit(".", 1)[-1]] += 1
                    if len(previews) < 5:
                        previews.append(
                            {
                                "source_review_id": review.id,
                                "variation_index": variation_index,
                                "changes": len(example.changes),
                                "record_hash": example.record_hash,
                            }
                        )
                    if args.execute:
                        session.add(
                            StudentTrainingExample(
                                id=str(uuid4()),
                                source_review_id=example.source_review_id,
                                source_job_id=example.source_job_id,
                                source_case_id=example.source_case_id,
                                source_document_id=example.source_document_id,
                                input_text=example.input_text,
                                target_json=canonical_json(example.target),
                                target_schema_version=TEACHER_SCHEMA_VERSION,
                                review_status="APPROVED",
                                is_synthetic=True,
                                is_active=True,
                                augmentation_version=AUGMENTATION_VERSION,
                                seed=example.seed,
                                variation_index=example.variation_index,
                                changes_json=canonical_json(example.changes),
                                record_hash=example.record_hash,
                                generated_by=GENERATED_BY,
                            )
                        )
                        summary["inserted"] += 1
            if not args.execute:
                session.rollback()
    finally:
        database.close()

    result: dict[str, Any] = {
        **summary,
        "augmentation_version": AUGMENTATION_VERSION,
        "seed": args.seed,
        "multiplier": args.multiplier,
        "review_status": "APPROVED",
        "is_synthetic": True,
        "changed_fields": dict(change_counts),
        "previews": previews,
        "executed": args.execute,
    }
    print("Synthetic augmentation:", "EXECUTED" if args.execute else "DRY RUN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        print("No database rows were changed. Add --execute after reviewing this report.")
    return result


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        raise SystemExit(f"Synthetic augmentation could not start: {exc}") from exc


if __name__ == "__main__":
    main()
