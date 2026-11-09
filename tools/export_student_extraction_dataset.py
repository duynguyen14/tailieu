from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import inspect, select, true
from sqlalchemy.orm import joinedload

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import (
    StudentTrainingExample,
    TeacherExtractionJob,
    TeacherExtractionReview,
)
from app.schemas.student_extraction import (
    STUDENT_MODEL_ID,
    STUDENT_SCHEMA_VERSION,
    build_student_user_prompt,
    compact_document_to_student_target,
    student_schema_descriptor,
    student_target_to_compact_document,
)
from app.schemas.teacher_extraction import (
    TEACHER_SCHEMA_VERSION,
    CompactDocumentExtraction,
)

DEFAULT_OUTPUT_DIR = Path("data/training/gemma3_270m_shortkey_v8")


@dataclass(frozen=True)
class ExportExample:
    review_id: str
    job_id: str
    case_id: str
    document_id: str
    original_filename: str
    input_hash: str
    reviewed_by: str | None
    reviewed_at: str | None
    input_text: str
    target: dict[str, Any]
    split: str
    record_hash: str
    is_synthetic: bool = False
    source_review_id: str | None = None
    augmentation_version: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export active approved real reviews and approved local synthetic "
            "examples as Gemma 3 270M text-to-compact-JSON SFT data."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument(
        "--exclude-synthetic",
        action="store_true",
        help="Export only real human-reviewed examples.",
    )
    return parser.parse_args()


def split_name(document_id: str) -> str:
    bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def record_hash(input_text: str, target: dict[str, Any]) -> str:
    material = {
        "schema_version": STUDENT_SCHEMA_VERSION,
        "input_text": input_text,
        "target": target,
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def normalize_teacher_payload_for_student(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove graph-only identities before validating/exporting student data.

    The compact production student never predicts person IDs or relationship
    links.  Teacher reviews can therefore contain stale links after a reviewer
    removes an applicant.  Normalize those internal fields deterministically;
    the human-reviewed business fields remain unchanged.
    """

    normalized = copy.deepcopy(payload)
    applicants = normalized.get("applicants")
    if isinstance(applicants, list):
        for index, applicant in enumerate(applicants, start=1):
            if not isinstance(applicant, dict):
                continue
            applicant["person_id"] = f"person_{index}"
            applicant["sequence_number"] = index
            applicant["related_to_person_id"] = None
    return normalized


def build_example(review: TeacherExtractionReview) -> ExportExample:
    job = review.job
    try:
        raw_target = json.loads(review.reviewed_output_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Review {review.id} contains invalid JSON: {exc}") from exc
    extraction = CompactDocumentExtraction.model_validate(
        normalize_teacher_payload_for_student(raw_target)
    )
    if extraction.case_id != job.case_id:
        raise ValueError(
            f"Review {review.id} case_id does not match job {job.id}."
        )
    if extraction.document_id != job.document_id:
        raise ValueError(
            f"Review {review.id} document_id does not match job {job.id}."
        )
    target = compact_document_to_student_target(extraction)
    # A round trip catches positional mapping mistakes before data reaches training.
    student_target_to_compact_document(
        target,
        case_id=job.case_id,
        document_id=job.document_id,
    )
    input_text = job.input_text.strip()
    if not input_text:
        raise ValueError(f"Teacher job {job.id} has empty input text.")
    return ExportExample(
        review_id=review.id,
        job_id=job.id,
        case_id=job.case_id,
        document_id=job.document_id,
        original_filename=job.document.original_filename,
        input_hash=job.input_hash,
        reviewed_by=review.reviewed_by,
        reviewed_at=review.reviewed_at.isoformat() if review.reviewed_at else None,
        input_text=input_text,
        target=target,
        split=split_name(job.document_id),
        record_hash=record_hash(input_text, target),
    )


def build_synthetic_example(row: StudentTrainingExample) -> ExportExample:
    try:
        raw_target = json.loads(row.target_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Synthetic example {row.id} contains invalid JSON: {exc}") from exc
    extraction = CompactDocumentExtraction.model_validate(
        normalize_teacher_payload_for_student(raw_target)
    )
    if extraction.case_id != row.source_case_id:
        raise ValueError(f"Synthetic example {row.id} has a mismatched case_id.")
    if extraction.document_id != row.source_document_id:
        raise ValueError(f"Synthetic example {row.id} has a mismatched document_id.")
    if split_name(row.source_document_id) != "train":
        raise ValueError(
            f"Synthetic example {row.id} was derived from a non-train document."
        )
    target = compact_document_to_student_target(extraction)
    student_target_to_compact_document(
        target,
        case_id=row.source_case_id,
        document_id=row.source_document_id,
    )
    input_text = row.input_text.strip()
    if not input_text:
        raise ValueError(f"Synthetic example {row.id} has empty input text.")
    return ExportExample(
        review_id=row.id,
        job_id=row.source_job_id,
        case_id=row.source_case_id,
        document_id=row.source_document_id,
        original_filename=row.source_document.original_filename,
        input_hash=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        reviewed_by=row.generated_by,
        reviewed_at=row.created_at.isoformat(),
        input_text=input_text,
        target=target,
        split="train",
        record_hash=record_hash(input_text, target),
        is_synthetic=True,
        source_review_id=row.source_review_id,
        augmentation_version=row.augmentation_version,
    )


def training_record(example: ExportExample) -> dict[str, Any]:
    return {
        "id": example.review_id,
        "case_id": example.case_id,
        "document_id": example.document_id,
        "schema_version": STUDENT_SCHEMA_VERSION,
        "messages": [
            {
                "role": "user",
                "content": build_student_user_prompt(example.input_text),
            },
            {
                "role": "assistant",
                "content": canonical_json(example.target),
            },
        ],
    }


def manifest_record(example: ExportExample) -> dict[str, Any]:
    return {
        "id": example.review_id,
        "job_id": example.job_id,
        "case_id": example.case_id,
        "document_id": example.document_id,
        "original_filename": example.original_filename,
        "input_hash": example.input_hash,
        "record_hash": example.record_hash,
        "split": example.split,
        "reviewed_by": example.reviewed_by,
        "reviewed_at": example.reviewed_at,
        "applicant_count": len(example.target["a"]),
        "input_characters": len(example.input_text),
        "target_characters": len(canonical_json(example.target)),
        "is_synthetic": example.is_synthetic,
        "source_review_id": example.source_review_id,
        "augmentation_version": example.augmentation_version,
    }


def resolve_output_dir(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_dataset(
    examples: list[ExportExample],
    *,
    output_dir: Path,
    invalid_items: list[dict[str, str]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[ExportExample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for example in examples:
        by_split[example.split].append(example)

    for split, split_examples in by_split.items():
        write_jsonl(
            output_dir / f"{split}.jsonl",
            [training_record(example) for example in split_examples],
        )
    write_jsonl(
        output_dir / "manifest.jsonl",
        [manifest_record(example) for example in examples],
    )
    (output_dir / "schema.json").write_text(
        json.dumps(student_schema_descriptor(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    applicant_distribution = Counter(
        len(example.target["a"]) for example in examples
    )
    fingerprint = hashlib.sha256(
        "".join(sorted(example.record_hash for example in examples)).encode("ascii")
    ).hexdigest()
    statistics = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": STUDENT_MODEL_ID,
        "schema_version": STUDENT_SCHEMA_VERSION,
        "dataset_fingerprint": fingerprint,
        "total_examples": len(examples),
        "counts_by_split": {
            split: len(split_examples) for split, split_examples in by_split.items()
        },
        "document_count": len({example.document_id for example in examples}),
        "case_count": len({example.case_id for example in examples}),
        "synthetic_count": sum(example.is_synthetic for example in examples),
        "real_count": sum(not example.is_synthetic for example in examples),
        "applicant_count_distribution": {
            str(count): total for count, total in sorted(applicant_distribution.items())
        },
        "average_input_characters": (
            round(sum(len(example.input_text) for example in examples) / len(examples), 2)
            if examples
            else 0
        ),
        "average_target_characters": (
            round(
                sum(len(canonical_json(example.target)) for example in examples)
                / len(examples),
                2,
            )
            if examples
            else 0
        ),
        "invalid_count": len(invalid_items),
        "invalid_items": invalid_items,
    }
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return statistics


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater.")
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError(
            "Student extraction export must run with DB_NAME=anotation_2."
        )
    database = Database(settings)
    examples: list[ExportExample] = []
    invalid_items: list[dict[str, str]] = []
    duplicate_hashes = 0
    seen_record_hashes: set[str] = set()
    try:
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
        if args.limit > 0:
            statement = statement.limit(args.limit)
        with database.session() as session:
            reviews = list(session.scalars(statement).unique().all())
            for review in reviews:
                try:
                    example = build_example(review)
                except (ValueError, ValidationError, TypeError) as exc:
                    invalid_items.append(
                        {"review_id": review.id, "error": str(exc)}
                    )
                    if args.fail_on_invalid:
                        raise RuntimeError(
                            f"Invalid approved teacher review {review.id}: {exc}"
                        ) from exc
                    continue
                if example.record_hash in seen_record_hashes:
                    duplicate_hashes += 1
                    continue
                seen_record_hashes.add(example.record_hash)
                examples.append(example)

            has_synthetic_table = inspect(database.engine).has_table(
                StudentTrainingExample.__tablename__
            )
            if not args.exclude_synthetic and has_synthetic_table:
                synthetic_statement = (
                    select(StudentTrainingExample)
                    .options(joinedload(StudentTrainingExample.source_document))
                    .where(
                        StudentTrainingExample.is_active == true(),
                        StudentTrainingExample.is_synthetic == true(),
                        StudentTrainingExample.review_status == "APPROVED",
                        StudentTrainingExample.target_schema_version
                        == TEACHER_SCHEMA_VERSION,
                    )
                    .order_by(
                        StudentTrainingExample.source_document_id,
                        StudentTrainingExample.source_review_id,
                        StudentTrainingExample.variation_index,
                    )
                )
                synthetic_rows = list(session.scalars(synthetic_statement).unique())
                for row in synthetic_rows:
                    try:
                        example = build_synthetic_example(row)
                    except (ValueError, ValidationError, TypeError) as exc:
                        invalid_items.append(
                            {"review_id": row.id, "error": str(exc)}
                        )
                        if args.fail_on_invalid:
                            raise RuntimeError(
                                f"Invalid approved synthetic example {row.id}: {exc}"
                            ) from exc
                        continue
                    if example.record_hash in seen_record_hashes:
                        duplicate_hashes += 1
                        continue
                    seen_record_hashes.add(example.record_hash)
                    examples.append(example)
    finally:
        database.close()

    counts = Counter(example.split for example in examples)
    summary: dict[str, Any] = {
        "discovered": len(examples) + len(invalid_items) + duplicate_hashes,
        "exported": len(examples),
        "duplicates": duplicate_hashes,
        "invalid": len(invalid_items),
        "counts_by_split": {
            split: counts.get(split, 0) for split in ("train", "val", "test")
        },
        "real": sum(not example.is_synthetic for example in examples),
        "synthetic": sum(example.is_synthetic for example in examples),
    }
    if args.dry_run:
        print("Student extraction export: DRY RUN")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("No files were written.")
        return summary

    output_dir = resolve_output_dir(args.output_dir)
    statistics = export_dataset(
        examples,
        output_dir=output_dir,
        invalid_items=invalid_items,
    )
    summary["output_dir"] = str(output_dir)
    summary["dataset_fingerprint"] = statistics["dataset_fingerprint"]
    print(
        f"Exported {len(examples)} approved teacher examples to {output_dir}; "
        f"train={counts.get('train', 0)}, val={counts.get('val', 0)}, "
        f"test={counts.get('test', 0)}, duplicates={duplicate_hashes}, "
        f"invalid={len(invalid_items)}"
    )
    return summary


def main() -> None:
    try:
        run(parse_args())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Student extraction export could not start: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
