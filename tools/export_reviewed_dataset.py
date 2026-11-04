from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from sqlalchemy import select

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import AnnotationDocument, AnnotationPage
from app.schemas.annotation import AnnotationLabel, ReviewStatus
from tools.document_classifier_config import DocumentClassifierSettings


@dataclass(frozen=True)
class SourceRecord:
    document_id: str
    page_id: str
    page_number: int
    original_filename: str
    image_relative_path: str
    label: str
    reviewed_by: str | None
    reviewed_at: str | None
    width: int
    height: int


@dataclass(frozen=True)
class ExportRecord:
    image_path: str
    label: str
    split: str
    document_id: str
    page_id: str
    page_number: int
    original_filename: str
    reviewed_by: str | None
    reviewed_at: str | None
    width: int
    height: int
    sha256: str


MANIFEST_FIELDS = [
    "image_path",
    "label",
    "split",
    "document_id",
    "page_id",
    "page_number",
    "original_filename",
    "reviewed_by",
    "reviewed_at",
    "width",
    "height",
    "sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reviewed annotation pages into a portable classifier dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override DATASET_EXPORT_DIR. The directory must be empty or not exist.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def load_reviewed_records(database: Database) -> list[SourceRecord]:
    with database.session() as session:
        rows = session.execute(
            select(AnnotationDocument, AnnotationPage)
            .join(AnnotationPage, AnnotationPage.document_id == AnnotationDocument.id)
            .where(
                AnnotationPage.review_status == ReviewStatus.REVIEWED.value,
                AnnotationPage.label.is_not(None),
            )
            .order_by(
                AnnotationDocument.created_at.asc(),
                AnnotationDocument.id.asc(),
                AnnotationPage.page_index.asc(),
            )
        ).all()

        return [
            SourceRecord(
                document_id=document.id,
                page_id=page.id,
                page_number=page.page_number,
                original_filename=document.original_filename,
                image_relative_path=page.image_relative_path,
                label=str(page.label),
                reviewed_by=page.reviewed_by,
                reviewed_at=page.reviewed_at.isoformat() if page.reviewed_at else None,
                width=page.width,
                height=page.height,
            )
            for document, page in rows
        ]


def split_document_ids(
    document_ids: list[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    shuffled = sorted(set(document_ids))
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count == 0:
        return {}

    val_count = round(count * val_ratio)
    test_count = round(count * test_ratio)
    if val_ratio > 0 and count >= 3:
        val_count = max(1, val_count)
    if test_ratio > 0 and count >= 3:
        test_count = max(1, test_count)
    while val_count + test_count >= count:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    # train_ratio participates in configuration validation; train receives the remainder.
    _ = train_ratio
    split_by_document: dict[str, str] = {}
    for index, document_id in enumerate(shuffled):
        if index < test_count:
            split = "test"
        elif index < test_count + val_count:
            split = "val"
        else:
            split = "train"
        split_by_document[document_id] = split
    return split_by_document


def ensure_training_label_coverage(
    records: list[tuple[SourceRecord, Path, str]],
    split_by_document: dict[str, str],
) -> dict[str, str]:
    """Ensure every observed label has at least one source document in train."""

    labels_by_document: dict[str, set[str]] = defaultdict(set)
    for record, _, _ in records:
        labels_by_document[record.document_id].add(record.label)

    observed_labels = set().union(*labels_by_document.values())
    train_labels: set[str] = set()
    for document_id, document_labels in labels_by_document.items():
        if split_by_document[document_id] == "train":
            train_labels.update(document_labels)
    missing_labels = observed_labels - train_labels
    while missing_labels:
        candidates = [
            document_id
            for document_id, labels in labels_by_document.items()
            if split_by_document[document_id] != "train" and labels & missing_labels
        ]
        if not candidates:
            raise RuntimeError(
                "Could not place every observed label in the training split: "
                + ", ".join(sorted(missing_labels))
            )
        candidates.sort(
            key=lambda document_id: (
                -len(labels_by_document[document_id] & missing_labels),
                document_id,
            )
        )
        selected_document = candidates[0]
        split_by_document[selected_document] = "train"
        train_labels.update(labels_by_document[selected_document])
        missing_labels = observed_labels - train_labels
    return split_by_document


def copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def write_csv(path: Path, records: list[ExportRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def export_dataset(
    *,
    output_dir: Path,
    config: DocumentClassifierSettings,
) -> None:
    app_settings = get_settings()
    if app_settings.db_name.lower() != config.dataset_source_database.strip().lower():
        raise RuntimeError(
            "Classifier dataset source database mismatch: "
            f"expected {config.dataset_source_database!r}, "
            f"configured DB_NAME is {app_settings.db_name!r}."
        )
    storage_root = app_settings.annotation_storage_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"Export path exists but is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Export directory is not empty: {output_dir}. "
            "Use a new directory to avoid overwriting an existing dataset."
        )

    database = Database(app_settings)
    try:
        source_records = load_reviewed_records(database)
    finally:
        database.close()

    if not source_records:
        raise RuntimeError("No REVIEWED pages with a non-null label were found.")

    supported_labels = {label.value for label in AnnotationLabel}
    invalid: list[str] = []
    unique_records: list[tuple[SourceRecord, Path, str]] = []
    seen_hashes: dict[str, SourceRecord] = {}
    duplicate_count = 0

    for record in source_records:
        if record.label not in supported_labels:
            invalid.append(f"{record.page_id}: unsupported label {record.label!r}")
            continue

        source_path = (storage_root / record.image_relative_path).resolve()
        if not source_path.is_relative_to(storage_root):
            invalid.append(f"{record.page_id}: image path escapes ANNOTATION_STORAGE_DIR")
            continue
        if not source_path.is_file():
            invalid.append(f"{record.page_id}: image is missing: {record.image_relative_path}")
            continue

        try:
            validate_image(source_path)
            image_hash = sha256_file(source_path)
        except (OSError, ValueError) as error:
            invalid.append(f"{record.page_id}: invalid image ({error})")
            continue

        previous = seen_hashes.get(image_hash)
        if previous:
            if previous.label != record.label:
                raise RuntimeError(
                    "The same image was reviewed with conflicting labels: "
                    f"{previous.page_id}={previous.label}, {record.page_id}={record.label}."
                )
            duplicate_count += 1
            continue

        seen_hashes[image_hash] = record
        unique_records.append((record, source_path, image_hash))

    if invalid and config.dataset_fail_on_invalid:
        preview = "\n".join(f"- {item}" for item in invalid[:20])
        remainder = len(invalid) - 20
        suffix = f"\n- ... and {remainder} more" if remainder > 0 else ""
        raise RuntimeError(f"Dataset validation failed:\n{preview}{suffix}")
    if not unique_records:
        raise RuntimeError("No valid unique images remain after validation.")

    split_by_document = split_document_ids(
        [record.document_id for record, _, _ in unique_records],
        train_ratio=config.dataset_train_ratio,
        val_ratio=config.dataset_val_ratio,
        test_ratio=config.dataset_test_ratio,
        seed=config.dataset_split_seed,
    )
    split_by_document = ensure_training_label_coverage(
        unique_records,
        split_by_document,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[ExportRecord] = []
    for record, source_path, image_hash in unique_records:
        split = split_by_document[record.document_id]
        suffix = source_path.suffix.lower() or ".png"
        filename = f"{record.document_id}_p{record.page_number:04d}_{record.page_id}{suffix}"
        relative_output = Path("images") / record.label / filename
        copy_file(source_path, output_dir / relative_output, config.dataset_copy_mode)
        exported.append(
            ExportRecord(
                image_path=relative_output.as_posix(),
                label=record.label,
                split=split,
                document_id=record.document_id,
                page_id=record.page_id,
                page_number=record.page_number,
                original_filename=record.original_filename,
                reviewed_by=record.reviewed_by,
                reviewed_at=record.reviewed_at,
                width=record.width,
                height=record.height,
                sha256=image_hash,
            )
        )

    exported.sort(key=lambda item: (item.split, item.label, item.document_id, item.page_number))
    write_csv(output_dir / "manifest.csv", exported)
    for split in ("train", "val", "test"):
        write_csv(output_dir / f"{split}.csv", [item for item in exported if item.split == split])

    labels = [label.value for label in AnnotationLabel]
    (output_dir / "labels.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "class_to_index": {label: index for index, label in enumerate(labels)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    by_split = Counter(item.split for item in exported)
    by_label = Counter(item.label for item in exported)
    by_split_label: dict[str, Counter[str]] = defaultdict(Counter)
    for item in exported:
        by_split_label[item.split][item.label] += 1
    statistics = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": config.dataset_version,
        "source_database": app_settings.db_name,
        "selection": {
            "page_review_status": ReviewStatus.REVIEWED.value,
            "label_required": True,
            "model_predictions_used": False,
        },
        "total_database_rows": len(source_records),
        "exported_unique_images": len(exported),
        "skipped_exact_duplicates": duplicate_count,
        "skipped_invalid_images": len(invalid),
        "document_count": len(split_by_document),
        "split_seed": config.dataset_split_seed,
        "configured_ratios": {
            "train": config.dataset_train_ratio,
            "val": config.dataset_val_ratio,
            "test": config.dataset_test_ratio,
        },
        "counts_by_split": dict(sorted(by_split.items())),
        "counts_by_label": {label: by_label.get(label, 0) for label in labels},
        "counts_by_split_and_label": {
            split: {label: by_split_label[split].get(label, 0) for label in labels}
            for split in ("train", "val", "test")
        },
        "labels_missing_from_validation": [
            label for label in labels if by_split_label["val"].get(label, 0) == 0
        ],
        "labels_missing_from_test": [
            label for label in labels if by_split_label["test"].get(label, 0) == 0
        ],
        "invalid_items": invalid,
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Exported {len(exported)} reviewed images to {output_dir}")
    print(f"Split counts: {dict(sorted(by_split.items()))}")


def main() -> None:
    args = parse_args()
    config = DocumentClassifierSettings()
    output_dir = (
        DocumentClassifierSettings.resolve_path(args.output_dir)
        if args.output_dir is not None
        else config.dataset_export_path
    )
    export_dataset(output_dir=output_dir, config=config)


if __name__ == "__main__":
    main()
