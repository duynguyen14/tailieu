from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from sqlalchemy import and_, func, or_, select, true
from sqlalchemy.orm import aliased

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    OcrReview,
    OcrReviewToken,
    OcrRun,
    OcrToken,
)
from tools.ocr_recognition_dataset_config import OcrRecognitionDatasetSettings


@dataclass(frozen=True)
class SourceToken:
    row_id: str
    row_created_at: datetime
    document_id: str
    page_id: str
    page_number: int
    original_filename: str
    review_id: str
    token_order: int
    source_token_id: str | None
    reviewed_text: str
    source_text: str | None
    x1: int
    y1: int
    x2: int
    y2: int
    source_x1: int | None
    source_y1: int | None
    source_x2: int | None
    source_y2: int | None
    source_confidence: float | None
    image_relative_path: str
    expected_width: int
    expected_height: int
    model_name: str
    model_version: str
    reviewed_by: str
    reviewed_at: str


@dataclass(frozen=True)
class ExportToken:
    image_path: str
    text: str
    split: str
    document_id: str
    page_id: str
    page_number: int
    original_filename: str
    review_id: str
    token_order: int
    source_token_id: str | None
    model_name: str
    model_version: str
    reviewed_by: str
    reviewed_at: str
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    crop_width: int
    crop_height: int
    source_confidence: float | None
    is_text_corrected: bool
    is_bbox_corrected: bool
    is_manually_added: bool
    sha256: str


MANIFEST_FIELDS = [field.name for field in ExportToken.__dataclass_fields__.values()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export active human-approved OCR review tokens as a PaddleOCR "
            "recognition dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override OCR_REC_DATASET_EXPORT_DIR. The target must be empty or "
            "must not exist."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate DB rows, source images, reviewed text and bbox crops, then "
            "print statistics without creating a dataset directory."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of reviewed OCR tokens fetched and processed per DB batch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted non-dry-run export from its checkpoint.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    """Make one safe PaddleOCR label line without changing visible characters."""

    return re.sub(r"\s+", " ", value).strip()


def split_document_ids(
    document_ids: list[str],
    *,
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

    result: dict[str, str] = {}
    for index, document_id in enumerate(shuffled):
        if index < test_count:
            split = "test"
        elif index < test_count + val_count:
            split = "val"
        else:
            split = "train"
        result[document_id] = split
    return result


def padded_crop_box(
    bbox: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox}")
    if x2 > image_width or y2 > image_height:
        raise ValueError(
            f"BBox {bbox} exceeds image size {image_width}x{image_height}."
        )
    # Padding is based on line height. Using line width would add excessive
    # whitespace to the long diplomatic-note lines in this dataset.
    padding = max(2, round((y2 - y1) * padding_ratio))
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width, x2 + padding),
        min(image_height, y2 + padding),
    )


def _within_snapshot(created_at, row_id, snapshot_created_at: datetime, snapshot_row_id: str):
    return or_(
        created_at < snapshot_created_at,
        and_(created_at == snapshot_created_at, row_id <= snapshot_row_id),
    )


def load_snapshot_cursor(database: Database) -> tuple[datetime, str] | None:
    with database.session() as session:
        row = session.execute(
            select(OcrReviewToken.created_at, OcrReviewToken.id)
            .join(OcrReview, OcrReview.id == OcrReviewToken.review_id)
            .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
            .where(
                OcrRun.status == "COMPLETED",
                OcrReview.status == "REVIEWED",
                OcrReview.is_active == true(),
                OcrReview.include_in_llm == true(),
                OcrReviewToken.include_in_extraction == true(),
            )
            .order_by(OcrReviewToken.created_at.desc(), OcrReviewToken.id.desc())
            .limit(1)
        ).first()
    return (row[0], row[1]) if row is not None else None


def load_document_ids(
    database: Database,
    *,
    snapshot_created_at: datetime,
    snapshot_row_id: str,
) -> list[str]:
    with database.session() as session:
        return list(
            session.scalars(
                select(AnnotationDocument.id)
                .join(AnnotationPage, AnnotationPage.document_id == AnnotationDocument.id)
                .join(OcrRun, OcrRun.page_id == AnnotationPage.id)
                .join(OcrReview, OcrReview.ocr_run_id == OcrRun.id)
                .join(OcrReviewToken, OcrReviewToken.review_id == OcrReview.id)
                .where(
                    OcrRun.status == "COMPLETED",
                    OcrReview.status == "REVIEWED",
                    OcrReview.is_active == true(),
                    OcrReview.include_in_llm == true(),
                    OcrReviewToken.include_in_extraction == true(),
                    _within_snapshot(
                        OcrReviewToken.created_at,
                        OcrReviewToken.id,
                        snapshot_created_at,
                        snapshot_row_id,
                    ),
                )
                .distinct()
                .order_by(AnnotationDocument.id.asc())
            ).all()
        )


def load_source_token_batch(
    database: Database,
    *,
    after_created_at: datetime | None,
    after_row_id: str | None,
    snapshot_created_at: datetime,
    snapshot_row_id: str,
    batch_size: int,
) -> list[SourceToken]:
    source = aliased(OcrToken)
    with database.session() as session:
        statement = (
            select(
                OcrReviewToken.id,
                OcrReviewToken.created_at,
                AnnotationDocument.id,
                AnnotationPage.id,
                AnnotationPage.page_number,
                AnnotationDocument.original_filename,
                OcrReview.id,
                OcrReviewToken.token_order,
                OcrReviewToken.source_token_id,
                OcrReviewToken.reviewed_text,
                source.text,
                OcrReviewToken.x1,
                OcrReviewToken.y1,
                OcrReviewToken.x2,
                OcrReviewToken.y2,
                source.x1,
                source.y1,
                source.x2,
                source.y2,
                source.confidence,
                OcrRun.processed_image_relative_path,
                AnnotationPage.image_relative_path,
                OcrRun.processed_width,
                OcrRun.processed_height,
                AnnotationPage.width,
                AnnotationPage.height,
                OcrRun.model_name,
                OcrRun.model_version,
                OcrReview.reviewed_by,
                OcrReview.reviewed_at,
            )
            .join(AnnotationPage, AnnotationPage.document_id == AnnotationDocument.id)
            .join(OcrRun, OcrRun.page_id == AnnotationPage.id)
            .join(OcrReview, OcrReview.ocr_run_id == OcrRun.id)
            .join(OcrReviewToken, OcrReviewToken.review_id == OcrReview.id)
            .outerjoin(source, source.id == OcrReviewToken.source_token_id)
            .where(
                OcrRun.status == "COMPLETED",
                OcrReview.status == "REVIEWED",
                OcrReview.is_active == true(),
                OcrReview.include_in_llm == true(),
                OcrReviewToken.include_in_extraction == true(),
                _within_snapshot(
                    OcrReviewToken.created_at,
                    OcrReviewToken.id,
                    snapshot_created_at,
                    snapshot_row_id,
                ),
            )
            .order_by(OcrReviewToken.created_at.asc(), OcrReviewToken.id.asc())
            .limit(batch_size)
        )
        if after_created_at is not None and after_row_id is not None:
            statement = statement.where(
                or_(
                    OcrReviewToken.created_at > after_created_at,
                    and_(
                        OcrReviewToken.created_at == after_created_at,
                        OcrReviewToken.id > after_row_id,
                    ),
                )
            )
        rows = session.execute(statement).all()

    records: list[SourceToken] = []
    for row in rows:
        (
            row_id,
            row_created_at,
            document_id,
            page_id,
            page_number,
            original_filename,
            review_id,
            token_order,
            source_token_id,
            reviewed_text,
            source_text,
            x1,
            y1,
            x2,
            y2,
            source_x1,
            source_y1,
            source_x2,
            source_y2,
            source_confidence,
            processed_image_relative_path,
            page_image_relative_path,
            processed_width,
            processed_height,
            page_width,
            page_height,
            model_name,
            model_version,
            reviewed_by,
            reviewed_at,
        ) = row
        records.append(
            SourceToken(
                row_id=row_id,
                row_created_at=row_created_at,
                document_id=document_id,
                page_id=page_id,
                page_number=page_number,
                original_filename=original_filename,
                review_id=review_id,
                token_order=token_order,
                source_token_id=source_token_id,
                reviewed_text=reviewed_text,
                source_text=source_text,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                source_x1=source_x1,
                source_y1=source_y1,
                source_x2=source_x2,
                source_y2=source_y2,
                source_confidence=(
                    float(source_confidence) if source_confidence is not None else None
                ),
                image_relative_path=(
                    processed_image_relative_path or page_image_relative_path
                ),
                expected_width=processed_width or page_width,
                expected_height=processed_height or page_height,
                model_name=model_name,
                model_version=model_version,
                reviewed_by=reviewed_by,
                reviewed_at=reviewed_at.isoformat(),
            )
        )
    return records


def write_manifest(path: Path, records: list[ExportToken]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def write_paddle_labels(path: Path, records: list[ExportToken]) -> None:
    content = "".join(f"{record.image_path}\t{record.text}\n" for record in records)
    path.write_text(content, encoding="utf-8", newline="\n")


def _recommended_max_text_length(max_length: int) -> int:
    # One extra block gives the production model room for slightly longer
    # future lines without silently truncating the current maximum.
    return max(32, math.ceil((max_length + 16) / 16) * 16)


def _checkpoint_signature(config: OcrRecognitionDatasetSettings, database_name: str) -> dict:
    return {
        "source_database": database_name,
        "dataset_version": config.ocr_rec_dataset_version,
        "split_seed": config.ocr_rec_dataset_split_seed,
        "train_ratio": config.ocr_rec_dataset_train_ratio,
        "val_ratio": config.ocr_rec_dataset_val_ratio,
        "test_ratio": config.ocr_rec_dataset_test_ratio,
        "padding_ratio": config.ocr_rec_dataset_padding_ratio,
        "deduplicate": config.ocr_rec_dataset_deduplicate,
    }


def _write_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_export_records(path: Path) -> list[ExportToken]:
    if not path.is_file():
        return []
    unique: dict[tuple[str, int], ExportToken] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = ExportToken(**json.loads(line))
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"Invalid export checkpoint record at line {line_number}: {error}"
                ) from error
            unique[(record.review_id, record.token_order)] = record
    records = list(unique.values())
    if len(records) != sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line):
        with path.open("w", encoding="utf-8", newline="\n") as target:
            for record in records:
                target.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return records


def _count_source_tokens(
    database: Database,
    *,
    snapshot_created_at: datetime,
    snapshot_row_id: str,
) -> int:
    with database.session() as session:
        return int(
            session.scalar(
                select(func.count(OcrReviewToken.id))
                .join(OcrReview, OcrReview.id == OcrReviewToken.review_id)
                .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
                .where(
                    OcrRun.status == "COMPLETED",
                    OcrReview.status == "REVIEWED",
                    OcrReview.is_active == true(),
                    OcrReview.include_in_llm == true(),
                    OcrReviewToken.include_in_extraction == true(),
                    _within_snapshot(
                        OcrReviewToken.created_at,
                        OcrReviewToken.id,
                        snapshot_created_at,
                        snapshot_row_id,
                    ),
                )
            )
            or 0
        )


def _process_source_batch(
    source_tokens: list[SourceToken],
    *,
    storage_root: Path,
    staging: Path | None,
    split_by_document: dict[str, str],
    padding_ratio: float,
    deduplicate: bool,
    seen_crop_hashes: dict[str, tuple[str, str]],
    processed_keys: set[tuple[str, int]],
) -> tuple[list[ExportToken], list[str], int]:
    invalid_items: list[str] = []
    exported: list[ExportToken] = []
    duplicate_count = 0
    grouped: dict[str, list[SourceToken]] = defaultdict(list)
    for record in source_tokens:
        if (record.review_id, record.token_order) not in processed_keys:
            grouped[record.image_relative_path].append(record)

    for relative_path in sorted(grouped):
        image_path = (storage_root / relative_path).resolve()
        if not image_path.is_relative_to(storage_root):
            invalid_items.append(f"Image path escapes storage: {relative_path}")
            continue
        if not image_path.is_file():
            invalid_items.append(f"Image is missing: {relative_path}")
            continue
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except OSError as error:
            invalid_items.append(f"Cannot open {relative_path}: {error}")
            continue

        try:
            expected_sizes = {
                (record.expected_width, record.expected_height)
                for record in grouped[relative_path]
            }
            if len(expected_sizes) != 1 or (image.width, image.height) not in expected_sizes:
                invalid_items.append(
                    f"Image size mismatch for {relative_path}: actual "
                    f"{image.width}x{image.height}, expected {sorted(expected_sizes)}"
                )
                continue
            for record in grouped[relative_path]:
                text = normalize_text(record.reviewed_text)
                if not text:
                    invalid_items.append(
                        f"{record.review_id}/{record.token_order}: empty reviewed text"
                    )
                    continue
                try:
                    crop_box = padded_crop_box(
                        (record.x1, record.y1, record.x2, record.y2),
                        image_width=image.width,
                        image_height=image.height,
                        padding_ratio=padding_ratio,
                    )
                except ValueError as error:
                    invalid_items.append(
                        f"{record.review_id}/{record.token_order}: {error}"
                    )
                    continue

                crop = image.crop(crop_box)
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")
                crop_bytes = buffer.getvalue()
                crop_hash = hashlib.sha256(crop_bytes).hexdigest()
                previous = seen_crop_hashes.get(crop_hash)
                if deduplicate and previous:
                    previous_text, previous_source = previous
                    if previous_text != text:
                        invalid_items.append(
                            "Conflicting labels for identical crop: "
                            f"{previous_source}={previous_text!r}, "
                            f"{record.review_id}/{record.token_order}={text!r}"
                        )
                    else:
                        duplicate_count += 1
                    crop.close()
                    continue

                seen_crop_hashes[crop_hash] = (
                    text,
                    f"{record.review_id}/{record.token_order}",
                )
                split = split_by_document[record.document_id]
                filename = (
                    f"{record.document_id}_p{record.page_number:04d}_"
                    f"{record.review_id}_t{record.token_order:04d}.png"
                )
                relative_crop_path = Path("crops") / split / filename
                if staging is not None:
                    destination = staging / relative_crop_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(crop_bytes)

                source_bbox = (
                    record.source_x1,
                    record.source_y1,
                    record.source_x2,
                    record.source_y2,
                )
                reviewed_bbox = (record.x1, record.y1, record.x2, record.y2)
                exported.append(
                    ExportToken(
                        image_path=relative_crop_path.as_posix(),
                        text=text,
                        split=split,
                        document_id=record.document_id,
                        page_id=record.page_id,
                        page_number=record.page_number,
                        original_filename=record.original_filename,
                        review_id=record.review_id,
                        token_order=record.token_order,
                        source_token_id=record.source_token_id,
                        model_name=record.model_name,
                        model_version=record.model_version,
                        reviewed_by=record.reviewed_by,
                        reviewed_at=record.reviewed_at,
                        bbox_x1=record.x1,
                        bbox_y1=record.y1,
                        bbox_x2=record.x2,
                        bbox_y2=record.y2,
                        crop_x1=crop_box[0],
                        crop_y1=crop_box[1],
                        crop_x2=crop_box[2],
                        crop_y2=crop_box[3],
                        crop_width=crop.width,
                        crop_height=crop.height,
                        source_confidence=record.source_confidence,
                        is_text_corrected=(
                            record.source_text is not None
                            and text != normalize_text(record.source_text)
                        ),
                        is_bbox_corrected=(
                            record.source_token_id is not None
                            and source_bbox != reviewed_bbox
                        ),
                        is_manually_added=record.source_token_id is None,
                        sha256=crop_hash,
                    )
                )
                crop.close()
        finally:
            image.close()
    return exported, invalid_items, duplicate_count


def export_dataset(
    *,
    output_dir: Path,
    config: OcrRecognitionDatasetSettings,
    dry_run: bool = False,
    batch_size: int = 1000,
    resume: bool = False,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if dry_run and resume:
        raise ValueError("--resume cannot be combined with --dry-run")

    app_settings = get_settings()
    expected_database = config.ocr_rec_dataset_source_database.strip()
    if app_settings.db_name.lower() != expected_database.lower():
        raise RuntimeError(
            "OCR recognition dataset source database mismatch: "
            f"expected {expected_database!r}, configured DB_NAME is "
            f"{app_settings.db_name!r}."
        )

    output_dir = output_dir.resolve()
    work_dir = output_dir.parent / f".{output_dir.name}.ocr-export-work"
    checkpoint_path = work_dir / "checkpoint.json"
    records_path = work_dir / "records.jsonl"
    signature = _checkpoint_signature(config, app_settings.db_name)
    database = Database(app_settings)
    try:
        if resume:
            if not work_dir.is_dir() or not checkpoint_path.is_file():
                raise RuntimeError(
                    f"No resumable OCR export checkpoint exists for {output_dir}."
                )
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if state.get("signature") != signature:
                raise RuntimeError(
                    "Export checkpoint configuration differs from the current ENV. "
                    "Restore the original configuration or start a new output version."
                )
            snapshot_created_at = datetime.fromisoformat(state["snapshot_created_at"])
            snapshot_row_id = state["snapshot_row_id"]
            last_created_at = (
                datetime.fromisoformat(state["last_created_at"])
                if state.get("last_created_at")
                else None
            )
            last_row_id = state.get("last_row_id")
            split_by_document = state["split_by_document"]
            duplicate_count = int(state.get("duplicate_count", 0))
            invalid_items = list(state.get("invalid_items", []))
            exported = _load_export_records(records_path)
            print(
                f"Resuming OCR export after {state.get('processed_source_tokens', 0)} "
                f"source tokens; existing crops={len(exported)}"
            )
        else:
            if not dry_run:
                if output_dir.exists() and not output_dir.is_dir():
                    raise RuntimeError(
                        f"Export path exists but is not a directory: {output_dir}"
                    )
                if output_dir.exists() and any(output_dir.iterdir()):
                    raise RuntimeError(
                        f"Export directory is not empty: {output_dir}. "
                        "Use a new versioned directory to preserve existing datasets."
                    )
                if work_dir.exists():
                    raise RuntimeError(
                        f"Interrupted export checkpoint already exists: {work_dir}. "
                        "Use --resume or choose a new --output-dir."
                    )
            snapshot = load_snapshot_cursor(database)
            if snapshot is None:
                raise RuntimeError("No active approved OCR review tokens were found.")
            snapshot_created_at, snapshot_row_id = snapshot
            document_ids = load_document_ids(
                database,
                snapshot_created_at=snapshot_created_at,
                snapshot_row_id=snapshot_row_id,
            )
            split_by_document = split_document_ids(
                document_ids,
                val_ratio=config.ocr_rec_dataset_val_ratio,
                test_ratio=config.ocr_rec_dataset_test_ratio,
                seed=config.ocr_rec_dataset_split_seed,
            )
            last_created_at = None
            last_row_id = None
            duplicate_count = 0
            invalid_items: list[str] = []
            exported: list[ExportToken] = []
            if not dry_run:
                output_dir.parent.mkdir(parents=True, exist_ok=True)
                if output_dir.exists():
                    output_dir.rmdir()
                work_dir.mkdir(parents=True)

        source_review_tokens = _count_source_tokens(
            database,
            snapshot_created_at=snapshot_created_at,
            snapshot_row_id=snapshot_row_id,
        )
        storage_root = app_settings.annotation_storage_path.resolve()
        staging = None if dry_run else work_dir
        seen_crop_hashes = {
            record.sha256: (
                record.text,
                f"{record.review_id}/{record.token_order}",
            )
            for record in exported
        }
        processed_keys = {
            (record.review_id, record.token_order) for record in exported
        }
        processed_source_tokens = int(
            state.get("processed_source_tokens", 0) if resume else 0
        )
        batch_number = 0

        if staging is not None and not resume:
            _write_checkpoint(
                checkpoint_path,
                {
                    "signature": signature,
                    "snapshot_created_at": snapshot_created_at.isoformat(),
                    "snapshot_row_id": snapshot_row_id,
                    "last_created_at": None,
                    "last_row_id": None,
                    "processed_source_tokens": 0,
                    "duplicate_count": 0,
                    "invalid_items": [],
                    "split_by_document": split_by_document,
                },
            )

        while True:
            source_batch = load_source_token_batch(
                database,
                after_created_at=last_created_at,
                after_row_id=last_row_id,
                snapshot_created_at=snapshot_created_at,
                snapshot_row_id=snapshot_row_id,
                batch_size=batch_size,
            )
            if not source_batch:
                break
            batch_number += 1
            batch_exported, batch_invalid, batch_duplicates = _process_source_batch(
                source_batch,
                storage_root=storage_root,
                staging=staging,
                split_by_document=split_by_document,
                padding_ratio=config.ocr_rec_dataset_padding_ratio,
                deduplicate=config.ocr_rec_dataset_deduplicate,
                seen_crop_hashes=seen_crop_hashes,
                processed_keys=processed_keys,
            )
            exported.extend(batch_exported)
            processed_keys.update(
                (record.review_id, record.token_order) for record in batch_exported
            )
            invalid_items.extend(batch_invalid)
            duplicate_count += batch_duplicates
            processed_source_tokens += len(source_batch)
            last_created_at = source_batch[-1].row_created_at
            last_row_id = source_batch[-1].row_id

            if staging is not None:
                with records_path.open("a", encoding="utf-8", newline="\n") as target:
                    for record in batch_exported:
                        target.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                _write_checkpoint(
                    checkpoint_path,
                    {
                        "signature": signature,
                        "snapshot_created_at": snapshot_created_at.isoformat(),
                        "snapshot_row_id": snapshot_row_id,
                        "last_created_at": last_created_at.isoformat(),
                        "last_row_id": last_row_id,
                        "processed_source_tokens": processed_source_tokens,
                        "duplicate_count": duplicate_count,
                        "invalid_items": invalid_items,
                        "split_by_document": split_by_document,
                    },
                )
            print(
                f"Batch {batch_number:04d}: fetched={len(source_batch)}, "
                f"crops={len(batch_exported)}, duplicates={batch_duplicates}, "
                f"invalid={len(batch_invalid)}, progress="
                f"{min(processed_source_tokens, source_review_tokens)}/"
                f"{source_review_tokens}"
            )

        if invalid_items and config.ocr_rec_dataset_fail_on_invalid:
            preview = "\n".join(f"- {item}" for item in invalid_items[:30])
            remainder = len(invalid_items) - 30
            suffix = f"\n- ... and {remainder} more" if remainder > 0 else ""
            raise RuntimeError(f"OCR recognition dataset validation failed:\n{preview}{suffix}")
        if not exported:
            raise RuntimeError("No valid OCR recognition crops remain after validation.")

        exported.sort(
            key=lambda item: (
                item.split,
                item.document_id,
                item.page_number,
                item.token_order,
            )
        )
        if staging is not None:
            write_manifest(staging / "manifest.csv", exported)
            for split in ("train", "val", "test"):
                split_records = [record for record in exported if record.split == split]
                write_manifest(staging / f"{split}.csv", split_records)
                write_paddle_labels(staging / f"{split}.txt", split_records)

        characters = Counter(character for item in exported for character in item.text)
        character_inventory = [
            {"character": character, "codepoint": ord(character), "count": count}
            for character, count in sorted(characters.items(), key=lambda item: ord(item[0]))
        ]
        if staging is not None:
            (staging / "character_inventory.json").write_text(
                json.dumps(character_inventory, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        by_split = Counter(record.split for record in exported)
        documents_by_split: dict[str, set[str]] = defaultdict(set)
        pages_by_split: dict[str, set[str]] = defaultdict(set)
        by_model = Counter(
            f"{record.model_name}::{record.model_version}" for record in exported
        )
        for record in exported:
            documents_by_split[record.split].add(record.document_id)
            pages_by_split[record.split].add(record.page_id)
        lengths = [len(record.text) for record in exported]
        statistics: dict[str, object] = {
            "created_at": datetime.now(UTC).isoformat(),
            "dry_run": dry_run,
            "source_database": app_settings.db_name,
            "dataset_version": config.ocr_rec_dataset_version,
            "source_review_tokens": source_review_tokens,
            "exported_unique_crops": len(exported),
            "skipped_exact_duplicates": duplicate_count,
            "invalid_items_count": len(invalid_items),
            "invalid_items": invalid_items,
            "document_count": len({record.document_id for record in exported}),
            "page_count": len({record.page_id for record in exported}),
            "counts_by_split": dict(sorted(by_split.items())),
            "documents_by_split": {
                split: len(ids) for split, ids in sorted(documents_by_split.items())
            },
            "pages_by_split": {
                split: len(ids) for split, ids in sorted(pages_by_split.items())
            },
            "corrected_text_crops": sum(item.is_text_corrected for item in exported),
            "corrected_bbox_crops": sum(item.is_bbox_corrected for item in exported),
            "manually_added_crops": sum(item.is_manually_added for item in exported),
            "text_length": {
                "minimum": min(lengths),
                "maximum": max(lengths),
                "average": round(sum(lengths) / len(lengths), 2),
                "recommended_max_text_length": _recommended_max_text_length(
                    max(lengths)
                ),
            },
            "unique_character_count": len(characters),
            "counts_by_source_model": dict(sorted(by_model.items())),
            "configuration": {
                "split_seed": config.ocr_rec_dataset_split_seed,
                "train_ratio": config.ocr_rec_dataset_train_ratio,
                "val_ratio": config.ocr_rec_dataset_val_ratio,
                "test_ratio": config.ocr_rec_dataset_test_ratio,
                "padding_ratio": config.ocr_rec_dataset_padding_ratio,
                "deduplicate": config.ocr_rec_dataset_deduplicate,
                "batch_size": batch_size,
            },
        }
        if staging is not None:
            (staging / "dataset_statistics.json").write_text(
                json.dumps(statistics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            checkpoint_path.unlink(missing_ok=True)
            records_path.unlink(missing_ok=True)
            staging.replace(output_dir)
        return statistics
    finally:
        database.close()


def main() -> None:
    args = parse_args()
    config = OcrRecognitionDatasetSettings()
    if args.output_dir is None:
        resolved_output = config.export_path
    elif args.output_dir.is_absolute():
        resolved_output = args.output_dir.resolve()
    else:
        from app.core.config import PROJECT_ROOT

        resolved_output = (PROJECT_ROOT / args.output_dir).resolve()

    statistics = export_dataset(
        output_dir=resolved_output,
        config=config,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    if args.dry_run:
        print(
            "DRY RUN validated "
            f"{statistics['exported_unique_crops']} approved OCR recognition crops. "
            "No dataset files were written."
        )
    else:
        print(
            "Exported "
            f"{statistics['exported_unique_crops']} approved OCR recognition crops "
            f"to {resolved_output}"
        )
    print(f"Split counts: {statistics['counts_by_split']}")
    print(
        "Corrected text / added / bbox-edited: "
        f"{statistics['corrected_text_crops']} / "
        f"{statistics['manually_added_crops']} / "
        f"{statistics['corrected_bbox_crops']}"
    )
    print(
        "Recommended max_text_length: "
        f"{statistics['text_length']['recommended_max_text_length']}"
    )


if __name__ == "__main__":
    main()
