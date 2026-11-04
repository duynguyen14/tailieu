from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from sqlalchemy import select, true
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


def load_source_tokens(database: Database) -> list[SourceToken]:
    source = aliased(OcrToken)
    with database.session() as session:
        rows = session.execute(
            select(
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
            )
            .order_by(
                AnnotationDocument.id.asc(),
                AnnotationPage.page_number.asc(),
                OcrReviewToken.token_order.asc(),
            )
        ).all()

    records: list[SourceToken] = []
    for row in rows:
        (
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


def export_dataset(
    *,
    output_dir: Path,
    config: OcrRecognitionDatasetSettings,
    dry_run: bool = False,
) -> dict[str, object]:
    app_settings = get_settings()
    expected_database = config.ocr_rec_dataset_source_database.strip()
    if app_settings.db_name.lower() != expected_database.lower():
        raise RuntimeError(
            "OCR recognition dataset source database mismatch: "
            f"expected {expected_database!r}, configured DB_NAME is "
            f"{app_settings.db_name!r}."
        )

    output_dir = output_dir.resolve()
    if not dry_run:
        if output_dir.exists() and not output_dir.is_dir():
            raise RuntimeError(f"Export path exists but is not a directory: {output_dir}")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Export directory is not empty: {output_dir}. "
                "Use a new versioned directory to preserve existing datasets."
            )

    database = Database(app_settings)
    try:
        source_tokens = load_source_tokens(database)
    finally:
        database.close()
    if not source_tokens:
        raise RuntimeError("No active approved OCR review tokens were found.")

    split_by_document = split_document_ids(
        [record.document_id for record in source_tokens],
        val_ratio=config.ocr_rec_dataset_val_ratio,
        test_ratio=config.ocr_rec_dataset_test_ratio,
        seed=config.ocr_rec_dataset_split_seed,
    )
    storage_root = app_settings.annotation_storage_path.resolve()
    staging: Path | None = None
    if not dry_run:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            output_dir.rmdir()  # The directory was verified empty above.
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}-staging-", dir=str(output_dir.parent)
            )
        )

    invalid_items: list[str] = []
    exported: list[ExportToken] = []
    seen_crop_hashes: dict[str, tuple[str, str]] = {}
    duplicate_count = 0

    try:
        grouped: dict[str, list[SourceToken]] = defaultdict(list)
        for record in source_tokens:
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
                        f"{image.width}x{image.height}, expected "
                        f"{sorted(expected_sizes)}"
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
                            padding_ratio=config.ocr_rec_dataset_padding_ratio,
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
                    if config.ocr_rec_dataset_deduplicate and previous:
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
            "source_review_tokens": len(source_tokens),
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
            },
        }
        if staging is not None:
            (staging / "dataset_statistics.json").write_text(
                json.dumps(statistics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            staging.replace(output_dir)
        return statistics
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise


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
