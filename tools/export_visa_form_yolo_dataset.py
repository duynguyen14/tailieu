from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy import select, true
from sqlalchemy.orm import selectinload

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    VisaFormAnnotation,
)
from app.schemas.visa_form_annotation import VisaFormAnnotationStatus
from app.services.visa_form_annotation import FIELD_CODES


@dataclass(frozen=True)
class YoloBox:
    field_code: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class SourceRecord:
    annotation_id: str
    document_id: str
    page_id: str
    page_number: int
    original_filename: str
    image_relative_path: str
    width: int
    height: int
    reviewed_by: str | None
    reviewed_at: str | None
    boxes: tuple[YoloBox, ...]


@dataclass(frozen=True)
class ExportRecord:
    annotation_id: str
    document_id: str
    page_id: str
    page_number: int
    original_filename: str
    split: str
    image_path: str
    label_path: str
    width: int
    height: int
    box_count: int
    reviewed_by: str | None
    reviewed_at: str | None
    image_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export active REVIEWED visa-form field annotations as a YOLO detection dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/visa_form_yolo_v1"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use hardlink only when source and output are on the same filesystem.",
    )
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Abort instead of skipping invalid images or annotations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and validate DB rows without creating the dataset.",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    ratios = (train, val, test)
    if any(value < 0 or value > 1 for value in ratios):
        raise ValueError("Split ratios must be between 0 and 1.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.")
    if train <= 0:
        raise ValueError("The training ratio must be greater than zero.")


def split_document_ids(
    document_ids: list[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    identifiers = sorted(set(document_ids))
    random.Random(seed).shuffle(identifiers)
    count = len(identifiers)
    if not count:
        return {}

    val_count = round(count * val_ratio)
    test_count = round(count * test_ratio)
    if count >= 3 and val_ratio > 0:
        val_count = max(1, val_count)
    if count >= 3 and test_ratio > 0:
        test_count = max(1, test_count)
    while val_count + test_count >= count:
        if test_count >= val_count and test_count:
            test_count -= 1
        elif val_count:
            val_count -= 1
        else:
            break

    result: dict[str, str] = {}
    for index, document_id in enumerate(identifiers):
        if index < test_count:
            result[document_id] = "test"
        elif index < test_count + val_count:
            result[document_id] = "val"
        else:
            result[document_id] = "train"
    return result


def pixel_bbox_to_yolo(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise ValueError(f"BBox {bbox} exceeds image dimensions {width}x{height}.")
    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reviewed_records(database: Database, limit: int | None = None) -> list[SourceRecord]:
    with database.session() as session:
        statement = (
            select(VisaFormAnnotation, AnnotationPage, AnnotationDocument)
            .join(AnnotationPage, AnnotationPage.id == VisaFormAnnotation.page_id)
            .join(AnnotationDocument, AnnotationDocument.id == AnnotationPage.document_id)
            .options(selectinload(VisaFormAnnotation.fields))
            .where(
                VisaFormAnnotation.is_active == true(),
                VisaFormAnnotation.status == VisaFormAnnotationStatus.REVIEWED.value,
            )
            .order_by(
                AnnotationDocument.created_at.asc(),
                AnnotationDocument.id.asc(),
                AnnotationPage.page_index.asc(),
            )
        )
        if limit is not None:
            statement = statement.limit(limit)
        rows = session.execute(statement).all()
        return [
            SourceRecord(
                annotation_id=annotation.id,
                document_id=document.id,
                page_id=page.id,
                page_number=page.page_number,
                original_filename=document.original_filename,
                image_relative_path=page.image_relative_path,
                width=page.width,
                height=page.height,
                reviewed_by=annotation.reviewed_by,
                reviewed_at=(
                    annotation.reviewed_at.isoformat() if annotation.reviewed_at else None
                ),
                boxes=tuple(
                    YoloBox(
                        field_code=field.field_code,
                        x1=field.x1,
                        y1=field.y1,
                        x2=field.x2,
                        y2=field.y2,
                    )
                    for field in annotation.fields
                ),
            )
            for annotation, page, document in rows
        ]


def validate_source_record(record: SourceRecord, source_path: Path) -> None:
    if not source_path.is_file():
        raise ValueError(f"Image is missing: {record.image_relative_path}")
    if record.width <= 0 or record.height <= 0:
        raise ValueError("DB image dimensions are invalid.")
    codes = [box.field_code for box in record.boxes]
    if len(codes) != len(FIELD_CODES) or set(codes) != set(FIELD_CODES):
        missing = sorted(set(FIELD_CODES) - set(codes))
        extra = sorted(set(codes) - set(FIELD_CODES))
        raise ValueError(
            f"Exactly 19 unique field boxes are required; missing={missing}, extra={extra}."
        )
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate field_code values were found.")
    for box in record.boxes:
        pixel_bbox_to_yolo((box.x1, box.y1, box.x2, box.y2), record.width, record.height)
    with Image.open(source_path) as image:
        actual_width, actual_height = image.size
        image.verify()
        if actual_width != record.width or actual_height != record.height:
            raise ValueError(
                f"Image size {actual_width}x{actual_height} differs from DB "
                f"{record.width}x{record.height}."
            )


def copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def write_data_yaml(path: Path) -> None:
    lines = [
        "# Portable definition. The trainer generates an absolute-path YAML at runtime.",
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {code}" for index, code in enumerate(FIELD_CODES))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_dataset(args: argparse.Namespace) -> dict[str, object]:
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    settings = get_settings()
    storage_root = settings.annotation_storage_path.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (ROOT / args.output_dir).resolve()
    )
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"Output path is not a directory: {output_dir}")
    if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )

    database = Database(settings)
    try:
        source_records = load_reviewed_records(database, args.limit)
    finally:
        database.close()
    if not source_records:
        raise RuntimeError("No active REVIEWED visa-form annotations were found.")

    valid: list[tuple[SourceRecord, Path, str]] = []
    invalid: list[str] = []
    seen_pages: set[str] = set()
    for record in source_records:
        try:
            if record.page_id in seen_pages:
                raise ValueError("Duplicate active REVIEWED annotation for the same page.")
            seen_pages.add(record.page_id)
            source_path = (storage_root / record.image_relative_path).resolve()
            if not source_path.is_relative_to(storage_root):
                raise ValueError("Image path escapes ANNOTATION_STORAGE_DIR.")
            validate_source_record(record, source_path)
            valid.append((record, source_path, sha256_file(source_path)))
        except (OSError, ValueError) as error:
            invalid.append(f"{record.annotation_id} page={record.page_id}: {error}")

    if invalid and args.fail_on_invalid:
        preview = "\n".join(f"- {item}" for item in invalid[:20])
        raise RuntimeError(f"Visa-form dataset validation failed:\n{preview}")
    if not valid:
        raise RuntimeError("No valid reviewed annotations remain after validation.")

    split_by_document = split_document_ids(
        [record.document_id for record, _, _ in valid],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_counts = Counter(split_by_document.values())
    if len(split_by_document) >= 3:
        for split, ratio in (
            ("val", args.val_ratio),
            ("test", args.test_ratio),
        ):
            if ratio > 0 and not split_counts[split]:
                raise RuntimeError(f"Could not create a non-empty {split} document split.")

    preview_stats: dict[str, object] = {
        "reviewed_rows": len(source_records),
        "valid_images": len(valid),
        "invalid_images": len(invalid),
        "document_count": len(split_by_document),
        "counts_by_split": dict(
            Counter(split_by_document[record.document_id] for record, _, _ in valid)
        ),
        "invalid_items": invalid,
    }
    if args.dry_run:
        print(json.dumps(preview_stats, ensure_ascii=False, indent=2))
        print("DRY RUN: database rows, images and all 19 boxes were validated; no files created.")
        return preview_stats

    output_dir.mkdir(parents=True, exist_ok=True)
    class_to_index = {code: index for index, code in enumerate(FIELD_CODES)}
    exported: list[ExportRecord] = []
    for record, source_path, image_hash in valid:
        split = split_by_document[record.document_id]
        suffix = source_path.suffix.lower() or ".png"
        stem = f"{record.document_id}_p{record.page_number:04d}_{record.page_id}"
        image_relative = Path("images") / split / f"{stem}{suffix}"
        label_relative = Path("labels") / split / f"{stem}.txt"
        copy_file(source_path, output_dir / image_relative, args.copy_mode)
        ordered_boxes = sorted(record.boxes, key=lambda item: class_to_index[item.field_code])
        lines: list[str] = []
        for box in ordered_boxes:
            normalized = pixel_bbox_to_yolo(
                (box.x1, box.y1, box.x2, box.y2), record.width, record.height
            )
            lines.append(
                f"{class_to_index[box.field_code]} "
                + " ".join(f"{value:.8f}" for value in normalized)
            )
        label_path = output_dir / label_relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        exported.append(
            ExportRecord(
                annotation_id=record.annotation_id,
                document_id=record.document_id,
                page_id=record.page_id,
                page_number=record.page_number,
                original_filename=record.original_filename,
                split=split,
                image_path=image_relative.as_posix(),
                label_path=label_relative.as_posix(),
                width=record.width,
                height=record.height,
                box_count=len(record.boxes),
                reviewed_by=record.reviewed_by,
                reviewed_at=record.reviewed_at,
                image_sha256=image_hash,
            )
        )

    exported.sort(key=lambda item: (item.split, item.document_id, item.page_number))
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for record in exported:
            target.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "labels": list(FIELD_CODES),
                "class_to_index": class_to_index,
                "index_to_class": {str(index): code for code, index in class_to_index.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_data_yaml(output_dir / "data.yaml")
    stats = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "YOLO detection",
        "selection": {"status": "REVIEWED", "is_active": True},
        "class_count": len(FIELD_CODES),
        "classes": list(FIELD_CODES),
        "box_count": sum(item.box_count for item in exported),
        "split_seed": args.seed,
        "configured_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        **preview_stats,
    }
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Exported {len(exported)} reviewed visa-form images and "
        f"{stats['box_count']} boxes to {output_dir}"
    )
    print(f"Split counts: {stats['counts_by_split']}; invalid skipped: {len(invalid)}")
    return stats


def main() -> None:
    args = parse_args()
    export_dataset(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Visa-form YOLO export failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
