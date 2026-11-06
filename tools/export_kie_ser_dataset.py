from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from sqlalchemy import select, true

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    KieAnnotationReview,
    KieAnnotationRun,
    KieEntityAnnotation,
    KiePageAnnotation,
    OcrReviewToken,
)
from app.services.storage import AnnotationStorage


DEFAULT_OUTPUT_DIR = Path("data/training/kie_ser_anotation_2_v1")
DATASET_VERSION = "kie-ser-human-v1"


@dataclass(frozen=True)
class ExportPage:
    page_annotation_id: str
    run_id: str
    case_id: str
    document_id: str
    original_filename: str
    page_id: str
    page_number: int
    page_order: int
    ocr_review_id: str
    ser_review_id: str
    reviewed_by: str
    reviewed_at: datetime
    image_relative_path: str
    image_width: int
    image_height: int
    orientation_angle: int
    preprocess_metadata: dict[str, Any] | None


@dataclass(frozen=True)
class ExportToken:
    id: str
    order: int
    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class EntitySpan:
    entity_id: str
    label: str
    field_path: str
    applicant_index: int | None
    token_id: str
    char_start: int
    char_end: int


@dataclass
class Segment:
    text: str
    source_token_id: str
    source_token_order: int
    char_start: int
    char_end: int
    pixel_bbox: list[int]
    entity_id: str | None = None
    label: str = "O"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export active human-approved SER snapshots as a LayoutLM-compatible "
            "BIO token-classification dataset."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help=(
            "Validate only DB tokens/spans/bboxes. Intended for diagnostics on a "
            "machine without ANNOTATION_STORAGE_DIR; cannot be used for a real export."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def split_document_ids(
    document_ids: Iterable[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(value < 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("Train/val/test ratios must be non-negative and sum to 1.")

    ids = sorted(set(document_ids))
    random.Random(seed).shuffle(ids)
    count = len(ids)
    if count == 0:
        return {}

    val_count = round(count * val_ratio)
    test_count = round(count * test_ratio)
    if count >= 3 and val_ratio > 0:
        val_count = max(1, val_count)
    if count >= 3 and test_ratio > 0:
        test_count = max(1, test_count)
    while val_count + test_count >= count:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    result: dict[str, str] = {}
    for index, document_id in enumerate(ids):
        if index < test_count:
            result[document_id] = "test"
        elif index < test_count + val_count:
            result[document_id] = "val"
        else:
            result[document_id] = "train"
    return result


def normalized_bbox(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int
) -> list[int]:
    x1, y1, x2, y2 = bbox
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox}")
    if x2 > image_width or y2 > image_height:
        raise ValueError(
            f"BBox {bbox} exceeds image dimensions {image_width}x{image_height}."
        )
    nx1 = max(0, min(999, round(x1 * 1000 / image_width)))
    ny1 = max(0, min(999, round(y1 * 1000 / image_height)))
    nx2 = max(nx1 + 1, min(1000, round(x2 * 1000 / image_width)))
    ny2 = max(ny1 + 1, min(1000, round(y2 * 1000 / image_height)))
    return [nx1, ny1, nx2, ny2]


def _segment_bbox(
    token_bbox: tuple[int, int, int, int], text_length: int, start: int, end: int
) -> list[int]:
    x1, y1, x2, y2 = token_bbox
    if text_length <= 0:
        raise ValueError("Cannot segment an empty OCR token.")
    width = x2 - x1
    left = x1 + math.floor(width * start / text_length)
    right = x1 + math.ceil(width * end / text_length)
    left = min(max(x1, left), x2 - 1)
    right = min(max(left + 1, right), x2)
    return [left, y1, right, y2]


def segment_ocr_token(
    token: ExportToken, spans: list[EntitySpan]
) -> list[Segment]:
    """Split one OCR line into trainable pieces at word and entity boundaries."""

    text = token.text
    if not text.strip():
        return []
    for span in spans:
        if span.char_start < 0 or span.char_end <= span.char_start or span.char_end > len(text):
            raise ValueError(
                f"Entity {span.entity_id} span [{span.char_start},{span.char_end}) "
                f"is outside token {token.id} length {len(text)}."
            )

    ownership: list[EntitySpan | None] = [None] * len(text)
    for span in spans:
        for position in range(span.char_start, span.char_end):
            if text[position].isspace():
                continue
            previous = ownership[position]
            if previous is not None and previous.entity_id != span.entity_id:
                raise ValueError(
                    f"Conflicting entities {previous.entity_id} and {span.entity_id} "
                    f"overlap token {token.id} at character {position}."
                )
            ownership[position] = span

    segments: list[Segment] = []
    for match in re.finditer(r"\S+", text):
        word_start, word_end = match.span()
        boundaries = {word_start, word_end}
        for span in spans:
            if span.char_start < word_end and span.char_end > word_start:
                boundaries.add(max(word_start, span.char_start))
                boundaries.add(min(word_end, span.char_end))
        ordered = sorted(boundaries)
        for start, end in zip(ordered, ordered[1:]):
            if start >= end or not text[start:end].strip():
                continue
            owners = {
                item.entity_id: item
                for item in ownership[start:end]
                if item is not None
            }
            if len(owners) > 1:
                raise ValueError(
                    f"Segment [{start},{end}) in token {token.id} has multiple entities."
                )
            owner = next(iter(owners.values()), None)
            # An entity boundary inside a word must cover this entire segment.
            if owner is not None and any(
                item is None for item in ownership[start:end] if not text[start:end].isspace()
            ):
                raise ValueError(
                    f"Entity {owner.entity_id} only partially owns segment "
                    f"[{start},{end}) in token {token.id}."
                )
            segments.append(
                Segment(
                    text=text[start:end],
                    source_token_id=token.id,
                    source_token_order=token.order,
                    char_start=start,
                    char_end=end,
                    pixel_bbox=_segment_bbox(token.bbox, len(text), start, end),
                    entity_id=owner.entity_id if owner else None,
                )
            )
    return segments


def assign_bio_labels(segments: list[Segment], entity_labels: dict[str, str]) -> None:
    seen: set[str] = set()
    for segment in segments:
        if segment.entity_id is None:
            segment.label = "O"
            continue
        label = entity_labels[segment.entity_id]
        prefix = "I" if segment.entity_id in seen else "B"
        segment.label = f"{prefix}-{label}"
        seen.add(segment.entity_id)


def validate_entity_identity(
    *,
    entity_id: str,
    label: str,
    field_path: str,
    applicant_index: int | None,
) -> None:
    note_match = re.fullmatch(r"note\.([a-z][a-z0-9_]*)", field_path)
    applicant_match = re.fullmatch(
        r"applicants\[(\d+)]\.([a-z][a-z0-9_]*)", field_path
    )
    if note_match:
        if applicant_index is not None:
            raise ValueError(
                f"Entity {entity_id} has note field_path but applicant_index={applicant_index}."
            )
        field_name = note_match.group(1)
    elif applicant_match:
        path_index = int(applicant_match.group(1))
        if applicant_index != path_index:
            raise ValueError(
                f"Entity {entity_id} field_path index {path_index} does not match "
                f"applicant_index={applicant_index}."
            )
        field_name = applicant_match.group(2)
    else:
        raise ValueError(f"Entity {entity_id} has invalid field_path '{field_path}'.")
    expected_label = {
        "note.issue_date": "NOTE_ISSUE_DATE",
        "note.issue_place": "NOTE_ISSUE_PLACE",
    }.get(field_path, field_name.upper())
    if label != expected_label:
        raise ValueError(
            f"Entity {entity_id} label {label} does not match field_path "
            f"{field_path}; expected {expected_label}."
        )


def _load_pages(database: Database) -> list[ExportPage]:
    with database.session() as session:
        rows = session.execute(
            select(
                KiePageAnnotation,
                KieAnnotationRun,
                AnnotationDocument,
                KieAnnotationReview,
            )
            .join(KieAnnotationRun, KieAnnotationRun.id == KiePageAnnotation.run_id)
            .join(AnnotationDocument, AnnotationDocument.id == KieAnnotationRun.document_id)
            .join(
                KieAnnotationReview,
                KieAnnotationReview.page_annotation_id == KiePageAnnotation.id,
            )
            .where(
                KieAnnotationRun.is_active == true(),
                KiePageAnnotation.ser_status == "APPROVED",
                KieAnnotationReview.mode == "SER",
                KieAnnotationReview.is_active == true(),
            )
            .order_by(
                AnnotationDocument.id,
                KiePageAnnotation.page_order,
                KiePageAnnotation.id,
            )
        ).all()

    result: list[ExportPage] = []
    seen_pages: set[str] = set()
    for page, run, document, review in rows:
        if page.id in seen_pages:
            raise RuntimeError(
                f"Page annotation {page.id} has multiple active SER review snapshots."
            )
        seen_pages.add(page.id)
        try:
            preprocess_metadata = (
                json.loads(page.preprocess_metadata_json)
                if page.preprocess_metadata_json
                else None
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Page annotation {page.id} has invalid preprocess metadata: {exc}"
            ) from exc
        result.append(
            ExportPage(
                page_annotation_id=page.id,
                run_id=run.id,
                case_id=run.case_id,
                document_id=run.document_id,
                original_filename=document.original_filename,
                page_id=page.page_id,
                page_number=page.page_number,
                page_order=page.page_order,
                ocr_review_id=page.ocr_review_id,
                ser_review_id=review.id,
                reviewed_by=review.reviewed_by,
                reviewed_at=review.reviewed_at,
                image_relative_path=page.image_relative_path,
                image_width=page.image_width,
                image_height=page.image_height,
                orientation_angle=page.orientation_angle,
                preprocess_metadata=preprocess_metadata,
            )
        )
    return result


def _load_tokens(
    database: Database, review_ids: list[str]
) -> dict[str, list[ExportToken]]:
    if not review_ids:
        return {}
    with database.session() as session:
        rows = session.execute(
            select(
                OcrReviewToken.review_id,
                OcrReviewToken.id,
                OcrReviewToken.token_order,
                OcrReviewToken.reviewed_text,
                OcrReviewToken.x1,
                OcrReviewToken.y1,
                OcrReviewToken.x2,
                OcrReviewToken.y2,
            )
            .where(
                OcrReviewToken.review_id.in_(review_ids),
                OcrReviewToken.include_in_extraction == true(),
            )
            .order_by(OcrReviewToken.review_id, OcrReviewToken.token_order)
        ).all()
    grouped: dict[str, list[ExportToken]] = defaultdict(list)
    for review_id, token_id, order, text, x1, y1, x2, y2 in rows:
        grouped[review_id].append(
            ExportToken(token_id, order, text, (x1, y1, x2, y2))
        )
    return grouped


def _load_entities(
    database: Database, page_annotation_ids: list[str]
) -> dict[str, list[KieEntityAnnotation]]:
    if not page_annotation_ids:
        return {}
    with database.session() as session:
        rows = list(
            session.scalars(
                select(KieEntityAnnotation)
                .where(
                    KieEntityAnnotation.page_annotation_id.in_(page_annotation_ids),
                    KieEntityAnnotation.is_active == true(),
                )
                .order_by(
                    KieEntityAnnotation.page_annotation_id,
                    KieEntityAnnotation.created_at,
                    KieEntityAnnotation.id,
                )
            )
        )
        # Detach before the session closes; all scalar fields are already loaded.
        session.expunge_all()
    grouped: dict[str, list[KieEntityAnnotation]] = defaultdict(list)
    for entity in rows:
        grouped[entity.page_annotation_id].append(entity)
    return grouped


def _copy_image(
    page: ExportPage,
    *,
    storage: AnnotationStorage,
    dataset_dir: Path,
    copy_images: bool,
    validate_image: bool = True,
) -> str:
    if not validate_image:
        return page.image_relative_path
    source = storage.resolve(page.image_relative_path)
    if not source.is_file():
        raise FileNotFoundError(f"OCR processed image was not found: {source}")
    with Image.open(source) as image:
        actual_size = image.size
    expected_size = (page.image_width, page.image_height)
    if actual_size != expected_size:
        raise ValueError(
            "OCR image dimensions do not match the SER coordinate system: "
            f"actual={actual_size}, expected={expected_size}, path={source}"
        )
    if not copy_images:
        return page.image_relative_path
    suffix = source.suffix.lower() or ".png"
    relative = Path("images") / page.document_id / f"{page.page_id}{suffix}"
    destination = dataset_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return relative.as_posix()


def _build_record(
    page: ExportPage,
    tokens: list[ExportToken],
    entities: list[KieEntityAnnotation],
    *,
    image_path: str,
) -> dict[str, Any]:
    if not tokens:
        raise ValueError("Approved SER page has no included OCR tokens.")
    token_by_id = {token.id: token for token in tokens}
    if len(token_by_id) != len(tokens):
        raise ValueError("OCR token IDs are not unique.")
    if len({token.order for token in tokens}) != len(tokens):
        raise ValueError("OCR token orders are not unique.")

    spans_by_token: dict[str, list[EntitySpan]] = defaultdict(list)
    entity_labels: dict[str, str] = {}
    entity_payloads: list[dict[str, Any]] = []
    label_names: set[str] = set()
    for entity in entities:
        label = entity.label.strip().upper()
        if not label or not re.fullmatch(r"[A-Z][A-Z0-9_]*", label):
            raise ValueError(f"Entity {entity.id} has invalid label '{entity.label}'.")
        field_path = (entity.field_path or "").strip()
        if not field_path:
            raise ValueError(f"Entity {entity.id} has no reviewed field_path.")
        validate_entity_identity(
            entity_id=entity.id,
            label=label,
            field_path=field_path,
            applicant_index=entity.applicant_index,
        )
        if entity.x1 < 0 or entity.y1 < 0 or entity.x2 > page.image_width or entity.y2 > page.image_height:
            raise ValueError(f"Entity {entity.id} bbox is outside the page image.")
        try:
            token_ids = json.loads(entity.token_ids_json)
            raw_spans = json.loads(entity.spans_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Entity {entity.id} has invalid JSON: {exc}") from exc
        if not isinstance(token_ids, list) or not isinstance(raw_spans, list) or not raw_spans:
            raise ValueError(f"Entity {entity.id} has no reviewed character spans.")
        selected_token_ids: list[str] = []
        normalized_spans: list[dict[str, Any]] = []
        for raw in raw_spans:
            token_id = str(raw.get("token_id", ""))
            token = token_by_id.get(token_id)
            if token is None:
                raise ValueError(
                    f"Entity {entity.id} references missing/excluded OCR token {token_id}."
                )
            start = int(raw.get("char_start", -1))
            end = int(raw.get("char_end", -1))
            span = EntitySpan(
                entity_id=entity.id,
                label=label,
                field_path=field_path,
                applicant_index=entity.applicant_index,
                token_id=token_id,
                char_start=start,
                char_end=end,
            )
            spans_by_token[token_id].append(span)
            selected_token_ids.append(token_id)
            normalized_spans.append(
                {
                    "token_id": token_id,
                    "char_start": start,
                    "char_end": end,
                    "text": token.text[start:end] if 0 <= start < end <= len(token.text) else "",
                }
            )
        if set(map(str, token_ids)) != set(selected_token_ids):
            raise ValueError(
                f"Entity {entity.id} token_ids do not match its reviewed spans."
            )
        expected_value = " ".join(item["text"] for item in normalized_spans).strip()
        if " ".join(entity.value.split()) != " ".join(expected_value.split()):
            raise ValueError(
                f"Entity {entity.id} value does not match its reviewed character "
                f"spans; value='{entity.value}', spans='{expected_value}'."
            )
        entity_labels[entity.id] = label
        label_names.add(label)
        entity_payloads.append(
            {
                "id": entity.id,
                "label": label,
                "value": entity.value,
                "field_path": field_path,
                "applicant_index": entity.applicant_index,
                "source_token_ids": list(dict.fromkeys(selected_token_ids)),
                "source_spans": normalized_spans,
                "pixel_bbox": [entity.x1, entity.y1, entity.x2, entity.y2],
                "bbox": normalized_bbox(
                    (entity.x1, entity.y1, entity.x2, entity.y2),
                    page.image_width,
                    page.image_height,
                ),
                "source": "HUMAN",
            }
        )

    segments: list[Segment] = []
    for token in sorted(tokens, key=lambda item: item.order):
        normalized_bbox(token.bbox, page.image_width, page.image_height)
        segments.extend(segment_ocr_token(token, spans_by_token.get(token.id, [])))
    if not segments:
        raise ValueError("Page has no non-empty trainable OCR segments.")
    assign_bio_labels(segments, entity_labels)

    segment_indices_by_entity: dict[str, list[int]] = defaultdict(list)
    for index, segment in enumerate(segments):
        if segment.entity_id:
            segment_indices_by_entity[segment.entity_id].append(index)
    for entity in entity_payloads:
        indices = segment_indices_by_entity.get(entity["id"], [])
        if not indices:
            raise ValueError(f"Entity {entity['id']} produced no labeled segment.")
        entity["segment_indices"] = indices

    return {
        "id": f"{page.case_id}:{page.page_id}",
        "dataset_version": DATASET_VERSION,
        "document_id": page.document_id,
        "original_filename": page.original_filename,
        "case_id": page.case_id,
        "run_id": page.run_id,
        "page_annotation_id": page.page_annotation_id,
        "page_id": page.page_id,
        "page_number": page.page_number,
        "page_order": page.page_order,
        "image_path": image_path,
        "image_storage_relative_path": page.image_relative_path,
        "image_width": page.image_width,
        "image_height": page.image_height,
        "orientation_angle": page.orientation_angle,
        "preprocess_metadata": page.preprocess_metadata,
        "ocr_review_id": page.ocr_review_id,
        "ser_review_id": page.ser_review_id,
        "ser_reviewed_by": page.reviewed_by,
        "ser_reviewed_at": page.reviewed_at.isoformat(),
        "words": [segment.text for segment in segments],
        "boxes": [
            normalized_bbox(tuple(segment.pixel_bbox), page.image_width, page.image_height)
            for segment in segments
        ],
        "pixel_boxes": [segment.pixel_bbox for segment in segments],
        "labels": [segment.label for segment in segments],
        "source_segments": [
            {
                "source_token_id": segment.source_token_id,
                "source_token_order": segment.source_token_order,
                "char_start": segment.char_start,
                "char_end": segment.char_end,
            }
            for segment in segments
        ],
        "entities": entity_payloads,
        "relations": [],
        "entity_labels": sorted(label_names),
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("SER dataset export must run with DB_NAME=anotation_2.")
    if args.skip_image_validation and not args.dry_run:
        raise ValueError("--skip-image-validation is allowed only with --dry-run.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (PROJECT_ROOT / args.output_dir).resolve()
    )
    if not output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output directory must stay inside the project root.")
    if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. Choose a new versioned directory."
        )

    database = Database(settings)
    try:
        pages = _load_pages(database)
        tokens_by_review = _load_tokens(database, [page.ocr_review_id for page in pages])
        entities_by_page = _load_entities(
            database, [page.page_annotation_id for page in pages]
        )
    finally:
        database.close()

    split_by_document = split_document_ids(
        (page.document_id for page in pages),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    storage = AnnotationStorage(settings.annotation_storage_path)
    staging_dir = output_dir.with_name(f"{output_dir.name}.tmp")
    if not args.dry_run:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    invalid_items: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    bio_counts: Counter[str] = Counter()
    segment_counts: list[int] = []
    try:
        for index, page in enumerate(pages, start=1):
            try:
                target_dir = staging_dir if not args.dry_run else output_dir
                image_path = _copy_image(
                    page,
                    storage=storage,
                    dataset_dir=target_dir,
                    copy_images=not args.no_copy_images and not args.dry_run,
                    validate_image=not args.skip_image_validation,
                )
                record = _build_record(
                    page,
                    tokens_by_review.get(page.ocr_review_id, []),
                    entities_by_page.get(page.page_annotation_id, []),
                    image_path=image_path,
                )
                split = split_by_document[page.document_id]
                records_by_split[split].append(record)
                segment_counts.append(len(record["words"]))
                label_counts.update(entity["label"] for entity in record["entities"])
                bio_counts.update(record["labels"])
                print(
                    f"[{index}/{len(pages)}] {('VALID' if args.dry_run else 'EXPORTED')} "
                    f"file='{page.original_filename}' page={page.page_number} "
                    f"split={split} segments={len(record['words'])} "
                    f"entities={len(record['entities'])}"
                )
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                item = {
                    "page_annotation_id": page.page_annotation_id,
                    "document_id": page.document_id,
                    "original_filename": page.original_filename,
                    "page_number": page.page_number,
                    "error": str(exc),
                }
                invalid_items.append(item)
                print(
                    f"[{index}/{len(pages)}] INVALID file='{page.original_filename}' "
                    f"page={page.page_number}: {exc}"
                )
                if args.fail_on_invalid:
                    raise RuntimeError(
                        f"Invalid approved SER page {page.page_annotation_id}: {exc}"
                    ) from exc

        labels = sorted(label_counts)
        document_counts = {
            split: len({record["document_id"] for record in records})
            for split, records in records_by_split.items()
        }
        summary = {
            "dataset_version": DATASET_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database": settings.db_name,
            "dry_run": args.dry_run,
            "image_validation_skipped": args.skip_image_validation,
            "seed": args.seed,
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "approved_pages_discovered": len(pages),
            "exported_pages": sum(len(records) for records in records_by_split.values()),
            "document_count": len(
                {
                    record["document_id"]
                    for records in records_by_split.values()
                    for record in records
                }
            ),
            "counts_by_split": {
                split: len(records) for split, records in records_by_split.items()
            },
            "document_counts_by_split": document_counts,
            "entity_count": sum(label_counts.values()),
            "entity_counts_by_label": dict(sorted(label_counts.items())),
            "bio_label_counts": dict(sorted(bio_counts.items())),
            "entity_labels": labels,
            "average_segments_per_page": (
                round(sum(segment_counts) / len(segment_counts), 3)
                if segment_counts
                else 0
            ),
            "maximum_segments_per_page": max(segment_counts, default=0),
            "pages_over_512_segments": sum(value > 512 for value in segment_counts),
            "invalid_count": len(invalid_items),
            "invalid_items": invalid_items,
            "bbox_coordinate_system": {
                "boxes": "LayoutLM integer coordinates in [0,1000]",
                "pixel_boxes": "OCR processed-image pixels",
                "image_contract": "Exact processed image used by the approved OCR review",
            },
        }

        if not args.dry_run:
            for split, records in records_by_split.items():
                _write_jsonl(staging_dir / f"{split}.jsonl", records)
            (staging_dir / "labels.json").write_text(
                json.dumps(
                    {
                        "outside": "O",
                        "entity_labels": labels,
                        "bio_labels": [
                            "O",
                            *[
                                bio
                                for label in labels
                                for bio in (f"B-{label}", f"I-{label}")
                            ],
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (staging_dir / "dataset_statistics.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            fingerprint = hashlib.sha256()
            for split in ("train", "val", "test"):
                fingerprint.update((staging_dir / f"{split}.jsonl").read_bytes())
            summary["dataset_fingerprint"] = fingerprint.hexdigest()
            (staging_dir / "dataset_statistics.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if output_dir.exists():
                output_dir.rmdir()
            staging_dir.replace(output_dir)
            summary["output_dir"] = str(output_dir)

        print("SER dataset export finished:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    except Exception:
        if not args.dry_run and staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def main() -> None:
    try:
        run(parse_args())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"SER dataset export could not start: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
