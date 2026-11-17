from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, select, true

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    VisaFormAnnotation,
    VisaFormFieldAnnotation,
    utc_now,
)
from app.schemas.annotation import AnnotationLabel, ReviewStatus
from app.schemas.visa_form_annotation import (
    VisaFormAnnotationSource,
    VisaFormAnnotationStatus,
)
from app.services.storage import AnnotationStorage
from app.visa_form_fields import FIELD_CODES, VISA_FORM_TEMPLATE_FIELDS

DEFAULT_MODEL_PATH = Path("models/yolo_form_visa/visa_form_yolo_v1_best.pt")
DEFAULT_CLASSES_PATH = Path("models/yolo_form_visa/visa_form_yolo_v1_classes.json")


@dataclass(frozen=True)
class PageCandidate:
    page_id: str
    document_id: str
    original_filename: str
    page_number: int
    image_relative_path: str
    width: int
    height: int
    active_annotation_id: str | None
    active_source: str | None
    active_status: str | None
    active_model_version: str | None


@dataclass(frozen=True)
class FieldPrediction:
    field_code: str
    field_order: int
    bbox: tuple[int, int, int, int]
    normalized_bbox: tuple[int, int, int, int]
    confidence: float | None
    used_fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PENDING visa-form bbox annotations with the reviewed YOLO model. "
            "The command is read-only unless --execute is supplied."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES_PATH)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--page-id", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly select read-only mode (this is also the default).",
    )
    parser.add_argument(
        "--replace-pending",
        action="store_true",
        help="Deactivate an existing non-REVIEWED annotation before inserting model output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate an active MODEL annotation made by the same model version.",
    )
    parser.add_argument(
        "--include-unreviewed-pages",
        action="store_true",
        help="Also process page labels that have not been human-reviewed.",
    )
    parser.add_argument(
        "--require-all-detections",
        action="store_true",
        help="Do not write a page when YOLO misses any of the 19 fields.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_class_mapping(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise RuntimeError(f"YOLO class mapping was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    class_to_index = payload.get("class_to_index")
    labels = payload.get("labels")
    if not isinstance(class_to_index, dict):
        if labels == list(FIELD_CODES):
            class_to_index = {code: index for index, code in enumerate(labels)}
        else:
            raise RuntimeError(f"Invalid class mapping: {path}")
    normalized = {str(code): int(index) for code, index in class_to_index.items()}
    expected = {code: index for index, code in enumerate(FIELD_CODES)}
    if normalized != expected:
        raise RuntimeError(
            "classes.json does not match the canonical 19-field class order."
        )
    return {index: code for code, index in normalized.items()}


def normalized_to_pixel(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, round(x1 * width / 1000))),
        max(0, min(height - 1, round(y1 * height / 1000))),
        max(1, min(width, round(x2 * width / 1000))),
        max(1, min(height, round(y2 * height / 1000))),
    )


def pixel_to_normalized(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(999, round(x1 * 1000 / width))),
        max(0, min(999, round(y1 * 1000 / height))),
        max(1, min(1000, round(x2 * 1000 / width))),
        max(1, min(1000, round(y2 * 1000 / height))),
    )


def select_best_detections(
    *,
    boxes: list[list[float]],
    class_ids: list[int],
    confidences: list[float],
    class_by_index: dict[int, str],
    width: int,
    height: int,
) -> dict[str, tuple[tuple[int, int, int, int], float]]:
    if not (len(boxes) == len(class_ids) == len(confidences)):
        raise ValueError("YOLO boxes, classes and confidences have different lengths.")
    selected: dict[str, tuple[tuple[int, int, int, int], float]] = {}
    for raw_box, class_id, confidence in zip(boxes, class_ids, confidences):
        field_code = class_by_index.get(int(class_id))
        if field_code is None or len(raw_box) != 4:
            continue
        x1 = max(0, min(width - 1, round(raw_box[0])))
        y1 = max(0, min(height - 1, round(raw_box[1])))
        x2 = max(1, min(width, round(raw_box[2])))
        y2 = max(1, min(height, round(raw_box[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        previous = selected.get(field_code)
        if previous is None or confidence > previous[1]:
            selected[field_code] = ((x1, y1, x2, y2), float(confidence))
    return selected


def build_field_predictions(
    selected: dict[str, tuple[tuple[int, int, int, int], float]],
    *,
    width: int,
    height: int,
) -> list[FieldPrediction]:
    predictions: list[FieldPrediction] = []
    for field_order, template in enumerate(VISA_FORM_TEMPLATE_FIELDS, start=1):
        detected = selected.get(template.code)
        if detected is None:
            bbox = normalized_to_pixel(template.normalized_bbox, width, height)
            confidence = None
            fallback = True
        else:
            bbox, confidence = detected
            fallback = False
        predictions.append(
            FieldPrediction(
                field_code=template.code,
                field_order=field_order,
                bbox=bbox,
                normalized_bbox=pixel_to_normalized(bbox, width, height),
                confidence=confidence,
                used_fallback=fallback,
            )
        )
    return predictions


def load_candidates(
    database: Database,
    *,
    model_version: str,
    limit: int | None,
    page_id: str | None,
    replace_pending: bool,
    force: bool,
    include_unreviewed_pages: bool,
) -> tuple[list[PageCandidate], dict[str, int]]:
    stats = {
        "discovered": 0,
        "reviewed_skipped": 0,
        "existing_skipped": 0,
        "same_model_skipped": 0,
    }
    with database.session() as session:
        active_join = and_(
            VisaFormAnnotation.page_id == AnnotationPage.id,
            VisaFormAnnotation.is_active == true(),
        )
        statement = (
            select(AnnotationPage, AnnotationDocument, VisaFormAnnotation)
            .join(AnnotationDocument, AnnotationDocument.id == AnnotationPage.document_id)
            .outerjoin(VisaFormAnnotation, active_join)
            .where(AnnotationPage.label == AnnotationLabel.VISA_APPLICATION_FORM.value)
            .order_by(
                AnnotationPage.created_at.desc(),
                AnnotationPage.page_index.asc(),
            )
        )
        if not include_unreviewed_pages:
            statement = statement.where(
                AnnotationPage.review_status == ReviewStatus.REVIEWED.value
            )
        if page_id:
            statement = statement.where(AnnotationPage.id == page_id)
        rows = session.execute(statement).all()
        reviewed_page_ids = set(
            session.scalars(
                select(VisaFormAnnotation.page_id).where(
                    VisaFormAnnotation.status == VisaFormAnnotationStatus.REVIEWED.value
                )
            ).all()
        )

        candidates: list[PageCandidate] = []
        for page, document, active in rows:
            stats["discovered"] += 1
            if page.id in reviewed_page_ids:
                stats["reviewed_skipped"] += 1
                continue
            if active is not None:
                same_model = (
                    active.source == VisaFormAnnotationSource.MODEL.value
                    and active.model_version == model_version
                )
                if same_model and not force:
                    stats["same_model_skipped"] += 1
                    continue
                if not replace_pending:
                    stats["existing_skipped"] += 1
                    continue
            candidates.append(
                PageCandidate(
                    page_id=page.id,
                    document_id=page.document_id,
                    original_filename=document.original_filename,
                    page_number=page.page_number,
                    image_relative_path=page.image_relative_path,
                    width=page.width,
                    height=page.height,
                    active_annotation_id=active.id if active else None,
                    active_source=active.source if active else None,
                    active_status=active.status if active else None,
                    active_model_version=active.model_version if active else None,
                )
            )
            if limit is not None and len(candidates) >= limit:
                break
    return candidates, stats


def write_prediction(
    database: Database,
    *,
    candidate: PageCandidate,
    predictions: list[FieldPrediction],
    model_name: str,
    model_version: str,
    processing_time_ms: int,
    replace_pending: bool,
    force: bool,
) -> str | None:
    with database.session() as session:
        reviewed_exists = session.scalar(
            select(VisaFormAnnotation.id)
            .where(
                VisaFormAnnotation.page_id == candidate.page_id,
                VisaFormAnnotation.status == VisaFormAnnotationStatus.REVIEWED.value,
            )
            .limit(1)
        )
        if reviewed_exists is not None:
            return None
        active = session.scalar(
            select(VisaFormAnnotation)
            .where(
                VisaFormAnnotation.page_id == candidate.page_id,
                VisaFormAnnotation.is_active == true(),
            )
            .with_for_update()
        )
        if active is not None:
            same_model = (
                active.source == VisaFormAnnotationSource.MODEL.value
                and active.model_version == model_version
            )
            if (same_model and not force) or not replace_pending:
                return None
            active.is_active = False
            active.updated_at = utc_now()

        annotation = VisaFormAnnotation(
            id=str(uuid4()),
            page_id=candidate.page_id,
            parent_annotation_id=active.id if active else None,
            source=VisaFormAnnotationSource.MODEL.value,
            status=VisaFormAnnotationStatus.PENDING.value,
            template_version=f"yolo:{model_version}",
            is_active=True,
            image_width=candidate.width,
            image_height=candidate.height,
            model_name=model_name,
            model_version=model_version,
            processing_time_ms=max(0, processing_time_ms),
        )
        session.add(annotation)
        session.flush()
        for prediction in predictions:
            x1, y1, x2, y2 = prediction.bbox
            nx1, ny1, nx2, ny2 = prediction.normalized_bbox
            confidence = (
                Decimal(f"{prediction.confidence:.6f}")
                if prediction.confidence is not None
                else None
            )
            annotation.fields.append(
                VisaFormFieldAnnotation(
                    id=str(uuid4()),
                    field_code=prediction.field_code,
                    field_order=prediction.field_order,
                    value_text=None,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    nx1=nx1,
                    ny1=ny1,
                    nx2=nx2,
                    ny2=ny2,
                    confidence=confidence,
                )
            )
        session.flush()
        return annotation.id


def result_values(result: Any) -> tuple[list[list[float]], list[int], list[float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], [], []
    return (
        boxes.xyxy.detach().cpu().tolist(),
        [int(value) for value in boxes.cls.detach().cpu().tolist()],
        [float(value) for value in boxes.conf.detach().cpu().tolist()],
    )


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.confidence <= 1:
        raise ValueError("--confidence must be greater than 0 and at most 1.")
    if not 0 < args.iou <= 1:
        raise ValueError("--iou must be greater than 0 and at most 1.")
    if args.image_size < 320 or args.batch_size < 1:
        raise ValueError("--image-size must be >= 320 and --batch-size must be >= 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.force and not args.replace_pending:
        raise ValueError("--force requires --replace-pending.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    model_path = resolve_path(args.model)
    classes_path = resolve_path(args.classes)
    if not model_path.is_file():
        raise RuntimeError(f"YOLO model was not found: {model_path}")
    class_by_index = load_class_mapping(classes_path)
    model_hash = sha256_file(model_path)
    model_version = args.model_version or f"visa-form-yolo-v1-{model_hash[:12]}"

    settings = get_settings()
    database = Database(settings)
    storage = AnnotationStorage(settings.annotation_storage_path)
    try:
        candidates, selection_stats = load_candidates(
            database,
            model_version=model_version,
            limit=args.limit,
            page_id=args.page_id,
            replace_pending=args.replace_pending,
            force=args.force,
            include_unreviewed_pages=args.include_unreviewed_pages,
        )
        mode = "EXECUTE" if args.execute else "DRY-RUN"
        print(
            f"{mode}: model={model_path}; version={model_version}; "
            f"eligible={len(candidates)}; selection={selection_stats}"
        )
        for index, candidate in enumerate(candidates, start=1):
            print(
                f"[{index}/{len(candidates)}] {mode} file={candidate.original_filename!r} "
                f"page={candidate.page_number} page_id={candidate.page_id} "
                f"active={candidate.active_status or 'NONE'}"
            )
        if not args.execute:
            print("DRY RUN: model was not loaded and the database was not changed.")
            return
        if not candidates:
            print("No eligible visa-form pages were found.")
            return

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required. Install requirements-visa-form-yolo-training.txt."
            ) from error
        model = YOLO(str(model_path))
        model_names = {int(index): str(name) for index, name in model.names.items()}
        if model_names != class_by_index:
            raise RuntimeError(
                f"Model class mapping differs from classes.json: {model_names}"
            )

        completed = 0
        failed = 0
        protected = 0
        fallback_fields = 0
        for offset in range(0, len(candidates), args.batch_size):
            batch = candidates[offset : offset + args.batch_size]
            image_paths: list[str] = []
            for candidate in batch:
                image_path = storage.resolve(candidate.image_relative_path)
                if not image_path.is_file():
                    raise RuntimeError(
                        f"Page image is missing for {candidate.page_id}: {image_path}"
                    )
                image_paths.append(str(image_path))
            started = time.perf_counter()
            results = model.predict(
                source=image_paths,
                conf=args.confidence,
                iou=args.iou,
                imgsz=args.image_size,
                device=args.device,
                batch=len(batch),
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            if len(results) != len(batch):
                raise RuntimeError("Ultralytics returned a different result count than input pages.")
            per_page_ms = round(elapsed_ms / len(batch))
            for candidate, result in zip(batch, results):
                try:
                    boxes, class_ids, confidences = result_values(result)
                    selected = select_best_detections(
                        boxes=boxes,
                        class_ids=class_ids,
                        confidences=confidences,
                        class_by_index=class_by_index,
                        width=candidate.width,
                        height=candidate.height,
                    )
                    missing = [code for code in FIELD_CODES if code not in selected]
                    if missing and args.require_all_detections:
                        failed += 1
                        print(
                            f"[FAILED] page={candidate.page_id} missing={','.join(missing)}"
                        )
                        continue
                    predictions = build_field_predictions(
                        selected, width=candidate.width, height=candidate.height
                    )
                    fallback_count = sum(item.used_fallback for item in predictions)
                    annotation_id = write_prediction(
                        database,
                        candidate=candidate,
                        predictions=predictions,
                        model_name=model_path.name,
                        model_version=model_version,
                        processing_time_ms=per_page_ms,
                        replace_pending=args.replace_pending,
                        force=args.force,
                    )
                    if annotation_id is None:
                        protected += 1
                        print(f"[SKIP-PROTECTED] page={candidate.page_id}")
                        continue
                    completed += 1
                    fallback_fields += fallback_count
                    print(
                        f"[COMPLETED] page={candidate.page_id} annotation={annotation_id} "
                        f"detected={len(selected)}/19 fallback={fallback_count} "
                        f"elapsed_ms={per_page_ms}"
                    )
                except Exception as error:
                    failed += 1
                    print(f"[FAILED] page={candidate.page_id} error={error}", file=sys.stderr)
        print(
            "Visa-form YOLO generation finished: "
            f"eligible={len(candidates)}, completed={completed}, failed={failed}, "
            f"protected={protected}, fallback_fields={fallback_fields}."
        )
    finally:
        database.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Visa-form YOLO generation could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error
