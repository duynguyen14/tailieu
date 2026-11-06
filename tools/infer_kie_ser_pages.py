from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
from sqlalchemy import select, true, update

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    KieAnnotationRun,
    KieEntityAnnotation,
    KiePageAnnotation,
    OcrReviewToken,
    utc_now,
)
from app.services.storage import AnnotationStorage
from tools.export_kie_ser_dataset import (
    ExportToken,
    Segment,
    normalized_bbox,
    segment_ocr_token,
)
from tools.predict_kie_ser import (
    DEFAULT_MODEL_DIR,
    LayoutLmV3SerPredictor,
    PredictedEntity,
    decode_bio_entities,
)


MODEL_SOURCE = "SER_MODEL_V1"


@dataclass(frozen=True)
class PendingPage:
    page_annotation_id: str
    run_id: str
    page_id: str
    page_number: int
    ocr_review_id: str
    image_relative_path: str
    image_width: int
    image_height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fine-tuned LayoutLMv3 SER model on pending KIE pages. "
            "Dry-run is the default; add --execute to replace only non-human "
            "proposal entities while leaving SER status PENDING for review."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--from-index", type=int, default=0)
    parser.add_argument("--to-index", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--minimum-confidence", type=float, default=0.50)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_inference_segments(tokens: list[ExportToken]) -> list[Segment]:
    segments: list[Segment] = []
    for token in sorted(tokens, key=lambda item: item.order):
        segments.extend(segment_ocr_token(token, []))
    return segments


def entity_storage_payload(
    entity: PredictedEntity,
    segments: list[Segment],
) -> dict[str, Any]:
    if entity.pixel_bbox is None:
        raise ValueError("Predicted entity has no pixel bbox.")
    selected = [segments[index] for index in entity.word_indices]
    token_ids = list(dict.fromkeys(item.source_token_id for item in selected))
    spans = [
        {
            "token_id": item.source_token_id,
            "char_start": item.char_start,
            "char_end": item.char_end,
            "text": item.text,
        }
        for item in selected
    ]
    return {
        "label": entity.label,
        "value": entity.value,
        "token_ids": token_ids,
        "spans": spans,
        "bbox": entity.pixel_bbox,
        "normalized_bbox": entity.bbox,
        "confidence": entity.confidence,
    }


def _load_pages(
    database: Database,
    *,
    from_index: int,
    to_index: int | None,
    limit: int,
    include_existing: bool,
) -> list[PendingPage]:
    filters = [
        KieAnnotationRun.is_active == true(),
        KiePageAnnotation.ser_status == "PENDING",
    ]
    if not include_existing:
        existing_prediction = (
            select(KieEntityAnnotation.id)
            .where(
                KieEntityAnnotation.page_annotation_id == KiePageAnnotation.id,
                KieEntityAnnotation.is_active == true(),
                KieEntityAnnotation.source == MODEL_SOURCE,
            )
            .exists()
        )
        filters.append(~existing_prediction)
    statement = (
        select(KiePageAnnotation, KieAnnotationRun)
        .join(KieAnnotationRun, KieAnnotationRun.id == KiePageAnnotation.run_id)
        .where(*filters)
        .order_by(
            KieAnnotationRun.created_at,
            KiePageAnnotation.page_order,
            KiePageAnnotation.id,
        )
        .offset(from_index)
    )
    if to_index is not None:
        statement = statement.limit(to_index - from_index + 1)
    elif limit > 0:
        statement = statement.limit(limit)
    with database.session() as session:
        rows = session.execute(statement).all()
    return [
        PendingPage(
            page_annotation_id=page.id,
            run_id=run.id,
            page_id=page.page_id,
            page_number=page.page_number,
            ocr_review_id=page.ocr_review_id,
            image_relative_path=page.image_relative_path,
            image_width=page.image_width,
            image_height=page.image_height,
        )
        for page, run in rows
    ]


def _load_tokens(database: Database, review_id: str) -> list[ExportToken]:
    with database.session() as session:
        rows = session.execute(
            select(
                OcrReviewToken.id,
                OcrReviewToken.token_order,
                OcrReviewToken.reviewed_text,
                OcrReviewToken.x1,
                OcrReviewToken.y1,
                OcrReviewToken.x2,
                OcrReviewToken.y2,
            )
            .where(
                OcrReviewToken.review_id == review_id,
                OcrReviewToken.include_in_extraction == true(),
            )
            .order_by(OcrReviewToken.token_order)
        ).all()
    return [
        ExportToken(token_id, order, text, (x1, y1, x2, y2))
        for token_id, order, text, x1, y1, x2, y2 in rows
        if text and text.strip()
    ]


def _has_existing_prediction(database: Database, page_id: str) -> bool:
    with database.session() as session:
        return session.scalar(
            select(KieEntityAnnotation.id)
            .where(
                KieEntityAnnotation.page_annotation_id == page_id,
                KieEntityAnnotation.is_active == true(),
                KieEntityAnnotation.source == MODEL_SOURCE,
            )
            .limit(1)
        ) is not None


def _has_human_entity(database: Database, page_id: str) -> bool:
    with database.session() as session:
        return session.scalar(
            select(KieEntityAnnotation.id)
            .where(
                KieEntityAnnotation.page_annotation_id == page_id,
                KieEntityAnnotation.is_active == true(),
                KieEntityAnnotation.source == "HUMAN",
            )
            .limit(1)
        ) is not None


def _save_predictions(
    database: Database,
    page: PendingPage,
    payloads: list[dict[str, Any]],
) -> None:
    with database.session() as session:
        locked_page = session.scalar(
            select(KiePageAnnotation)
            .where(KiePageAnnotation.id == page.page_annotation_id)
            .with_for_update()
        )
        if locked_page is None:
            raise RuntimeError(f"KIE page disappeared: {page.page_annotation_id}")
        if locked_page.ser_status != "PENDING":
            raise RuntimeError(
                f"KIE page {page.page_annotation_id} is no longer SER PENDING."
            )
        human_id = session.scalar(
            select(KieEntityAnnotation.id)
            .where(
                KieEntityAnnotation.page_annotation_id == page.page_annotation_id,
                KieEntityAnnotation.is_active == true(),
                KieEntityAnnotation.source == "HUMAN",
            )
            .limit(1)
        )
        if human_id:
            raise RuntimeError("Refusing to replace an active HUMAN entity.")
        now = utc_now()
        session.execute(
            update(KieEntityAnnotation)
            .where(
                KieEntityAnnotation.page_annotation_id == page.page_annotation_id,
                KieEntityAnnotation.is_active == true(),
                KieEntityAnnotation.source != "HUMAN",
            )
            .values(is_active=False, updated_at=now)
        )
        for index, payload in enumerate(payloads, start=1):
            x1, y1, x2, y2 = payload["bbox"]
            nx1, ny1, nx2, ny2 = payload["normalized_bbox"]
            session.add(
                KieEntityAnnotation(
                    id=str(uuid4()),
                    page_annotation_id=page.page_annotation_id,
                    entity_key=f"ser_model_v1_{index:04d}",
                    label=payload["label"],
                    value=payload["value"],
                    field_path=None,
                    applicant_index=None,
                    token_ids_json=json.dumps(payload["token_ids"]),
                    spans_json=json.dumps(payload["spans"], ensure_ascii=False),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    nx1=nx1,
                    ny1=ny1,
                    nx2=nx2,
                    ny2=ny2,
                    confidence=Decimal(str(round(payload["confidence"], 6))),
                    source=MODEL_SOURCE,
                    candidate_rank=1,
                    is_active=True,
                )
            )
        locked_page.updated_at = now


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 0 or args.from_index < 0:
        raise ValueError("--limit and --from-index cannot be negative.")
    if args.to_index is not None and args.to_index < args.from_index:
        raise ValueError("--to-index must be greater than or equal to --from-index.")
    if not 0 <= args.minimum_confidence <= 1:
        raise ValueError("--minimum-confidence must be between 0 and 1.")
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("KIE inference must run with DB_NAME=anotation_2.")

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device_id)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA PyTorch is unavailable.")
    effective_device = (
        "cuda:0"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model_dir = (
        args.model_dir.resolve()
        if args.model_dir.is_absolute()
        else (Path(__file__).resolve().parents[1] / args.model_dir).resolve()
    )
    predictor = LayoutLmV3SerPredictor(
        model_dir,
        device=effective_device,
        maximum_length=args.max_sequence_length,
    )
    database = Database(settings)
    storage = AnnotationStorage(settings.annotation_storage_path)
    summary = {
        "discovered": 0,
        "completed": 0,
        "skipped_existing": 0,
        "skipped_human": 0,
        "failed": 0,
        "entities": 0,
        "execute": args.execute,
        "device": effective_device,
    }
    try:
        pages = _load_pages(
            database,
            from_index=args.from_index,
            to_index=args.to_index,
            limit=args.limit,
            include_existing=args.force,
        )
        summary["discovered"] = len(pages)
        for position, page in enumerate(pages, start=1):
            try:
                if _has_human_entity(database, page.page_annotation_id):
                    summary["skipped_human"] += 1
                    print(f"[{position}/{len(pages)}] SKIP HUMAN page={page.page_id}")
                    continue
                if _has_existing_prediction(database, page.page_annotation_id) and not args.force:
                    summary["skipped_existing"] += 1
                    print(f"[{position}/{len(pages)}] SKIP EXISTING page={page.page_id}")
                    continue
                tokens = _load_tokens(database, page.ocr_review_id)
                segments = build_inference_segments(tokens)
                if not segments:
                    raise RuntimeError("Page has no included OCR words.")
                image_path = storage.resolve(page.image_relative_path)
                if not image_path.is_file():
                    raise FileNotFoundError(f"OCR image was not found: {image_path}")
                with Image.open(image_path) as image:
                    if image.size != (page.image_width, page.image_height):
                        raise RuntimeError(
                            f"Image size {image.size} does not match KIE coordinates "
                            f"{page.image_width}x{page.image_height}."
                        )
                words = [item.text for item in segments]
                pixel_boxes = [item.pixel_bbox for item in segments]
                boxes = [
                    normalized_bbox(tuple(item.pixel_bbox), page.image_width, page.image_height)
                    for item in segments
                ]
                labels, confidences, chunks, latency_ms = predictor.predict(
                    image_path=image_path,
                    words=words,
                    boxes=boxes,
                )
                predicted = [
                    entity
                    for entity in decode_bio_entities(
                        words,
                        boxes,
                        labels,
                        confidences,
                        pixel_boxes=pixel_boxes,
                    )
                    if entity.confidence >= args.minimum_confidence
                ]
                payloads = [entity_storage_payload(entity, segments) for entity in predicted]
                if args.execute:
                    _save_predictions(database, page, payloads)
                summary["completed"] += 1
                summary["entities"] += len(payloads)
                action = "SAVED" if args.execute else "DRY"
                print(
                    f"[{position}/{len(pages)}] {action} page={page.page_id} "
                    f"words={len(words)} entities={len(payloads)} chunks={chunks} "
                    f"time={latency_ms:.0f}ms"
                )
            except Exception as exc:
                summary["failed"] += 1
                print(f"[{position}/{len(pages)}] FAILED page={page.page_id}: {exc}")
        print("KIE SER pending-page inference finished:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        database.close()


def main() -> None:
    try:
        run(parse_args())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"KIE SER pending-page inference could not start: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
