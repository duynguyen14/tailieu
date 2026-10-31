from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import select, true, update

from app.core.config import DevicePreference, get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationPage,
    DocumentCase,
    DocumentCasePage,
    OcrRun,
    OcrToken,
    utc_now,
)
from app.models.paddle_ocr_engine import OcrTokenPrediction, PaddleOcrEngine
from app.schemas.annotation import ReviewStatus
from app.services.storage import AnnotationStorage


@dataclass
class OcrSummary:
    discovered: int = 0
    completed: int = 0
    failed: int = 0
    tokens: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPU-first OCR and store grounded text tokens with source boxes."
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all eligible pages")
    parser.add_argument("--page-id", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    return parser.parse_args()


def eligible_pages(
    database: Database,
    *,
    page_types: list[str],
    page_id: str | None,
    case_id: str | None,
    limit: int,
    force: bool,
) -> list[AnnotationPage]:
    with database.session() as session:
        active_case_page = (
            select(DocumentCasePage.id)
            .join(DocumentCase, DocumentCase.id == DocumentCasePage.case_id)
            .where(
                DocumentCasePage.page_id == AnnotationPage.id,
                DocumentCase.is_active == true(),
            )
        )
        if case_id:
            active_case_page = active_case_page.where(DocumentCase.id == case_id)
        statement = (
            select(AnnotationPage)
            .where(
                AnnotationPage.review_status == ReviewStatus.REVIEWED.value,
                AnnotationPage.label.in_(page_types),
                active_case_page.exists(),
            )
            .order_by(AnnotationPage.created_at.asc(), AnnotationPage.page_index.asc())
        )
        if page_id:
            statement = statement.where(AnnotationPage.id == page_id)
        if not force:
            active_ocr_exists = (
                select(OcrRun.id)
                .where(
                    OcrRun.page_id == AnnotationPage.id,
                    OcrRun.is_active == true(),
                )
                .exists()
            )
            statement = statement.where(~active_ocr_exists)
        if limit > 0:
            statement = statement.limit(limit)
        return list(session.scalars(statement).all())


def start_run(
    database: Database,
    *,
    page_id: str,
    model_name: str,
    model_version: str,
    device: str,
) -> str:
    run_id = str(uuid4())
    with database.session() as session:
        session.execute(
            update(OcrRun)
            .where(OcrRun.page_id == page_id, OcrRun.is_active == true())
            .values(is_active=False, updated_at=utc_now())
        )
        session.add(
            OcrRun(
                id=run_id,
                page_id=page_id,
                status="PROCESSING",
                is_active=True,
                model_name=model_name,
                model_version=model_version,
                device=device,
            )
        )
    return run_id


def complete_run(
    database: Database,
    *,
    run_id: str,
    tokens: list[OcrTokenPrediction],
    processing_time_ms: int,
) -> None:
    with database.session() as session:
        run = session.get(OcrRun, run_id)
        if run is None:
            raise RuntimeError(f"OCR run disappeared: {run_id}")
        run.tokens.clear()
        for token in tokens:
            run.tokens.append(
                OcrToken(
                    id=str(uuid4()),
                    token_index=token.token_index,
                    token_id=token.token_id,
                    text=token.text,
                    confidence=Decimal(str(token.confidence)),
                    x1=token.bbox[0],
                    y1=token.bbox[1],
                    x2=token.bbox[2],
                    y2=token.bbox[3],
                    polygon_json=json.dumps(token.polygon, ensure_ascii=False),
                )
            )
        run.status = "COMPLETED"
        run.processing_time_ms = processing_time_ms
        run.error_message = None
        run.updated_at = utc_now()


def fail_run(
    database: Database, *, run_id: str, error: Exception, processing_time_ms: int
) -> None:
    with database.session() as session:
        run = session.get(OcrRun, run_id)
        if run is None:
            return
        run.status = "FAILED"
        run.is_active = False
        run.processing_time_ms = processing_time_ms
        run.error_message = str(error)[:4000]
        run.updated_at = utc_now()


def run(args: argparse.Namespace) -> OcrSummary:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    settings = get_settings()
    database = Database(settings)
    storage = AnnotationStorage(settings.annotation_storage_path)
    summary = OcrSummary()
    try:
        pages = eligible_pages(
            database,
            page_types=settings.ocr_target_page_type_list,
            page_id=args.page_id,
            case_id=args.case_id,
            limit=args.limit,
            force=args.force,
        )
        summary.discovered = len(pages)
        if not pages:
            print("No eligible reviewed pages were found for OCR.")
            return summary

        device = DevicePreference(args.device) if args.device else None
        engine = PaddleOcrEngine(settings, device_override=device)
        print(
            f"Loaded {engine.model_name} on {engine.effective_device}; pages={len(pages)}"
        )
        for position, page in enumerate(pages, start=1):
            started = time.perf_counter()
            run_id = start_run(
                database,
                page_id=page.id,
                model_name=engine.model_name,
                model_version=settings.ocr_model_version,
                device=engine.effective_device,
            )
            try:
                tokens = engine.predict(storage.resolve(page.image_relative_path))
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                complete_run(
                    database,
                    run_id=run_id,
                    tokens=tokens,
                    processing_time_ms=elapsed_ms,
                )
                summary.completed += 1
                summary.tokens += len(tokens)
                print(
                    f"[{position}/{len(pages)}] page={page.id} "
                    f"tokens={len(tokens)} time={elapsed_ms}ms"
                )
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                fail_run(database, run_id=run_id, error=exc, processing_time_ms=elapsed_ms)
                summary.failed += 1
                print(f"[{position}/{len(pages)}] FAILED page={page.id}: {exc}")
        return summary
    finally:
        database.close()


def main() -> None:
    summary = run(parse_args())
    print(
        "Grounded OCR finished: "
        f"discovered={summary.discovered}, completed={summary.completed}, "
        f"failed={summary.failed}, tokens={summary.tokens}"
    )
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
