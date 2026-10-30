from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import DevicePreference, get_settings
from app.db.database import Database
from app.models.paddle_table_structure import PaddleTableStructureEngine
from app.services.storage import AnnotationStorage
from app.services.table_annotation import (
    TableAnnotationConflictError,
    TableAnnotationService,
)


@dataclass
class InferenceSummary:
    discovered: int = 0
    predicted: int = 0
    skipped: int = 0
    failed: int = 0
    tables: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run table detection and structure inference for reviewed APPLICANT_TABLE pages."
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--page-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> InferenceSummary:
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    settings = get_settings()
    database = Database(settings)
    storage = AnnotationStorage(settings.annotation_storage_path)
    service = TableAnnotationService(database=database, storage=storage)
    summary = InferenceSummary()
    try:
        pages = service.eligible_pages(
            limit=args.limit,
            page_id=args.page_id,
            force=args.force,
        )
        summary.discovered = len(pages)
        if not pages:
            print("No eligible reviewed APPLICANT_TABLE pages were found.")
            return summary

        device_override = DevicePreference(args.device) if args.device else None
        engine = PaddleTableStructureEngine(settings, device_override=device_override)
        print(
            f"Loaded {engine.model_name} on {engine.effective_device or 'auto'}; "
            f"pages={len(pages)}"
        )
        for position, page in enumerate(pages, start=1):
            annotation_id: str | None = None
            started = time.perf_counter()
            try:
                annotation_id = service.start_model_annotation(
                    page_id=page.id,
                    model_name=engine.model_name,
                    model_version=settings.table_model_version,
                    force=args.force,
                )
                image_path = storage.resolve(page.image_relative_path)
                tables = engine.predict(Path(image_path))
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                service.complete_model_annotation(
                    annotation_id=annotation_id,
                    tables=tables,
                    processing_time_ms=elapsed_ms,
                )
                summary.predicted += 1
                summary.tables += len(tables)
                print(
                    f"[{position}/{len(pages)}] page={page.id} "
                    f"tables={len(tables)} time={elapsed_ms}ms"
                )
            except TableAnnotationConflictError as exc:
                summary.skipped += 1
                print(f"[{position}/{len(pages)}] SKIPPED page={page.id}: {exc}")
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                if annotation_id is not None:
                    service.fail_model_annotation(
                        annotation_id=annotation_id,
                        error_message=str(exc),
                        processing_time_ms=elapsed_ms,
                    )
                summary.failed += 1
                print(f"[{position}/{len(pages)}] FAILED page={page.id}: {exc}")
        return summary
    finally:
        database.close()


def main() -> None:
    summary = run(parse_args())
    print(
        "Table inference finished: "
        f"discovered={summary.discovered}, predicted={summary.predicted}, "
        f"skipped={summary.skipped}, failed={summary.failed}, tables={summary.tables}"
    )
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
