from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import or_, select

from app.core.config import DevicePreference, Settings, get_settings
from app.db.database import Database
from app.db.models import AnnotationPage, utc_now
from app.models.onnx_page_classifier import DetailedPagePrediction, OnnxPageClassifier
from app.schemas.annotation import ReviewStatus
from app.services.storage import AnnotationStorage


@dataclass
class PredictionSummary:
    discovered: int = 0
    predicted: int = 0
    skipped: int = 0
    failed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict labels for UNLABELED annotation pages without marking them reviewed."
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all eligible pages")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--page-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--model-version", default=None)
    return parser.parse_args()


def _chunks(items: list[AnnotationPage], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _eligible_pages(
    database: Database,
    *,
    model_version: str,
    page_id: str | None,
    limit: int,
    force: bool,
) -> list[AnnotationPage]:
    with database.session() as session:
        statement = (
            select(AnnotationPage)
            .where(AnnotationPage.review_status == ReviewStatus.UNLABELED.value)
            .order_by(AnnotationPage.created_at.asc(), AnnotationPage.page_index.asc())
        )
        if page_id:
            statement = statement.where(AnnotationPage.id == page_id)
        if not force:
            statement = statement.where(
                or_(
                    AnnotationPage.model_prediction_version.is_(None),
                    AnnotationPage.model_prediction_version != model_version,
                )
            )
        if limit > 0:
            statement = statement.limit(limit)
        return list(session.scalars(statement).all())


def _save_predictions(
    database: Database,
    *,
    pages: list[AnnotationPage],
    predictions: list[DetailedPagePrediction],
    model_version: str,
) -> tuple[int, int]:
    saved = 0
    skipped = 0
    with database.session() as session:
        for page, prediction in zip(pages, predictions, strict=True):
            stored = session.get(AnnotationPage, page.id)
            if stored is None or stored.review_status != ReviewStatus.UNLABELED.value:
                skipped += 1
                continue
            stored.label = prediction.label.value
            stored.model_predicted_label = prediction.label.value
            stored.model_prediction_confidence = prediction.confidence
            stored.model_prediction_topk_json = json.dumps(
                [
                    {"label": item.label.value, "confidence": item.confidence}
                    for item in prediction.top_predictions
                ],
                ensure_ascii=False,
            )
            stored.model_prediction_version = model_version
            stored.model_predicted_at = utc_now()
            stored.reviewed_by = None
            stored.review_note = None
            stored.reviewed_at = None
            stored.updated_at = utc_now()
            saved += 1
    return saved, skipped


def run(args: argparse.Namespace) -> PredictionSummary:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    settings = get_settings()
    batch_size = args.batch_size or settings.page_classifier_batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    model_path = Settings.resolve_optional_project_path(
        args.model or settings.page_classifier_model_path
    )
    labels_path = Settings.resolve_optional_project_path(
        args.labels or settings.page_classifier_labels_path
    )
    if model_path is None or labels_path is None:
        raise RuntimeError("PAGE_CLASSIFIER_MODEL_PATH and labels path must be configured.")
    model_version = args.model_version or settings.page_classifier_model_version
    device = DevicePreference(args.device) if args.device else settings.device

    database = Database(settings)
    storage = AnnotationStorage(settings.annotation_storage_path)
    summary = PredictionSummary()
    try:
        pages = _eligible_pages(
            database,
            model_version=model_version,
            page_id=args.page_id,
            limit=args.limit,
            force=args.force,
        )
        summary.discovered = len(pages)
        if not pages:
            print("No eligible UNLABELED pages were found.")
            return summary

        classifier = OnnxPageClassifier(
            model_path=model_path,
            labels_path=labels_path,
            device=device,
            gpu_device_id=settings.gpu_device_id,
            allow_cpu_fallback=settings.allow_cpu_fallback,
        )
        print(
            f"Loaded classifier on {classifier.effective_device}; pages={len(pages)}; "
            f"batch_size={batch_size}; version={model_version}"
        )
        processed = 0
        for batch in _chunks(pages, batch_size):
            try:
                paths = [storage.resolve(page.image_relative_path) for page in batch]
                predictions = classifier.predict_paths(paths, top_k=3)
                saved, skipped = _save_predictions(
                    database,
                    pages=batch,
                    predictions=predictions,
                    model_version=model_version,
                )
                summary.predicted += saved
                summary.skipped += skipped
            except Exception as exc:
                print(f"Batch inference failed; retrying each page separately: {exc}")
                for page in batch:
                    try:
                        prediction = classifier.predict_paths(
                            [storage.resolve(page.image_relative_path)], top_k=3
                        )
                        saved, skipped = _save_predictions(
                            database,
                            pages=[page],
                            predictions=prediction,
                            model_version=model_version,
                        )
                        summary.predicted += saved
                        summary.skipped += skipped
                    except Exception as page_error:
                        summary.failed += 1
                        print(f"FAILED page={page.id}: {page_error}")
            processed += len(batch)
            print(
                f"Predicted {processed}/{len(pages)} pages "
                f"(saved={summary.predicted}, skipped={summary.skipped}, "
                f"failed={summary.failed})"
            )
        return summary
    finally:
        database.close()


def main() -> None:
    summary = run(parse_args())
    print(
        "Page prediction finished: "
        f"discovered={summary.discovered}, predicted={summary.predicted}, "
        f"skipped={summary.skipped}, failed={summary.failed}"
    )
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
