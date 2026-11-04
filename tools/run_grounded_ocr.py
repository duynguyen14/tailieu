from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, true, update

from app.core.config import DevicePreference, OcrEngineKind, get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationPage,
    DocumentCase,
    DocumentCasePage,
    OcrLayoutRegion,
    OcrReview,
    OcrRun,
    OcrToken,
    utc_now,
)
from app.models.ocr_preprocessing import (
    PreparedOcrImage,
    load_rgb_image,
    ocr_quality_score,
    orientation_probe,
    png_bytes,
    prepare_ocr_image,
    recommended_upscale,
)
from app.models.ocr_region_refinement import refine_ocr_tokens
from app.models.paddle_layout_detector import (
    LayoutRegionPrediction,
    PaddleLayoutDetector,
    token_exclusion_reason,
)
from app.models.paddle_ocr_engine import OcrTokenPrediction, PaddleOcrEngine
from app.models.paddle_orientation import (
    OrientationPrediction,
    PaddleOrientationClassifier,
)
from app.models.tesseract_ocr_engine import TesseractOcrEngine
from app.schemas.annotation import ReviewStatus
from app.services.storage import AnnotationStorage


@dataclass
class OcrSummary:
    discovered: int = 0
    completed: int = 0
    failed: int = 0
    tokens: int = 0


@dataclass
class EvaluatedCandidate:
    prepared: PreparedOcrImage
    tokens: list[OcrTokenPrediction]
    score: float


def _unique_angles(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def candidate_angles(
    prediction: OrientationPrediction,
    *,
    min_confidence: float,
    fallback_all_angles: bool,
) -> list[int]:
    if prediction.confidence >= min_confidence:
        return [prediction.angle]
    if not fallback_all_angles:
        return [prediction.angle]
    return _unique_angles([prediction.predicted_angle, 0, 90, 180, 270])


def _write_candidate(path: Path, prepared: PreparedOcrImage) -> None:
    path.write_bytes(png_bytes(prepared.image))


def effective_preprocess_version(settings) -> str:
    version = f"{settings.ocr_preprocess_version}|engine={settings.ocr_engine.value}"
    if settings.ocr_enable_selective_refinement:
        version += f"|refinement={settings.ocr_refinement_version}"
    return version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPU-first OCR and store grounded text tokens with source boxes."
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all eligible pages")
    parser.add_argument("--page-id", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--pending-review-only",
        action="store_true",
        help=(
            "OCR pages without a review decision that have not completed the "
            "current preprocessing version"
        ),
    )
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
    pending_review_only: bool,
    preprocess_version: str,
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
        if pending_review_only:
            historical_review_exists = (
                select(OcrReview.id)
                .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
                .where(OcrRun.page_id == AnnotationPage.id)
                .exists()
            )
            current_preprocess_exists = (
                select(OcrRun.id)
                .where(
                    OcrRun.page_id == AnnotationPage.id,
                    OcrRun.status == "COMPLETED",
                    OcrRun.is_active == true(),
                    OcrRun.preprocess_version == preprocess_version,
                )
                .exists()
            )
            statement = statement.where(
                ~historical_review_exists,
                ~current_preprocess_exists,
            )
        elif not force:
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
    regions: list[LayoutRegionPrediction],
    auto_ignore_overlap: float,
    orientation: OrientationPrediction,
    processed_image_relative_path: str | None,
    processed_width: int,
    processed_height: int,
    preprocess_metadata: dict[str, object],
    preprocess_version: str,
    processing_time_ms: int,
) -> None:
    with database.session() as session:
        run = session.get(OcrRun, run_id)
        if run is None:
            raise RuntimeError(f"OCR run disappeared: {run_id}")
        run.tokens.clear()
        run.layout_regions.clear()
        for region_index, region in enumerate(regions, start=1):
            run.layout_regions.append(
                OcrLayoutRegion(
                    id=str(uuid4()),
                    region_index=region_index,
                    region_type=region.region_type,
                    confidence=Decimal(str(region.confidence)),
                    x1=region.bbox[0],
                    y1=region.bbox[1],
                    x2=region.bbox[2],
                    y2=region.bbox[3],
                )
            )
        for token in tokens:
            ignored_reason = token_exclusion_reason(
                token.bbox, regions, threshold=auto_ignore_overlap
            )
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
                    include_in_extraction=ignored_reason is None,
                    ignored_reason=ignored_reason,
                )
            )
        run.status = "COMPLETED"
        run.orientation_angle = orientation.angle
        run.orientation_confidence = Decimal(str(orientation.confidence))
        run.processed_image_relative_path = processed_image_relative_path
        run.processed_width = processed_width
        run.processed_height = processed_height
        run.preprocess_metadata_json = json.dumps(
            preprocess_metadata, ensure_ascii=False, separators=(",", ":")
        )
        run.preprocess_version = preprocess_version
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
    if args.force and args.pending_review_only:
        raise ValueError("--force cannot be combined with --pending-review-only")
    settings = get_settings()
    preprocess_version = effective_preprocess_version(settings)
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
            pending_review_only=args.pending_review_only,
            preprocess_version=preprocess_version,
        )
        summary.discovered = len(pages)
        if not pages:
            print("No eligible reviewed pages were found for OCR.")
            return summary

        device = DevicePreference(args.device) if args.device else None
        if settings.ocr_engine == OcrEngineKind.TESSERACT:
            engine = TesseractOcrEngine(settings)
        else:
            engine = PaddleOcrEngine(settings, device_override=device)
        use_paddle_aux_models = (
            settings.ocr_engine == OcrEngineKind.PADDLE
            or settings.tesseract_use_paddle_aux_models
        )
        auxiliary_device = (
            engine.effective_device
            if settings.ocr_engine == OcrEngineKind.PADDLE
            else (
                f"gpu:{settings.ocr_runtime_gpu_device_id}"
                if settings.ocr_runtime_device == DevicePreference.CUDA
                else settings.ocr_runtime_device.value
            )
        )
        orientation_classifier = (
            PaddleOrientationClassifier(settings, device=auxiliary_device)
            if settings.ocr_enable_orientation and use_paddle_aux_models
            else None
        )
        layout_detector = (
            PaddleLayoutDetector(settings, device=auxiliary_device)
            if settings.ocr_enable_exclusion_detection and use_paddle_aux_models
            else None
        )
        orientation_model_name = (
            orientation_classifier.model_name if orientation_classifier else "disabled"
        )
        layout_model_name = layout_detector.model_name if layout_detector else "disabled"
        refinement_model_name = (
            settings.ocr_refinement_version
            if settings.ocr_enable_selective_refinement
            else "disabled"
        )
        print(
            f"Loaded {engine.model_name} on {engine.effective_device}; pages={len(pages)}; "
            f"orientation={orientation_model_name}; "
            f"exclusion_layout={layout_model_name}; "
            f"refinement={refinement_model_name}"
        )
        for position, page in enumerate(pages, start=1):
            started = time.perf_counter()
            run_id = start_run(
                database,
                page_id=page.id,
                model_name=(
                    f"{engine.model_name}"
                    f"|orientation={orientation_model_name}"
                    f"|layout={layout_model_name}"
                    f"|refinement={refinement_model_name}"
                ),
                model_version=(
                    settings.tesseract_model_version
                    if settings.ocr_engine == OcrEngineKind.TESSERACT
                    else settings.ocr_model_version
                ),
                device=engine.effective_device,
            )
            processed_relative_path: str | None = None
            source_image = None
            selected: EvaluatedCandidate | None = None
            try:
                source_image_path = storage.resolve(page.image_relative_path)
                source_image = load_rgb_image(source_image_path)
                probe_image, probe_detection = orientation_probe(
                    source_image,
                    thumbnail_max_side=settings.ocr_content_thumbnail_max_side,
                    padding_ratio=settings.ocr_content_padding_ratio,
                    min_components=settings.ocr_content_min_components,
                    min_content_area_ratio=settings.ocr_content_min_area_ratio,
                    max_crop_area_ratio=settings.ocr_content_max_crop_area_ratio,
                )
                orientation_prediction = (
                    orientation_classifier.predict_image(probe_image)
                    if orientation_classifier
                    else OrientationPrediction(
                        angle=0,
                        confidence=(
                            0.0
                            if settings.ocr_engine == OcrEngineKind.TESSERACT
                            and settings.tesseract_try_all_orientations
                            else 1.0
                        ),
                        predicted_angle=0,
                    )
                )
                probe_image.close()
                angles = candidate_angles(
                    orientation_prediction,
                    min_confidence=settings.ocr_orientation_min_confidence,
                    fallback_all_angles=(
                        settings.ocr_orientation_fallback_all_angles
                        and (
                            orientation_classifier is not None
                            or (
                                settings.ocr_engine == OcrEngineKind.TESSERACT
                                and settings.tesseract_try_all_orientations
                            )
                        )
                    ),
                )
                prepare_kwargs = {
                    "enable_content_crop": settings.ocr_enable_content_crop,
                    "thumbnail_max_side": settings.ocr_content_thumbnail_max_side,
                    "padding_ratio": settings.ocr_content_padding_ratio,
                    "min_components": settings.ocr_content_min_components,
                    "min_content_area_ratio": settings.ocr_content_min_area_ratio,
                    "max_crop_area_ratio": settings.ocr_content_max_crop_area_ratio,
                    "max_processed_side": settings.ocr_upscale_max_processed_side,
                }
                candidate_scores: dict[str, float] = {}
                with tempfile.TemporaryDirectory(prefix="grounded-ocr-") as temp_dir:
                    temp_root = Path(temp_dir)
                    for angle in angles:
                        prepared = prepare_ocr_image(
                            source_image,
                            rotation=angle,
                            scale=1.0,
                            **prepare_kwargs,
                        )
                        candidate_path = temp_root / f"angle-{angle}.png"
                        _write_candidate(candidate_path, prepared)
                        candidate_tokens = engine.predict(candidate_path)
                        score = ocr_quality_score(candidate_tokens)
                        candidate_scores[str(angle)] = round(score, 6)
                        candidate = EvaluatedCandidate(
                            prepared=prepared,
                            tokens=candidate_tokens,
                            score=score,
                        )
                        if selected is None or candidate.score > selected.score:
                            if selected is not None:
                                selected.prepared.image.close()
                            selected = candidate
                        else:
                            prepared.image.close()

                    if selected is None:
                        raise RuntimeError("OCR preprocessing produced no candidate image.")

                    upscale_attempted = 1.0
                    if settings.ocr_enable_small_text_upscale:
                        upscale_attempted = recommended_upscale(
                            selected.tokens,
                            target_text_height=settings.ocr_upscale_target_text_height,
                            max_factor=settings.ocr_upscale_max_factor,
                        )
                    if upscale_attempted > 1.05:
                        upscaled = prepare_ocr_image(
                            source_image,
                            rotation=selected.prepared.rotation,
                            scale=upscale_attempted,
                            **prepare_kwargs,
                        )
                        upscaled_path = temp_root / "selected-upscaled.png"
                        _write_candidate(upscaled_path, upscaled)
                        upscaled_tokens = engine.predict(upscaled_path)
                        upscaled_score = ocr_quality_score(upscaled_tokens)
                        candidate_scores[
                            f"{upscaled.rotation}@{upscaled.scale:.3f}x"
                        ] = round(upscaled_score, 6)
                        # A tiny score variation is normal after interpolation.
                        # Prefer the larger image when it preserves quality, as
                        # its boxes are easier to review and correct.
                        if upscaled_score >= selected.score - 0.05:
                            selected.prepared.image.close()
                            selected = EvaluatedCandidate(
                                prepared=upscaled,
                                tokens=upscaled_tokens,
                                score=upscaled_score,
                            )
                        else:
                            upscaled.image.close()

                    final_path = temp_root / "selected-final.png"
                    _write_candidate(final_path, selected.prepared)
                    tokens = selected.tokens
                    table_metadata: dict[str, object] = {"enabled": False}
                    if settings.ocr_engine == OcrEngineKind.TESSERACT:
                        table_result = engine.enhance_tables(final_path, tokens)
                        tokens = table_result.tokens
                        table_metadata = table_result.metadata
                    refinement_metadata: dict[str, object] = {
                        "enabled": False,
                        "version": settings.ocr_refinement_version,
                        "attempted": 0,
                        "replaced": 0,
                        "unresolved": 0,
                        "items": [],
                    }
                    if settings.ocr_enable_selective_refinement:
                        refinement = refine_ocr_tokens(
                            selected.prepared.image,
                            tokens,
                            temp_dir=temp_root / "refinement",
                            predict_many=engine.predict_refinement_many,
                            scales=settings.ocr_refinement_scale_list,
                            confidence_threshold=(
                                settings.ocr_refinement_confidence_threshold
                            ),
                            candidate_min_confidence=(
                                settings.ocr_refinement_candidate_min_confidence
                            ),
                            agreement_similarity=(
                                settings.ocr_refinement_agreement_similarity
                            ),
                            min_confidence_gain=(
                                settings.ocr_refinement_min_confidence_gain
                            ),
                            max_regions=settings.ocr_refinement_max_regions,
                            max_seconds=settings.ocr_refinement_max_seconds,
                            padding_ratio=settings.ocr_refinement_padding_ratio,
                            max_crop_side=settings.ocr_refinement_max_crop_side,
                            protect_digits=settings.ocr_refinement_protect_digits,
                            version=settings.ocr_refinement_version,
                        )
                        tokens = refinement.tokens
                        refinement_metadata = {
                            "enabled": True,
                            **refinement.metadata,
                        }
                    regions = (
                        layout_detector.predict(final_path) if layout_detector else []
                    )
                    processed_width, processed_height = selected.prepared.processed_size
                    if selected.prepared.changed:
                        processed_relative_path = (
                            f"ocr_preprocessed/{page.document_id}/{page.id}/{run_id}.png"
                        )
                        storage.write_bytes(
                            processed_relative_path, final_path.read_bytes()
                        )

                    preprocess_metadata = selected.prepared.metadata()
                    preprocess_metadata.update(
                        {
                            "orientation_probe_bbox": (
                                list(probe_detection.bbox)
                                if probe_detection.bbox
                                else None
                            ),
                            "orientation_probe_components": (
                                probe_detection.component_count
                            ),
                            "orientation_predicted_angle": (
                                orientation_prediction.predicted_angle
                            ),
                            "orientation_confidence": round(
                                orientation_prediction.confidence, 6
                            ),
                            "orientation_candidates": angles,
                            "candidate_scores": candidate_scores,
                            "selected_quality_score": round(selected.score, 6),
                            "upscale_attempted": round(upscale_attempted, 6),
                            "table_cell_ocr": table_metadata,
                            "selective_refinement": refinement_metadata,
                        }
                    )
                    orientation = OrientationPrediction(
                        angle=selected.prepared.rotation,
                        confidence=orientation_prediction.confidence,
                        predicted_angle=orientation_prediction.predicted_angle,
                    )
                    selected.prepared.image.close()
                source_image.close()
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                complete_run(
                    database,
                    run_id=run_id,
                    tokens=tokens,
                    regions=regions,
                    auto_ignore_overlap=settings.ocr_auto_ignore_overlap,
                    orientation=orientation,
                    processed_image_relative_path=processed_relative_path,
                    processed_width=processed_width,
                    processed_height=processed_height,
                    preprocess_metadata=preprocess_metadata,
                    preprocess_version=preprocess_version,
                    processing_time_ms=elapsed_ms,
                )
                summary.completed += 1
                summary.tokens += len(tokens)
                print(
                    f"[{position}/{len(pages)}] page={page.id} "
                    f"tokens={len(tokens)} exclusion_regions={len(regions)} "
                    f"orientation={orientation.angle} "
                    f"scale={preprocess_metadata['scale']} "
                    f"refined={refinement_metadata.get('replaced', 0)}/"
                    f"{refinement_metadata.get('attempted', 0)} "
                    f"time={elapsed_ms}ms"
                )
            except Exception as exc:
                if selected is not None:
                    selected.prepared.image.close()
                if source_image is not None:
                    source_image.close()
                if processed_relative_path:
                    storage.remove_file(processed_relative_path)
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
