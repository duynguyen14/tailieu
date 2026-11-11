from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select, true, update

from app.core.config import DevicePreference, OcrEngineKind, Settings, get_settings
from app.db.database import Database
from app.db.models import AnnotationPage, OcrRun, OcrToken, utc_now
from app.services.storage import AnnotationStorage

LOGGER = logging.getLogger(__name__)


def get_existing_page_tokens(
    database: Database, page_id: str
) -> list[OcrToken] | None:
    """Fetch completed tokens from the active OCR run for a page, if any."""
    with database.session() as session:
        active_run = session.scalar(
            select(OcrRun)
            .where(
                OcrRun.page_id == page_id,
                OcrRun.is_active == true(),
                OcrRun.status == "COMPLETED",
            )
            .order_by(OcrRun.created_at.desc(), OcrRun.id.desc())
            .limit(1)
        )
        if active_run is None:
            return None

        tokens = list(
            session.scalars(
                select(OcrToken)
                .where(OcrToken.ocr_run_id == active_run.id)
                .order_by(OcrToken.token_index.asc())
            ).all()
        )
        # Detach/keep tokens data
        return tokens


def run_ocr_for_page(
    page: AnnotationPage,
    storage: AnnotationStorage,
    settings: Settings | None = None,
    database: Database | None = None,
    *,
    force_rerun: bool = False,
    save_to_db: bool = True,
) -> list[Any]:
    """Retrieves or executes OCR for an annotation page.

    1. Checks if an active completed OcrRun already exists in the database.
    2. If not found (or force_rerun=True), executes PaddleOCR or Tesseract on the page image.
    3. Persists the OcrRun and OcrTokens to the database for future runs.
    4. Falls back gracefully to an empty token list if OCR runtime is unavailable.
    """
    settings = settings or get_settings()

    # Step 1: Check existing OCR tokens in database
    if not force_rerun and database is not None:
        existing = get_existing_page_tokens(database, page.id)
        if existing:
            return existing

    # Step 2: Initialize OCR engine
    try:
        if settings.ocr_engine == OcrEngineKind.TESSERACT:
            from app.models.tesseract_ocr_engine import TesseractOcrEngine

            engine: Any = TesseractOcrEngine(settings)
        else:
            from app.models.paddle_ocr_engine import PaddleOcrEngine

            engine = PaddleOcrEngine(settings)
    except Exception as exc:
        LOGGER.warning("OCR engine initialization failed for page %s: %s", page.id, exc)
        return []

    # Step 3: Run OCR prediction on the image
    image_path = storage.resolve(page.image_relative_path)
    if not image_path.exists():
        LOGGER.warning("Page image file does not exist: %s", image_path)
        return []

    start_time = time.perf_counter()
    try:
        from app.models.ocr_preprocessing import load_rgb_image, prepare_ocr_image

        source_img = load_rgb_image(image_path)
        with tempfile.TemporaryDirectory(prefix="visa-ocr-") as tmp_dir:
            prepared = prepare_ocr_image(
                source_img,
                rotation=0,
                scale=1.0,
                enable_content_crop=False,
            )
            candidate_path = Path(tmp_dir) / "page.png"
            prepared.image.save(candidate_path, format="PNG")
            prepared.image.close()
            source_img.close()

            predicted_tokens = engine.predict(candidate_path)
    except Exception as exc:
        LOGGER.warning("OCR prediction failed for page %s: %s", page.id, exc)
        return []

    elapsed_ms = round((time.perf_counter() - start_time) * 1000)

    # Step 4: Persist OcrRun and tokens into the database
    if save_to_db and database is not None:
        try:
            with database.session() as session:
                # Deactivate any previous runs
                session.execute(
                    update(OcrRun)
                    .where(OcrRun.page_id == page.id, OcrRun.is_active == true())
                    .values(is_active=False, updated_at=utc_now())
                )

                run_id = str(uuid4())
                ocr_run = OcrRun(
                    id=run_id,
                    page_id=page.id,
                    status="COMPLETED",
                    is_active=True,
                    model_name=getattr(engine, "model_name", "ocr-engine"),
                    model_version=(
                        settings.tesseract_model_version
                        if settings.ocr_engine == OcrEngineKind.TESSERACT
                        else settings.ocr_model_version
                    ),
                    device=getattr(engine, "effective_device", "unknown"),
                    processing_time_ms=elapsed_ms,
                    processed_width=page.width,
                    processed_height=page.height,
                    preprocess_version=settings.ocr_preprocess_version,
                )
                session.add(ocr_run)
                session.flush()

                for idx, tok in enumerate(predicted_tokens, start=1):
                    x1, y1, x2, y2 = tok.bbox
                    token_record = OcrToken(
                        id=str(uuid4()),
                        ocr_run_id=run_id,
                        token_index=idx,
                        token_id=getattr(tok, "token_id", f"tok_{idx}"),
                        text=tok.text,
                        confidence=tok.confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        polygon_json=None,
                        include_in_extraction=True,
                    )
                    session.add(token_record)
                session.commit()
        except Exception as exc:
            LOGGER.error("Failed to save OCR run to database for page %s: %s", page.id, exc)

    return predicted_tokens
