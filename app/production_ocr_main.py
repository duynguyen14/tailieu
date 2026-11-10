from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.production_ocr_routes import router as production_ocr_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.services.pdf_renderer import PdfRenderer
from app.services.production_ocr import ProductionOcrService

LOGGER = logging.getLogger(__name__)


def create_ocr_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    renderer = PdfRenderer(
        dpi=settings.pdf_dpi,
        max_pages=settings.max_pages,
        max_render_pixels=settings.max_render_pixels,
    )
    service = ProductionOcrService(settings, renderer)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if settings.production_ocr_service_enabled:
                try:
                    service.initialize()
                except Exception:
                    LOGGER.exception("Production OCR model preload failed")
            yield
        finally:
            service.close()

    app = FastAPI(
        title="Internal Production OCR Service",
        version="0.1.0",
        description="Private PDF classification, OCR and text reconstruction service.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.production_ocr_service = service
    app.include_router(production_ocr_router)
    return app


app = create_ocr_app()
