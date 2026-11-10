from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.production_classification_routes import router as production_classification_router
from app.api.production_extraction_routes import router as production_extraction_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.services.pdf_renderer import PdfRenderer
from app.services.production_extraction import ProductionExtractionService

LOGGER = logging.getLogger(__name__)


def create_production_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    renderer = PdfRenderer(
        dpi=settings.pdf_dpi,
        max_pages=settings.max_pages,
        max_render_pixels=settings.max_render_pixels,
    )
    service = ProductionExtractionService(settings, renderer)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if settings.production_extraction_enabled:
                try:
                    service.initialize()
                except Exception:
                    LOGGER.exception("Production student model preload failed")
            yield
        finally:
            service.close()

    app = FastAPI(
        title="Production Document Extraction API",
        version="0.1.0",
        description="Public PDF to validated business JSON API.",
        lifespan=lifespan,
    )
    cors_origins = settings.cors_origin_list
    allow_all_origins = "*" in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all_origins else cors_origins,
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.production_extraction_service = service
    app.include_router(production_classification_router, prefix=settings.api_prefix)
    app.include_router(production_extraction_router, prefix=settings.api_prefix)
    return app


app = create_production_app()
