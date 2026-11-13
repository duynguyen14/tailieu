from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.passport_mrz_routes import router as passport_mrz_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.services.passport_mrz.service import PassportMrzService

LOGGER = logging.getLogger(__name__)


def create_passport_mrz_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    service = PassportMrzService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if settings.passport_mrz_service_enabled and settings.passport_mrz_preload_models:
                try:
                    service.initialize()
                except Exception:
                    LOGGER.exception("Passport MRZ model preload failed")
            yield
        finally:
            service.close()

    app = FastAPI(
        title="Internal Passport MRZ Service",
        version="0.1.0",
        description="Private ONNX MRZ detection and Paddle CTC recognition service.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.passport_mrz_service = service
    app.include_router(passport_mrz_router)
    return app


app = create_passport_mrz_app()
