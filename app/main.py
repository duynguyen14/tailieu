from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.annotation_routes import router as annotation_router
from app.api.ocr_review_routes import router as ocr_review_router
from app.api.kie_review_routes import router as kie_review_router
from app.api.page_label_audit_routes import router as page_label_audit_router
from app.api.routes import router
from app.api.table_annotation_routes import router as table_annotation_router
from app.api.teacher_extraction_routes import router as teacher_extraction_router
from app.api.text_reconstruction_routes import router as text_reconstruction_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.runtime import resolve_runtime
from app.db.database import Database
from app.models.factory import build_model_bundle
from app.services.annotation import AnnotationService
from app.services.ocr_review import OcrReviewService
from app.services.kie_review import KieReviewService
from app.services.page_label_audit import PageLabelAuditService
from app.services.pdf_renderer import PdfRenderer
from app.services.pipeline import DocumentPipeline
from app.services.storage import AnnotationStorage
from app.services.table_annotation import TableAnnotationService
from app.services.teacher_extraction_review import TeacherExtractionReviewService
from app.services.text_reconstruction_review import TextReconstructionReviewService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    runtime = resolve_runtime(settings)
    models = build_model_bundle(settings)
    renderer = PdfRenderer(
        dpi=settings.pdf_dpi,
        max_pages=settings.max_pages,
        max_render_pixels=settings.max_render_pixels,
    )
    database = Database(settings)
    if settings.auto_create_db:
        database.create_tables()
    annotation_storage = AnnotationStorage(settings.annotation_storage_path)
    annotation_service = AnnotationService(
        database=database,
        storage=annotation_storage,
        renderer=renderer,
        max_file_size_bytes=settings.max_file_size_bytes,
    )
    table_annotation_service = TableAnnotationService(
        database=database,
        storage=annotation_storage,
    )
    page_label_audit_service = PageLabelAuditService(
        database=database,
        storage=annotation_storage,
    )
    ocr_review_service = OcrReviewService(
        database=database,
        storage=annotation_storage,
        line_tolerance_ratio=settings.text_reconstruction_line_tolerance,
    )
    text_reconstruction_review_service = TextReconstructionReviewService(
        database=database,
        storage=annotation_storage,
    )
    teacher_extraction_review_service = TeacherExtractionReviewService(
        database=database,
        ocr_review_service=ocr_review_service,
    )
    kie_review_service = KieReviewService(database=database, storage=annotation_storage)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            database.close()

    pipeline = DocumentPipeline(
        settings=settings,
        runtime=runtime,
        models=models,
        renderer=renderer,
    )

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Synchronous document annotation and extraction API.",
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
    app.state.runtime = runtime
    app.state.models = models
    app.state.pipeline = pipeline
    app.state.database = database
    app.state.annotation_storage = annotation_storage
    app.state.annotation_service = annotation_service
    app.state.table_annotation_service = table_annotation_service
    app.state.page_label_audit_service = page_label_audit_service
    app.state.ocr_review_service = ocr_review_service
    app.state.text_reconstruction_review_service = text_reconstruction_review_service
    app.state.teacher_extraction_review_service = teacher_extraction_review_service
    app.state.kie_review_service = kie_review_service
    app.include_router(router, prefix=settings.api_prefix)
    app.include_router(annotation_router, prefix=settings.api_prefix)
    app.include_router(page_label_audit_router, prefix=settings.api_prefix)
    app.include_router(ocr_review_router, prefix=settings.api_prefix)
    app.include_router(table_annotation_router, prefix=settings.api_prefix)
    app.include_router(text_reconstruction_router, prefix=settings.api_prefix)
    app.include_router(teacher_extraction_router, prefix=settings.api_prefix)
    app.include_router(kie_review_router, prefix=settings.api_prefix)
    return app


app = create_app()
