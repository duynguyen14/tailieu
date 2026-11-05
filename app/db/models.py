from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Unicode,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class AnnotationDocument(Base):
    __tablename__ = "annotation_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    pdf_relative_path: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="UNLABELED", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    pages: Mapped[list[AnnotationPage]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="AnnotationPage.page_index",
    )


class AnnotationPage(Base):
    __tablename__ = "annotation_pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_index", name="uq_document_page_index"),
        UniqueConstraint("document_id", "page_number", name="uq_document_page_number"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    image_relative_path: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False, default="image/png")
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model_predicted_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_prediction_confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(9, 6), nullable=True
    )
    model_prediction_topk_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_prediction_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_predicted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    review_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="UNLABELED", index=True
    )
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    document: Mapped[AnnotationDocument] = relationship(back_populates="pages")


class PageLabelAudit(Base):
    __tablename__ = "page_label_audits"
    __table_args__ = (
        CheckConstraint(
            "audit_status IN ('PENDING', 'REVIEWED')",
            name="ck_page_label_audit_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    original_label: Mapped[str] = mapped_column(String(64), nullable=False)
    audited_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    audit_status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="PENDING", index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    source_reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    audited_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    audit_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    audited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    page: Mapped[AnnotationPage] = relationship()


class DocumentCase(Base):
    __tablename__ = "document_cases"
    __table_args__ = (
        CheckConstraint(
            "status IN ('AUTO_GROUPED', 'NEEDS_REVIEW', 'REVIEWED')",
            name="ck_document_case_status",
        ),
        CheckConstraint("case_index >= 1", name="ck_document_case_index"),
        CheckConstraint(
            "start_page_number >= 1 AND end_page_number >= start_page_number",
            name="ck_document_case_page_range",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    case_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    start_page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    end_page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    document: Mapped[AnnotationDocument] = relationship()
    pages: Mapped[list[DocumentCasePage]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="DocumentCasePage.page_order",
    )
    extraction_annotations: Mapped[list[CaseExtractionAnnotation]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="CaseExtractionAnnotation.created_at",
    )
    text_reconstructions: Mapped[list[CaseTextReconstruction]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="CaseTextReconstruction.created_at",
    )


class DocumentCasePage(Base):
    __tablename__ = "document_case_pages"
    __table_args__ = (
        UniqueConstraint("case_id", "page_id", name="uq_document_case_page"),
        UniqueConstraint("case_id", "page_order", name="uq_document_case_page_order"),
        CheckConstraint("page_order >= 1", name="ck_document_case_page_order"),
        CheckConstraint(
            "page_role IN ('NOTE', 'APPLICANT_DATA')",
            name="ck_document_case_page_role",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        ForeignKey("document_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_pages.id"), nullable=False, index=True
    )
    page_order: Mapped[int] = mapped_column(Integer, nullable=False)
    page_role: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    case: Mapped[DocumentCase] = relationship(back_populates="pages")
    page: Mapped[AnnotationPage] = relationship()


class CaseExtractionAnnotation(Base):
    __tablename__ = "case_extraction_annotations"
    __table_args__ = (
        CheckConstraint(
            "review_status IN ('NEEDS_LABEL', 'REVIEWED')",
            name="ck_case_extraction_review_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        ForeignKey("document_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    target_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="NEEDS_LABEL", index=True
    )
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    case: Mapped[DocumentCase] = relationship(back_populates="extraction_annotations")


class CaseTextReconstruction(Base):
    __tablename__ = "case_text_reconstructions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('GENERATED', 'REVIEWED')",
            name="ck_case_text_reconstruction_status",
        ),
        CheckConstraint("page_count >= 1", name="ck_case_text_reconstruction_pages"),
        CheckConstraint("token_count >= 0", name="ck_case_text_reconstruction_tokens"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        ForeignKey("document_cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_reconstruction_id: Mapped[str | None] = mapped_column(
        ForeignKey("case_text_reconstructions.id"), nullable=True
    )
    algorithm_version: Mapped[str] = mapped_column(String(100), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="GENERATED", index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    reconstructed_text: Mapped[str] = mapped_column(Unicode(), nullable=False)
    structure_json: Mapped[str] = mapped_column(Unicode(), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    case: Mapped[DocumentCase] = relationship(back_populates="text_reconstructions")
    teacher_jobs: Mapped[list[TeacherExtractionJob]] = relationship(
        back_populates="reconstruction",
        cascade="all, delete-orphan",
        order_by="TeacherExtractionJob.created_at",
    )


class TeacherExtractionJob(Base):
    __tablename__ = "teacher_extraction_jobs"
    __table_args__ = (
        UniqueConstraint("input_hash", name="uq_teacher_extraction_job_input_hash"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')",
            name="ck_teacher_extraction_job_status",
        ),
        CheckConstraint(
            "attempt_count >= 0", name="ck_teacher_extraction_job_attempt_count"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    reconstruction_id: Mapped[str] = mapped_column(
        ForeignKey("case_text_reconstructions.id"), nullable=False, index=True
    )
    case_id: Mapped[str] = mapped_column(
        ForeignKey("document_cases.id"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_documents.id"), nullable=False, index=True
    )
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    input_text: Mapped[str] = mapped_column(Unicode(), nullable=False)
    source_structure_json: Mapped[str] = mapped_column(Unicode(), nullable=False)
    reconstruction_version: Mapped[str] = mapped_column(String(100), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    teacher_model: Mapped[str] = mapped_column(String(255), nullable=False)
    teacher_endpoint: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    reconstruction: Mapped[CaseTextReconstruction] = relationship(
        back_populates="teacher_jobs"
    )
    case: Mapped[DocumentCase] = relationship()
    document: Mapped[AnnotationDocument] = relationship()
    attempts: Mapped[list[TeacherExtractionAttempt]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="TeacherExtractionAttempt.attempt_number",
    )
    reviews: Mapped[list[TeacherExtractionReview]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="TeacherExtractionReview.created_at",
    )


class TeacherExtractionAttempt(Base):
    __tablename__ = "teacher_extraction_attempts"
    __table_args__ = (
        UniqueConstraint(
            "job_id", "attempt_number", name="uq_teacher_extraction_attempt_number"
        ),
        CheckConstraint(
            "attempt_number >= 1", name="ck_teacher_extraction_attempt_number"
        ),
        CheckConstraint(
            "validation_status IN "
            "('VALID', 'INVALID', 'PARSE_ERROR', 'HTTP_ERROR', 'TRANSPORT_ERROR')",
            name="ck_teacher_extraction_attempt_validation",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("teacher_extraction_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    request_json: Mapped[str] = mapped_column(Unicode(), nullable=False)
    raw_response_json: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    assistant_content: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    parsed_output_json: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    validation_status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    validation_errors_json: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    job: Mapped[TeacherExtractionJob] = relationship(back_populates="attempts")


class TeacherExtractionReview(Base):
    __tablename__ = "teacher_extraction_reviews"
    __table_args__ = (
        CheckConstraint(
            "review_status IN ('PENDING', 'APPROVED', 'REJECTED')",
            name="ck_teacher_extraction_review_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("teacher_extraction_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("teacher_extraction_attempts.id"), nullable=False, index=True
    )
    parent_review_id: Mapped[str | None] = mapped_column(
        ForeignKey("teacher_extraction_reviews.id"), nullable=True
    )
    reviewed_output_json: Mapped[str] = mapped_column(Unicode(), nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    job: Mapped[TeacherExtractionJob] = relationship(back_populates="reviews")
    source_attempt: Mapped[TeacherExtractionAttempt] = relationship()


class OcrRun(Base):
    __tablename__ = "ocr_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PROCESSING', 'COMPLETED', 'FAILED')",
            name="ck_ocr_run_status",
        ),
        CheckConstraint(
            "processing_time_ms IS NULL OR processing_time_ms >= 0",
            name="ck_ocr_run_processing_time",
        ),
        CheckConstraint(
            "orientation_angle IN (0, 90, 180, 270)",
            name="ck_ocr_run_orientation_angle",
        ),
        CheckConstraint(
            "orientation_confidence IS NULL OR "
            "(orientation_confidence >= 0 AND orientation_confidence <= 1)",
            name="ck_ocr_run_orientation_confidence",
        ),
        CheckConstraint(
            "processed_width IS NULL OR processed_width > 0",
            name="ck_ocr_run_processed_width",
        ),
        CheckConstraint(
            "processed_height IS NULL OR processed_height > 0",
            name="ck_ocr_run_processed_height",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    device: Mapped[str] = mapped_column(String(32), nullable=False)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    orientation_angle: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    orientation_confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(9, 6), nullable=True
    )
    processed_image_relative_path: Mapped[str | None] = mapped_column(
        String(1000), nullable=True
    )
    processed_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    preprocess_metadata_json: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    preprocess_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    page: Mapped[AnnotationPage] = relationship()
    tokens: Mapped[list[OcrToken]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="OcrToken.token_index",
    )
    layout_regions: Mapped[list[OcrLayoutRegion]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="OcrLayoutRegion.region_index",
    )


class OcrToken(Base):
    __tablename__ = "ocr_tokens"
    __table_args__ = (
        UniqueConstraint("ocr_run_id", "token_index", name="uq_ocr_token_index"),
        UniqueConstraint("ocr_run_id", "token_id", name="uq_ocr_token_id"),
        CheckConstraint("token_index >= 1", name="ck_ocr_token_index"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_ocr_token_confidence",
        ),
        CheckConstraint(
            "x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1",
            name="ck_ocr_token_bbox",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ocr_run_id: Mapped[str] = mapped_column(
        ForeignKey("ocr_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_id: Mapped[str] = mapped_column(String(20), nullable=False)
    # OCR text is multilingual. ``Text`` binds as VARCHAR/TEXT through the
    # SQL Server pyodbc dialect and can replace Vietnamese characters with
    # ``?`` before they reach an NVARCHAR column.  Unicode() binds as
    # NVARCHAR(max), matching the production schema.
    text: Mapped[str] = mapped_column(Unicode(), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(9, 6), nullable=False)
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    polygon_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    include_in_extraction: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    ignored_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    run: Mapped[OcrRun] = relationship(back_populates="tokens")


class OcrLayoutRegion(Base):
    __tablename__ = "ocr_layout_regions"
    __table_args__ = (
        UniqueConstraint("ocr_run_id", "region_index", name="uq_ocr_layout_region_index"),
        CheckConstraint("region_index >= 1", name="ck_ocr_layout_region_index"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_ocr_layout_region_confidence",
        ),
        CheckConstraint(
            "x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1",
            name="ck_ocr_layout_region_bbox",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ocr_run_id: Mapped[str] = mapped_column(
        ForeignKey("ocr_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    region_index: Mapped[int] = mapped_column(Integer, nullable=False)
    region_type: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(9, 6), nullable=False)
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    run: Mapped[OcrRun] = relationship(back_populates="layout_regions")


class OcrReview(Base):
    __tablename__ = "ocr_reviews"
    __table_args__ = (
        CheckConstraint("status IN ('REVIEWED')", name="ck_ocr_review_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ocr_run_id: Mapped[str] = mapped_column(
        ForeignKey("ocr_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_review_id: Mapped[str | None] = mapped_column(
        ForeignKey("ocr_reviews.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="REVIEWED")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    include_in_llm: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    reviewed_by: Mapped[str] = mapped_column(String(100), nullable=False)
    review_note: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class OcrReviewToken(Base):
    __tablename__ = "ocr_review_tokens"
    __table_args__ = (
        UniqueConstraint("review_id", "token_order", name="uq_ocr_review_token_order"),
        CheckConstraint("token_order >= 1", name="ck_ocr_review_token_order"),
        CheckConstraint(
            "x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1",
            name="ck_ocr_review_token_bbox",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    review_id: Mapped[str] = mapped_column(
        ForeignKey("ocr_reviews.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_token_id: Mapped[str | None] = mapped_column(
        ForeignKey("ocr_tokens.id"), nullable=True, index=True
    )
    token_order: Mapped[int] = mapped_column(Integer, nullable=False)
    # Keep this explicitly Unicode even when the physical SQL Server column
    # already is NVARCHAR.  The ORM type controls pyodbc parameter binding.
    reviewed_text: Mapped[str] = mapped_column(Unicode(), nullable=False)
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    include_in_extraction: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    ignored_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class OcrReviewRegion(Base):
    __tablename__ = "ocr_review_regions"
    __table_args__ = (
        UniqueConstraint("review_id", "region_order", name="uq_ocr_review_region_order"),
        CheckConstraint("region_order >= 1", name="ck_ocr_review_region_order"),
        CheckConstraint(
            "x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1",
            name="ck_ocr_review_region_bbox",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    review_id: Mapped[str] = mapped_column(
        ForeignKey("ocr_reviews.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_region_id: Mapped[str | None] = mapped_column(
        ForeignKey("ocr_layout_regions.id"), nullable=True, index=True
    )
    region_order: Mapped[int] = mapped_column(Integer, nullable=False)
    region_type: Mapped[str] = mapped_column(String(32), nullable=False)
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class TableAnnotation(Base):
    __tablename__ = "table_annotations"
    __table_args__ = (
        CheckConstraint("source IN ('MODEL', 'HUMAN')", name="ck_table_annotation_source"),
        CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'PREDICTED', 'REVIEWED', 'FAILED')",
            name="ck_table_annotation_status",
        ),
        CheckConstraint(
            "processing_time_ms IS NULL OR processing_time_ms >= 0",
            name="ck_table_annotation_processing_time",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_annotation_id: Mapped[str | None] = mapped_column(
        ForeignKey("table_annotations.id"), nullable=True
    )
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    page: Mapped[AnnotationPage] = relationship()
    regions: Mapped[list[TableRegion]] = relationship(
        back_populates="annotation",
        cascade="all, delete-orphan",
        order_by="TableRegion.table_index",
    )


class TableRegion(Base):
    __tablename__ = "table_regions"
    __table_args__ = (
        UniqueConstraint("annotation_id", "table_index", name="uq_table_region_index"),
        CheckConstraint("table_index >= 0", name="ck_table_region_index"),
        CheckConstraint(
            "x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1",
            name="ck_table_region_bbox",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_table_region_confidence",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    annotation_id: Mapped[str] = mapped_column(
        ForeignKey("table_annotations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    table_index: Mapped[int] = mapped_column(Integer, nullable=False)
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(9, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    annotation: Mapped[TableAnnotation] = relationship(back_populates="regions")
    elements: Mapped[list[TableStructureElement]] = relationship(
        back_populates="table_region",
        cascade="all, delete-orphan",
        order_by="TableStructureElement.element_type, "
        "TableStructureElement.row_index, TableStructureElement.column_index",
    )


class TableStructureElement(Base):
    __tablename__ = "table_structure_elements"
    __table_args__ = (
        CheckConstraint(
            "element_type IN ('HEADER_ROW', 'DATA_ROW', 'COLUMN', 'CELL')",
            name="ck_table_structure_element_type",
        ),
        CheckConstraint("row_span >= 1 AND column_span >= 1", name="ck_table_element_span"),
        CheckConstraint(
            "x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1",
            name="ck_table_element_bbox",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_table_element_confidence",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    table_region_id: Mapped[str] = mapped_column(
        ForeignKey("table_regions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    element_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    row_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    row_span: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    column_span: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(9, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    table_region: Mapped[TableRegion] = relationship(back_populates="elements")


class KieAnnotationRun(Base):
    __tablename__ = "kie_annotation_runs"
    __table_args__ = (
        CheckConstraint("status IN ('GENERATED', 'IN_REVIEW', 'COMPLETED')", name="ck_kie_run_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    teacher_review_id: Mapped[str] = mapped_column(ForeignKey("teacher_extraction_reviews.id"), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("document_cases.id"), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("annotation_documents.id"), nullable=False, index=True)
    algorithm_version: Mapped[str] = mapped_column(String(100), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="GENERATED", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class KiePageAnnotation(Base):
    __tablename__ = "kie_page_annotations"
    __table_args__ = (
        UniqueConstraint("run_id", "page_id", name="uq_kie_page_run_page"),
        CheckConstraint("ser_status IN ('PENDING', 'APPROVED')", name="ck_kie_page_ser_status"),
        CheckConstraint("re_status IN ('BLOCKED', 'PENDING', 'APPROVED')", name="ck_kie_page_re_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("kie_annotation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    page_id: Mapped[str] = mapped_column(ForeignKey("annotation_pages.id"), nullable=False, index=True)
    ocr_review_id: Mapped[str] = mapped_column(ForeignKey("ocr_reviews.id"), nullable=False, index=True)
    page_order: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    image_relative_path: Mapped[str] = mapped_column(String(500), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    orientation_angle: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    preprocess_metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ser_status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING", index=True)
    re_status: Mapped[str] = mapped_column(String(20), nullable=False, default="BLOCKED", index=True)
    ser_reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    re_reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ser_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    re_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class KieFieldProposal(Base):
    __tablename__ = "kie_field_proposals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("kie_annotation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    field_path: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    expected_value: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    match_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    chosen_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidates_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class KieEntityAnnotation(Base):
    __tablename__ = "kie_entity_annotations"
    __table_args__ = (
        CheckConstraint("x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1", name="ck_kie_entity_bbox"),
        CheckConstraint("nx1 >= 0 AND ny1 >= 0 AND nx2 <= 1000 AND ny2 <= 1000 AND nx2 > nx1 AND ny2 > ny1", name="ck_kie_entity_normalized_bbox"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_annotation_id: Mapped[str] = mapped_column(ForeignKey("kie_page_annotations.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_entity_id: Mapped[str | None] = mapped_column(ForeignKey("kie_entity_annotations.id"), nullable=True)
    field_proposal_id: Mapped[str | None] = mapped_column(ForeignKey("kie_field_proposals.id"), nullable=True, index=True)
    entity_key: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    value: Mapped[str] = mapped_column(Unicode(), nullable=False)
    field_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    applicant_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)
    nx1: Mapped[int] = mapped_column(Integer, nullable=False)
    ny1: Mapped[int] = mapped_column(Integer, nullable=False)
    nx2: Mapped[int] = mapped_column(Integer, nullable=False)
    ny2: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(9, 6), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="MODEL")
    candidate_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class KieRelationAnnotation(Base):
    __tablename__ = "kie_relation_annotations"
    __table_args__ = (CheckConstraint("head_entity_id <> tail_entity_id", name="ck_kie_relation_not_self"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_annotation_id: Mapped[str] = mapped_column(ForeignKey("kie_page_annotations.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_relation_id: Mapped[str | None] = mapped_column(ForeignKey("kie_relation_annotations.id"), nullable=True)
    head_entity_id: Mapped[str] = mapped_column(ForeignKey("kie_entity_annotations.id"), nullable=False, index=True)
    tail_entity_id: Mapped[str] = mapped_column(ForeignKey("kie_entity_annotations.id"), nullable=False, index=True)
    relation_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    applicant_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(9, 6), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="MODEL")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class KieAnnotationReview(Base):
    __tablename__ = "kie_annotation_reviews"
    __table_args__ = (CheckConstraint("mode IN ('SER', 'RE')", name="ck_kie_review_mode"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    page_annotation_id: Mapped[str] = mapped_column(ForeignKey("kie_page_annotations.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_review_id: Mapped[str | None] = mapped_column(ForeignKey("kie_annotation_reviews.id"), nullable=True)
    mode: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    reviewed_by: Mapped[str] = mapped_column(String(100), nullable=False)
    review_note: Mapped[str | None] = mapped_column(Unicode(), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
