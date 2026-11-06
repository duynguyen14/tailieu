from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import func, select, true, update
from sqlalchemy.orm import joinedload

from app.db.database import Database
from app.db.models import (
    TeacherExtractionJob,
    TeacherExtractionReview,
    utc_now,
)
from app.schemas.student_extraction import (
    STUDENT_MODEL_ID,
    STUDENT_SCHEMA_VERSION,
    build_student_user_prompt,
    compact_document_to_student_target,
    student_schema_descriptor,
)
from app.schemas.teacher_extraction import (
    CompactDocumentExtraction,
    TeacherExtractionAttemptInfo,
    TeacherExtractionCaseInfo,
    TeacherExtractionJobInfo,
    TeacherExtractionReviewDetail,
    TeacherExtractionReviewInfo,
    TeacherExtractionReviewRequest,
    TeacherExtractionReviewSaveResponse,
    TeacherExtractionReviewStats,
    TeacherStudentTrainingPreview,
)
from app.services.ocr_review import OcrReviewService


class TeacherExtractionReviewNotFoundError(LookupError):
    pass


class TeacherExtractionReviewConflictError(RuntimeError):
    pass


class TeacherExtractionReviewValidationError(ValueError):
    pass


class TeacherExtractionReviewService:
    def __init__(
        self,
        *,
        database: Database,
        ocr_review_service: OcrReviewService,
    ) -> None:
        self._database = database
        self._ocr_review_service = ocr_review_service

    def get_statistics(self) -> TeacherExtractionReviewStats:
        with self._database.session() as session:
            return self._statistics(session)

    def get_next_review(
        self,
        *,
        from_index: int | None = None,
        to_index: int | None = None,
    ) -> tuple[
        TeacherExtractionReviewDetail | None,
        TeacherExtractionReviewStats,
        int | None,
    ]:
        with self._database.session() as session:
            stats = self._statistics(session)
            current_index: int | None = None
            active_review_exists = (
                select(TeacherExtractionReview.id)
                .where(
                    TeacherExtractionReview.job_id == TeacherExtractionJob.id,
                    TeacherExtractionReview.is_active == true(),
                )
                .exists()
            )
            ordered_jobs = (
                select(TeacherExtractionJob.id)
                .where(
                    TeacherExtractionJob.is_active == true(),
                    active_review_exists,
                )
                .order_by(
                    TeacherExtractionJob.created_at.asc(),
                    TeacherExtractionJob.id.asc(),
                )
            )

            if from_index is not None and to_index is not None:
                # Rank immutable jobs before filtering their current review
                # status. Saving a new review version must not shift another
                # reviewer's manually assigned interval.
                range_job_ids = list(
                    session.scalars(
                        ordered_jobs
                        .offset(from_index)
                        .limit(to_index - from_index + 1)
                    ).all()
                )
                review = None
                if range_job_ids:
                    review = session.scalar(
                        self._detail_statement()
                        .join(
                            TeacherExtractionJob,
                            TeacherExtractionJob.id == TeacherExtractionReview.job_id,
                        )
                        .where(
                            TeacherExtractionReview.is_active == true(),
                            TeacherExtractionReview.review_status == "PENDING",
                            TeacherExtractionJob.is_active == true(),
                            TeacherExtractionJob.id.in_(range_job_ids),
                        )
                        .order_by(
                            TeacherExtractionJob.created_at.asc(),
                            TeacherExtractionJob.id.asc(),
                        )
                        .limit(1)
                    )
                    if review is not None:
                        current_index = from_index + range_job_ids.index(review.job_id)
            else:
                review = session.scalar(
                    self._detail_statement()
                    .join(
                        TeacherExtractionJob,
                        TeacherExtractionJob.id == TeacherExtractionReview.job_id,
                    )
                    .where(
                        TeacherExtractionReview.is_active == true(),
                        TeacherExtractionReview.review_status == "PENDING",
                        TeacherExtractionJob.is_active == true(),
                    )
                    .order_by(
                        TeacherExtractionJob.created_at.asc(),
                        TeacherExtractionJob.id.asc(),
                    )
                    .limit(1)
                )
            return (
                self._detail(session, review) if review is not None else None,
                stats,
                current_index,
            )

    def get_detail(self, review_id: str) -> TeacherExtractionReviewDetail:
        with self._database.session() as session:
            review = session.scalar(
                self._detail_statement().where(TeacherExtractionReview.id == review_id)
            )
            if review is None:
                raise TeacherExtractionReviewNotFoundError(
                    "Teacher extraction review was not found."
                )
            return self._detail(session, review)

    def review(
        self,
        *,
        review_id: str,
        payload: TeacherExtractionReviewRequest,
    ) -> TeacherExtractionReviewSaveResponse:
        reviewer = payload.reviewed_by.strip()
        if not reviewer:
            raise TeacherExtractionReviewValidationError(
                "Reviewer name cannot be empty."
            )

        with self._database.session() as session:
            current = session.scalar(
                self._detail_statement()
                .where(TeacherExtractionReview.id == review_id)
                .with_for_update()
            )
            if current is None:
                raise TeacherExtractionReviewNotFoundError(
                    "Teacher extraction review was not found."
                )
            if not current.is_active:
                raise TeacherExtractionReviewConflictError(
                    "This teacher review is no longer the active version."
                )
            if not current.job.is_active:
                raise TeacherExtractionReviewConflictError(
                    "This teacher review belongs to an obsolete teacher job."
                )
            # The review form is the source of truth. Persist its JSON without
            # running the teacher schema validator again; only immutable source
            # identifiers are enforced by the server.
            reviewed_output = dict(payload.reviewed_output)
            reviewed_output["case_id"] = current.job.case_id
            reviewed_output["document_id"] = current.job.document_id
            reviewed_output.setdefault("schema_version", current.job.schema_version)

            now = utc_now()
            session.execute(
                update(TeacherExtractionReview)
                .where(
                    TeacherExtractionReview.job_id == current.job_id,
                    TeacherExtractionReview.is_active == true(),
                )
                .values(is_active=False, updated_at=now)
            )
            session.flush()
            replacement = TeacherExtractionReview(
                id=str(uuid4()),
                job_id=current.job_id,
                source_attempt_id=current.source_attempt_id,
                parent_review_id=current.id,
                reviewed_output_json=json.dumps(
                    reviewed_output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                review_status=payload.decision,
                is_active=True,
                reviewed_by=reviewer,
                review_note=(
                    payload.review_note.strip() if payload.review_note else None
                ),
                reviewed_at=now,
            )
            session.add(replacement)
            session.flush()
            return TeacherExtractionReviewSaveResponse(
                review=TeacherExtractionReviewInfo(
                    id=replacement.id,
                    parent_review_id=replacement.parent_review_id,
                    review_status=replacement.review_status,
                    is_active=replacement.is_active,
                    reviewed_by=replacement.reviewed_by,
                    review_note=replacement.review_note,
                    reviewed_at=replacement.reviewed_at,
                    created_at=replacement.created_at,
                    updated_at=replacement.updated_at,
                ),
                reviewed_output=reviewed_output,
            )

    @staticmethod
    def _detail_statement():
        return select(TeacherExtractionReview).options(
            joinedload(TeacherExtractionReview.job).joinedload(
                TeacherExtractionJob.document
            ),
            joinedload(TeacherExtractionReview.source_attempt),
        )

    @staticmethod
    def _json_value(value: str | None) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _detail(
        self, session, review: TeacherExtractionReview
    ) -> TeacherExtractionReviewDetail:
        job = review.job
        attempt = review.source_attempt
        output = self._json_value(review.reviewed_output_json)
        if not isinstance(output, dict):
            raise TeacherExtractionReviewConflictError(
                "Stored teacher review output is not a JSON object."
            )
        preview_error: str | None = None
        try:
            extraction = CompactDocumentExtraction.model_validate(output)
            compact_target = compact_document_to_student_target(extraction)
        except ValidationError as exc:
            # A reviewer must still be able to reopen and repair a loosely
            # structured JSON version. Approval/export can be handled after the
            # form has brought it back to the canonical shape.
            compact_target = {}
            preview_error = str(exc)

        approved_ocr_pages = self._ocr_review_service.get_document_pages_in_session(
            session,
            document_id=job.document_id,
            included_only=True,
        )

        descriptor = student_schema_descriptor()
        if preview_error:
            descriptor["preview_error"] = preview_error
        return TeacherExtractionReviewDetail(
            review=TeacherExtractionReviewInfo(
                id=review.id,
                parent_review_id=review.parent_review_id,
                review_status=review.review_status,
                is_active=review.is_active,
                reviewed_by=review.reviewed_by,
                review_note=review.review_note,
                reviewed_at=review.reviewed_at,
                created_at=review.created_at,
                updated_at=review.updated_at,
            ),
            job=TeacherExtractionJobInfo(
                id=job.id,
                reconstruction_id=job.reconstruction_id,
                input_hash=job.input_hash,
                schema_version=job.schema_version,
                prompt_version=job.prompt_version,
                teacher_model=job.teacher_model,
                teacher_endpoint=job.teacher_endpoint,
                status=job.status,
                attempt_count=job.attempt_count,
                completed_at=job.completed_at,
                created_at=job.created_at,
            ),
            attempt=TeacherExtractionAttemptInfo(
                id=attempt.id,
                attempt_number=attempt.attempt_number,
                validation_status=attempt.validation_status,
                http_status=attempt.http_status,
                latency_ms=attempt.latency_ms,
                prompt_tokens=attempt.prompt_tokens,
                completion_tokens=attempt.completion_tokens,
                total_tokens=attempt.total_tokens,
                finish_reason=attempt.finish_reason,
                error_message=attempt.error_message,
                created_at=attempt.created_at,
            ),
            case=TeacherExtractionCaseInfo(
                case_id=job.case_id,
                document_id=job.document_id,
                original_filename=job.document.original_filename,
            ),
            input_text=job.input_text,
            reviewed_output=output,
            approved_ocr_pages=approved_ocr_pages,
            student_training=TeacherStudentTrainingPreview(
                model_id=STUDENT_MODEL_ID,
                schema_version=STUDENT_SCHEMA_VERSION,
                input_text=job.input_text,
                user_prompt=build_student_user_prompt(job.input_text),
                compact_target=compact_target,
                schema_descriptor=descriptor,
            ),
            raw_response=self._json_value(attempt.raw_response_json),
            validation_errors=self._json_value(attempt.validation_errors_json),
        )

    @staticmethod
    def _statistics(session) -> TeacherExtractionReviewStats:
        rows = session.execute(
            select(
                TeacherExtractionReview.review_status,
                func.count(TeacherExtractionReview.id),
            )
            .join(
                TeacherExtractionJob,
                TeacherExtractionJob.id == TeacherExtractionReview.job_id,
            )
            .where(
                TeacherExtractionReview.is_active == true(),
                TeacherExtractionJob.is_active == true(),
            )
            .group_by(TeacherExtractionReview.review_status)
        ).all()
        counts = {status: count for status, count in rows}
        pending = counts.get("PENDING", 0)
        approved = counts.get("APPROVED", 0)
        rejected = counts.get("REJECTED", 0)
        return TeacherExtractionReviewStats(
            pending=pending,
            approved=approved,
            rejected=rejected,
            total=pending + approved + rejected,
        )
