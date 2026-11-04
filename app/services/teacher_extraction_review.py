from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import func, select, true, update
from sqlalchemy.orm import joinedload

from app.db.database import Database
from app.db.models import (
    TeacherExtractionAttempt,
    TeacherExtractionJob,
    TeacherExtractionReview,
    utc_now,
)
from app.schemas.teacher_extraction import (
    CompactDocumentExtraction,
    TeacherExtractionAttemptInfo,
    TeacherExtractionCaseInfo,
    TeacherExtractionJobInfo,
    TeacherExtractionReviewDetail,
    TeacherExtractionReviewInfo,
    TeacherExtractionReviewRequest,
    TeacherExtractionReviewStats,
)


class TeacherExtractionReviewNotFoundError(LookupError):
    pass


class TeacherExtractionReviewConflictError(RuntimeError):
    pass


class TeacherExtractionReviewValidationError(ValueError):
    pass


class TeacherExtractionReviewService:
    def __init__(self, *, database: Database) -> None:
        self._database = database

    def get_statistics(self) -> TeacherExtractionReviewStats:
        with self._database.session() as session:
            return self._statistics(session)

    def get_next_review(
        self,
    ) -> tuple[
        TeacherExtractionReviewDetail | None,
        TeacherExtractionReviewStats,
    ]:
        with self._database.session() as session:
            stats = self._statistics(session)
            review = session.scalar(
                self._detail_statement()
                .where(
                    TeacherExtractionReview.is_active == true(),
                    TeacherExtractionReview.review_status == "PENDING",
                )
                .order_by(
                    TeacherExtractionReview.created_at.asc(),
                    TeacherExtractionReview.id.asc(),
                )
                .limit(1)
            )
            return self._detail(review) if review is not None else None, stats

    def get_detail(self, review_id: str) -> TeacherExtractionReviewDetail:
        with self._database.session() as session:
            review = session.scalar(
                self._detail_statement().where(TeacherExtractionReview.id == review_id)
            )
            if review is None:
                raise TeacherExtractionReviewNotFoundError(
                    "Teacher extraction review was not found."
                )
            return self._detail(review)

    def review(
        self,
        *,
        review_id: str,
        payload: TeacherExtractionReviewRequest,
    ) -> TeacherExtractionReviewDetail:
        reviewer = payload.reviewed_by.strip()
        if not reviewer:
            raise TeacherExtractionReviewValidationError(
                "Reviewer name cannot be empty."
            )

        try:
            validated = CompactDocumentExtraction.model_validate(
                payload.reviewed_output
            )
        except ValidationError as exc:
            raise TeacherExtractionReviewValidationError(
                f"Compact JSON is invalid: {exc}"
            ) from exc

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
            if validated.case_id != current.job.case_id:
                raise TeacherExtractionReviewValidationError(
                    "JSON case_id does not match the teacher job."
                )
            if validated.document_id != current.job.document_id:
                raise TeacherExtractionReviewValidationError(
                    "JSON document_id does not match the teacher job."
                )

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
                    validated.model_dump(mode="json"),
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
            session.expire_all()
            stored = session.scalar(
                self._detail_statement().where(
                    TeacherExtractionReview.id == replacement.id
                )
            )
            if stored is None:
                raise TeacherExtractionReviewConflictError(
                    "SQL Server did not persist the new review version."
                )
            return self._detail(stored)

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
        self, review: TeacherExtractionReview
    ) -> TeacherExtractionReviewDetail:
        job = review.job
        attempt = review.source_attempt
        output = self._json_value(review.reviewed_output_json)
        if not isinstance(output, dict):
            raise TeacherExtractionReviewConflictError(
                "Stored teacher review output is not a JSON object."
            )
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
            .where(TeacherExtractionReview.is_active == true())
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
