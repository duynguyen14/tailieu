from __future__ import annotations

import base64
import json
from uuid import uuid4

from sqlalchemy import func, select, true, update
from sqlalchemy.orm import joinedload, selectinload

from app.db.database import Database
from app.db.models import (
    CaseTextReconstruction,
    DocumentCase,
    DocumentCasePage,
    utc_now,
)
from app.schemas.text_reconstruction import (
    TextReconstructionCase,
    TextReconstructionDetail,
    TextReconstructionInfo,
    TextReconstructionPage,
    TextReconstructionReviewRequest,
    TextReconstructionStats,
)
from app.services.storage import AnnotationStorage


class TextReconstructionNotFoundError(LookupError):
    pass


class TextReconstructionConflictError(RuntimeError):
    pass


class TextReconstructionValidationError(ValueError):
    pass


class TextReconstructionReviewService:
    def __init__(self, *, database: Database, storage: AnnotationStorage) -> None:
        self._database = database
        self._storage = storage

    def get_statistics(self) -> TextReconstructionStats:
        with self._database.session() as session:
            return self._statistics(session)

    def get_next_review(
        self,
    ) -> tuple[TextReconstructionDetail | None, TextReconstructionStats]:
        with self._database.session() as session:
            stats = self._statistics(session)
            reconstruction = session.scalar(
                self._detail_statement()
                .where(
                    CaseTextReconstruction.is_active == true(),
                    CaseTextReconstruction.status == "GENERATED",
                )
                .order_by(
                    CaseTextReconstruction.created_at.asc(),
                    CaseTextReconstruction.id.asc(),
                )
                .limit(1)
            )
            return (
                self._detail(reconstruction) if reconstruction is not None else None,
                stats,
            )

    def get_detail(self, reconstruction_id: str) -> TextReconstructionDetail:
        with self._database.session() as session:
            reconstruction = session.scalar(
                self._detail_statement().where(
                    CaseTextReconstruction.id == reconstruction_id
                )
            )
            if reconstruction is None:
                raise TextReconstructionNotFoundError(
                    "Text reconstruction was not found."
                )
            return self._detail(reconstruction)

    def review(
        self,
        *,
        reconstruction_id: str,
        payload: TextReconstructionReviewRequest,
    ) -> TextReconstructionDetail:
        reviewer = payload.reviewed_by.strip()
        text = payload.reconstructed_text.strip()
        if not reviewer:
            raise TextReconstructionValidationError("Reviewer name cannot be empty.")
        if not text:
            raise TextReconstructionValidationError(
                "Reconstructed text cannot be empty."
            )

        with self._database.session() as session:
            current = session.scalar(
                self._detail_statement()
                .where(CaseTextReconstruction.id == reconstruction_id)
                .with_for_update()
            )
            if current is None:
                raise TextReconstructionNotFoundError(
                    "Text reconstruction was not found."
                )
            if not current.is_active:
                raise TextReconstructionConflictError(
                    "This reconstruction is no longer the active version."
                )
            if current.status not in {"GENERATED", "REVIEWED"}:
                raise TextReconstructionConflictError(
                    "Only generated or reviewed reconstruction can be reviewed."
                )

            now = utc_now()
            session.execute(
                update(CaseTextReconstruction)
                .where(
                    CaseTextReconstruction.case_id == current.case_id,
                    CaseTextReconstruction.is_active == true(),
                )
                .values(is_active=False, updated_at=now)
            )
            session.flush()
            reviewed = CaseTextReconstruction(
                id=str(uuid4()),
                case_id=current.case_id,
                parent_reconstruction_id=current.id,
                algorithm_version=current.algorithm_version,
                source_fingerprint=current.source_fingerprint,
                status="REVIEWED",
                is_active=True,
                reconstructed_text=text,
                structure_json=current.structure_json,
                page_count=current.page_count,
                token_count=current.token_count,
                reviewed_by=reviewer,
                review_note=(payload.review_note.strip() if payload.review_note else None),
                reviewed_at=now,
            )
            session.add(reviewed)
            session.flush()
            session.expire_all()
            stored = session.scalar(
                self._detail_statement().where(
                    CaseTextReconstruction.id == reviewed.id
                )
            )
            if stored is None or stored.reconstructed_text != text:
                raise TextReconstructionConflictError(
                    "SQL Server did not persist reconstructed text verbatim; "
                    "the transaction was rolled back."
                )
            return self._detail(stored)

    @staticmethod
    def _detail_statement():
        return select(CaseTextReconstruction).options(
            joinedload(CaseTextReconstruction.case).joinedload(DocumentCase.document),
            joinedload(CaseTextReconstruction.case)
            .selectinload(DocumentCase.pages)
            .joinedload(DocumentCasePage.page),
        )

    def _detail(
        self, reconstruction: CaseTextReconstruction
    ) -> TextReconstructionDetail:
        case = reconstruction.case
        pages = [
            TextReconstructionPage(
                id=case_page.page.id,
                page_number=case_page.page.page_number,
                page_type=case_page.page.label,
                page_role=case_page.page_role,
                width=case_page.page.width,
                height=case_page.page.height,
                mime_type=case_page.page.mime_type,
                image_base64=base64.b64encode(
                    self._storage.read_bytes(case_page.page.image_relative_path)
                ).decode("ascii"),
            )
            for case_page in sorted(case.pages, key=lambda item: item.page_order)
        ]
        try:
            structure = json.loads(reconstruction.structure_json)
        except json.JSONDecodeError:
            structure = {"warnings": ["INVALID_STRUCTURE_JSON"]}
        return TextReconstructionDetail(
            reconstruction=TextReconstructionInfo(
                id=reconstruction.id,
                case_id=reconstruction.case_id,
                parent_reconstruction_id=reconstruction.parent_reconstruction_id,
                algorithm_version=reconstruction.algorithm_version,
                source_fingerprint=reconstruction.source_fingerprint,
                status=reconstruction.status,
                is_active=reconstruction.is_active,
                page_count=reconstruction.page_count,
                token_count=reconstruction.token_count,
                reviewed_by=reconstruction.reviewed_by,
                review_note=reconstruction.review_note,
                reviewed_at=reconstruction.reviewed_at,
                created_at=reconstruction.created_at,
                updated_at=reconstruction.updated_at,
            ),
            case=TextReconstructionCase(
                case_id=case.id,
                document_id=case.document_id,
                original_filename=case.document.original_filename,
                case_index=case.case_index,
                case_status=case.status,
            ),
            pages=pages,
            reconstructed_text=reconstruction.reconstructed_text,
            structure=structure,
        )

    @staticmethod
    def _statistics(session) -> TextReconstructionStats:
        pending = session.scalar(
            select(func.count(CaseTextReconstruction.id)).where(
                CaseTextReconstruction.is_active == true(),
                CaseTextReconstruction.status == "GENERATED",
            )
        ) or 0
        reviewed = session.scalar(
            select(func.count(CaseTextReconstruction.id)).where(
                CaseTextReconstruction.is_active == true(),
                CaseTextReconstruction.status == "REVIEWED",
            )
        ) or 0
        return TextReconstructionStats(
            pending=pending,
            reviewed=reviewed,
            total=pending + reviewed,
        )
