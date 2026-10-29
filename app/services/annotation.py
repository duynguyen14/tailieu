from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.db.database import Database
from app.db.models import AnnotationDocument, AnnotationPage, utc_now
from app.schemas.annotation import AnnotationLabel, ReviewStatus
from app.services.exceptions import DocumentValidationError
from app.services.pdf_renderer import PdfRenderer
from app.services.storage import AnnotationStorage


class AnnotationNotFoundError(LookupError):
    pass


class AnnotationService:
    def __init__(
        self,
        *,
        database: Database,
        storage: AnnotationStorage,
        renderer: PdfRenderer,
        max_file_size_bytes: int,
    ) -> None:
        self._database = database
        self._storage = storage
        self._renderer = renderer
        self._max_file_size_bytes = max_file_size_bytes

    def import_pdf(
        self,
        *,
        source: BinaryIO,
        original_filename: str,
    ) -> tuple[bool, AnnotationDocument]:
        filename = (original_filename or "document.pdf").replace("\\", "/").split("/")[-1]
        filename = Path(filename).name
        if Path(filename).suffix.lower() != ".pdf":
            raise DocumentValidationError("Only PDF files are supported.")

        document_id = str(uuid4())
        staging_relative_path = f"_staging/{document_id}.pdf"
        staging_path = self._storage.resolve(staging_relative_path)
        staging_path.parent.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha256()
        total_bytes = 0
        first_bytes = b""
        try:
            with staging_path.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    if not first_bytes:
                        first_bytes = chunk[:5]
                    total_bytes += len(chunk)
                    if total_bytes > self._max_file_size_bytes:
                        raise DocumentValidationError(
                            "Uploaded PDF exceeds the configured size limit."
                        )
                    digest.update(chunk)
                    output.write(chunk)

            if total_bytes == 0:
                raise DocumentValidationError("The uploaded file is empty.")
            if first_bytes != b"%PDF-":
                raise DocumentValidationError("The uploaded file does not have a PDF signature.")

            sha256 = digest.hexdigest()
            existing = self._find_document_by_sha256(sha256)
            if existing is not None:
                self._storage.remove_file(staging_relative_path)
                return False, existing

            document_root = f"documents/{document_id}"
            pdf_relative_path = f"{document_root}/source/original.pdf"
            pdf_path = self._storage.resolve(pdf_relative_path)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            staging_path.replace(pdf_path)

            try:
                rendered_pages = self._renderer.render(pdf_path)
                page_records: list[AnnotationPage] = []
                for rendered in rendered_pages:
                    image_relative_path = (
                        f"{document_root}/pages/page_{rendered.page_number:04d}.png"
                    )
                    stored_relative_path = self._storage.write_bytes(
                        image_relative_path,
                        rendered.image_bytes,
                    )
                    page_records.append(
                        AnnotationPage(
                            id=str(uuid4()),
                            document_id=document_id,
                            page_index=rendered.page_index,
                            page_number=rendered.page_number,
                            image_relative_path=stored_relative_path,
                            mime_type=rendered.mime_type,
                            width=rendered.width,
                            height=rendered.height,
                            review_status=ReviewStatus.UNLABELED.value,
                        )
                    )

                document = AnnotationDocument(
                    id=document_id,
                    original_filename=filename,
                    pdf_relative_path=self._storage.normalize(pdf_relative_path),
                    sha256=sha256,
                    page_count=len(page_records),
                    review_status=ReviewStatus.UNLABELED.value,
                    pages=page_records,
                )
                with self._database.session() as session:
                    session.add(document)
                return True, document
            except Exception:
                self._storage.remove_tree(document_root)
                raise
        except Exception:
            self._storage.remove_file(staging_relative_path)
            raise

    def list_documents(
        self,
        *,
        offset: int,
        limit: int,
        review_status: ReviewStatus | None = None,
    ) -> tuple[list[AnnotationDocument], int]:
        with self._database.session() as session:
            filters = []
            if review_status is not None:
                filters.append(AnnotationDocument.review_status == review_status.value)

            total = session.scalar(select(func.count(AnnotationDocument.id)).where(*filters)) or 0
            statement = (
                select(AnnotationDocument)
                .options(selectinload(AnnotationDocument.pages))
                .where(*filters)
                .order_by(AnnotationDocument.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            documents = list(session.scalars(statement).unique().all())
            return documents, total

    def get_document(self, document_id: str) -> AnnotationDocument:
        with self._database.session() as session:
            statement = (
                select(AnnotationDocument)
                .options(selectinload(AnnotationDocument.pages))
                .where(AnnotationDocument.id == document_id)
            )
            document = session.scalar(statement)
            if document is None:
                raise AnnotationNotFoundError("Annotation document was not found.")
            return document

    def get_page(
        self, document_id: str, page_number: int
    ) -> tuple[AnnotationDocument, AnnotationPage]:
        document = self.get_document(document_id)
        page = next((item for item in document.pages if item.page_number == page_number), None)
        if page is None:
            raise AnnotationNotFoundError("Annotation page was not found.")
        return document, page

    def read_page_image(self, page: AnnotationPage) -> bytes:
        return self._storage.read_bytes(page.image_relative_path)

    def get_next_unreviewed_page(
        self,
    ) -> tuple[AnnotationDocument | None, AnnotationPage | None, int, int, int]:
        """Return the oldest unreviewed page and aggregate queue counts."""

        with self._database.session() as session:
            total = session.scalar(select(func.count(AnnotationPage.id))) or 0
            reviewed = (
                session.scalar(
                    select(func.count(AnnotationPage.id)).where(
                        AnnotationPage.review_status == ReviewStatus.REVIEWED.value
                    )
                )
                or 0
            )
            pending = (
                session.scalar(
                    select(func.count(AnnotationPage.id)).where(
                        AnnotationPage.review_status == ReviewStatus.UNLABELED.value
                    )
                )
                or 0
            )
            row = session.execute(
                select(AnnotationDocument, AnnotationPage)
                .join(
                    AnnotationPage,
                    AnnotationPage.document_id == AnnotationDocument.id,
                )
                .where(AnnotationPage.review_status == ReviewStatus.UNLABELED.value)
                .order_by(
                    AnnotationDocument.created_at.asc(),
                    AnnotationPage.page_index.asc(),
                )
                .limit(1)
            ).first()
            if row is None:
                return None, None, pending, reviewed, total
            document, page = row
            return document, page, pending, reviewed, total

    def review_page(
        self,
        *,
        document_id: str,
        page_number: int,
        label: AnnotationLabel,
        reviewed_by: str,
        review_note: str | None,
    ) -> tuple[str, AnnotationPage]:
        with self._database.session() as session:
            document = session.get(AnnotationDocument, document_id)
            if document is None:
                raise AnnotationNotFoundError("Annotation document was not found.")

            page = session.scalar(
                select(AnnotationPage).where(
                    AnnotationPage.document_id == document_id,
                    AnnotationPage.page_number == page_number,
                )
            )
            if page is None:
                raise AnnotationNotFoundError("Annotation page was not found.")

            page.label = label.value
            page.review_status = ReviewStatus.REVIEWED.value
            page.reviewed_by = reviewed_by.strip()
            page.review_note = review_note.strip() if review_note else None
            page.reviewed_at = utc_now()
            page.updated_at = utc_now()
            session.flush()

            reviewed_pages = (
                session.scalar(
                    select(func.count(AnnotationPage.id)).where(
                        AnnotationPage.document_id == document_id,
                        AnnotationPage.review_status == ReviewStatus.REVIEWED.value,
                    )
                )
                or 0
            )
            if reviewed_pages >= document.page_count:
                document.review_status = ReviewStatus.REVIEWED.value
            elif reviewed_pages > 0:
                document.review_status = ReviewStatus.IN_PROGRESS.value
            else:
                document.review_status = ReviewStatus.UNLABELED.value
            document.updated_at = utc_now()
            session.flush()
            return document.review_status, page

    def _find_document_by_sha256(self, sha256: str) -> AnnotationDocument | None:
        with self._database.session() as session:
            statement = (
                select(AnnotationDocument)
                .options(selectinload(AnnotationDocument.pages))
                .where(AnnotationDocument.sha256 == sha256)
            )
            return session.scalar(statement)
