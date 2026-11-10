from __future__ import annotations

from pathlib import Path

from app.domain.types import RenderedPage
from app.services.exceptions import DocumentValidationError


class PdfRenderer:
    def __init__(self, *, dpi: int, max_pages: int, max_render_pixels: int) -> None:
        self._dpi = dpi
        self._max_pages = max_pages
        self._max_render_pixels = max_render_pixels

    def render(self, pdf_path: Path) -> list[RenderedPage]:
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise RuntimeError("PyMuPDF is required to render PDF files.") from exc

        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise DocumentValidationError("The uploaded file is not a readable PDF.") from exc

        try:
            if document.needs_pass:
                raise DocumentValidationError("Password-protected PDFs are not supported.")
            if document.page_count == 0:
                raise DocumentValidationError("The PDF contains no pages.")
            if document.page_count > self._max_pages:
                raise DocumentValidationError(
                    f"PDF has {document.page_count} pages; maximum is {self._max_pages}."
                )

            zoom = self._dpi / 72.0
            matrix = pymupdf.Matrix(zoom, zoom)
            pages: list[RenderedPage] = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                expected_width = max(1, round(page.rect.width * zoom))
                expected_height = max(1, round(page.rect.height * zoom))
                expected_pixels = expected_width * expected_height
                if expected_pixels > self._max_render_pixels:
                    raise DocumentValidationError(
                        f"Page {page_index + 1} would render to {expected_pixels} pixels; "
                        f"maximum is {self._max_render_pixels}."
                    )
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pages.append(
                    RenderedPage(
                        page_index=page_index,
                        page_number=page_index + 1,
                        width=pixmap.width,
                        height=pixmap.height,
                        image_bytes=pixmap.pil_tobytes(format="PNG", compress_level=1),
                    )
                )
            return pages
        finally:
            document.close()
