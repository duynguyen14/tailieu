from __future__ import annotations

from app.domain.types import RenderedPage
from app.models.interfaces import NoteExtractor, OCREngine, PageClassifier, TableExtractor
from app.schemas.document import (
    NoteExtraction,
    OCRToken,
    PageClassification,
    PageType,
    TableExtraction,
)


class MockPageClassifier(PageClassifier):
    """Deterministic page labels used to exercise the pipeline without model files."""

    def __init__(self, page_types: list[str]) -> None:
        self._page_types = page_types

    def classify(self, pages: list[RenderedPage]) -> list[PageClassification]:
        results: list[PageClassification] = []
        for page in pages:
            configured = (
                self._page_types[page.page_index]
                if page.page_index < len(self._page_types)
                else PageType.OTHER.value
            )
            try:
                page_type = PageType(configured)
            except ValueError:
                page_type = PageType.OTHER
            results.append(PageClassification(page_type=page_type, confidence=1.0))
        return results


class MockOCREngine(OCREngine):
    def recognize(self, page: RenderedPage) -> list[OCRToken]:
        return []


class MockNoteExtractor(NoteExtractor):
    def extract(self, page: RenderedPage, tokens: list[OCRToken]) -> NoteExtraction:
        return NoteExtraction()


class MockTableExtractor(TableExtractor):
    def extract(self, page: RenderedPage, tokens: list[OCRToken]) -> TableExtraction:
        return TableExtraction()
