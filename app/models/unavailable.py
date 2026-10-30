from __future__ import annotations

from app.domain.types import RenderedPage
from app.models.interfaces import NoteExtractor, OCREngine, PageClassifier, TableExtractor
from app.schemas.document import NoteExtraction, OCRToken, PageClassification, TableExtraction


class ModelAdapterUnavailableError(RuntimeError):
    pass


class _UnavailableAdapter:
    def __init__(self, backend: str) -> None:
        self._backend = backend

    def _raise(self, component: str) -> None:
        raise ModelAdapterUnavailableError(
            f"{component} adapter for MODEL_BACKEND={self._backend} is not implemented yet. "
            "Use MODEL_BACKEND=mock to exercise the API skeleton."
        )


class UnavailablePageClassifier(_UnavailableAdapter, PageClassifier):
    def classify(self, pages: list[RenderedPage]) -> list[PageClassification]:
        self._raise("page_classifier")


class UnavailableOCREngine(_UnavailableAdapter, OCREngine):
    def recognize(self, page: RenderedPage) -> list[OCRToken]:
        self._raise("ocr")


class UnavailableNoteExtractor(_UnavailableAdapter, NoteExtractor):
    def extract(self, page: RenderedPage, tokens: list[OCRToken]) -> NoteExtraction:
        self._raise("note_extractor")


class UnavailableTableExtractor(_UnavailableAdapter, TableExtractor):
    def extract(self, page: RenderedPage, tokens: list[OCRToken]) -> TableExtraction:
        self._raise("table_structure")
