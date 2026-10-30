from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.types import RenderedPage
from app.schemas.document import (
    NoteExtraction,
    OCRToken,
    PageClassification,
    TableExtraction,
)


class PageClassifier(ABC):
    @abstractmethod
    def classify(self, pages: list[RenderedPage]) -> list[PageClassification]:
        """Classify a batch of rendered pages in input order."""


class OCREngine(ABC):
    @abstractmethod
    def recognize(self, page: RenderedPage) -> list[OCRToken]:
        """Return OCR tokens with pixel-space bounding boxes."""


class NoteExtractor(ABC):
    @abstractmethod
    def extract(self, page: RenderedPage, tokens: list[OCRToken]) -> NoteExtraction:
        """Extract structured diplomatic-note fields."""


class TableExtractor(ABC):
    @abstractmethod
    def extract(self, page: RenderedPage, tokens: list[OCRToken]) -> TableExtraction:
        """Extract table structure and applicant rows."""
