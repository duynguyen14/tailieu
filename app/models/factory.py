from __future__ import annotations

from dataclasses import dataclass

from app.core.config import ModelBackend, Settings
from app.models.interfaces import NoteExtractor, OCREngine, PageClassifier, TableExtractor
from app.models.mock import (
    MockNoteExtractor,
    MockOCREngine,
    MockPageClassifier,
    MockTableExtractor,
)
from app.models.unavailable import (
    UnavailableNoteExtractor,
    UnavailableOCREngine,
    UnavailablePageClassifier,
    UnavailableTableExtractor,
)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    page_classifier: PageClassifier
    ocr: OCREngine
    note_extractor: NoteExtractor
    table_extractor: TableExtractor
    adapters_loaded: bool


def build_model_bundle(settings: Settings) -> ModelBundle:
    if settings.model_backend == ModelBackend.MOCK:
        return ModelBundle(
            page_classifier=MockPageClassifier(settings.mock_page_type_list),
            ocr=MockOCREngine(),
            note_extractor=MockNoteExtractor(),
            table_extractor=MockTableExtractor(),
            adapters_loaded=True,
        )

    backend = settings.model_backend.value
    return ModelBundle(
        page_classifier=UnavailablePageClassifier(backend),
        ocr=UnavailableOCREngine(backend),
        note_extractor=UnavailableNoteExtractor(backend),
        table_extractor=UnavailableTableExtractor(backend),
        adapters_loaded=False,
    )
