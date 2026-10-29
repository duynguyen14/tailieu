from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from app.core.config import ModelBackend, Settings
from app.core.runtime import RuntimeSelection
from app.models.factory import ModelBundle
from app.models.unavailable import ModelAdapterUnavailableError
from app.schemas.document import (
    DocumentData,
    DocumentExtractionResponse,
    ExtractionStatus,
    ImageSize,
    PageClassification,
    PageExtraction,
    PageType,
    RuntimeInfo,
)
from app.services.exceptions import PipelineUnavailableError
from app.services.pdf_renderer import PdfRenderer


class DocumentPipeline:
    """Synchronous orchestration. No queue, background task, or async model call."""

    def __init__(
        self,
        *,
        settings: Settings,
        runtime: RuntimeSelection,
        models: ModelBundle,
        renderer: PdfRenderer,
    ) -> None:
        self._settings = settings
        self._runtime = runtime
        self._models = models
        self._renderer = renderer

    def extract(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> DocumentExtractionResponse:
        started_at = time.perf_counter()
        document_warnings: list[str] = []

        if self._settings.model_backend == ModelBackend.MOCK:
            document_warnings.append(
                "MODEL_BACKEND=mock: classifications are configured by MOCK_PAGE_TYPES; "
                "OCR and extraction results are placeholders."
            )

        pages = self._renderer.render(pdf_path)
        try:
            classifications = self._classify_pages(pages)
            page_results = [
                self._process_page(page, classification)
                for page, classification in zip(pages, classifications, strict=True)
            ]
        except ModelAdapterUnavailableError as exc:
            raise PipelineUnavailableError(str(exc)) from exc

        note_fields = {}
        applicants = []
        for page_result in page_results:
            if page_result.note is not None:
                note_fields.update(page_result.note.fields)
                applicants.extend(page_result.note.applicants)
            if page_result.table is not None:
                applicants.extend(row.fields for row in page_result.table.rows)

        all_warnings = document_warnings + [
            warning for page_result in page_results for warning in page_result.warnings
        ]
        status = ExtractionStatus.NEEDS_REVIEW if all_warnings else ExtractionStatus.COMPLETED
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        return DocumentExtractionResponse(
            document_id=str(uuid4()),
            filename=filename,
            sha256=sha256,
            status=status,
            processing_time_ms=round(elapsed_ms, 3),
            runtime=RuntimeInfo(
                backend=self._runtime.backend,
                requested_device=self._runtime.requested_device,
                effective_device=self._runtime.effective_device,
            ),
            pages=page_results,
            data=DocumentData(note_fields=note_fields, applicants=applicants),
            warnings=all_warnings,
        )

    def _classify_pages(self, pages) -> list[PageClassification]:
        if not self._settings.enable_page_classifier:
            return [PageClassification(page_type=PageType.OTHER, confidence=0.0) for _ in pages]
        classifications = self._models.page_classifier.classify(pages)
        if len(classifications) != len(pages):
            raise RuntimeError(
                "Page classifier returned a different number of results than input pages."
            )
        return classifications

    def _process_page(self, page, classification: PageClassification) -> PageExtraction:
        warnings: list[str] = []
        tokens = []
        note = None
        table = None

        should_ocr = classification.page_type in {
            PageType.DIPLOMATIC_NOTE,
            PageType.APPLICANT_TABLE,
        }
        if should_ocr and self._settings.enable_ocr:
            tokens = self._models.ocr.recognize(page)
        elif should_ocr:
            warnings.append("OCR is disabled; extraction will not contain recognized text.")

        if classification.page_type == PageType.DIPLOMATIC_NOTE:
            if self._settings.enable_note_extractor:
                note = self._models.note_extractor.extract(page, tokens)
            else:
                warnings.append("Note extractor is disabled.")
        elif classification.page_type == PageType.APPLICANT_TABLE:
            if self._settings.enable_table_structure:
                table = self._models.table_extractor.extract(page, tokens)
            else:
                warnings.append("Table structure extraction is disabled.")

        if classification.confidence < self._settings.low_confidence_threshold:
            warnings.append(
                f"Page classification confidence {classification.confidence:.3f} is below "
                f"threshold {self._settings.low_confidence_threshold:.3f}."
            )

        return PageExtraction(
            page_index=page.page_index,
            page_number=page.page_number,
            page_type=classification.page_type,
            classification_confidence=classification.confidence,
            image_size=ImageSize(width=page.width, height=page.height),
            ocr_tokens=tokens,
            note=note,
            table=table,
            warnings=warnings,
        )
