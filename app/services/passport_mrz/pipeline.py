from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.schemas.production_extraction import PassportMrzCheck, PassportMrzPageResult
from app.services.passport_mrz.cropper import MrzLineCrop, crop_mrz_lines, enhance_lines
from app.services.passport_mrz.detector import OnnxMrzDetector
from app.services.passport_mrz.parser import MrzParseResult, parse_mrz
from app.services.passport_mrz.recognizer import PaddleMrzRecognizer


@dataclass
class _PageWork:
    page_number: int
    detector_confidence: float = 0.0
    detector_ms: float = 0.0
    crop_ms: float = 0.0
    deskew_angle: float = 0.0
    lines: list[MrzLineCrop] | None = None
    ocr_results: list[dict[str, Any]] | None = None
    warning: str | None = None


class PassportMrzPipeline:
    def __init__(self, detector: OnnxMrzDetector, recognizer: PaddleMrzRecognizer) -> None:
        self.detector = detector
        self.recognizer = recognizer

    def read_batch(
        self, pages: list[tuple[int, np.ndarray]]
    ) -> list[PassportMrzPageResult]:
        work_items: list[_PageWork] = []
        flat_lines: list[np.ndarray] = []
        line_ranges: list[tuple[int, int]] = []
        for page_number, image in pages:
            work = _PageWork(page_number=page_number)
            work_items.append(work)
            detection, work.detector_ms = self.detector.detect(image)
            if detection is None:
                work.warning = "MRZ region was not detected."
                line_ranges.append((len(flat_lines), len(flat_lines)))
                continue
            work.detector_confidence = detection.confidence
            crop_started = time.perf_counter()
            try:
                _, work.lines, work.deskew_angle = crop_mrz_lines(image, detection.bbox)
            except Exception as exc:
                work.warning = str(exc)
                work.lines = []
            work.crop_ms = (time.perf_counter() - crop_started) * 1000
            start = len(flat_lines)
            flat_lines.extend(line.image for line in work.lines or [])
            line_ranges.append((start, len(flat_lines)))

        ocr_results, total_ocr_ms = self.recognizer.recognize(flat_lines)
        for work, (start, end) in zip(work_items, line_ranges, strict=True):
            work.ocr_results = ocr_results[start:end]

        # One conservative second pass for checksum failures. All suspicious
        # lines are batched together so multiple passports do not serialize CTC.
        fallback_images: list[np.ndarray] = []
        fallback_ranges: dict[int, tuple[int, int]] = {}
        for index, work in enumerate(work_items):
            candidate = _parse_candidate(work.ocr_results or [])
            if work.lines and (candidate is None or not candidate.checksum_pass):
                start = len(fallback_images)
                fallback_images.extend(line.image for line in enhance_lines(work.lines))
                fallback_ranges[index] = (start, len(fallback_images))
        fallback_results, fallback_ms = self.recognizer.recognize(fallback_images)
        total_ocr_ms += fallback_ms
        for index, (start, end) in fallback_ranges.items():
            original = work_items[index].ocr_results or []
            enhanced = fallback_results[start:end]
            if _candidate_score(enhanced) > _candidate_score(original):
                work_items[index].ocr_results = enhanced

        per_line_ms = total_ocr_ms / max(1, len(flat_lines) + len(fallback_images))
        return [self._result(work, per_line_ms) for work in work_items]

    def _result(self, work: _PageWork, per_line_ms: float) -> PassportMrzPageResult:
        results = work.ocr_results or []
        parsed = _parse_candidate(results)
        raw_lines = [str(item.get("normalized_text") or "") for item in results]
        accepted_scores = [
            float(item.get("confidence") or 0.0)
            for item in results
            if item.get("accepted")
        ]
        ocr_confidence = (
            sum(accepted_scores) / len(accepted_scores) if accepted_scores else 0.0
        )
        warnings = [work.warning] if work.warning else []
        if results and parsed is None:
            warnings.append("OCR lines could not be parsed as ICAO MRZ.")
        if parsed is not None and not parsed.checksum_pass:
            warnings.append("MRZ was parsed but ICAO checksum validation failed.")
        confidence = 0.45 * ocr_confidence + 0.20 * work.detector_confidence
        if parsed is not None:
            confidence += 0.35 * parsed.format_score
        return PassportMrzPageResult(
            page_number=work.page_number,
            found=parsed is not None,
            document_type=parsed.document_type if parsed else None,
            checksum_pass=bool(parsed and parsed.checksum_pass),
            confidence=round(min(0.999, confidence), 6),
            detector_confidence=round(work.detector_confidence, 6),
            average_ocr_confidence=round(ocr_confidence, 6),
            lines=parsed.lines if parsed else raw_lines,
            fields=parsed.fields if parsed else {},
            checks=(
                [PassportMrzCheck(**item.__dict__) for item in parsed.checks]
                if parsed
                else []
            ),
            processing={
                "detector_ms": round(work.detector_ms, 3),
                "crop_ms": round(work.crop_ms, 3),
                "ocr_ms": round(per_line_ms * len(results), 3),
                "deskew_angle": round(work.deskew_angle, 3),
            },
            warnings=warnings,
        )


def _parse_candidate(results: list[dict[str, Any]]) -> MrzParseResult | None:
    lines = [
        str(item.get("normalized_text") or "")
        for item in results
        if item.get("accepted")
    ]
    parsed = parse_mrz(lines)
    if parsed is not None and parsed.checksum_pass:
        return parsed
    corrected = _correct_numeric_positions(lines)
    corrected_parsed = parse_mrz(corrected)
    if corrected_parsed is not None and corrected_parsed.checksum_pass:
        return corrected_parsed
    return parsed


def _candidate_score(results: list[dict[str, Any]]) -> tuple[int, int, float]:
    parsed = _parse_candidate(results)
    average = sum(float(item.get("confidence") or 0.0) for item in results) / max(
        1, len(results)
    )
    return (
        int(bool(parsed and parsed.checksum_pass)),
        sum(item.passed for item in parsed.checks) if parsed else 0,
        average,
    )


def _correct_numeric_positions(lines: list[str]) -> list[str]:
    if len(lines) < 2 or len(lines[1]) < 28:
        return lines
    replacements = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }
    characters = list(lines[1])
    for index in set(range(13, 20)) | set(range(21, 28)) | {9, 43}:
        if index < len(characters) and characters[index] in replacements:
            characters[index] = replacements[characters[index]]
    return [lines[0], "".join(characters), *lines[2:]]
