from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.models.paddle_ocr_engine import OcrModelUnavailableError, OcrTokenPrediction

_DIGITS = re.compile(r"\d")


@dataclass(frozen=True)
class TableEnhancementResult:
    tokens: list[OcrTokenPrediction]
    metadata: dict[str, object]


def parse_tesseract_data(data: dict[str, list[Any]]) -> list[OcrTokenPrediction]:
    """Convert Tesseract TSV data into stable line-level grounded tokens."""

    grouped: dict[tuple[int, int, int, int], list[dict[str, object]]] = defaultdict(list)
    count = len(data.get("text", []))
    for index in range(count):
        text = str(data["text"][index]).strip()
        if not text:
            continue
        try:
            confidence = float(data.get("conf", [])[index])
            left = int(data.get("left", [])[index])
            top = int(data.get("top", [])[index])
            width = int(data.get("width", [])[index])
            height = int(data.get("height", [])[index])
        except (IndexError, TypeError, ValueError):
            continue
        if confidence < 0 or width <= 0 or height <= 0:
            continue
        key = (
            int(data.get("page_num", [1] * count)[index]),
            int(data.get("block_num", [0] * count)[index]),
            int(data.get("par_num", [0] * count)[index]),
            int(data.get("line_num", [index] * count)[index]),
        )
        grouped[key].append(
            {
                "text": text,
                "confidence": confidence / 100.0,
                "bbox": (left, top, left + width, top + height),
            }
        )

    candidates: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for words in grouped.values():
        words.sort(key=lambda word: (word["bbox"][0], word["bbox"][1]))  # type: ignore[index]
        text = " ".join(str(word["text"]) for word in words).strip()
        boxes = [word["bbox"] for word in words]
        x1 = min(box[0] for box in boxes)  # type: ignore[index]
        y1 = min(box[1] for box in boxes)  # type: ignore[index]
        x2 = max(box[2] for box in boxes)  # type: ignore[index]
        y2 = max(box[3] for box in boxes)  # type: ignore[index]
        weights = [max(1, len(str(word["text"]))) for word in words]
        confidence = sum(
            float(word["confidence"]) * weight
            for word, weight in zip(words, weights, strict=True)
        ) / sum(weights)
        candidates.append((text, confidence, (x1, y1, x2, y2)))

    candidates.sort(key=lambda item: (item[2][1], item[2][0], item[2][3]))
    return [
        OcrTokenPrediction(
            token_id=f"t_{index:04d}",
            token_index=index,
            text=text,
            confidence=max(0.0, min(1.0, confidence)),
            bbox=bbox,
            polygon=[
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ],
        )
        for index, (text, confidence, bbox) in enumerate(candidates, start=1)
    ]


def detect_table_cells(
    image: Image.Image, *, max_cells: int
) -> list[tuple[int, int, int, int]]:
    # Keep Tesseract dependencies optional for installations that continue to
    # use the default Paddle backend.
    import cv2
    import numpy as np

    if max_cells <= 0:
        return []
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    if width < 160 or height < 160:
        return []
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, width // 35), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, height // 35))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.bitwise_or(horizontal, vertical)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    page_area = width * height
    raw: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, cell_width, cell_height = cv2.boundingRect(contour)
        area = cell_width * cell_height
        if cell_width < 28 or cell_height < 16:
            continue
        if area < page_area * 0.00015 or area > page_area * 0.20:
            continue
        if cell_width > width * 0.98 or cell_height > height * 0.98:
            continue
        raw.append((x, y, x + cell_width, y + cell_height))

    raw.sort(key=lambda box: ((box[2] - box[0]) * (box[3] - box[1]), box[1], box[0]))
    cells: list[tuple[int, int, int, int]] = []
    for candidate in raw:
        if any(
            abs(candidate[0] - current[0]) <= 3
            and abs(candidate[1] - current[1]) <= 3
            and abs(candidate[2] - current[2]) <= 3
            and abs(candidate[3] - current[3]) <= 3
            for current in cells
        ):
            continue
        # Reject a table/container rectangle when smaller cells already cover it.
        contained = sum(
            candidate[0] <= current[0]
            and candidate[1] <= current[1]
            and candidate[2] >= current[2]
            and candidate[3] >= current[3]
            for current in cells
        )
        if contained >= 2:
            continue
        cells.append(candidate)
        if len(cells) >= max_cells:
            break
    if len(cells) < 4:
        return []
    return sorted(cells, key=lambda box: (box[1], box[0]))


def _center_in_box(
    token: OcrTokenPrediction, box: tuple[int, int, int, int]
) -> bool:
    center_x = (token.bbox[0] + token.bbox[2]) / 2
    center_y = (token.bbox[1] + token.bbox[3]) / 2
    return box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]


def _digits(value: str) -> str:
    return "".join(_DIGITS.findall(value))


class TesseractOcrEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.effective_device = "cpu"
        try:
            import pytesseract
        except ImportError as exc:
            raise OcrModelUnavailableError(
                "pytesseract is not installed. Install requirements-ocr-tesseract.txt."
            ) from exc
        self._pytesseract = pytesseract
        command = settings.tesseract_cmd_path
        fast_data = settings.tesseract_fast_data_path
        best_data = settings.tesseract_best_data_path
        if not command.is_file():
            raise OcrModelUnavailableError(
                f"Tesseract executable was not found: {command}"
            )
        if not fast_data.is_dir():
            raise OcrModelUnavailableError(
                f"Tesseract fast tessdata directory was not found: {fast_data}"
            )
        missing = [
            language
            for language in settings.tesseract_language_list
            if not (fast_data / f"{language}.traineddata").is_file()
        ]
        if missing:
            raise OcrModelUnavailableError(
                "Missing Tesseract fast language model(s): " + ", ".join(missing)
            )
        if settings.tesseract_enable_vie_refinement:
            best_model = best_data / "vie.traineddata"
            if not best_model.is_file():
                raise OcrModelUnavailableError(
                    f"Vietnamese tessdata_best model was not found: {best_model}"
                )
        pytesseract.pytesseract.tesseract_cmd = str(command)
        os.environ["OMP_THREAD_LIMIT"] = str(settings.tesseract_threads)
        self._fast_data = fast_data
        self._best_data = best_data

    @property
    def model_name(self) -> str:
        languages = "+".join(self._settings.tesseract_language_list)
        suffix = "+vie_best" if self._settings.tesseract_enable_vie_refinement else ""
        return f"tesseract5-lstm:{languages}{suffix}"

    def _config(self, *, data_dir: Path, psm: int) -> str:
        data_dir_value = str(data_dir)
        if any(character.isspace() for character in data_dir_value):
            raise OcrModelUnavailableError(
                "Tesseract tessdata path must not contain spaces when used through "
                f"pytesseract on Windows: {data_dir}. Configure "
                "TESSERACT_FAST_DATA_DIR/TESSERACT_BEST_DATA_DIR to a path without "
                "spaces."
            )
        return (
            f"--tessdata-dir {data_dir_value} "
            f"--oem {self._settings.tesseract_oem} --psm {psm} --dpi 300"
        )

    def _predict(
        self,
        image: str | Path | Image.Image,
        *,
        languages: str,
        data_dir: Path,
        psm: int,
    ) -> list[OcrTokenPrediction]:
        input_image = str(image) if isinstance(image, Path) else image
        data = self._pytesseract.image_to_data(
            input_image,
            lang=languages,
            config=self._config(data_dir=data_dir, psm=psm),
            output_type=self._pytesseract.Output.DICT,
            timeout=self._settings.tesseract_timeout_seconds,
        )
        return parse_tesseract_data(data)

    def predict(self, image_path: Path) -> list[OcrTokenPrediction]:
        if not image_path.is_file():
            raise FileNotFoundError(f"OCR page image does not exist: {image_path}")
        return self._predict(
            image_path,
            languages="+".join(self._settings.tesseract_language_list),
            data_dir=self._fast_data,
            psm=self._settings.tesseract_page_segmentation_mode,
        )

    def predict_many(
        self, image_paths: list[Path]
    ) -> list[list[OcrTokenPrediction]]:
        return [self.predict(path) for path in image_paths]

    def predict_refinement_many(
        self, image_paths: list[Path]
    ) -> list[list[OcrTokenPrediction]]:
        if not self._settings.tesseract_enable_vie_refinement:
            return self.predict_many(image_paths)

        def read_region(path: Path) -> list[OcrTokenPrediction]:
            return self._predict(
                path,
                languages="vie",
                data_dir=self._best_data,
                psm=self._settings.tesseract_refinement_page_segmentation_mode,
            )

        # pytesseract starts one native Tesseract process per crop. A very small
        # pool overlaps process startup without overloading the CPU or making
        # the OCR result depend on a high worker count.
        worker_count = min(2, len(image_paths))
        if worker_count <= 1:
            return [read_region(path) for path in image_paths]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(read_region, image_paths))

    def enhance_tables(
        self,
        image_path: Path,
        tokens: list[OcrTokenPrediction],
    ) -> TableEnhancementResult:
        if not self._settings.tesseract_enable_table_cells:
            return TableEnhancementResult(tokens=tokens, metadata={"enabled": False})
        started = time.perf_counter()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        try:
            cells = detect_table_cells(
                image, max_cells=self._settings.tesseract_table_max_cells
            )
            if not cells:
                return TableEnhancementResult(
                    tokens=tokens,
                    metadata={
                        "enabled": True,
                        "detected_cells": 0,
                        "replaced_cells": 0,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000),
                        "items": [],
                    },
                )
            current = list(tokens)
            audit: list[dict[str, object]] = []
            replacements: list[tuple[tuple[int, int, int, int], OcrTokenPrediction]] = []
            for cell_index, box in enumerate(cells, start=1):
                if (
                    time.perf_counter() - started
                    > self._settings.tesseract_table_budget_seconds
                ):
                    break
                x1, y1, x2, y2 = box
                inset = max(2, min(x2 - x1, y2 - y1) // 30)
                crop = image.crop((x1 + inset, y1 + inset, x2 - inset, y2 - inset))
                try:
                    reread = self._predict(
                        crop,
                        languages="+".join(self._settings.tesseract_language_list),
                        data_dir=self._fast_data,
                        psm=6,
                    )
                finally:
                    crop.close()
                reread_text = " ".join(item.text for item in reread).strip()
                reread_confidence = (
                    sum(item.confidence for item in reread) / len(reread)
                    if reread
                    else 0.0
                )
                overlapping = [token for token in current if _center_in_box(token, box)]
                original_text = " ".join(token.text for token in overlapping).strip()
                original_confidence = (
                    sum(token.confidence for token in overlapping) / len(overlapping)
                    if overlapping
                    else 0.0
                )
                reason = "accepted"
                accepted = bool(reread_text)
                if not reread_text:
                    reason = "empty_cell_reread"
                    accepted = False
                elif _digits(original_text) and _digits(original_text) != _digits(
                    reread_text
                ):
                    reason = "digit_mismatch_protected"
                    accepted = False
                elif not overlapping and reread_confidence < 0.60:
                    reason = "new_cell_confidence_low"
                    accepted = False
                elif overlapping and reread_confidence + 0.03 < original_confidence:
                    reason = "cell_reread_confidence_lower"
                    accepted = False
                elif (
                    original_text
                    and len("".join(character for character in reread_text if character.isalnum()))
                    < len(
                        "".join(
                            character
                            for character in original_text
                            if character.isalnum()
                        )
                    )
                    * 0.55
                ):
                    reason = "cell_reread_lost_text"
                    accepted = False
                if accepted:
                    replacements.append(
                        (
                            box,
                            OcrTokenPrediction(
                                token_id=f"cell_{cell_index:04d}",
                                token_index=cell_index,
                                text=reread_text,
                                confidence=reread_confidence,
                                bbox=box,
                                polygon=[
                                    [x1, y1],
                                    [x2, y1],
                                    [x2, y2],
                                    [x1, y2],
                                ],
                            ),
                        )
                    )
                audit.append(
                    {
                        "cell": cell_index,
                        "bbox": list(box),
                        "original_text": original_text,
                        "original_confidence": round(original_confidence, 6),
                        "reread_text": reread_text,
                        "confidence": round(reread_confidence, 6),
                        "accepted": accepted,
                        "reason": reason,
                    }
                )

            for box, replacement in replacements:
                current = [token for token in current if not _center_in_box(token, box)]
                current.append(replacement)
            current.sort(key=lambda token: (token.bbox[1], token.bbox[0]))
            normalized = [
                OcrTokenPrediction(
                    token_id=f"t_{index:04d}",
                    token_index=index,
                    text=token.text,
                    confidence=token.confidence,
                    bbox=token.bbox,
                    polygon=token.polygon,
                )
                for index, token in enumerate(current, start=1)
            ]
            return TableEnhancementResult(
                tokens=normalized,
                metadata={
                    "enabled": True,
                    "detected_cells": len(cells),
                    "processed_cells": len(audit),
                    "replaced_cells": len(replacements),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "items": audit,
                },
            )
        finally:
            image.close()
