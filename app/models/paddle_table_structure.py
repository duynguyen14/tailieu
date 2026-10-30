from __future__ import annotations

import math
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from app.core.config import DevicePreference, Settings
from app.schemas.table_annotation import (
    TableElementPayload,
    TableElementType,
    TableRegionPayload,
)


class TableModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CellSpec:
    row_index: int
    column_index: int
    row_span: int
    column_span: int


class _TableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.row_index = -1
        self.cells: list[CellSpec] = []
        self._occupied: set[tuple[int, int]] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.row_index += 1
            return
        if tag not in {"td", "th"}:
            return
        if self.row_index < 0:
            self.row_index = 0
        attr_map = {name.lower(): value for name, value in attrs}
        row_span = _positive_int(attr_map.get("rowspan"), default=1)
        column_span = _positive_int(attr_map.get("colspan"), default=1)
        column_index = 0
        while (self.row_index, column_index) in self._occupied:
            column_index += 1
        self.cells.append(
            CellSpec(
                row_index=self.row_index,
                column_index=column_index,
                row_span=row_span,
                column_span=column_span,
            )
        )
        for row in range(self.row_index, self.row_index + row_span):
            for column in range(column_index, column_index + column_span):
                self._occupied.add((row, column))


def _positive_int(value: str | None, *, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def parse_table_cells(structure_tokens: list[str]) -> list[CellSpec]:
    parser = _TableHtmlParser()
    parser.feed("".join(str(token) for token in structure_tokens))
    return parser.cells


def _result_payload(result: Any) -> dict[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR returned an unsupported result object.")
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _first_result(output: Any) -> dict[str, Any]:
    try:
        result = next(iter(output))
    except StopIteration as exc:
        raise RuntimeError("PaddleOCR returned no prediction result.") from exc
    return _result_payload(result)


def _rect_from_coordinate(
    coordinate: Any, *, width: int, height: int
) -> tuple[int, int, int, int] | None:
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 4:
        return None
    if len(coordinate) == 4 and all(isinstance(value, (int, float)) for value in coordinate):
        xs = [float(coordinate[0]), float(coordinate[2])]
        ys = [float(coordinate[1]), float(coordinate[3])]
    else:
        flat: list[float] = []
        for value in coordinate:
            if isinstance(value, (list, tuple)):
                flat.extend(float(item) for item in value)
            else:
                flat.append(float(value))
        if len(flat) < 8 or len(flat) % 2 != 0:
            return None
        xs = flat[0::2]
        ys = flat[1::2]
    x1 = max(0, min(width - 1, math.floor(min(xs))))
    y1 = max(0, min(height - 1, math.floor(min(ys))))
    x2 = max(1, min(width, math.ceil(max(xs))))
    y2 = max(1, min(height, math.ceil(max(ys))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _union_rect(rectangles: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(item[0] for item in rectangles),
        min(item[1] for item in rectangles),
        max(item[2] for item in rectangles),
        max(item[3] for item in rectangles),
    )


class PaddleTableStructureEngine:
    """PaddleOCR layout + SLANet adapter used only by the offline inference tool."""

    def __init__(
        self,
        settings: Settings,
        *,
        device_override: DevicePreference | None = None,
    ) -> None:
        self._settings = settings
        requested_device = device_override or settings.table_device
        self.effective_device = self._paddle_device(requested_device)
        try:
            self._layout_model, self._structure_model = self._build_models(
                self.effective_device
            )
        except Exception as first_error:
            if (
                requested_device == DevicePreference.CUDA
                and settings.table_allow_cpu_fallback
            ):
                self.effective_device = "cpu"
                try:
                    self._layout_model, self._structure_model = self._build_models("cpu")
                    return
                except Exception:
                    pass
            raise TableModelUnavailableError(
                "Could not initialize PaddleOCR table models. Install a compatible "
                "PaddlePaddle engine and requirements-table.txt, then verify the model "
                f"directories. Original error: {first_error}"
            ) from first_error

    @property
    def model_name(self) -> str:
        return (
            f"{self._settings.table_layout_model_name}+"
            f"{self._settings.table_structure_model_name}"
        )

    def _paddle_device(self, preference: DevicePreference) -> str | None:
        if preference == DevicePreference.AUTO:
            return None
        if preference == DevicePreference.CUDA:
            return f"gpu:{self._settings.table_gpu_device_id}"
        return "cpu"

    def _build_models(self, device: str | None):
        try:
            from paddleocr import LayoutDetection, TableStructureRecognition
        except ImportError as exc:
            raise TableModelUnavailableError(
                "PaddleOCR is not installed. Install the PaddlePaddle inference engine "
                "and then run: python -m pip install -r requirements-table.txt"
            ) from exc

        layout_dir = Settings.resolve_optional_project_path(
            self._settings.table_layout_model_dir
        )
        structure_dir = Settings.resolve_optional_project_path(
            self._settings.table_structure_model_dir
        )
        for name, path in (("layout", layout_dir), ("structure", structure_dir)):
            if path is not None and not path.is_dir():
                raise RuntimeError(f"Configured {name} model directory does not exist: {path}")

        common = {
            "device": device,
            "enable_hpi": self._settings.table_enable_hpi,
            "cpu_threads": self._settings.table_cpu_threads,
        }
        layout_model = LayoutDetection(
            model_name=self._settings.table_layout_model_name,
            model_dir=str(layout_dir) if layout_dir else None,
            threshold=self._settings.table_layout_threshold,
            **common,
        )
        structure_model = TableStructureRecognition(
            model_name=self._settings.table_structure_model_name,
            model_dir=str(structure_dir) if structure_dir else None,
            **common,
        )
        return layout_model, structure_model

    def predict(self, image_path: Path) -> list[TableRegionPayload]:
        if not image_path.is_file():
            raise FileNotFoundError(f"Page image does not exist: {image_path}")
        with Image.open(image_path) as source:
            page_image = ImageOps.exif_transpose(source).convert("RGB")
        page_width, page_height = page_image.size

        layout_payload = _first_result(
            self._layout_model.predict(
                input=str(image_path),
                batch_size=1,
                threshold=self._settings.table_layout_threshold,
            )
        )
        detected: list[tuple[tuple[int, int, int, int], float]] = []
        for item in layout_payload.get("boxes", []):
            if not isinstance(item, dict) or str(item.get("label", "")).lower() != "table":
                continue
            score = float(item.get("score", 0.0))
            if score < self._settings.table_layout_threshold:
                continue
            bbox = _rect_from_coordinate(
                item.get("coordinate"), width=page_width, height=page_height
            )
            if bbox is not None:
                detected.append((bbox, score))
        detected.sort(key=lambda item: (item[0][1], item[0][0]))

        regions: list[TableRegionPayload] = []
        for table_index, (table_bbox, table_score) in enumerate(detected):
            crop = page_image.crop(table_bbox)
            structure_payload = _first_result(
                self._structure_model.predict(input=_pil_to_numpy(crop), batch_size=1)
            )
            elements = self._structure_elements(
                structure_payload=structure_payload,
                table_bbox=table_bbox,
                crop_size=crop.size,
            )
            regions.append(
                TableRegionPayload(
                    table_index=table_index,
                    bbox=table_bbox,
                    confidence=table_score,
                    elements=elements,
                )
            )
        return regions

    def _structure_elements(
        self,
        *,
        structure_payload: dict[str, Any],
        table_bbox: tuple[int, int, int, int],
        crop_size: tuple[int, int],
    ) -> list[TableElementPayload]:
        raw_boxes = structure_payload.get("bbox") or structure_payload.get("boxes") or []
        structure_tokens = structure_payload.get("structure") or []
        specs = parse_table_cells([str(token) for token in structure_tokens])
        structure_score = float(structure_payload.get("structure_score", 0.0))
        crop_width, crop_height = crop_size
        offset_x, offset_y = table_bbox[0], table_bbox[1]

        cells: list[tuple[CellSpec, tuple[int, int, int, int]]] = []
        for index, raw_box in enumerate(raw_boxes):
            local_bbox = _rect_from_coordinate(
                raw_box, width=crop_width, height=crop_height
            )
            if local_bbox is None:
                continue
            spec = (
                specs[index]
                if index < len(specs)
                else CellSpec(index, 0, 1, 1)
            )
            page_bbox = (
                local_bbox[0] + offset_x,
                local_bbox[1] + offset_y,
                local_bbox[2] + offset_x,
                local_bbox[3] + offset_y,
            )
            cells.append((spec, page_bbox))

        if not cells:
            return []

        rows: dict[int, list[tuple[int, int, int, int]]] = {}
        columns: dict[int, list[tuple[int, int, int, int]]] = {}
        for spec, bbox in cells:
            for row in range(spec.row_index, spec.row_index + spec.row_span):
                rows.setdefault(row, []).append(bbox)
            for column in range(
                spec.column_index, spec.column_index + spec.column_span
            ):
                columns.setdefault(column, []).append(bbox)

        first_row = min(rows)
        elements: list[TableElementPayload] = []
        for row_index, rectangles in sorted(rows.items()):
            is_header = (
                self._settings.table_assume_first_row_header and row_index == first_row
            )
            elements.append(
                TableElementPayload(
                    element_type=(
                        TableElementType.HEADER_ROW
                        if is_header
                        else TableElementType.DATA_ROW
                    ),
                    row_index=row_index,
                    bbox=_union_rect(rectangles),
                    confidence=structure_score,
                )
            )
        for column_index, rectangles in sorted(columns.items()):
            elements.append(
                TableElementPayload(
                    element_type=TableElementType.COLUMN,
                    column_index=column_index,
                    bbox=_union_rect(rectangles),
                    confidence=structure_score,
                )
            )
        for spec, bbox in cells:
            elements.append(
                TableElementPayload(
                    element_type=TableElementType.CELL,
                    row_index=spec.row_index,
                    column_index=spec.column_index,
                    row_span=spec.row_span,
                    column_span=spec.column_span,
                    bbox=bbox,
                    confidence=structure_score,
                )
            )
        return elements


def _pil_to_numpy(image: Image.Image):
    try:
        import numpy as np
    except ImportError as exc:
        raise TableModelUnavailableError(
            "NumPy is required by the PaddleOCR table adapter."
        ) from exc
    return np.asarray(image)
