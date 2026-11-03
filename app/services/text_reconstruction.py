from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Literal

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ReconstructionToken:
    id: str
    text: str
    bbox: BBox
    order: int

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class ReconstructionElement:
    element_type: Literal["HEADER_ROW", "DATA_ROW", "COLUMN", "CELL"]
    bbox: BBox
    row_index: int | None = None
    column_index: int | None = None
    row_span: int = 1
    column_span: int = 1


@dataclass(frozen=True)
class ReconstructionTable:
    table_index: int
    bbox: BBox
    elements: tuple[ReconstructionElement, ...] = ()


@dataclass(frozen=True)
class PageReconstructionInput:
    page_id: str
    page_number: int
    page_type: str
    page_role: str
    tokens: tuple[ReconstructionToken, ...]
    tables: tuple[ReconstructionTable, ...] = ()


@dataclass(frozen=True)
class PageReconstructionResult:
    page_id: str
    page_number: int
    page_type: str
    page_role: str
    text: str
    token_count: int
    table_count: int
    structured_table_count: int
    warnings: tuple[str, ...] = ()

    def metadata(self) -> dict:
        return {
            "page_id": self.page_id,
            "page_number": self.page_number,
            "page_type": self.page_type,
            "page_role": self.page_role,
            "token_count": self.token_count,
            "table_count": self.table_count,
            "structured_table_count": self.structured_table_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class BboxTextLineResult:
    """One visual line reconstructed only from reviewed text and bbox geometry."""

    line_index: int
    bbox: BBox
    text: str
    token_ids: tuple[str, ...]


@dataclass
class _Line:
    tokens: list[ReconstructionToken] = field(default_factory=list)

    @property
    def center_y(self) -> float:
        return sum(token.center_y for token in self.tokens) / len(self.tokens)

    @property
    def y1(self) -> int:
        return min(token.bbox[1] for token in self.tokens)

    @property
    def y2(self) -> int:
        return max(token.bbox[3] for token in self.tokens)


@dataclass(frozen=True)
class _Block:
    y: int
    kind: Literal["TEXT", "TABLE"]
    text: str


_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _vertical_overlap_ratio(token: ReconstructionToken, line: _Line) -> float:
    overlap = max(0, min(token.bbox[3], line.y2) - max(token.bbox[1], line.y1))
    denominator = max(1, min(token.height, line.y2 - line.y1))
    return overlap / denominator


def group_token_lines(
    tokens: Sequence[ReconstructionToken], *, tolerance_ratio: float = 0.55
) -> list[list[ReconstructionToken]]:
    """Group OCR boxes into visual lines, then sort each line left-to-right."""

    clean = [token for token in tokens if normalize_text(token.text)]
    if not clean:
        return []
    typical_height = median(max(1, token.height) for token in clean)
    tolerance = max(3.0, typical_height * tolerance_ratio)
    lines: list[_Line] = []
    for token in sorted(clean, key=lambda item: (item.center_y, item.bbox[0], item.order)):
        candidates = [
            line
            for line in lines
            if abs(token.center_y - line.center_y) <= tolerance
            or _vertical_overlap_ratio(token, line) >= 0.40
        ]
        if candidates:
            target = min(candidates, key=lambda line: abs(token.center_y - line.center_y))
            target.tokens.append(token)
        else:
            lines.append(_Line(tokens=[token]))
    lines.sort(key=lambda line: (line.y1, line.center_y))
    return [
        sorted(line.tokens, key=lambda item: (item.bbox[0], item.center_y, item.order))
        for line in lines
    ]


def line_text(tokens: Sequence[ReconstructionToken]) -> str:
    return " ".join(
        text for token in tokens if (text := normalize_text(token.text))
    )


def reconstruct_bbox_lines(
    tokens: Sequence[ReconstructionToken], *, tolerance_ratio: float = 0.55
) -> list[BboxTextLineResult]:
    """Expose bbox-only line grouping for diagnostics and visual review."""

    results: list[BboxTextLineResult] = []
    for line_index, line_tokens in enumerate(
        group_token_lines(tokens, tolerance_ratio=tolerance_ratio), start=1
    ):
        results.append(
            BboxTextLineResult(
                line_index=line_index,
                bbox=(
                    min(token.bbox[0] for token in line_tokens),
                    min(token.bbox[1] for token in line_tokens),
                    max(token.bbox[2] for token in line_tokens),
                    max(token.bbox[3] for token in line_tokens),
                ),
                text=line_text(line_tokens),
                token_ids=tuple(token.id for token in line_tokens),
            )
        )
    return results


def _center_inside(token: ReconstructionToken, bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= token.center_x <= x2 and y1 <= token.center_y <= y2


def _tokens_in_bbox(
    tokens: Sequence[ReconstructionToken], bbox: BBox
) -> list[ReconstructionToken]:
    return [token for token in tokens if _center_inside(token, bbox)]


def _cell_text(
    tokens: Sequence[ReconstructionToken], bbox: BBox, tolerance_ratio: float
) -> str:
    cell_tokens = _tokens_in_bbox(tokens, bbox)
    lines = group_token_lines(cell_tokens, tolerance_ratio=tolerance_ratio)
    return " ".join(filter(None, (line_text(line) for line in lines)))


def _virtual_cells(
    rows: Sequence[ReconstructionElement], columns: Sequence[ReconstructionElement]
) -> list[ReconstructionElement]:
    cells: list[ReconstructionElement] = []
    for row in rows:
        for column in columns:
            x1 = max(row.bbox[0], column.bbox[0])
            y1 = max(row.bbox[1], column.bbox[1])
            x2 = min(row.bbox[2], column.bbox[2])
            y2 = min(row.bbox[3], column.bbox[3])
            if x2 <= x1 or y2 <= y1:
                continue
            cells.append(
                ReconstructionElement(
                    element_type="CELL",
                    row_index=row.row_index,
                    column_index=column.column_index,
                    bbox=(x1, y1, x2, y2),
                )
            )
    return cells


def reconstruct_table(
    table: ReconstructionTable,
    tokens: Sequence[ReconstructionToken],
    *,
    tolerance_ratio: float = 0.55,
    assume_first_row_header: bool = True,
) -> tuple[str, bool, list[str]]:
    """Return table text, whether rows/cells were reconstructed, and warnings."""

    table_tokens = _tokens_in_bbox(tokens, table.bbox)
    header_rows = {
        element.row_index
        for element in table.elements
        if element.element_type == "HEADER_ROW" and element.row_index is not None
    }
    rows = [
        element
        for element in table.elements
        if element.element_type in {"HEADER_ROW", "DATA_ROW"}
        and element.row_index is not None
    ]
    columns = [
        element
        for element in table.elements
        if element.element_type == "COLUMN" and element.column_index is not None
    ]
    cells = [
        element
        for element in table.elements
        if element.element_type == "CELL"
        and element.row_index is not None
        and element.column_index is not None
    ]
    if not cells and rows and columns:
        cells = _virtual_cells(rows, columns)

    lines = [f"[TABLE {table.table_index + 1}]"]
    warnings: list[str] = []
    if cells:
        row_indexes = sorted({cell.row_index for cell in cells if cell.row_index is not None})
        if assume_first_row_header and not header_rows and row_indexes:
            header_rows.add(row_indexes[0])
        max_column = max(cell.column_index or 0 for cell in cells)
        for row_index in row_indexes:
            row_cells: list[str] = []
            for column_index in range(max_column + 1):
                candidates = [
                    cell
                    for cell in cells
                    if cell.row_index == row_index and cell.column_index == column_index
                ]
                value = ""
                if candidates:
                    value = _cell_text(
                        table_tokens, candidates[0].bbox, tolerance_ratio
                    )
                row_cells.append(value)
            prefix = "[HEADER]" if row_index in header_rows else f"[ROW {row_index}]"
            lines.append(f"{prefix} " + " | ".join(row_cells))
        return "\n".join(lines), True, warnings

    if rows:
        for row in sorted(rows, key=lambda item: (item.row_index or 0, item.bbox[1])):
            value = _cell_text(table_tokens, row.bbox, tolerance_ratio)
            prefix = (
                "[HEADER]"
                if row.element_type == "HEADER_ROW"
                else f"[ROW {row.row_index}]"
            )
            lines.append(f"{prefix} {value}")
        warnings.append("TABLE_COLUMNS_UNAVAILABLE")
        return "\n".join(lines), True, warnings

    fallback_lines = group_token_lines(table_tokens, tolerance_ratio=tolerance_ratio)
    lines.append("[TABLE_TEXT]")
    lines.extend(filter(None, (line_text(line) for line in fallback_lines)))
    warnings.append("TABLE_STRUCTURE_UNAVAILABLE")
    return "\n".join(lines), False, warnings


def reconstruct_page(
    page: PageReconstructionInput,
    *,
    tolerance_ratio: float = 0.55,
    assume_first_row_header: bool = True,
) -> PageReconstructionResult:
    tables = sorted(page.tables, key=lambda item: (item.bbox[1], item.bbox[0]))
    table_token_ids = {
        token.id
        for table in tables
        for token in page.tokens
        if _center_inside(token, table.bbox)
    }
    plain_tokens = [token for token in page.tokens if token.id not in table_token_ids]
    blocks = [
        _Block(y=min(token.bbox[1] for token in line), kind="TEXT", text=line_text(line))
        for line in group_token_lines(plain_tokens, tolerance_ratio=tolerance_ratio)
        if line_text(line)
    ]
    warnings: list[str] = []
    structured_tables = 0
    for table in tables:
        table_text, structured, table_warnings = reconstruct_table(
            table,
            page.tokens,
            tolerance_ratio=tolerance_ratio,
            assume_first_row_header=assume_first_row_header,
        )
        blocks.append(_Block(y=table.bbox[1], kind="TABLE", text=table_text))
        warnings.extend(table_warnings)
        structured_tables += int(structured)
    blocks.sort(key=lambda block: (block.y, 0 if block.kind == "TEXT" else 1))

    output = [
        f"[PAGE {page.page_number}]",
        f"[TYPE: {page.page_type}]",
        f"[ROLE: {page.page_role}]",
    ]
    previous_kind: str | None = None
    for block in blocks:
        if block.kind == "TEXT" and previous_kind != "TEXT":
            output.append("[TEXT]")
        output.append(block.text)
        previous_kind = block.kind
    return PageReconstructionResult(
        page_id=page.page_id,
        page_number=page.page_number,
        page_type=page.page_type,
        page_role=page.page_role,
        text="\n".join(output).strip(),
        token_count=len(page.tokens),
        table_count=len(tables),
        structured_table_count=structured_tables,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def reconstruct_case(
    pages: Sequence[PageReconstructionInput],
    *,
    tolerance_ratio: float = 0.55,
    assume_first_row_header: bool = True,
) -> tuple[str, list[PageReconstructionResult]]:
    results = [
        reconstruct_page(
            page,
            tolerance_ratio=tolerance_ratio,
            assume_first_row_header=assume_first_row_header,
        )
        for page in sorted(pages, key=lambda item: item.page_number)
    ]
    return "\n\n".join(result.text for result in results), results
