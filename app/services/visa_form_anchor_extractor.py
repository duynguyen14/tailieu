from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Default template bounding boxes (in 0..1000 normalized coordinates)
# Used as baseline reference for anchor positions and fallback interpolation.
TEMPLATE_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "full_name": {
        "order": 1,
        "label": "Họ và tên",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 165, 175, 205),
        "baseline_value": (145, 176, 390, 200),
    },
    "date_of_birth": {
        "order": 2,
        "label": "Ngày sinh",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 205, 165, 240),
        "baseline_value": (160, 211, 350, 235),
    },
    "place_of_birth": {
        "order": 3,
        "label": "Nơi sinh",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 240, 165, 275),
        "baseline_value": (160, 247, 390, 271),
    },
    "gender": {
        "order": 4,
        "label": "Giới tính",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 275, 140, 312),
        "baseline_value": (135, 283, 300, 307),
    },
    "nationality": {
        "order": 5,
        "label": "Quốc tịch",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 312, 165, 348),
        "baseline_value": (160, 318, 350, 343),
    },
    "contact_address": {
        "order": 6,
        "label": "Địa chỉ liên hệ",
        "column": "left",
        "field_type": "block",
        "baseline_anchor": (60, 355, 150, 390),
        "baseline_value": (75, 375, 440, 420),
    },
    "contact_phone": {
        "order": 7,
        "label": "Điện thoại liên hệ",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 415, 150, 448),
        "baseline_value": (145, 420, 355, 445),
    },
    "contact_email": {
        "order": 8,
        "label": "Email liên hệ",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 448, 115, 482),
        "baseline_value": (95, 453, 410, 480),
    },
    "sponsor_name": {
        "order": 9,
        "label": "Tên người/cơ quan mời",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 495, 130, 532),
        "baseline_value": (125, 500, 420, 532),
    },
    "sponsor_phone": {
        "order": 10,
        "label": "Điện thoại người/cơ quan mời",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (60, 532, 140, 568),
        "baseline_value": (135, 538, 365, 566),
    },
    "passport_number": {
        "order": 11,
        "label": "Số hộ chiếu",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (510, 205, 630, 240),
        "baseline_value": (625, 211, 870, 236),
    },
    "passport_expiration": {
        "order": 12,
        "label": "Hạn hộ chiếu",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (510, 240, 640, 275),
        "baseline_value": (635, 247, 875, 272),
    },
    "number_of_entries": {
        "order": 13,
        "label": "Số lần nhập cảnh",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (510, 275, 640, 312),
        "baseline_value": (635, 283, 875, 308),
    },
    "entry_date": {
        "order": 14,
        "label": "Ngày dự kiến nhập cảnh",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (510, 312, 655, 348),
        "baseline_value": (650, 319, 885, 344),
    },
    "exit_date": {
        "order": 15,
        "label": "Ngày dự kiến xuất cảnh",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (510, 348, 655, 382),
        "baseline_value": (650, 354, 885, 379),
    },
    "submission_place": {
        "order": 16,
        "label": "Nơi nộp hồ sơ",
        "column": "right",
        "field_type": "block",
        "baseline_anchor": (510, 390, 680, 425),
        "baseline_value": (515, 405, 890, 440),
    },
    "collection_place": {
        "order": 17,
        "label": "Nơi nhận thị thực",
        "column": "right",
        "field_type": "block",
        "baseline_anchor": (510, 455, 680, 492),
        "baseline_value": (515, 474, 890, 508),
    },
    "purpose_of_entry": {
        "order": 18,
        "label": "Mục đích nhập cảnh",
        "column": "right",
        "field_type": "block",
        "baseline_anchor": (510, 505, 650, 545),
        "baseline_value": (625, 520, 975, 565),
    },
    "application_date": {
        "order": 19,
        "label": "Ngày làm đơn",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (550, 680, 650, 725),
        "baseline_value": (650, 690, 900, 725),
    },
}

FIELD_CODES_ORDERED = tuple(
    sorted(TEMPLATE_FIELD_METADATA.keys(), key=lambda k: TEMPLATE_FIELD_METADATA[k]["order"])
)

FORMAT_HINT_PAT = re.compile(r"\(?\s*dd[\/\.\-]mm[\/\.\-]yyyy\s*\)?", re.IGNORECASE)


def strip_accents_and_lower(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    cleaned = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    cleaned = cleaned.replace("đ", "d").replace("Đ", "d")
    return cleaned.lower()


@dataclass
class ExtractedToken:
    text: str
    clean_text: str
    x1: int
    y1: int
    x2: int
    y2: int
    nx1: int
    ny1: int
    nx2: int
    ny2: int
    confidence: float = 1.0
    is_anchor: bool = False
    assigned_field: str | None = None

    @property
    def ny_center(self) -> float:
        return (self.ny1 + self.ny2) / 2.0

    @property
    def nx_center(self) -> float:
        return (self.nx1 + self.nx2) / 2.0


@dataclass
class TokenLine:
    tokens: list[ExtractedToken] = field(default_factory=list)

    @property
    def ny1(self) -> int:
        return min((t.ny1 for t in self.tokens), default=0)

    @property
    def ny2(self) -> int:
        return max((t.ny2 for t in self.tokens), default=0)

    @property
    def ny_center(self) -> float:
        return (self.ny1 + self.ny2) / 2.0

    @property
    def height(self) -> int:
        return max(1, self.ny2 - self.ny1)


@dataclass
class DetectedAnchor:
    field_code: str
    nx1: int
    ny1: int
    nx2: int
    ny2: int
    tokens: list[ExtractedToken]
    confidence: float = 1.0

    @property
    def ny_center(self) -> float:
        return (self.ny1 + self.ny2) / 2.0

    @property
    def nx_center(self) -> float:
        return (self.nx1 + self.nx2) / 2.0


@dataclass
class ExtractedFieldResult:
    field_code: str
    field_order: int
    label: str
    bbox: tuple[int, int, int, int]
    normalized_bbox: tuple[int, int, int, int]
    value_text: str | None
    anchor_found: bool
    anchor_normalized_bbox: tuple[int, int, int, int] | None


def normalize_tokens(raw_tokens: Sequence[Any], width: int, height: int) -> list[ExtractedToken]:
    results: list[ExtractedToken] = []
    w = max(1, width)
    h = max(1, height)

    for item in raw_tokens:
        text: str = ""
        conf: float = 1.0
        x1, y1, x2, y2 = 0, 0, 0, 0

        if hasattr(item, "text"):
            text = getattr(item, "text", "") or ""
            conf = getattr(item, "confidence", 1.0)
            if hasattr(item, "bbox") and getattr(item, "bbox"):
                x1, y1, x2, y2 = getattr(item, "bbox")
            elif hasattr(item, "x1"):
                x1 = getattr(item, "x1", 0)
                y1 = getattr(item, "y1", 0)
                x2 = getattr(item, "x2", 0)
                y2 = getattr(item, "y2", 0)
        elif isinstance(item, dict):
            text = str(item.get("text", "") or "")
            conf = float(item.get("confidence", 1.0))
            if "bbox" in item and item["bbox"]:
                x1, y1, x2, y2 = item["bbox"]
            else:
                x1 = int(item.get("x1", 0))
                y1 = int(item.get("y1", 0))
                x2 = int(item.get("x2", 0))
                y2 = int(item.get("y2", 0))

        text = text.strip()
        if not text:
            continue

        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))

        nx1 = max(0, min(998, round(x1 * 1000 / w)))
        ny1 = max(0, min(998, round(y1 * 1000 / h)))
        nx2 = max(nx1 + 1, min(1000, round(x2 * 1000 / w)))
        ny2 = max(ny1 + 1, min(1000, round(y2 * 1000 / h)))

        clean = strip_accents_and_lower(text)
        results.append(
            ExtractedToken(
                text=text,
                clean_text=clean,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                nx1=nx1,
                ny1=ny1,
                nx2=nx2,
                ny2=ny2,
                confidence=float(conf),
            )
        )
    return results


def group_tokens_into_lines(tokens: list[ExtractedToken]) -> list[TokenLine]:
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda t: (t.ny1, t.nx1))
    lines: list[TokenLine] = []

    for tok in sorted_tokens:
        matched_line: TokenLine | None = None
        for line in lines:
            line_height = max(8, line.height)
            tok_height = max(8, tok.ny2 - tok.ny1)
            min_h = min(line_height, tok_height)
            overlap = max(0, min(line.ny2, tok.ny2) - max(line.ny1, tok.ny1))

            if overlap >= 0.40 * min_h or abs(tok.ny_center - line.ny_center) <= 8:
                matched_line = line
                break

        if matched_line is not None:
            matched_line.tokens.append(tok)
        else:
            lines.append(TokenLine(tokens=[tok]))

    for line in lines:
        line.tokens.sort(key=lambda t: t.nx1)

    lines.sort(key=lambda l: l.ny1)
    return lines


class VisaFormAnchorExtractor:
    """Detects bilingual anchor labels, computes page layout scale/shift, and

    extracts tight bounding boxes + values for Vietnamese Visa Application Forms
    (Mẫu NA1).
    """

    def __init__(self, raw_tokens: Sequence[Any], image_width: int, image_height: int) -> None:
        self.image_width = max(1, image_width)
        self.image_height = max(1, image_height)
        self.tokens = normalize_tokens(raw_tokens, self.image_width, self.image_height)
        self.lines = group_tokens_into_lines(self.tokens)
        self.detected_anchors: dict[str, DetectedAnchor] = {}
        self.section6_ny: int | None = None
        self.section7_ny: int | None = None

    def extract_fields(self) -> list[ExtractedFieldResult]:
        if not self.tokens:
            return self._fallback_all_fields()

        # Step 1: Detect section headers first (Section 6 Contact, Section 7 Sponsor)
        self._detect_section_headers()

        # Step 2: Detect all 19 field anchors
        self._detect_all_field_anchors()

        # Step 3: Compute global coordinate transformation (scale & shift) from detected anchors
        scale_x, shift_x, scale_y, shift_y = self._fit_coordinate_transform()

        # Step 4: Extract bounding box and value_text for each field
        results: list[ExtractedFieldResult] = []
        for code in FIELD_CODES_ORDERED:
            meta = TEMPLATE_FIELD_METADATA[code]
            field_res = self._extract_single_field(
                code=code,
                meta=meta,
                scale_x=scale_x,
                shift_x=shift_x,
                scale_y=scale_y,
                shift_y=shift_y,
            )
            results.append(field_res)

        return results

    def _detect_section_headers(self) -> None:
        sec6_pat = re.compile(r"(?:6\s*[\.\)]?\s*)?contact\s*information|thong\s*tin\s*lien\s*he")
        sec7_pat = re.compile(
            r"(?:7\s*[\.\)]?\s*)?sponsor\s*information|thong\s*tin\s*co\s*quan|ca\s*nhan\s*moi\s*don"
        )

        for line in self.lines:
            line_str = " ".join(t.clean_text for t in line.tokens if t.nx_center < 500)
            if self.section6_ny is None and sec6_pat.search(line_str):
                self.section6_ny = line.ny1
                for t in line.tokens:
                    if t.nx_center < 500 and any(
                        w in t.clean_text
                        for w in ("contact", "information", "thong", "tin", "lien", "he")
                    ):
                        t.is_anchor = True

            if self.section7_ny is None and sec7_pat.search(line_str):
                self.section7_ny = line.ny1
                for t in line.tokens:
                    if t.nx_center < 500 and any(
                        w in t.clean_text
                        for w in ("sponsor", "information", "co", "quan", "moi", "don")
                    ):
                        t.is_anchor = True

        if self.section6_ny is None:
            self.section6_ny = 345
        if self.section7_ny is None:
            self.section7_ny = 480

    def _detect_all_field_anchors(self) -> None:
        anchor_rules: dict[str, dict[str, Any]] = {
            "full_name": {
                "col": "left",
                "y_range": (110, 240),
                "patterns": [
                    re.compile(r"(?:1\s*[\.\)]?\s*)?full\s*name"),
                    re.compile(r"ten\s*day\s*du"),
                    re.compile(r"ho\s*(?:va\s*)?ten"),
                    re.compile(r"1\s*[\.\)]\s*full"),
                ],
            },
            "date_of_birth": {
                "col": "left",
                "y_range": (160, 275),
                "patterns": [
                    re.compile(r"(?:2\s*[\.\)]?\s*)?date\s*of\s*birth"),
                    re.compile(r"ngay\s*sinh"),
                    re.compile(r"2\s*[\.\)]\s*date"),
                ],
            },
            "place_of_birth": {
                "col": "left",
                "y_range": (200, 310),
                "patterns": [
                    re.compile(r"(?:3\s*[\.\)]?\s*)?place\s*of\s*birth"),
                    re.compile(r"noi\s*sinh"),
                    re.compile(r"3\s*[\.\)]\s*place"),
                ],
            },
            "gender": {
                "col": "left",
                "y_range": (235, 345),
                "patterns": [
                    re.compile(r"(?:4\s*[\.\)]?\s*)?gender"),
                    re.compile(r"gioi\s*tinh"),
                    re.compile(r"4\s*[\.\)]\s*gender"),
                    re.compile(r"sex"),
                ],
            },
            "nationality": {
                "col": "left",
                "y_range": (270, 380),
                "patterns": [
                    re.compile(r"(?:5\s*[\.\)]?\s*)?nationality"),
                    re.compile(r"quoc\s*tich"),
                    re.compile(r"5\s*[\.\)]\s*nationality"),
                ],
            },
            "contact_address": {
                "col": "left",
                "y_range": (self.section6_ny or 340, (self.section7_ny or 480) - 40),
                "patterns": [
                    re.compile(r"address"),
                    re.compile(r"dia\s*chi"),
                ],
            },
            "contact_phone": {
                "col": "left",
                "y_range": ((self.section6_ny or 340) + 15, (self.section7_ny or 480) - 15),
                "patterns": [
                    re.compile(r"telephone"),
                    re.compile(r"dien\s*thoai"),
                    re.compile(r"phone"),
                    re.compile(r"tel"),
                ],
            },
            "contact_email": {
                "col": "left",
                "y_range": ((self.section6_ny or 340) + 35, (self.section7_ny or 480)),
                "patterns": [
                    re.compile(r"email"),
                ],
            },
            "sponsor_name": {
                "col": "left",
                "y_range": (self.section7_ny or 480, 560),
                "patterns": [
                    re.compile(r"name"),
                    re.compile(r"ten"),
                ],
            },
            "sponsor_phone": {
                "col": "left",
                "y_range": ((self.section7_ny or 480) + 15, 620),
                "patterns": [
                    re.compile(r"telephone"),
                    re.compile(r"dien\s*thoai"),
                    re.compile(r"tel"),
                ],
            },
            "passport_number": {
                "col": "right",
                "y_range": (150, 275),
                "patterns": [
                    re.compile(r"(?:8\s*[\.\)]?\s*)?passport\s*number"),
                    re.compile(r"so\s*ho\s*chieu"),
                    re.compile(r"passport"),
                ],
            },
            "passport_expiration": {
                "col": "right",
                "y_range": (200, 310),
                "patterns": [
                    re.compile(r"date\s*of\s*expiry"),
                    re.compile(r"ngay\s*het\s*han"),
                    re.compile(r"het\s*han\s*ho\s*chieu"),
                    re.compile(r"expiry"),
                    re.compile(r"expiration"),
                ],
            },
            "number_of_entries": {
                "col": "right",
                "y_range": (235, 345),
                "patterns": [
                    re.compile(r"(?:9\s*[\.\)]?\s*)?number\s*of\s*entries"),
                    re.compile(r"so\s*lan\s*nhap\s*canh"),
                    re.compile(r"entries"),
                ],
            },
            "entry_date": {
                "col": "right",
                "y_range": (270, 380),
                "patterns": [
                    re.compile(r"(?:10\s*[\.\)]?\s*)?proposed\s*date\s*of\s*entry"),
                    re.compile(r"ngay\s*du\s*kien\s*nhap\s*canh"),
                    re.compile(r"du\s*kien\s*nhap\s*canh"),
                    re.compile(r"date\s*of\s*entry"),
                ],
            },
            "exit_date": {
                "col": "right",
                "y_range": (310, 415),
                "patterns": [
                    re.compile(r"(?:11\s*[\.\)]?\s*)?proposed\s*date\s*of\s*exit"),
                    re.compile(r"ngay\s*du\s*kien\s*xuat\s*canh"),
                    re.compile(r"du\s*kien\s*xuat\s*canh"),
                    re.compile(r"date\s*of\s*exit"),
                ],
            },
            "submission_place": {
                "col": "right",
                "y_range": (350, 460),
                "patterns": [
                    re.compile(r"(?:12\s*[\.\)]?\s*)?place\s*of\s*visa\s*submission"),
                    re.compile(r"noi\s*nop\s*ho\s*so"),
                    re.compile(r"visa\s*submission"),
                ],
            },
            "collection_place": {
                "col": "right",
                "y_range": (420, 525),
                "patterns": [
                    re.compile(r"(?:13\s*[\.\)]?\s*)?place\s*of\s*visa\s*collection"),
                    re.compile(r"noi\s*nhan\s*thi\s*thuc"),
                    re.compile(r"visa\s*collection"),
                ],
            },
            "purpose_of_entry": {
                "col": "right",
                "y_range": (470, 600),
                "patterns": [
                    re.compile(r"(?:14\s*[\.\)]?\s*)?purpose\s*of\s*entry"),
                    re.compile(r"muc\s*dich\s*nhap\s*canh"),
                ],
            },
            "application_date": {
                "col": "right",
                "y_range": (620, 850),
                "patterns": [
                    re.compile(r"date\s*[\/\:]?\s*ngay"),
                    re.compile(r"date"),
                    re.compile(r"ngay"),
                ],
            },
        }

        for code in FIELD_CODES_ORDERED:
            rule = anchor_rules[code]
            is_right = rule["col"] == "right"
            y_min, y_max = rule["y_range"]
            patterns = rule["patterns"]

            matched_anchor = self._find_anchor_in_lines(
                patterns=patterns,
                is_right_col=is_right,
                y_min=y_min,
                y_max=y_max,
                field_code=code,
            )
            if matched_anchor:
                self.detected_anchors[code] = matched_anchor
                for tok in matched_anchor.tokens:
                    tok.is_anchor = True
                    tok.assigned_field = code

    def _find_anchor_in_lines(
        self,
        patterns: list[re.Pattern],
        is_right_col: bool,
        y_min: int,
        y_max: int,
        field_code: str,
    ) -> DetectedAnchor | None:
        anchor_tokens: list[ExtractedToken] = []

        for line in self.lines:
            if line.ny_center < y_min or line.ny_center > y_max:
                continue

            col_tokens = [
                t
                for t in line.tokens
                if (t.nx_center >= 460 if is_right_col else t.nx_center < 500)
                and (t.assigned_field is None or t.assigned_field == field_code)
            ]
            if not col_tokens:
                continue

            n = len(col_tokens)
            anchor_indices: set[int] = set()

            # Pass 1: Match multi-token exact spans (from longest to shortest)
            for span_len in range(min(5, n), 0, -1):
                for i in range(n - span_len + 1):
                    if any(idx in anchor_indices for idx in range(i, i + span_len)):
                        continue
                    span = col_tokens[i : i + span_len]
                    span_text = " ".join(t.clean_text for t in span).strip(" :/-.,")
                    if any(p.fullmatch(span_text) for p in patterns):
                        for idx in range(i, i + span_len):
                            anchor_indices.add(idx)

            if not anchor_indices:
                continue

            # Pass 2: Absorb date format hints like (DD/MM/YYYY) adjacent to anchor
            for i in range(n):
                if i not in anchor_indices and FORMAT_HINT_PAT.search(col_tokens[i].clean_text):
                    if (i > 0 and i - 1 in anchor_indices) or (i + 1 < n and i + 1 in anchor_indices):
                        anchor_indices.add(i)

            # Pass 3: Absorb in-between delimiters (e.g. '/' or ':')
            for i in range(n):
                if i not in anchor_indices and col_tokens[i].clean_text in (":", "/", "-", "."):
                    if (i > 0 and i - 1 in anchor_indices) and (i + 1 < n and i + 1 in anchor_indices):
                        anchor_indices.add(i)

            # Pass 4: Absorb trailing punctuation immediately after anchor (e.g. ':')
            for i in range(n):
                if i not in anchor_indices and (i > 0 and i - 1 in anchor_indices):
                    if col_tokens[i].clean_text in (":", "/", "-", "."):
                        anchor_indices.add(i)

            for idx in sorted(anchor_indices):
                anchor_tokens.append(col_tokens[idx])

        if not anchor_tokens:
            return None

        nx1 = min(t.nx1 for t in anchor_tokens)
        ny1 = min(t.ny1 for t in anchor_tokens)
        nx2 = max(t.nx2 for t in anchor_tokens)
        ny2 = max(t.ny2 for t in anchor_tokens)
        return DetectedAnchor(
            field_code=field_code,
            nx1=nx1,
            ny1=ny1,
            nx2=nx2,
            ny2=ny2,
            tokens=anchor_tokens,
        )

    def _fit_coordinate_transform(self) -> tuple[float, float, float, float]:
        """Calculates global scale and shift factors relative to baseline template coordinates.

        Returns (scale_x, shift_x, scale_y, shift_y).
        """
        pairs_y: list[tuple[float, float]] = []
        pairs_x: list[tuple[float, float]] = []

        for code, anchor in self.detected_anchors.items():
            base_meta = TEMPLATE_FIELD_METADATA[code]
            base_anchor = base_meta["baseline_anchor"]
            base_cy = (base_anchor[1] + base_anchor[3]) / 2.0
            base_cx = (base_anchor[0] + base_anchor[2]) / 2.0
            pairs_y.append((base_cy, anchor.ny_center))
            pairs_x.append((base_cx, anchor.nx_center))

        if len(pairs_y) < 2:
            if len(pairs_y) == 1:
                shift_y = pairs_y[0][1] - pairs_y[0][0]
                shift_x = pairs_x[0][1] - pairs_x[0][0]
                return 1.0, shift_x, 1.0, shift_y
            return 1.0, 0.0, 1.0, 0.0

        # Linear regression for scale and shift
        base_y_mean = sum(p[0] for p in pairs_y) / len(pairs_y)
        det_y_mean = sum(p[1] for p in pairs_y) / len(pairs_y)

        num = sum((p[0] - base_y_mean) * (p[1] - det_y_mean) for p in pairs_y)
        den = sum((p[0] - base_y_mean) ** 2 for p in pairs_y)

        scale_y = (num / den) if den > 1e-4 else 1.0
        scale_y = max(0.80, min(1.25, scale_y))
        shift_y = det_y_mean - scale_y * base_y_mean

        base_x_mean = sum(p[0] for p in pairs_x) / len(pairs_x)
        det_x_mean = sum(p[1] for p in pairs_x) / len(pairs_x)
        scale_x = 1.0
        shift_x = det_x_mean - base_x_mean

        return scale_x, shift_x, scale_y, shift_y

    def _extract_single_field(
        self,
        code: str,
        meta: dict[str, Any],
        scale_x: float,
        shift_x: float,
        scale_y: float,
        shift_y: float,
    ) -> ExtractedFieldResult:
        field_type = meta["field_type"]
        is_right = meta["column"] == "right"
        col_left_bound = 500 if is_right else 50
        col_right_bound = 980 if is_right else 485

        anchor = self.detected_anchors.get(code)
        anchor_found = anchor is not None

        if anchor:
            ax1, ay1, ax2, ay2 = anchor.nx1, anchor.ny1, anchor.nx2, anchor.ny2
            anchor_bbox = (ax1, ay1, ax2, ay2)
        else:
            base_anc = meta["baseline_anchor"]
            ax1 = int(round(base_anc[0] * scale_x + shift_x))
            ay1 = int(round(base_anc[1] * scale_y + shift_y))
            ax2 = int(round(base_anc[2] * scale_x + shift_x))
            ay2 = int(round(base_anc[3] * scale_y + shift_y))
            anchor_bbox = None

        next_anchor_top = self._find_next_anchor_top(
            current_code=code,
            is_right=is_right,
            scale_y=scale_y,
            shift_y=shift_y,
        )

        if field_type == "inline":
            res = self._extract_inline_field(
                code=code,
                meta=meta,
                ax1=ax1,
                ay1=ay1,
                ax2=ax2,
                ay2=ay2,
                col_right=col_right_bound,
                next_anchor_top=next_anchor_top,
                anchor_found=anchor_found,
                anchor_bbox=anchor_bbox,
            )
        else:
            res = self._extract_block_field(
                code=code,
                meta=meta,
                ax1=ax1,
                ay1=ay1,
                ax2=ax2,
                ay2=ay2,
                col_left=col_left_bound,
                col_right=col_right_bound,
                next_anchor_top=next_anchor_top,
                anchor_found=anchor_found,
                anchor_bbox=anchor_bbox,
            )
        return res

    def _find_next_anchor_top(
        self,
        current_code: str,
        is_right: bool,
        scale_y: float,
        shift_y: float,
    ) -> int:
        col_codes = [
            c for c in FIELD_CODES_ORDERED if (TEMPLATE_FIELD_METADATA[c]["column"] == "right") == is_right
        ]
        curr_idx = col_codes.index(current_code)
        for next_code in col_codes[curr_idx + 1 :]:
            if next_code in self.detected_anchors:
                return self.detected_anchors[next_code].ny1
            next_base = TEMPLATE_FIELD_METADATA[next_code]["baseline_anchor"]
            return int(round(next_base[1] * scale_y + shift_y))
        return 1000

    def _extract_inline_field(
        self,
        code: str,
        meta: dict[str, Any],
        ax1: int,
        ay1: int,
        ax2: int,
        ay2: int,
        col_right: int,
        next_anchor_top: int,
        anchor_found: bool,
        anchor_bbox: tuple[int, int, int, int] | None,
    ) -> ExtractedFieldResult:
        h = max(14, ay2 - ay1)
        y_search_min = max(0, ay1 - 6)
        y_search_max = min(next_anchor_top - 2, ay2 + 5)
        x_search_min = max(ax1 + 5, ax2 - 5)

        val_tokens: list[ExtractedToken] = []
        for t in self.tokens:
            if t.is_anchor:
                continue
            if t.nx_center >= x_search_min and t.nx2 <= col_right + 15:
                if t.ny1 < next_anchor_top - 1:
                    if t.ny1 >= y_search_min - 4 and t.ny2 <= y_search_max + 4:
                        val_tokens.append(t)
                    elif abs(t.ny_center - (ay1 + ay2) / 2.0) <= max(10, h * 0.50):
                        val_tokens.append(t)

        val_tokens.sort(key=lambda t: t.nx1)

        if val_tokens:
            nx1 = max(0, min(t.nx1 for t in val_tokens) - 4)
            ny1 = max(0, min(t.ny1 for t in val_tokens) - 3)
            nx2 = min(1000, max(t.nx2 for t in val_tokens) + 4)
            ny2 = min(1000, max(t.ny2 for t in val_tokens) + 3)
            value_text = " ".join(t.text for t in val_tokens).strip() or None
        else:
            nx1 = min(col_right - 40, ax2 + 8)
            nx2 = min(col_right, nx1 + 220)
            ny1 = ay1
            ny2 = ay2
            value_text = None

        nx1, ny1, nx2, ny2 = self._clamp_norm_box(nx1, ny1, nx2, ny2)
        px1, py1, px2, py2 = self._norm_to_pixel(nx1, ny1, nx2, ny2)

        return ExtractedFieldResult(
            field_code=code,
            field_order=meta["order"],
            label=meta["label"],
            bbox=(px1, py1, px2, py2),
            normalized_bbox=(nx1, ny1, nx2, ny2),
            value_text=value_text,
            anchor_found=anchor_found,
            anchor_normalized_bbox=anchor_bbox,
        )

    def _extract_block_field(
        self,
        code: str,
        meta: dict[str, Any],
        ax1: int,
        ay1: int,
        ax2: int,
        ay2: int,
        col_left: int,
        col_right: int,
        next_anchor_top: int,
        anchor_found: bool,
        anchor_bbox: tuple[int, int, int, int] | None,
    ) -> ExtractedFieldResult:
        val_tokens: list[ExtractedToken] = []
        for t in self.tokens:
            if t.is_anchor:
                continue
            if t.nx1 >= col_left - 15 and t.nx2 <= col_right + 15:
                if t.ny1 < next_anchor_top - 1:
                    if (t.ny1 >= ay1 - 5 and t.nx1 >= ax2 - 5) or (t.ny1 >= ay2 - 5):
                        val_tokens.append(t)

        val_tokens.sort(key=lambda t: (t.ny1, t.nx1))

        if val_tokens:
            nx1 = max(0, min(t.nx1 for t in val_tokens) - 5)
            ny1 = max(0, min(t.ny1 for t in val_tokens) - 3)
            nx2 = min(1000, max(t.nx2 for t in val_tokens) + 5)
            ny2 = min(1000, max(t.ny2 for t in val_tokens) + 3)
            value_text = " ".join(t.text for t in val_tokens).strip() or None
        else:
            nx1 = col_left
            nx2 = col_right
            ny1 = ay2 + 5
            ny2 = min(next_anchor_top - 5, ny1 + 45)
            value_text = None

        nx1, ny1, nx2, ny2 = self._clamp_norm_box(nx1, ny1, nx2, ny2)
        px1, py1, px2, py2 = self._norm_to_pixel(nx1, ny1, nx2, ny2)

        return ExtractedFieldResult(
            field_code=code,
            field_order=meta["order"],
            label=meta["label"],
            bbox=(px1, py1, px2, py2),
            normalized_bbox=(nx1, ny1, nx2, ny2),
            value_text=value_text,
            anchor_found=anchor_found,
            anchor_normalized_bbox=anchor_bbox,
        )

    def _fallback_all_fields(self) -> list[ExtractedFieldResult]:
        results: list[ExtractedFieldResult] = []
        for code in FIELD_CODES_ORDERED:
            meta = TEMPLATE_FIELD_METADATA[code]
            base_val = meta["baseline_value"]
            nx1, ny1, nx2, ny2 = self._clamp_norm_box(
                base_val[0], base_val[1], base_val[2], base_val[3]
            )
            px1, py1, px2, py2 = self._norm_to_pixel(nx1, ny1, nx2, ny2)
            results.append(
                ExtractedFieldResult(
                    field_code=code,
                    field_order=meta["order"],
                    label=meta["label"],
                    bbox=(px1, py1, px2, py2),
                    normalized_bbox=(nx1, ny1, nx2, ny2),
                    value_text=None,
                    anchor_found=False,
                    anchor_normalized_bbox=None,
                )
            )
        return results

    def _clamp_norm_box(self, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        cx1 = max(0, min(998, int(x1)))
        cy1 = max(0, min(998, int(y1)))
        cx2 = max(cx1 + 1, min(1000, int(x2)))
        cy2 = max(cy1 + 1, min(1000, int(y2)))
        return cx1, cy1, cx2, cy2

    def _norm_to_pixel(
        self, nx1: int, ny1: int, nx2: int, ny2: int
    ) -> tuple[int, int, int, int]:
        w, h = self.image_width, self.image_height
        px1 = max(0, min(w - 2, round(nx1 * w / 1000)))
        py1 = max(0, min(h - 2, round(ny1 * h / 1000)))
        px2 = max(px1 + 1, min(w, round(nx2 * w / 1000)))
        py2 = max(py1 + 1, min(h, round(ny2 * h / 1000)))
        return px1, py1, px2, py2


def extract_visa_form_fields(
    raw_tokens: Sequence[Any], image_width: int, image_height: int
) -> list[ExtractedFieldResult]:
    """Top-level convenience entry point for extracting 19 fields from raw OCR tokens."""
    extractor = VisaFormAnchorExtractor(raw_tokens, image_width, image_height)
    return extractor.extract_fields()
