from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Default template bounding boxes (in 0..1000 normalized coordinates)
# Baseline reference calibrated to standard Mẫu NA1 forms.
TEMPLATE_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "full_name": {
        "order": 1,
        "label": "Họ và tên",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 200, 220, 225),
        "baseline_value": (215, 204, 420, 225),
    },
    "date_of_birth": {
        "order": 2,
        "label": "Ngày sinh",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 235, 210, 260),
        "baseline_value": (215, 239, 320, 260),
    },
    "place_of_birth": {
        "order": 3,
        "label": "Nơi sinh",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 270, 220, 295),
        "baseline_value": (225, 274, 300, 295),
    },
    "gender": {
        "order": 4,
        "label": "Giới tính",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 305, 175, 330),
        "baseline_value": (180, 306, 250, 328),
    },
    "nationality": {
        "order": 5,
        "label": "Quốc tịch",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 338, 210, 363),
        "baseline_value": (215, 341, 300, 362),
    },
    "contact_address": {
        "order": 6,
        "label": "Địa chỉ liên hệ",
        "column": "left",
        "field_type": "block",
        "baseline_anchor": (90, 405, 180, 430),
        "baseline_value": (170, 408, 380, 450),
    },
    "contact_phone": {
        "order": 7,
        "label": "Điện thoại liên hệ",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 440, 210, 465),
        "baseline_value": (210, 443, 310, 463),
    },
    "contact_email": {
        "order": 8,
        "label": "Email liên hệ",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 475, 185, 500),
        "baseline_value": (185, 476, 380, 497),
    },
    "sponsor_name": {
        "order": 9,
        "label": "Tên người/cơ quan mời",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 525, 175, 550),
        "baseline_value": (175, 529, 390, 558),
    },
    "sponsor_phone": {
        "order": 10,
        "label": "Điện thoại người/cơ quan mời",
        "column": "left",
        "field_type": "inline",
        "baseline_anchor": (90, 560, 205, 585),
        "baseline_value": (205, 563, 420, 592),
    },
    "passport_number": {
        "order": 11,
        "label": "Số hộ chiếu",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (520, 235, 680, 260),
        "baseline_value": (680, 237, 780, 258),
    },
    "passport_expiration": {
        "order": 12,
        "label": "Hạn hộ chiếu",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (520, 270, 660, 295),
        "baseline_value": (660, 270, 750, 292),
    },
    "number_of_entries": {
        "order": 13,
        "label": "Số lần nhập cảnh",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (520, 305, 685, 330),
        "baseline_value": (685, 305, 770, 325),
    },
    "entry_date": {
        "order": 14,
        "label": "Ngày dự kiến nhập cảnh",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (520, 335, 720, 360),
        "baseline_value": (720, 338, 810, 359),
    },
    "exit_date": {
        "order": 15,
        "label": "Ngày dự kiến xuất cảnh",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (520, 370, 715, 395),
        "baseline_value": (715, 372, 805, 393),
    },
    "submission_place": {
        "order": 16,
        "label": "Nơi nộp hồ sơ",
        "column": "right",
        "field_type": "block",
        "baseline_anchor": (520, 405, 730, 435),
        "baseline_value": (550, 440, 720, 460),
    },
    "collection_place": {
        "order": 17,
        "label": "Nơi nhận thị thực",
        "column": "right",
        "field_type": "block",
        "baseline_anchor": (520, 470, 730, 500),
        "baseline_value": (550, 505, 860, 540),
    },
    "purpose_of_entry": {
        "order": 18,
        "label": "Mục đích nhập cảnh",
        "column": "right",
        "field_type": "block",
        "baseline_anchor": (520, 540, 680, 570),
        "baseline_value": (700, 540, 920, 610),
    },
    "application_date": {
        "order": 19,
        "label": "Ngày làm đơn",
        "column": "right",
        "field_type": "inline",
        "baseline_anchor": (620, 710, 730, 735),
        "baseline_value": (735, 710, 830, 730),
    },
}

FIELD_CODES_ORDERED = tuple(
    sorted(TEMPLATE_FIELD_METADATA.keys(), key=lambda k: TEMPLATE_FIELD_METADATA[k]["order"])
)

FORMAT_HINT_PAT = re.compile(r"\(?\s*d{1,2}[\/\.\-]mm[\/\.\-]yyyy\s*\)?", re.IGNORECASE)

COMPOUND_PREFIX_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(?:1\s*[\.\)]?\s*)?(?:full\s*name|ten\s*day\s*du|ho\s*(?:va\s*)?ten)\s*[:\s]+", re.I),
    re.compile(r"^(?:2\s*[\.\)]?\s*)?(?:date\s*of\s*birth|ngay\s*sinh)\s*(?:\([^\)]*\))?\s*[:\s]+", re.I),
    re.compile(r"^(?:3\s*[\.\)]?\s*)?(?:place\s*of\s*birth|noi\s*sinh)\s*[:\s]+", re.I),
    re.compile(r"^(?:4\s*[\.\)]?\s*)?(?:gender|gioi\s*tinh|sex)\s*[:\s]+", re.I),
    re.compile(r"^(?:5\s*[\.\)]?\s*)?(?:nationality|quoc\s*tich)\s*[:\s]+", re.I),
    re.compile(r"^(?:address|dia\s*chi)\s*[:\s]+", re.I),
    re.compile(r"^(?:telephone|dien\s*thoai|phone|tel)\s*[:\s]+", re.I),
    re.compile(r"^(?:email)\s*[:\s]+", re.I),
    re.compile(r"^(?:8\s*[\.\)]?\s*)?(?:passport\s*number|so\s*ho\s*chieu)\s*[:\s]+", re.I),
    re.compile(r"^(?:date\s*of\s*expiry|ngay\s*het\s*han|het\s*han\s*ho\s*chieu|expiry|expiration)\s*(?:\([^\)]*\))?\s*[:\s]+", re.I),
    re.compile(r"^(?:9\s*[\.\)]?\s*)?(?:number\s*of\s*entries|so\s*lan\s*nhap\s*canh)\s*[:\s]+", re.I),
    re.compile(r"^(?:10\s*[\.\)]?\s*)?(?:proposed\s*date\s*of\s*entry|ngay\s*du\s*kien\s*nhap\s*canh|du\s*kien\s*nhap\s*canh)\s*(?:\([^\)]*\))?\s*[:\s]+", re.I),
    re.compile(r"^(?:11\s*[\.\)]?\s*)?(?:proposed\s*date\s*of\s*exit|ngay\s*du\s*kien\s*xuat\s*canh|du\s*kien\s*xuat\s*canh)\s*(?:\([^\)]*\))?\s*[:\s]+", re.I),
    re.compile(r"^(?:12\s*[\.\)]?\s*)?(?:place\s*of\s*visa\s*submission|noi\s*nop\s*ho\s*so)\s*[:\s]+", re.I),
    re.compile(r"^(?:13\s*[\.\)]?\s*)?(?:place\s*of\s*visa\s*collection|noi\s*nhan\s*thi\s*thuc)\s*[:\s]+", re.I),
    re.compile(r"^(?:14\s*[\.\)]?\s*)?(?:purpose\s*of\s*entry|muc\s*dich\s*nhap\s*canh)\s*[:\s]+", re.I),
    re.compile(r"^(?:date\s*[\/\:]?\s*ngay|ngay\s*lam\s*don)\s*[:\s]+", re.I),
]


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
    primary_nx2: int = 0
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

        clean = strip_accents_and_lower(text)

        # Check if line contains both anchor pattern and value text
        matched_prefix_end = None
        for pat in COMPOUND_PREFIX_PATTERNS:
            m = pat.match(clean)
            if m:
                val_text = text[m.end() :].strip()
                if val_text:
                    matched_prefix_end = m.end()
                break

        if matched_prefix_end:
            # Compound token: split into anchor token + value token
            total_chars = max(1, len(text))
            ratio = matched_prefix_end / total_chars
            split_x = max(x1 + 1, min(x2 - 1, x1 + round((x2 - x1) * ratio)))

            p_text = text[:matched_prefix_end].strip()
            v_text = text[matched_prefix_end:].strip()

            # Token A (anchor)
            anx1 = max(0, min(998, round(x1 * 1000 / w)))
            any1 = max(0, min(998, round(y1 * 1000 / h)))
            anx2 = max(anx1 + 1, min(1000, round(split_x * 1000 / w)))
            any2 = max(any1 + 1, min(1000, round(y2 * 1000 / h)))
            results.append(
                ExtractedToken(
                    text=p_text,
                    clean_text=strip_accents_and_lower(p_text),
                    x1=x1,
                    y1=y1,
                    x2=split_x,
                    y2=y2,
                    nx1=anx1,
                    ny1=any1,
                    nx2=anx2,
                    ny2=any2,
                    confidence=float(conf),
                    is_anchor=True,
                )
            )

            # Token B (value)
            vnx1 = max(0, min(998, round(split_x * 1000 / w)))
            vny1 = max(0, min(998, round(y1 * 1000 / h)))
            vnx2 = max(vnx1 + 1, min(1000, round(x2 * 1000 / w)))
            vny2 = max(vny1 + 1, min(1000, round(y2 * 1000 / h)))

            # If value token is just format hint (e.g. '(DD/MM/YYYY)'), mark as anchor
            is_hint = bool(FORMAT_HINT_PAT.fullmatch(v_text))
            results.append(
                ExtractedToken(
                    text=v_text,
                    clean_text=strip_accents_and_lower(v_text),
                    x1=split_x,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    nx1=vnx1,
                    ny1=vny1,
                    nx2=vnx2,
                    ny2=vny2,
                    confidence=float(conf),
                    is_anchor=is_hint,
                )
            )
        else:
            nx1 = max(0, min(998, round(x1 * 1000 / w)))
            ny1 = max(0, min(998, round(y1 * 1000 / h)))
            nx2 = max(nx1 + 1, min(1000, round(x2 * 1000 / w)))
            ny2 = max(ny1 + 1, min(1000, round(y2 * 1000 / h)))
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

        # Step 1: Detect section headers (Section 6 Contact, Section 7 Sponsor)
        self._detect_section_headers()

        # Step 2: Detect all 19 field anchors
        self._detect_all_field_anchors()

        # Step 3: Compute coordinate transformation (scale & shift) from detected anchors
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
            self.section6_ny = 375
        if self.section7_ny is None:
            self.section7_ny = 495

    def _detect_all_field_anchors(self) -> None:
        anchor_rules: dict[str, dict[str, Any]] = {
            "full_name": {
                "col": "left",
                "y_range": (160, 240),
                "patterns": [
                    re.compile(r"(?:1\s*[\.\)]?\s*)?full\s*name"),
                    re.compile(r"ten\s*day\s*du"),
                    re.compile(r"ho\s*(?:va\s*)?ten"),
                    re.compile(r"1\s*[\.\)]\s*full"),
                ],
            },
            "date_of_birth": {
                "col": "left",
                "y_range": (205, 275),
                "patterns": [
                    re.compile(r"(?:2\s*[\.\)]?\s*)?date\s*of\s*birth"),
                    re.compile(r"ngay\s*sinh"),
                    re.compile(r"2\s*[\.\)]\s*date"),
                ],
            },
            "place_of_birth": {
                "col": "left",
                "y_range": (245, 310),
                "patterns": [
                    re.compile(r"(?:3\s*[\.\)]?\s*)?place\s*of\s*birth"),
                    re.compile(r"noi\s*sinh"),
                    re.compile(r"3\s*[\.\)]\s*place"),
                ],
            },
            "gender": {
                "col": "left",
                "y_range": (280, 345),
                "patterns": [
                    re.compile(r"(?:4\s*[\.\)]?\s*)?gender"),
                    re.compile(r"gioi\s*tinh"),
                    re.compile(r"4\s*[\.\)]\s*gender"),
                    re.compile(r"sex"),
                ],
            },
            "nationality": {
                "col": "left",
                "y_range": (315, 380),
                "patterns": [
                    re.compile(r"(?:5\s*[\.\)]?\s*)?nationality"),
                    re.compile(r"quoc\s*tich"),
                    re.compile(r"5\s*[\.\)]\s*nationality"),
                ],
            },
            "contact_address": {
                "col": "left",
                "y_range": (self.section6_ny or 375, (self.section7_ny or 495) - 40),
                "patterns": [
                    re.compile(r"address"),
                    re.compile(r"dia\s*chi"),
                ],
            },
            "contact_phone": {
                "col": "left",
                "y_range": ((self.section6_ny or 375) + 30, (self.section7_ny or 495) - 10),
                "patterns": [
                    re.compile(r"telephone"),
                    re.compile(r"dien\s*thoai"),
                    re.compile(r"phone"),
                    re.compile(r"tel"),
                ],
            },
            "contact_email": {
                "col": "left",
                "y_range": ((self.section6_ny or 375) + 60, (self.section7_ny or 495) + 15),
                "patterns": [
                    re.compile(r"email"),
                ],
            },
            "sponsor_name": {
                "col": "left",
                "y_range": (self.section7_ny or 495, 575),
                "patterns": [
                    re.compile(r"name"),
                    re.compile(r"ten"),
                ],
            },
            "sponsor_phone": {
                "col": "left",
                "y_range": ((self.section7_ny or 495) + 30, 625),
                "patterns": [
                    re.compile(r"telephone"),
                    re.compile(r"dien\s*thoai"),
                    re.compile(r"tel"),
                ],
            },
            "passport_number": {
                "col": "right",
                "y_range": (205, 275),
                "patterns": [
                    re.compile(r"(?:8\s*[\.\)]?\s*)?passport\s*number"),
                    re.compile(r"so\s*ho\s*chieu"),
                    re.compile(r"passport"),
                ],
            },
            "passport_expiration": {
                "col": "right",
                "y_range": (245, 310),
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
                "y_range": (280, 345),
                "patterns": [
                    re.compile(r"(?:9\s*[\.\)]?\s*)?number\s*of\s*entries"),
                    re.compile(r"so\s*lan\s*nhap\s*canh"),
                    re.compile(r"entries"),
                ],
            },
            "entry_date": {
                "col": "right",
                "y_range": (315, 380),
                "patterns": [
                    re.compile(r"(?:10\s*[\.\)]?\s*)?proposed\s*date\s*of\s*entry"),
                    re.compile(r"ngay\s*du\s*kien\s*nhap\s*canh"),
                    re.compile(r"du\s*kien\s*nhap\s*canh"),
                    re.compile(r"date\s*of\s*entry"),
                ],
            },
            "exit_date": {
                "col": "right",
                "y_range": (350, 415),
                "patterns": [
                    re.compile(r"(?:11\s*[\.\)]?\s*)?proposed\s*date\s*of\s*exit"),
                    re.compile(r"ngay\s*du\s*kien\s*xuat\s*canh"),
                    re.compile(r"du\s*kien\s*xuat\s*canh"),
                    re.compile(r"date\s*of\s*exit"),
                ],
            },
            "submission_place": {
                "col": "right",
                "y_range": (385, 465),
                "patterns": [
                    re.compile(r"(?:12\s*[\.\)]?\s*)?place\s*of\s*visa\s*submission"),
                    re.compile(r"noi\s*nop\s*ho\s*so"),
                    re.compile(r"visa\s*submission"),
                ],
            },
            "collection_place": {
                "col": "right",
                "y_range": (455, 535),
                "patterns": [
                    re.compile(r"(?:13\s*[\.\)]?\s*)?place\s*of\s*visa\s*collection"),
                    re.compile(r"noi\s*nhan\s*thi\s*thuc"),
                    re.compile(r"visa\s*collection"),
                ],
            },
            "purpose_of_entry": {
                "col": "right",
                "y_range": (515, 620),
                "patterns": [
                    re.compile(r"(?:14\s*[\.\)]?\s*)?purpose\s*of\s*entry"),
                    re.compile(r"muc\s*dich\s*nhap\s*canh"),
                ],
            },
            "application_date": {
                "col": "right",
                "y_range": (670, 800),
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
                # Also check if any single token was pre-flagged as anchor matching patterns
                for i, tok in enumerate(col_tokens):
                    if tok.is_anchor and any(p.search(tok.clean_text) for p in patterns):
                        anchor_indices.add(i)

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

        # Determine primary line tokens (the topmost line of the anchor)
        min_line_y = min(t.ny_center for t in anchor_tokens)
        primary_tokens = [t for t in anchor_tokens if abs(t.ny_center - min_line_y) <= 8]
        primary_nx2 = max(t.nx2 for t in primary_tokens)

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
            primary_nx2=primary_nx2,
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

        base_y_mean = sum(p[0] for p in pairs_y) / len(pairs_y)
        det_y_mean = sum(p[1] for p in pairs_y) / len(pairs_y)

        num = sum((p[0] - base_y_mean) * (p[1] - det_y_mean) for p in pairs_y)
        den = sum((p[0] - base_y_mean) ** 2 for p in pairs_y)

        scale_y = (num / den) if den > 1e-4 else 1.0
        scale_y = max(0.85, min(1.20, scale_y))
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
        col_left_bound = 480 if is_right else 50
        col_right_bound = 980 if is_right else 485

        anchor = self.detected_anchors.get(code)
        anchor_found = anchor is not None

        if anchor:
            ax1, ay1, ax2, ay2 = anchor.nx1, anchor.ny1, anchor.nx2, anchor.ny2
            anchor_bbox = (ax1, ay1, ax2, ay2)
            primary_nx2 = anchor.primary_nx2 or ax2
        else:
            base_anc = meta["baseline_anchor"]
            ax1 = int(round(base_anc[0] * scale_x + shift_x))
            ay1 = int(round(base_anc[1] * scale_y + shift_y))
            ax2 = int(round(base_anc[2] * scale_x + shift_x))
            ay2 = int(round(base_anc[3] * scale_y + shift_y))
            anchor_bbox = None
            primary_nx2 = ax2

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
                primary_nx2=primary_nx2,
                col_left=col_left_bound,
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
        primary_nx2: int,
        col_left: int,
        col_right: int,
        next_anchor_top: int,
        anchor_found: bool,
        anchor_bbox: tuple[int, int, int, int] | None,
    ) -> ExtractedFieldResult:
        is_right = meta["column"] == "right"
        anchor = self.detected_anchors.get(code)
        val_tokens: list[ExtractedToken] = []

        # Find the line containing the primary anchor
        target_ny = ay1 + 10
        if anchor and anchor.tokens:
            target_ny = min(t.ny_center for t in anchor.tokens)

        # 1. Search for value tokens on that primary line
        search_x_min = primary_nx2 - 2
        for t in self.tokens:
            if t.is_anchor:
                continue
            if is_right:
                if t.nx1 < col_left - 10:
                    continue
            else:
                if t.nx2 > col_right + 10:
                    continue

            # Check if on same line (within 12 normalized units vertically)
            if abs(t.ny_center - target_ny) <= 12 and t.nx_center >= search_x_min and t.ny1 >= ay1 - 4:
                if t.ny1 < next_anchor_top - 2:
                    val_tokens.append(t)

        val_tokens.sort(key=lambda t: t.nx1)

        # Filter out format hints if any slipped through
        val_tokens = [t for t in val_tokens if not FORMAT_HINT_PAT.fullmatch(t.text.strip())]

        # Special handling for gender if no token found on same line
        if code == "gender" and not val_tokens:
            for t in self.tokens:
                if not t.is_anchor and t.nx2 <= col_right and abs(t.ny_center - target_ny) <= 15:
                    if t.clean_text in ("male", "female", "nam", "nu", "m", "f"):
                        val_tokens = [t]
                        break

        # Special handling for application_date if not found
        if code == "application_date" and not val_tokens:
            for t in self.tokens:
                if not t.is_anchor and t.nx1 >= col_left and abs(t.ny_center - target_ny) <= 15:
                    if re.search(r"\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4}", t.text):
                        val_tokens = [t]
                        break

        if val_tokens:
            nx1 = max(0, min(t.nx1 for t in val_tokens) - 4)
            ny1 = max(0, min(t.ny1 for t in val_tokens) - 3)
            nx2 = min(1000, max(t.nx2 for t in val_tokens) + 4)
            ny2 = min(1000, max(t.ny2 for t in val_tokens) + 3)
            value_text = " ".join(t.text for t in val_tokens).strip() or None
            # Strip trailing format hints from value_text
            if value_text:
                value_text = FORMAT_HINT_PAT.sub("", value_text).strip() or None
        else:
            # Blank or missing field: synthesized tight bounding box
            if code in ("sponsor_name", "sponsor_phone"):
                nx1 = min(col_right - 40, primary_nx2 + 8)
                nx2 = min(col_right, nx1 + 220)
                ny1 = ay1
                ny2 = min(ay2 + 10, next_anchor_top - 5)
            else:
                nx1 = min(col_right - 40, primary_nx2 + 8)
                nx2 = min(col_right, nx1 + 200)
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
        is_right = meta["column"] == "right"

        # Explicit block boundaries and stops
        stop_keywords = ("statement", "trach nhiem", "knowledge", "phap luat", "checklist", "truoc")
        for t in self.tokens:
            if t.is_anchor:
                continue

            # Column filter
            if is_right:
                if t.nx1 < col_left - 15:
                    continue
            else:
                if t.nx2 > col_right + 15:
                    continue

            # Legal declaration stop condition for purpose_of_entry
            if code == "purpose_of_entry":
                if any(kw in t.clean_text for kw in stop_keywords) or t.ny1 >= 640:
                    continue

            # Exclude Vietnamese sublabels for contact_address
            if code == "contact_address":
                if "dia chi" in t.clean_text or "dja chi" in t.clean_text:
                    continue

            # Vertical range
            if t.ny1 >= ay1 - 4 and t.ny_center < next_anchor_top - 4:
                # Include tokens on anchor line if to the right of anchor, or any token below anchor line
                if (t.ny1 >= ay1 - 4 and t.nx1 >= ax2 - 5) or (t.ny1 >= ay2 - 5):
                    val_tokens.append(t)

        val_tokens.sort(key=lambda t: (t.ny1, t.nx1))

        if val_tokens:
            nx1 = max(0, min(t.nx1 for t in val_tokens) - 5)
            ny1 = max(0, min(t.ny1 for t in val_tokens) - 3)
            nx2 = min(1000, max(t.nx2 for t in val_tokens) + 5)
            ny2 = min(1000, max(t.ny2 for t in val_tokens) + 3)
            value_text = " ".join(t.text for t in val_tokens).strip() or None
        else:
            nx1 = col_left + 10
            nx2 = min(col_right, nx1 + 250)
            ny1 = ay2 + 5
            ny2 = min(next_anchor_top - 5, ny1 + 35)
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
