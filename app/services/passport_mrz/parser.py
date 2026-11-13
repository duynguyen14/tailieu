from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

MRZ_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
WEIGHTS = (7, 3, 1)


@dataclass(frozen=True)
class CheckResult:
    name: str
    expected: str
    actual: str | None
    passed: bool


@dataclass(frozen=True)
class MrzParseResult:
    document_type: str
    lines: list[str]
    fields: dict[str, str | None]
    checks: list[CheckResult]
    checksum_pass: bool
    format_score: float


def normalize_mrz_text(text: str) -> str:
    replacements = {"«": "<", "‹": "<", ">": "<", " ": "", "\t": "", "\r": ""}
    normalized = text.upper()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return "".join(character for character in normalized if character in MRZ_CHARS)


def _normalize_lines(raw_lines: list[str]) -> list[str]:
    return [line for raw in raw_lines if len(line := normalize_mrz_text(raw)) >= 20]


def _char_value(character: str) -> int:
    if character == "<":
        return 0
    if "0" <= character <= "9":
        return ord(character) - ord("0")
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    return 0


def check_digit(data: str) -> str:
    total = sum(
        _char_value(character) * WEIGHTS[index % 3]
        for index, character in enumerate(data)
    )
    return str(total % 10)


def _compact_date(value: str | None) -> str | None:
    if not value or not re.fullmatch(r"\d{6}", value):
        return None
    year, month, day = int(value[:2]), int(value[2:4]), int(value[4:6])
    # timezone.utc keeps this service compatible with the Linux Python 3.10 host.
    current_year = int(datetime.now(timezone.utc).strftime("%y"))  # noqa: UP017
    century = 2000 if year <= current_year + 10 else 1900
    try:
        return datetime(century + year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _clean(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value.replace("<", " ").strip())
    return cleaned or None


def _split_names(value: str) -> tuple[str | None, str | None]:
    parts = value.split("<<", 1)
    return _clean(parts[0]), _clean(parts[1]) if len(parts) > 1 else None


def _check(name: str, payload: str, actual: str | None) -> CheckResult:
    expected = check_digit(payload)
    return CheckResult(name=name, expected=expected, actual=actual, passed=actual == expected)


def _build_result(
    document_type: str,
    lines: list[str],
    fields: dict[str, str | None],
    checks: list[CheckResult],
) -> MrzParseResult:
    passed = sum(item.passed for item in checks)
    total_characters = max(1, sum(len(line) for line in lines))
    valid_characters = sum(character in MRZ_CHARS for line in lines for character in line)
    checksum_score = passed / max(1, len(checks))
    format_score = (0.55 * checksum_score) + (0.45 * valid_characters / total_characters)
    return MrzParseResult(
        document_type=document_type,
        lines=lines,
        fields=fields,
        checks=checks,
        checksum_pass=passed == len(checks),
        format_score=round(format_score, 4),
    )


def _parse_td3(lines: list[str]) -> MrzParseResult:
    line_one = lines[0].ljust(44, "<")[:44]
    line_two = lines[1].ljust(44, "<")[:44]
    surname, given_names = _split_names(line_one[5:44])
    checks = [
        _check("document_number", line_two[0:9], line_two[9:10]),
        _check("birth_date", line_two[13:19], line_two[19:20]),
        _check("expiry_date", line_two[21:27], line_two[27:28]),
        _check(
            "composite",
            line_two[0:10] + line_two[13:20] + line_two[21:43],
            line_two[43:44],
        ),
    ]
    fields = {
        "document_code": line_one[0:2],
        "issuing_country": line_one[2:5],
        "surname": surname,
        "given_names": given_names,
        "document_number": _clean(line_two[0:9]),
        "nationality": line_two[10:13],
        "birth_date": _compact_date(line_two[13:19]),
        "sex": _clean(line_two[20:21]),
        "expiry_date": _compact_date(line_two[21:27]),
        "optional_data": _clean(line_two[28:43]),
    }
    return _build_result("TD3", [line_one, line_two], fields, checks)


def _parse_mrv(lines: list[str], *, width: int, document_type: str) -> MrzParseResult:
    line_one = lines[0].ljust(width, "<")[:width]
    line_two = lines[1].ljust(width, "<")[:width]
    surname, given_names = _split_names(line_one[5:width])
    checks = [
        _check("document_number", line_two[0:9], line_two[9:10]),
        _check("birth_date", line_two[13:19], line_two[19:20]),
        _check("expiry_date", line_two[21:27], line_two[27:28]),
    ]
    fields = {
        "document_code": line_one[0:2],
        "issuing_country": line_one[2:5],
        "surname": surname,
        "given_names": given_names,
        "document_number": _clean(line_two[0:9]),
        "nationality": line_two[10:13],
        "birth_date": _compact_date(line_two[13:19]),
        "sex": _clean(line_two[20:21]),
        "expiry_date": _compact_date(line_two[21:27]),
        "optional_data": _clean(line_two[28:width]),
    }
    return _build_result(document_type, [line_one, line_two], fields, checks)


def _parse_td1(lines: list[str]) -> MrzParseResult:
    line_one = lines[0].ljust(30, "<")[:30]
    line_two = lines[1].ljust(30, "<")[:30]
    line_three = lines[2].ljust(30, "<")[:30]
    surname, given_names = _split_names(line_three)
    checks = [
        _check("document_number", line_one[5:14], line_one[14:15]),
        _check("birth_date", line_two[0:6], line_two[6:7]),
        _check("expiry_date", line_two[8:14], line_two[14:15]),
        _check(
            "composite",
            line_one[5:30] + line_two[0:7] + line_two[8:15] + line_two[18:29],
            line_two[29:30],
        ),
    ]
    fields = {
        "document_code": line_one[0:2],
        "issuing_country": line_one[2:5],
        "surname": surname,
        "given_names": given_names,
        "document_number": _clean(line_one[5:14]),
        "nationality": line_two[15:18],
        "birth_date": _compact_date(line_two[0:6]),
        "sex": _clean(line_two[7:8]),
        "expiry_date": _compact_date(line_two[8:14]),
        "optional_data": _clean(line_one[15:30] + line_two[18:29]),
    }
    return _build_result("TD1", [line_one, line_two, line_three], fields, checks)


def parse_mrz(raw_lines: list[str]) -> MrzParseResult | None:
    lines = _normalize_lines(raw_lines)
    if len(lines) >= 3:
        td1_lines = [line for line in lines if 27 <= len(line) <= 33]
        if len(td1_lines) >= 3:
            return _parse_td1(td1_lines[:3])

    two_lines = [line for line in lines if len(line) >= 30]
    if len(two_lines) < 2:
        return None
    selected = two_lines[:2]
    if selected[0].startswith("V"):
        return _parse_mrv(
            selected,
            width=36 if max(map(len, selected)) <= 38 else 44,
            document_type="MRV-B" if max(map(len, selected)) <= 38 else "MRV-A",
        )
    if max(map(len, selected)) <= 38:
        return _parse_mrv(selected, width=36, document_type="MRV-B")
    return _parse_td3(selected)
