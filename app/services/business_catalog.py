from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from app.core.config import Settings

MappingStatus = Literal["NULL", "EXACT", "ALIAS", "PATTERN", "FUZZY", "UNMAPPED"]


def normalize_catalog_alias(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    # Keep letters and digits from every writing system.  The previous ASCII-only
    # regex silently reduced aliases such as 日本, 中国 or Arabic names to an
    # empty string, making native-language catalog entries impossible to match.
    searchable = "".join(
        character if character.isalnum() else " " for character in without_marks
    )
    return " ".join(searchable.casefold().split())


def _fuzzy_similarity(left: str, right: str) -> float:
    direct = SequenceMatcher(None, left, right, autojunk=False).ratio()
    token_sorted_left = " ".join(sorted(left.split()))
    token_sorted_right = " ".join(sorted(right.split()))
    token_sorted = SequenceMatcher(
        None, token_sorted_left, token_sorted_right, autojunk=False
    ).ratio()
    return round(max(direct, token_sorted) * 100.0, 2)


@dataclass(frozen=True)
class CatalogMappingDecision:
    field: str
    applicant_index: int | None
    raw_value: str | None
    canonical_value: str | None
    status: MappingStatus
    matched_alias: str | None = None
    match_score: float | None = None
    second_best_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "applicant_index": self.applicant_index,
            "raw_value": self.raw_value,
            "canonical_value": self.canonical_value,
            "status": self.status,
            "matched_alias": self.matched_alias,
            "match_score": self.match_score,
            "second_best_score": self.second_best_score,
        }


@dataclass(frozen=True)
class _Catalog:
    aliases: dict[str, str]
    alias_labels: dict[str, str]
    canonical_values: frozenset[str]
    on_unmapped: Literal["null", "preserve"]
    canonical_pattern: re.Pattern[str] | None
    compact_uppercase: bool
    pattern_values_only: bool
    fuzzy_enabled: bool
    fuzzy_threshold: float
    fuzzy_min_margin: float
    fuzzy_min_length: int
    fuzzy_candidates: tuple[tuple[str, str, str], ...]
    fuzzy_cache: dict[
        str, tuple[str | None, MappingStatus, str | None, float | None, float | None]
    ] = dataclass_field(default_factory=dict, compare=False, repr=False)

    def match(
        self, value: Any
    ) -> tuple[str | None, MappingStatus, str | None, float | None, float | None]:
        if value is None:
            return None, "NULL", None, None, None
        cleaned = str(value).strip()
        if not cleaned:
            return None, "NULL", None, None, None
        # Canonical codes are case-sensitive here so an ISO code such as NGA
        # (Nigeria) cannot collide with a localized alias such as "Nga"
        # (Vietnamese for Russia) after case folding.
        if cleaned in self.canonical_values:
            return cleaned, "EXACT", cleaned, 100.0, None
        normalized = normalize_catalog_alias(cleaned)
        canonical = self.aliases.get(normalized)
        if canonical is not None:
            status: MappingStatus = "EXACT" if cleaned == canonical else "ALIAS"
            return canonical, status, self.alias_labels.get(normalized), 100.0, None
        candidate = re.sub(r"[\s._/-]+", "", cleaned).upper()
        if (
            self.compact_uppercase
            and self.canonical_pattern is not None
            and self.canonical_pattern.fullmatch(candidate)
            and (not self.pattern_values_only or candidate in self.canonical_values)
        ):
            return candidate, "PATTERN", None, 100.0, None
        if (
            self.fuzzy_enabled
            and len(normalized) >= self.fuzzy_min_length
            and self.fuzzy_candidates
        ):
            cached = self.fuzzy_cache.get(normalized)
            if cached is None:
                cached = self._fuzzy_match(normalized)
                if len(self.fuzzy_cache) >= 1024:
                    self.fuzzy_cache.clear()
                self.fuzzy_cache[normalized] = cached
            if cached[1] == "FUZZY":
                return cached
            fuzzy_score = cached[3]
            fuzzy_second_best = cached[4]
        else:
            fuzzy_score = None
            fuzzy_second_best = None
        if self.on_unmapped == "preserve":
            return cleaned, "UNMAPPED", None, fuzzy_score, fuzzy_second_best
        return None, "UNMAPPED", None, fuzzy_score, fuzzy_second_best

    def map(self, value: Any) -> tuple[str | None, MappingStatus]:
        canonical, status, _, _, _ = self.match(value)
        return canonical, status

    def _fuzzy_match(
        self, normalized: str
    ) -> tuple[str | None, MappingStatus, str | None, float | None, float | None]:
        best_by_canonical: dict[str, tuple[float, str]] = {}
        for alias, canonical, label in self.fuzzy_candidates:
            # SequenceMatcher cannot exceed this length-only upper bound.  Skip
            # impossible candidates before running the more expensive matcher.
            max_possible = 200.0 * min(len(normalized), len(alias)) / (
                len(normalized) + len(alias)
            )
            if max_possible < self.fuzzy_threshold:
                continue
            score = _fuzzy_similarity(normalized, alias)
            previous = best_by_canonical.get(canonical)
            if previous is None or score > previous[0]:
                best_by_canonical[canonical] = (score, label)
        ranked = sorted(
            (
                (score, canonical, label)
                for canonical, (score, label) in best_by_canonical.items()
            ),
            reverse=True,
        )
        if not ranked:
            return None, "UNMAPPED", None, None, None
        best_score, best_canonical, best_label = ranked[0]
        second_best_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if (
            best_score >= self.fuzzy_threshold
            and best_score - second_best_score >= self.fuzzy_min_margin
        ):
            return (
                best_canonical,
                "FUZZY",
                best_label,
                best_score,
                second_best_score,
            )
        return None, "UNMAPPED", None, best_score, second_best_score


class BusinessCatalogMapper:
    """Immutable, versioned business catalog cached entirely in process RAM."""

    APPLICANT_FIELDS = {
        "sex": "gender",
        "nat": "nationality",
        "pt": "passport_type",
        "vt": "visa_type",
        "ne": "number_of_entries",
    }
    NOTE_FIELDS = {"snd": "sender_organization"}

    def __init__(self, *, version: str, catalogs: dict[str, _Catalog]) -> None:
        self.version = version
        self._catalogs = catalogs

    @classmethod
    def from_settings(cls, settings: Settings) -> BusinessCatalogMapper:
        if not settings.production_catalog_mapping_enabled:
            return cls(version="disabled", catalogs={})
        path = Settings.resolve_project_path(settings.production_catalog_path)
        if not path.is_file():
            raise FileNotFoundError(f"Production business catalog was not found: {path}")
        return cls.from_path(path)

    @classmethod
    def from_path(cls, path: Path) -> BusinessCatalogMapper:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Business catalog root must be a JSON object.")
        version = str(payload.get("version") or "").strip()
        if not version:
            raise ValueError("Business catalog version is required.")
        raw_catalogs = payload.get("catalogs")
        if not isinstance(raw_catalogs, dict):
            raise ValueError("Business catalog must contain a catalogs object.")

        catalogs: dict[str, _Catalog] = {}
        for field, raw_catalog in raw_catalogs.items():
            if not isinstance(raw_catalog, dict):
                raise ValueError(f"Catalog {field!r} must be an object.")
            on_unmapped = str(raw_catalog.get("on_unmapped", "null"))
            if on_unmapped not in {"null", "preserve"}:
                raise ValueError(f"Catalog {field!r} has invalid on_unmapped policy.")
            pattern_text = raw_catalog.get("canonical_pattern")
            pattern = re.compile(str(pattern_text)) if pattern_text else None
            aliases: dict[str, str] = {}
            alias_labels: dict[str, str] = {}
            canonical_values: set[str] = set()
            items = raw_catalog.get("items", [])
            if not isinstance(items, list):
                raise ValueError(f"Catalog {field!r} items must be an array.")
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError(f"Catalog {field!r} item must be an object.")
                canonical = str(item.get("canonical") or "").strip()
                if not canonical:
                    raise ValueError(f"Catalog {field!r} item canonical is required.")
                canonical_values.add(canonical)
                raw_aliases = item.get("aliases", [])
                if not isinstance(raw_aliases, list):
                    raise ValueError(f"Catalog {field!r} aliases must be an array.")
                # Do not inject the canonical code into the normalized alias map.
                # It is matched case-sensitively before alias normalization above.
                for alias in raw_aliases:
                    normalized = normalize_catalog_alias(str(alias))
                    previous = aliases.get(normalized)
                    if previous is not None and previous != canonical:
                        raise ValueError(
                            f"Catalog {field!r} alias {alias!r} maps to both "
                            f"{previous!r} and {canonical!r}."
                        )
                    aliases[normalized] = canonical
                    alias_labels.setdefault(normalized, str(alias).strip())
            raw_fuzzy = raw_catalog.get("fuzzy", {})
            if not isinstance(raw_fuzzy, dict):
                raise ValueError(f"Catalog {field!r} fuzzy configuration must be an object.")
            fuzzy_enabled = bool(raw_fuzzy.get("enabled", False))
            fuzzy_threshold = float(raw_fuzzy.get("threshold", 90.0))
            fuzzy_min_margin = float(raw_fuzzy.get("min_margin", 8.0))
            fuzzy_min_length = int(raw_fuzzy.get("min_length", 4))
            if not 0.0 <= fuzzy_threshold <= 100.0:
                raise ValueError(f"Catalog {field!r} fuzzy threshold must be 0..100.")
            if not 0.0 <= fuzzy_min_margin <= 100.0:
                raise ValueError(f"Catalog {field!r} fuzzy min_margin must be 0..100.")
            if fuzzy_min_length < 1:
                raise ValueError(f"Catalog {field!r} fuzzy min_length must be positive.")
            fuzzy_candidates = tuple(
                (alias, canonical, alias_labels[alias])
                for alias, canonical in aliases.items()
                if len(alias) >= fuzzy_min_length
            )
            catalogs[str(field)] = _Catalog(
                aliases=aliases,
                alias_labels=alias_labels,
                canonical_values=frozenset(canonical_values),
                on_unmapped=on_unmapped,  # type: ignore[arg-type]
                canonical_pattern=pattern,
                compact_uppercase=bool(raw_catalog.get("compact_uppercase", False)),
                pattern_values_only=bool(raw_catalog.get("pattern_values_only", False)),
                fuzzy_enabled=fuzzy_enabled,
                fuzzy_threshold=fuzzy_threshold,
                fuzzy_min_margin=fuzzy_min_margin,
                fuzzy_min_length=fuzzy_min_length,
                fuzzy_candidates=fuzzy_candidates,
            )
        return cls(version=version, catalogs=catalogs)

    def map_student_target(
        self, target: dict[str, Any]
    ) -> tuple[dict[str, Any], list[CatalogMappingDecision]]:
        output = deepcopy(target)
        if not self._catalogs:
            return output, []
        decisions: list[CatalogMappingDecision] = []
        note = output.get("n")
        if isinstance(note, dict):
            for alias, field in self.NOTE_FIELDS.items():
                self._map_value(note, alias, field, None, decisions)
        applicants = output.get("a")
        if isinstance(applicants, list):
            for index, applicant in enumerate(applicants, start=1):
                if not isinstance(applicant, dict):
                    continue
                for alias, field in self.APPLICANT_FIELDS.items():
                    self._map_value(applicant, alias, field, index, decisions)
        return output, decisions

    def map_value(
        self, field: str, value: Any
    ) -> tuple[str | None, MappingStatus]:
        """Map one specialist value without exposing raw catalog internals."""
        catalog = self._catalogs.get(field)
        if catalog is None:
            if value is None or not str(value).strip():
                return None, "NULL"
            return str(value).strip(), "UNMAPPED"
        return catalog.map(value)

    def _map_value(
        self,
        payload: dict[str, Any],
        alias: str,
        field: str,
        applicant_index: int | None,
        decisions: list[CatalogMappingDecision],
    ) -> None:
        catalog = self._catalogs.get(field)
        if catalog is None:
            return
        raw = payload.get(alias)
        canonical, status, matched_alias, score, second_best_score = catalog.match(raw)
        payload[alias] = canonical
        decisions.append(
            CatalogMappingDecision(
                field=field,
                applicant_index=applicant_index,
                raw_value=None if raw is None else str(raw),
                canonical_value=canonical,
                status=status,
                matched_alias=matched_alias,
                match_score=score,
                second_best_score=second_best_score,
            )
        )

