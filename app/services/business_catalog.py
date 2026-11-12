from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.core.config import Settings

MappingStatus = Literal["NULL", "EXACT", "ALIAS", "PATTERN", "UNMAPPED"]


def normalize_catalog_alias(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^0-9a-zA-Z]+", " ", without_marks).casefold().split())


@dataclass(frozen=True)
class CatalogMappingDecision:
    field: str
    applicant_index: int | None
    raw_value: str | None
    canonical_value: str | None
    status: MappingStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "applicant_index": self.applicant_index,
            "raw_value": self.raw_value,
            "canonical_value": self.canonical_value,
            "status": self.status,
        }


@dataclass(frozen=True)
class _Catalog:
    aliases: dict[str, str]
    canonical_values: frozenset[str]
    on_unmapped: Literal["null", "preserve"]
    canonical_pattern: re.Pattern[str] | None
    compact_uppercase: bool

    def map(self, value: Any) -> tuple[str | None, MappingStatus]:
        if value is None:
            return None, "NULL"
        cleaned = str(value).strip()
        if not cleaned:
            return None, "NULL"
        normalized = normalize_catalog_alias(cleaned)
        canonical = self.aliases.get(normalized)
        if canonical is not None:
            status: MappingStatus = "EXACT" if cleaned == canonical else "ALIAS"
            return canonical, status
        candidate = re.sub(r"[\s._/-]+", "", cleaned).upper()
        if (
            self.compact_uppercase
            and self.canonical_pattern is not None
            and self.canonical_pattern.fullmatch(candidate)
        ):
            return candidate, "PATTERN"
        if self.on_unmapped == "preserve":
            return cleaned, "UNMAPPED"
        return None, "UNMAPPED"


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
                for alias in [canonical, *raw_aliases]:
                    normalized = normalize_catalog_alias(str(alias))
                    previous = aliases.get(normalized)
                    if previous is not None and previous != canonical:
                        raise ValueError(
                            f"Catalog {field!r} alias {alias!r} maps to both "
                            f"{previous!r} and {canonical!r}."
                        )
                    aliases[normalized] = canonical
            catalogs[str(field)] = _Catalog(
                aliases=aliases,
                canonical_values=frozenset(canonical_values),
                on_unmapped=on_unmapped,  # type: ignore[arg-type]
                canonical_pattern=pattern,
                compact_uppercase=bool(raw_catalog.get("compact_uppercase", False)),
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
        canonical, status = catalog.map(raw)
        payload[alias] = canonical
        decisions.append(
            CatalogMappingDecision(
                field=field,
                applicant_index=applicant_index,
                raw_value=None if raw is None else str(raw),
                canonical_value=canonical,
                status=status,
            )
        )

