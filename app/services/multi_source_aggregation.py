from __future__ import annotations

import re
from typing import Any

from app.schemas.production_extraction import (
    MultiValueBusinessApplicant,
    MultiValueBusinessNote,
    MultiValueBusinessOutput,
    PassportMrzPageResult,
)
from app.services.business_catalog import (
    BusinessCatalogMapper,
    normalize_catalog_alias,
)

APPLICANT_FIELDS = {
    "nm": "full_name",
    "dob": "date_of_birth",
    "sex": "gender",
    "nat": "nationality",
    "pp": "passport_number",
    "pt": "passport_type",
    "pid": "passport_issue_date",
    "pex": "passport_expiration",
    "vt": "visa_type",
    "ne": "number_of_entries",
    "in": "entry_date",
    "out": "exit_date",
}


def _value_list(value: Any) -> list[str | None]:
    if value is None or not str(value).strip():
        return [None]
    return [str(value).strip()]


def _add_value(values: list[str | None], value: str | None) -> None:
    if value is None or not str(value).strip():
        if not values:
            values.append(None)
        return
    cleaned = str(value).strip()
    if cleaned not in values:
        values[:] = [item for item in values if item is not None]
        values.append(cleaned)


def _passport_key(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _identity_key(name: str | None, birth_date: str | None) -> tuple[str, str]:
    return normalize_catalog_alias(name or ""), (birth_date or "").strip()


def _full_name(fields: dict[str, str | None]) -> str | None:
    parts = [fields.get("surname"), fields.get("given_names")]
    value = " ".join(part.strip() for part in parts if part and part.strip())
    return value or None


def _blank_applicant() -> MultiValueBusinessApplicant:
    return MultiValueBusinessApplicant()


def _from_student(applicant: dict[str, Any]) -> MultiValueBusinessApplicant:
    payload = {
        field_name: _value_list(applicant.get(alias))
        for alias, field_name in APPLICANT_FIELDS.items()
    }
    return MultiValueBusinessApplicant.model_validate(payload)


def _find_applicant(
    applicants: list[MultiValueBusinessApplicant],
    *,
    passport_number: str | None,
    full_name: str | None,
    birth_date: str | None,
) -> MultiValueBusinessApplicant | None:
    passport_key = _passport_key(passport_number)
    if passport_key:
        matches = [
            applicant
            for applicant in applicants
            if any(
                _passport_key(value) == passport_key
                for value in applicant.passport_number
                if value
            )
        ]
        if len(matches) == 1:
            return matches[0]

    identity_key = _identity_key(full_name, birth_date)
    if all(identity_key):
        matches = [
            applicant
            for applicant in applicants
            if any(
                _identity_key(name, dob) == identity_key
                for name in applicant.full_name
                if name
                for dob in applicant.date_of_birth
                if dob
            )
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def build_multi_source_output(
    mapped_target: dict[str, Any],
    passport_results: list[PassportMrzPageResult],
    catalog_mapper: BusinessCatalogMapper,
) -> MultiValueBusinessOutput:
    note = mapped_target.get("n")
    note = note if isinstance(note, dict) else {}
    output = MultiValueBusinessOutput(
        note=MultiValueBusinessNote(
            note_number=_value_list(note.get("no")),
            sender_organization=_value_list(note.get("snd")),
            request_type_code=_value_list(note.get("rt")),
        ),
        applicants=[
            _from_student(item)
            for item in mapped_target.get("a", [])
            if isinstance(item, dict)
        ],
    )

    for result in passport_results:
        if not result.found:
            continue
        fields = result.fields
        full_name = _full_name(fields)
        passport_number = fields.get("document_number")
        birth_date = fields.get("birth_date")
        applicant = _find_applicant(
            output.applicants,
            passport_number=passport_number,
            full_name=full_name,
            birth_date=birth_date,
        )
        if applicant is None:
            applicant = _blank_applicant()
            output.applicants.append(applicant)

        gender, _ = catalog_mapper.map_value("gender", fields.get("sex"))
        nationality, _ = catalog_mapper.map_value(
            "nationality", fields.get("nationality")
        )
        document_code = (fields.get("document_code") or "").replace("<", "")[:1]
        passport_type, _ = catalog_mapper.map_value(
            "passport_type", document_code or None
        )

        _add_value(applicant.full_name, full_name)
        _add_value(applicant.date_of_birth, birth_date)
        _add_value(applicant.gender, gender)
        _add_value(applicant.nationality, nationality)
        _add_value(applicant.passport_number, passport_number)
        _add_value(applicant.passport_type, passport_type)
        _add_value(applicant.passport_expiration, fields.get("expiry_date"))

    return output
