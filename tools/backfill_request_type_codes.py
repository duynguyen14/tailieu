from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import select, true
from sqlalchemy.orm import joinedload

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import TeacherExtractionJob, TeacherExtractionReview, utc_now
from app.schemas.teacher_extraction import RequestTypeCode

SYSTEM_REVIEWER = "system:request-type-code-v1"
REVIEW_NOTE = "Added normalized note.request_type_code (request-type-code-v1)."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill normalized request type codes into active approved teacher "
            "JSON while preserving the original free-text request_type."
        )
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create versioned approved review snapshots. Omit for a read-only preview.",
    )
    parser.add_argument("--reviewed-by", default=SYSTEM_REVIEWER)
    return parser.parse_args()


def _fold(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    ).casefold()


_VISA_TERMS = r"(?:visa|thi thuc|visado)"
_NEW_PASSPORT_TERMS = (
    r"(?:new passport|ho chieu moi|doi ho chieu|replacement passport|"
    r"nouveau passeport)"
)
_EXPLICIT_NON_VISA_ID_REQUEST = re.compile(
    r"issuance of temporary residence permits? and diplomatic ids?"
)
_RAW_EXTENSION = re.compile(
    r"(?:gia han|extend|extension|renewal|renew.{0,40}visa|renouvellement|"
    r"prolongement|extension de visado|extension|stay permit|temporary residen|"
    r"residence permit)"
)
_TEXT_EXTENSION = re.compile(
    r"(?:gia han(?: thi thuc| tam tru)?|"
    r"extend(?:ing|ed|s)? .{0,80}(?:visa|residence permit|stay permit)|"
    r"extension of .{0,80}(?:visa|residence|stay permit)|"
    r"renewal of .{0,50}visa|renew the .{0,50}visa|"
    r"renouvellement de visa|prolongement du visa|"
    r"solicitud de extension de visado)",
    re.DOTALL,
)
_TEXT_ISSUANCE = re.compile(
    r"(?:request|de nghi|xin).{0,180}(?:issue|issuance|grant|cap)"
    r".{0,100}(?:visa|thi thuc)|"
    r"(?:cap thi thuc|issue .{0,40}visa|issuing .{0,40}visa|"
    r"grant .{0,40}visa)",
    re.DOTALL,
)


def classify_request_type(
    request_type: str | None,
    reconstructed_text: str,
) -> RequestTypeCode:
    """Map reviewed free text plus its source into the four business categories.

    The reviewed request_type wins over generic wording in the source. This is
    important because diplomatic notes routinely contain a closing phrase such
    as "renew ... assurances", which is not a visa-extension request.
    """

    raw = _fold(request_type)
    source = _fold(reconstructed_text)

    passport_after_visa = re.search(
        _VISA_TERMS + r".{0,240}" + _NEW_PASSPORT_TERMS,
        source,
        re.DOTALL,
    )
    visa_after_passport = re.search(
        _NEW_PASSPORT_TERMS + r".{0,240}" + _VISA_TERMS,
        source,
        re.DOTALL,
    )
    if passport_after_visa or visa_after_passport:
        return "CHUYEN_THI_THUC"

    # A request to issue residence permits/IDs is not a visa issuance or renewal.
    if _EXPLICIT_NON_VISA_ID_REQUEST.search(raw):
        return "KHAC"

    if _RAW_EXTENSION.search(raw):
        return "GIA_HAN"

    # A reviewed free-text visa request is authoritative. Do not let generic
    # closing language in the full document turn it into an extension.
    if re.search(_VISA_TERMS, raw):
        return "CAP_MOI"

    # Null/vague historical labels need a constrained source-text fallback.
    if _TEXT_EXTENSION.search(source):
        return "GIA_HAN"
    if _TEXT_ISSUANCE.search(source):
        return "CAP_MOI"
    return "KHAC"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def append_review_note(previous: str | None) -> str:
    if not previous:
        return REVIEW_NOTE
    if REVIEW_NOTE in previous:
        return previous
    return f"{previous.rstrip()}\n{REVIEW_NOTE}"


@dataclass
class Summary:
    discovered: int = 0
    eligible: int = 0
    created: int = 0
    unchanged: int = 0
    invalid: int = 0


def run(args: argparse.Namespace) -> tuple[Summary, Counter[str]]:
    settings = get_settings()
    database = Database(settings)
    summary = Summary()
    counts: Counter[str] = Counter()
    try:
        with database.session() as session:
            statement = (
                select(TeacherExtractionReview)
                .join(
                    TeacherExtractionJob,
                    TeacherExtractionJob.id == TeacherExtractionReview.job_id,
                )
                .options(
                    joinedload(TeacherExtractionReview.job).joinedload(
                        TeacherExtractionJob.document
                    )
                )
                .where(
                    TeacherExtractionReview.is_active == true(),
                    TeacherExtractionReview.review_status == "APPROVED",
                    TeacherExtractionJob.is_active == true(),
                )
                .order_by(
                    TeacherExtractionReview.reviewed_at,
                    TeacherExtractionReview.id,
                )
            )
            if args.limit > 0:
                statement = statement.limit(args.limit)
            reviews = list(session.scalars(statement).unique())
            summary.discovered = len(reviews)

            for index, review in enumerate(reviews, start=1):
                try:
                    raw = json.loads(review.reviewed_output_json)
                    if not isinstance(raw, dict):
                        raise ValueError("reviewed_output_json root is not an object")
                    note = raw.get("note")
                    if not isinstance(note, dict):
                        raise ValueError("reviewed_output_json.note is not an object")
                    code = classify_request_type(
                        note.get("request_type"),
                        review.job.input_text,
                    )
                    counts[code] += 1
                    summary.eligible += 1
                    if note.get("request_type_code") == code:
                        summary.unchanged += 1
                        action = "UNCHANGED"
                    else:
                        note["request_type_code"] = code
                        # Historical approved reviews can contain legacy fields that
                        # no longer satisfy today's full Pydantic schema. This
                        # migration is deliberately additive: preserve every
                        # existing value verbatim and add only request_type_code.
                        persisted_json = canonical_json(raw)
                        if args.execute:
                            now = utc_now()
                            review.is_active = False
                            review.updated_at = now
                            session.add(
                                TeacherExtractionReview(
                                    id=str(uuid4()),
                                    job_id=review.job_id,
                                    source_attempt_id=review.source_attempt_id,
                                    parent_review_id=review.id,
                                    reviewed_output_json=persisted_json,
                                    review_status="APPROVED",
                                    is_active=True,
                                    reviewed_by=args.reviewed_by,
                                    review_note=append_review_note(review.review_note),
                                    reviewed_at=now,
                                    created_at=now,
                                    updated_at=now,
                                )
                            )
                            summary.created += 1
                            action = "CREATED"
                        else:
                            action = "DRY"
                    print(
                        f"[{index}/{len(reviews)}] {action} "
                        f"file={review.job.document.original_filename!r} code={code}"
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    summary.invalid += 1
                    print(
                        f"[{index}/{len(reviews)}] INVALID "
                        f"file={review.job.document.original_filename!r}: {exc}"
                    )
            if not args.execute:
                session.rollback()
    finally:
        database.close()
    return summary, counts


def main() -> None:
    args = parse_args()
    try:
        summary, counts = run(args)
    except Exception as exc:
        raise SystemExit(f"Request type backfill could not start: {exc}") from exc
    print("Request type code backfill:", "EXECUTED" if args.execute else "DRY RUN")
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    print("Counts:")
    for code in ("CAP_MOI", "GIA_HAN", "CHUYEN_THI_THUC", "KHAC"):
        print(f"  {code}: {counts[code]}")
    if not args.execute:
        print("No database rows were changed. Add --execute to create review snapshots.")


if __name__ == "__main__":
    main()
