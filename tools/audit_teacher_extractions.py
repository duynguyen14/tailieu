from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select, true
from sqlalchemy.orm import joinedload

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import TeacherExtractionJob, TeacherExtractionReview
from app.schemas.teacher_extraction import CompactDocumentExtraction


DEFAULT_OUTPUT_DIR = Path("output/teacher_extraction_audit")

AUDIT_SYSTEM_PROMPT = """You are a strict document extraction quality auditor.
Compare SOURCE_TEXT against CANDIDATE_JSON. The source is OCR text and may contain
minor OCR errors or bilingual repetitions. Audit factual support, field assignment,
applicant grouping and completeness. Do not treat harmless punctuation, spacing,
capitalization, equivalent translations or unambiguous date normalization as errors.

An applicant is a person for whom the document requests a visa, visa renewal,
entry/residence permission, or whose personal/passport details are presented as a
subject of that request. Do not add officials, signers, senders, recipients, hosts,
relatives or referenced people merely because their names occur. A related person
may be referenced by related_to_person_id only if that person is legitimately an
applicant in the same extraction. Never invent reciprocal relationships.

Hard rules:
- current_value MUST be copied exactly from CANDIDATE_JSON at path. If the candidate
  value is null, current_value must be null. Never put a source value in current_value.
- Do not report an issue when the candidate already contains the same value.
- ISO YYYY-MM-DD normalization is correct and must not be reported as a formatting error.
- Never report capitalization, spacing, punctuation or name casing differences.
- Report MISSING_VALUE only when the source explicitly and unambiguously states the
  value and the candidate field is null/empty. Do not infer nationality from an embassy,
  sender country or a person's title. Do not infer sequence_number from list position.
- Do not duplicate a value across occupation, job_title and organization. A combined
  explicit job title already stored in job_title is acceptable.
- Visa collection location and port of entry are different. Do not use an airport/border
  merely mentioned for visa-on-arrival as proposed_place_of_visa_collection.

Use these issue codes only:
UNSUPPORTED_VALUE, MISSING_VALUE, WRONG_FIELD, WRONG_APPLICANT,
MISSING_APPLICANT, EXTRA_APPLICANT, WRONG_RELATION, DATE_CONFUSION, OTHER.

Return JSON only:
{
  "verdict": "PASS" or "NEEDS_REVIEW",
  "confidence": number from 0 to 1,
  "summary": "short Vietnamese summary",
  "issues": [
    {
      "severity": "ERROR" or "WARNING",
      "code": "one allowed code",
      "path": "JSON path",
      "current_value": any JSON value or null,
      "suggested_value": any JSON value or null,
      "evidence": "short exact source evidence",
      "message": "Vietnamese explanation"
    }
  ]
}
PASS requires an empty issues list. Do not rewrite the full candidate JSON."""


class AuditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["ERROR", "WARNING"]
    code: Literal[
        "UNSUPPORTED_VALUE",
        "MISSING_VALUE",
        "WRONG_FIELD",
        "WRONG_APPLICANT",
        "MISSING_APPLICANT",
        "EXTRA_APPLICANT",
        "WRONG_RELATION",
        "DATE_CONFUSION",
        "OTHER",
    ]
    path: str
    current_value: Any = None
    suggested_value: Any = None
    evidence: str = ""
    message: str


class AuditDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "NEEDS_REVIEW"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    issues: list[AuditIssue] = Field(default_factory=list)


@dataclass(frozen=True)
class AuditCandidate:
    review_id: str
    job_id: str
    case_id: str
    document_id: str
    original_filename: str
    review_status: str
    input_text: str
    candidate_json: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only semantic audit of reconstructed text versus teacher JSON."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--status", choices=("ALL", "PENDING", "APPROVED"), default="ALL")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1400)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Auditor response does not contain a JSON object.") from None
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Auditor response root is not an object.")
    return parsed


def load_candidates(args: argparse.Namespace) -> list[AuditCandidate]:
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("Teacher audit must run with DB_NAME=anotation_2.")
    database = Database(settings)
    try:
        statement = (
            select(TeacherExtractionReview)
            .join(TeacherExtractionJob, TeacherExtractionJob.id == TeacherExtractionReview.job_id)
            .options(
                joinedload(TeacherExtractionReview.job).joinedload(
                    TeacherExtractionJob.document
                )
            )
            .where(
                TeacherExtractionReview.is_active == true(),
                TeacherExtractionJob.is_active == true(),
                TeacherExtractionJob.status == "COMPLETED",
            )
            .order_by(
                TeacherExtractionReview.created_at.asc(),
                TeacherExtractionReview.id.asc(),
            )
        )
        if args.status != "ALL":
            statement = statement.where(
                TeacherExtractionReview.review_status == args.status
            )
        if args.limit > 0:
            statement = statement.limit(args.limit)
        with database.session() as session:
            reviews = list(session.scalars(statement).unique().all())
            candidates: list[AuditCandidate] = []
            for review in reviews:
                try:
                    raw = json.loads(review.reviewed_output_json)
                    validated = CompactDocumentExtraction.model_validate(raw)
                except (json.JSONDecodeError, ValidationError) as exc:
                    raw = {
                        "_stored_json_validation_error": str(exc),
                        "_raw": review.reviewed_output_json,
                    }
                else:
                    raw = validated.model_dump(mode="json")
                candidates.append(
                    AuditCandidate(
                        review_id=review.id,
                        job_id=review.job_id,
                        case_id=review.job.case_id,
                        document_id=review.job.document_id,
                        original_filename=review.job.document.original_filename,
                        review_status=review.review_status,
                        input_text=review.job.input_text,
                        candidate_json=raw,
                    )
                )
            return candidates
    finally:
        database.close()


def audit_messages(candidate: AuditCandidate) -> list[dict[str, str]]:
    user_content = (
        f"CASE_ID: {candidate.case_id}\n"
        f"DOCUMENT_ID: {candidate.document_id}\n\n"
        "SOURCE_TEXT:\n"
        f"{candidate.input_text}\n\n"
        "CANDIDATE_JSON:\n"
        f"{json.dumps(candidate.candidate_json, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


_PATH_PART_RE = re.compile(r"([^.[\]]+)|\[(\d+)\]")


def value_at_path(root: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = root
    for match in _PATH_PART_RE.finditer(path):
        key, index = match.groups()
        if key is not None:
            if not isinstance(current, dict) or key not in current:
                return False, None
            current = current[key]
        else:
            position = int(index)
            if not isinstance(current, list) or position >= len(current):
                return False, None
            current = current[position]
    return True, current


def comparable(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"[^0-9a-z]+", "", str(value).casefold())


def sanitize_decision(
    decision: AuditDecision,
    candidate_json: dict[str, Any],
) -> AuditDecision:
    cleaned: list[AuditIssue] = []
    for issue in decision.issues:
        exists, actual = value_at_path(candidate_json, issue.path)
        if not exists:
            continue
        issue.current_value = actual
        actual_key = comparable(actual)
        suggested_key = comparable(issue.suggested_value)
        if issue.code == "MISSING_VALUE" and actual_key:
            continue
        if issue.code in {"UNSUPPORTED_VALUE", "WRONG_FIELD", "DATE_CONFUSION"}:
            if actual_key and suggested_key and actual_key == suggested_key:
                continue
        if issue.code in {"UNSUPPORTED_VALUE", "WRONG_FIELD"} and not actual_key:
            issue.code = "MISSING_VALUE"
        cleaned.append(issue)
    decision.issues = cleaned
    decision.verdict = "NEEDS_REVIEW" if cleaned else "PASS"
    return decision


async def audit_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    candidate: AuditCandidate,
    *,
    endpoint: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": audit_messages(candidate),
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    started = asyncio.get_running_loop().time()
    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            async with semaphore:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            decision = AuditDecision.model_validate(extract_json_object(content))
            decision = sanitize_decision(decision, candidate.candidate_json)
            latency_ms = round((asyncio.get_running_loop().time() - started) * 1000)
            return {
                "review_id": candidate.review_id,
                "job_id": candidate.job_id,
                "case_id": candidate.case_id,
                "document_id": candidate.document_id,
                "original_filename": candidate.original_filename,
                "source_review_status": candidate.review_status,
                "audit_status": "COMPLETED",
                "latency_ms": latency_ms,
                "verdict": decision.verdict,
                "confidence": decision.confidence,
                "summary": decision.summary,
                "issues": [issue.model_dump(mode="json") for issue in decision.issues],
                "usage": body.get("usage"),
            }
        except (httpx.HTTPError, KeyError, ValueError, ValidationError) as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(1.0)
    latency_ms = round((asyncio.get_running_loop().time() - started) * 1000)
    return {
        "review_id": candidate.review_id,
        "job_id": candidate.job_id,
        "case_id": candidate.case_id,
        "document_id": candidate.document_id,
        "original_filename": candidate.original_filename,
        "source_review_status": candidate.review_status,
        "audit_status": "FAILED",
        "latency_ms": latency_ms,
        "error": str(last_error),
        "issues": [],
    }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            existing[record["review_id"]] = record
    return existing


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    completed = [item for item in results if item["audit_status"] == "COMPLETED"]
    failed = [item for item in results if item["audit_status"] == "FAILED"]
    verdict_counts = Counter(item.get("verdict", "FAILED") for item in results)
    issue_codes: Counter[str] = Counter()
    issue_paths: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    for item in completed:
        for issue in item["issues"]:
            issue_codes[issue["code"]] += 1
            issue_paths[issue["path"]] += 1
            severity_counts[issue["severity"]] += 1
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "total": len(results),
        "completed": len(completed),
        "failed": len(failed),
        "verdict_counts": dict(verdict_counts),
        "severity_counts": dict(severity_counts),
        "issue_codes": dict(issue_codes.most_common()),
        "issue_paths": dict(issue_paths.most_common()),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Teacher extraction audit",
        "",
        f"- Total: {len(results)}",
        f"- Completed: {len(completed)}",
        f"- Failed: {len(failed)}",
        f"- PASS: {verdict_counts.get('PASS', 0)}",
        f"- NEEDS_REVIEW: {verdict_counts.get('NEEDS_REVIEW', 0)}",
        f"- ERROR issues: {severity_counts.get('ERROR', 0)}",
        f"- WARNING issues: {severity_counts.get('WARNING', 0)}",
        "",
        "## Issue types",
        "",
    ]
    for code, count in issue_codes.most_common():
        lines.append(f"- {code}: {count}")
    lines.extend(["", "## Cases requiring review", ""])
    for item in completed:
        if item.get("verdict") != "NEEDS_REVIEW":
            continue
        lines.extend(
            [
                f"### {item['original_filename']} — case `{item['case_id']}`",
                "",
                item.get("summary", ""),
                "",
            ]
        )
        for issue in item["issues"]:
            lines.append(
                f"- **{issue['severity']} {issue['code']}** `{issue['path']}`: "
                f"{issue['message']} Evidence: {issue.get('evidence') or 'N/A'}"
            )
        lines.append("")
    (output_dir / "audit_report.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


async def run_async(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater.")
    if args.concurrency < 1 or args.concurrency > 16:
        raise ValueError("--concurrency must be between 1 and 16.")
    settings = get_settings()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "audit_results.jsonl"
    existing = load_existing(results_path) if args.resume else {}
    candidates = load_candidates(args)
    pending = [
        item
        for item in candidates
        if existing.get(item.review_id, {}).get("audit_status") != "COMPLETED"
    ]
    print(
        f"Teacher audit: discovered={len(candidates)}, existing={len(existing)}, "
        f"pending={len(pending)}, concurrency={args.concurrency}"
    )
    api_key = (
        settings.teacher_api_key.get_secret_value().strip()
        if settings.teacher_api_key
        else None
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(settings.teacher_timeout_seconds)
    new_results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            asyncio.create_task(
                audit_one(
                    client,
                    semaphore,
                    candidate,
                    endpoint=settings.teacher_chat_completions_url,
                    model=settings.teacher_model,
                    api_key=api_key,
                    max_tokens=args.max_tokens,
                )
            )
            for candidate in pending
        ]
        mode = "a" if args.resume and results_path.exists() else "w"
        with results_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for position, task in enumerate(asyncio.as_completed(tasks), start=1):
                result = await task
                new_results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{position}/{len(tasks)}] {result['audit_status']} "
                    f"file={result['original_filename']!r} "
                    f"verdict={result.get('verdict', 'N/A')} "
                    f"issues={len(result.get('issues', []))} "
                    f"time={result['latency_ms']}ms"
                )
    combined_by_review = dict(existing)
    combined_by_review.update({item["review_id"]: item for item in new_results})
    combined = list(combined_by_review.values())
    write_summary(combined, output_dir)
    print(f"Audit results: {results_path}")
    print(f"Audit report: {output_dir / 'audit_report.md'}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(run_async(parse_args()))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Teacher audit could not start: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
