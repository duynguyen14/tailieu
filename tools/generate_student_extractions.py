from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import select, true, update

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import (
    TeacherExtractionAttempt,
    TeacherExtractionJob,
    TeacherExtractionReview,
    utc_now,
)
from app.schemas.student_extraction import (
    STUDENT_APPLICANT_KEYS,
    STUDENT_NOTE_KEYS,
    STUDENT_SCHEMA_VERSION,
    build_student_user_prompt,
    student_target_to_compact_document,
)
from app.schemas.teacher_extraction import TEACHER_SCHEMA_VERSION
from tools.evaluate_student_extractor import (
    generation_stop_token_ids,
    load_runtime,
    strip_markdown_fence,
)
from tools.generate_teacher_extractions import (
    ReconstructionCandidate,
    canonical_json,
    select_candidates,
)

DEFAULT_MODEL_DIR = Path("models/student_extractor/gemma3_270m_shortkey_v6/merged_model")
PROMPT_VERSION = f"student-local-{STUDENT_SCHEMA_VERSION}"
LOCAL_ENDPOINT = "local://transformers"


@dataclass
class StudentGenerationSummary:
    discovered: int = 0
    eligible: int = 0
    created_jobs: int = 0
    completed: int = 0
    failed: int = 0
    skipped_approved: int = 0
    skipped_existing: int = 0


@dataclass(frozen=True)
class LocalGenerationResult:
    validation_status: str
    request_json: str
    raw_response_json: str | None
    assistant_content: str | None
    parsed_output_json: str | None
    validation_errors_json: str | None
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    error_message: str | None

    @property
    def valid(self) -> bool:
        return self.validation_status == "VALID" and self.parsed_output_json is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the merged local Gemma student model on eligible reconstructed "
            "documents and enqueue its full business JSON for human review."
        )
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--case-id")
    selector.add_argument("--document-id")
    selector.add_argument("--filename")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-device-id", default="0")
    parser.add_argument(
        "--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum remaining cases to process; 0 means all.",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Also use GENERATED reconstructed text; default requires REVIEWED text.",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry an existing FAILED job for this exact model/input snapshot.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run inference and write jobs/reviews. Omit for a read-only preview.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_student_target(value: Any) -> dict[str, Any]:
    """Fill omitted known short keys with null while rejecting malformed rows.

    Gemma occasionally omits a key whose value is null. The target schema is
    positional-by-key, so adding only the known absent keys is deterministic and
    does not invent a business value. Unknown model keys are deliberately dropped.
    """

    if not isinstance(value, dict):
        raise ValueError("Student response root must be a JSON object.")
    note = value.get("n")
    applicants = value.get("a")
    if not isinstance(note, dict):
        raise ValueError("Student response n must be a JSON object.")
    if not isinstance(applicants, list):
        raise ValueError("Student response a must be a JSON array.")
    normalized_applicants: list[dict[str, Any]] = []
    for index, row in enumerate(applicants):
        if not isinstance(row, dict):
            raise ValueError(f"Student response a[{index}] must be a JSON object.")
        normalized_applicants.append(
            {alias: row.get(alias) for alias in STUDENT_APPLICANT_KEYS.values()}
        )
    return {
        "n": {alias: note.get(alias) for alias in STUDENT_NOTE_KEYS.values()},
        "a": normalized_applicants,
    }


def model_fingerprint(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "generation_config.json", "merge_metadata.json"):
        path = model_dir / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        raise RuntimeError(f"Merged model weights were not found: {weights}")
    stat = weights.stat()
    digest.update(f"model.safetensors:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(STUDENT_SCHEMA_VERSION.encode("utf-8"))
    return digest.hexdigest()


def student_input_hash(
    candidate: ReconstructionCandidate, *, model_id: str, fingerprint: str
) -> str:
    payload = {
        "reconstruction_id": candidate.reconstruction_id,
        "source_fingerprint": candidate.source_fingerprint,
        "reconstruction_version": candidate.reconstruction_version,
        "schema_version": STUDENT_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": model_id,
        "model_fingerprint": fingerprint,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def approved_case_ids(database: Database) -> set[str]:
    statement = (
        select(TeacherExtractionJob.case_id)
        .join(
            TeacherExtractionReview,
            TeacherExtractionReview.job_id == TeacherExtractionJob.id,
        )
        .where(
            TeacherExtractionReview.is_active == true(),
            TeacherExtractionReview.review_status == "APPROVED",
        )
        .distinct()
    )
    with database.session() as session:
        return set(session.scalars(statement).all())


def select_remaining_candidates(
    database: Database, args: argparse.Namespace
) -> tuple[list[ReconstructionCandidate], int, int]:
    # Apply the limit only after excluding approved human snapshots. Otherwise
    # approved rows at the head of the query would consume the requested batch.
    selector_args = SimpleNamespace(
        case_id=args.case_id,
        document_id=args.document_id,
        filename=args.filename,
        include_generated=args.include_generated,
        limit=0,
    )
    discovered = select_candidates(database, selector_args)
    approved = approved_case_ids(database)
    remaining = [item for item in discovered if item.case_id not in approved]
    if args.limit > 0:
        remaining = remaining[: args.limit]
    return remaining, len(discovered), sum(item.case_id in approved for item in discovered)


def create_or_get_job(
    database: Database,
    candidate: ReconstructionCandidate,
    *,
    input_hash: str,
    model_id: str,
) -> tuple[TeacherExtractionJob, bool]:
    with database.session() as session:
        approved = session.scalar(
            select(TeacherExtractionReview.id)
            .join(
                TeacherExtractionJob,
                TeacherExtractionJob.id == TeacherExtractionReview.job_id,
            )
            .where(
                TeacherExtractionJob.case_id == candidate.case_id,
                TeacherExtractionReview.is_active == true(),
                TeacherExtractionReview.review_status == "APPROVED",
            )
        )
        if approved is not None:
            raise RuntimeError(
                f"Case {candidate.case_id} already has an active approved human review."
            )
        existing = session.scalar(
            select(TeacherExtractionJob).where(TeacherExtractionJob.input_hash == input_hash)
        )
        if existing is not None:
            return existing, False

        # Only pending/failed machine work is superseded. Approved human reviews
        # were checked above and are never deactivated by this tool.
        session.execute(
            update(TeacherExtractionJob)
            .where(
                TeacherExtractionJob.case_id == candidate.case_id,
                TeacherExtractionJob.is_active == true(),
            )
            .values(is_active=False, updated_at=utc_now())
        )
        job = TeacherExtractionJob(
            id=str(uuid4()),
            reconstruction_id=candidate.reconstruction_id,
            case_id=candidate.case_id,
            document_id=candidate.document_id,
            input_hash=input_hash,
            input_text=candidate.reconstructed_text,
            source_structure_json=candidate.structure_json,
            reconstruction_version=candidate.reconstruction_version,
            schema_version=TEACHER_SCHEMA_VERSION,
            prompt_version=PROMPT_VERSION,
            teacher_model=model_id,
            teacher_endpoint=LOCAL_ENDPOINT,
            status="PENDING",
            is_active=True,
            attempt_count=0,
        )
        session.add(job)
        session.flush()
        return job, True


def begin_attempt(database: Database, job_id: str) -> int:
    with database.session() as session:
        job = session.scalar(
            select(TeacherExtractionJob)
            .where(TeacherExtractionJob.id == job_id)
            .with_for_update()
        )
        if job is None:
            raise RuntimeError(f"Student job disappeared: {job_id}")
        job.attempt_count += 1
        job.status = "RUNNING"
        job.last_error = None
        job.completed_at = None
        job.updated_at = utc_now()
        session.flush()
        return job.attempt_count


def save_result(
    database: Database,
    *,
    job_id: str,
    attempt_number: int,
    result: LocalGenerationResult,
) -> None:
    with database.session() as session:
        job = session.scalar(
            select(TeacherExtractionJob)
            .where(TeacherExtractionJob.id == job_id)
            .with_for_update()
        )
        if job is None:
            raise RuntimeError(f"Student job disappeared: {job_id}")
        attempt = TeacherExtractionAttempt(
            id=str(uuid4()),
            job_id=job.id,
            attempt_number=attempt_number,
            request_json=result.request_json,
            raw_response_json=result.raw_response_json,
            assistant_content=result.assistant_content,
            parsed_output_json=result.parsed_output_json,
            validation_status=result.validation_status,
            validation_errors_json=result.validation_errors_json,
            http_status=None,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=(
                result.prompt_tokens + result.completion_tokens
                if result.prompt_tokens is not None
                and result.completion_tokens is not None
                else None
            ),
            finish_reason="stop" if result.valid else None,
            error_message=result.error_message,
        )
        session.add(attempt)
        session.flush()
        if result.valid:
            previous = session.scalar(
                select(TeacherExtractionReview).where(
                    TeacherExtractionReview.job_id == job.id,
                    TeacherExtractionReview.is_active == true(),
                )
            )
            if previous is not None:
                previous.is_active = False
                previous.updated_at = utc_now()
                session.flush()
            session.add(
                TeacherExtractionReview(
                    id=str(uuid4()),
                    job_id=job.id,
                    source_attempt_id=attempt.id,
                    parent_review_id=previous.id if previous else None,
                    reviewed_output_json=result.parsed_output_json,
                    review_status="PENDING",
                    is_active=True,
                )
            )
            job.status = "COMPLETED"
            job.completed_at = utc_now()
            job.last_error = None
        else:
            job.status = "FAILED"
            job.completed_at = None
            job.last_error = result.error_message or result.validation_status
        job.updated_at = utc_now()


def generate_one(
    candidate: ReconstructionCandidate,
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    runtime_device: str,
    model_id: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> LocalGenerationResult:
    user_text = build_student_user_prompt(candidate.reconstructed_text)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    request_json = canonical_json(
        {
            "model": model_id,
            "schema_version": STUDENT_SCHEMA_VERSION,
            "case_id": candidate.case_id,
            "document_id": candidate.document_id,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": max_new_tokens,
        }
    )
    if prompt_tokens > max_input_tokens:
        error = f"Prompt has {prompt_tokens} tokens; limit is {max_input_tokens}."
        return LocalGenerationResult(
            validation_status="INVALID",
            request_json=request_json,
            raw_response_json=None,
            assistant_content=None,
            parsed_output_json=None,
            validation_errors_json=canonical_json([{"message": error}]),
            latency_ms=0,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            error_message=error,
        )

    encoded = {key: value.to(runtime_device) for key, value in encoded.items()}
    stop_ids = generation_stop_token_ids(tokenizer, model)
    if runtime_device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )
    if runtime_device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = round((time.perf_counter() - started) * 1000)
    output_ids = generated[0, prompt_tokens:]
    completion_tokens = int(output_ids.shape[-1])
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    raw_response_json = canonical_json({"raw_output": raw_output})

    try:
        raw_target = json.loads(strip_markdown_fence(raw_output))
        normalized = normalize_student_target(raw_target)
        full_output = student_target_to_compact_document(
            normalized,
            case_id=candidate.case_id,
            document_id=candidate.document_id,
        )
        parsed_output_json = canonical_json(full_output.model_dump(mode="json"))
        return LocalGenerationResult(
            validation_status="VALID",
            request_json=request_json,
            raw_response_json=raw_response_json,
            assistant_content=raw_output,
            parsed_output_json=parsed_output_json,
            validation_errors_json=None,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_message=None,
        )
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        errors = (
            exc.errors(include_url=False, include_context=False)
            if isinstance(exc, ValidationError)
            else [{"message": str(exc)}]
        )
        return LocalGenerationResult(
            validation_status=("INVALID" if isinstance(exc, ValidationError) else "PARSE_ERROR"),
            request_json=request_json,
            raw_response_json=raw_response_json,
            assistant_content=raw_output,
            parsed_output_json=None,
            validation_errors_json=canonical_json(errors),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_message=str(exc),
        )


def run(args: argparse.Namespace) -> StudentGenerationSummary:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0:
        raise ValueError("Token limits must be greater than zero")
    settings = get_settings()
    if settings.db_name.lower() != "anotation_2":
        raise RuntimeError("Student extraction must run with DB_NAME=anotation_2.")
    model_dir = resolve_path(args.model_dir)
    fingerprint = model_fingerprint(model_dir)
    model_id = f"local-gemma3-270m:{fingerprint[:12]}"
    database = Database(settings)
    summary = StudentGenerationSummary()
    try:
        candidates, discovered, approved_count = select_remaining_candidates(database, args)
        summary.discovered = discovered
        summary.eligible = len(candidates)
        summary.skipped_approved = approved_count
        if not candidates:
            print("No remaining eligible reconstructed cases were found.")
            return summary

        candidate_rows: list[tuple[ReconstructionCandidate, str, TeacherExtractionJob | None]] = []
        for candidate in candidates:
            input_hash = student_input_hash(
                candidate, model_id=model_id, fingerprint=fingerprint
            )
            with database.session() as session:
                existing = session.scalar(
                    select(TeacherExtractionJob).where(
                        TeacherExtractionJob.input_hash == input_hash
                    )
                )
            candidate_rows.append((candidate, input_hash, existing))

        if not args.execute:
            for index, (candidate, _, existing) in enumerate(candidate_rows, start=1):
                status = existing.status if existing is not None else "NEW"
                print(
                    f"[{index}/{len(candidate_rows)}] DRY-RUN "
                    f"file={candidate.original_filename!r} case={candidate.case_id} "
                    f"text_status={candidate.reconstruction_status} job={status}"
                )
            return summary

        runtime_args = SimpleNamespace(
            device=args.device,
            gpu=str(args.gpu_device_id),
            offline=args.offline,
            model_dir=model_dir,
            precision=args.precision,
        )
        torch, tokenizer, model, runtime_device = load_runtime(runtime_args)
        print(
            f"Loaded {model_id} on {runtime_device}; eligible={len(candidate_rows)}; "
            f"approved_skipped={approved_count}"
        )
        for index, (candidate, input_hash, existing) in enumerate(candidate_rows, start=1):
            if existing is not None:
                if existing.status == "FAILED" and args.retry_failed:
                    job = existing
                else:
                    summary.skipped_existing += 1
                    print(
                        f"[{index}/{len(candidate_rows)}] SKIP {existing.status.lower()} "
                        f"file={candidate.original_filename!r} job={existing.id}"
                    )
                    continue
            else:
                job, created = create_or_get_job(
                    database,
                    candidate,
                    input_hash=input_hash,
                    model_id=model_id,
                )
                if created:
                    summary.created_jobs += 1
            attempt_number = begin_attempt(database, job.id)
            print(
                f"[{index}/{len(candidate_rows)}] INFER file={candidate.original_filename!r} "
                f"job={job.id} attempt={attempt_number}"
            )
            result = generate_one(
                candidate,
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                runtime_device=runtime_device,
                model_id=model_id,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )
            save_result(
                database,
                job_id=job.id,
                attempt_number=attempt_number,
                result=result,
            )
            if result.valid:
                summary.completed += 1
                print(
                    f"[{index}/{len(candidate_rows)}] COMPLETED "
                    f"latency={result.latency_ms}ms tokens={result.completion_tokens}"
                )
            else:
                summary.failed += 1
                print(
                    f"[{index}/{len(candidate_rows)}] FAILED "
                    f"status={result.validation_status} error={result.error_message}"
                )
        return summary
    finally:
        database.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    summary = run(parse_args())
    print(
        "Student extraction generation finished: "
        f"discovered={summary.discovered}, eligible={summary.eligible}, "
        f"approved_skipped={summary.skipped_approved}, "
        f"created_jobs={summary.created_jobs}, completed={summary.completed}, "
        f"failed={summary.failed}, existing_skipped={summary.skipped_existing}"
    )
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"Student extraction generation could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error
