from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.schemas.student_extraction import student_target_to_compact_document
from app.services.student_model_runtime import (
    apply_student_chat_template,
    parse_student_json_output,
)
from tools.evaluate_student_extractor import (
    PredictionResult,
    build_metrics,
    flatten_fields,
    safe_ratio,
    write_results,
)
from tools.train_student_extractor import DatasetRecord, read_training_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI-compatible student extractor API on a JSONL split."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8004/v1")
    parser.add_argument("--api-model", default="student-extractor")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Student API returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Student API request failed: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Student API response must be a JSON object.")
    return data


def evaluate_record(
    record: DatasetRecord,
    *,
    tokenizer: Any,
    api_url: str,
    api_model: str,
    max_new_tokens: int,
    timeout: float,
) -> PredictionResult:
    expected = json.loads(record.assistant_text)
    prompt = apply_student_chat_template(
        tokenizer,
        [{"role": "user", "content": record.user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens_local = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    started = time.perf_counter()
    response = post_json(
        api_url,
        {
            "model": api_model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "stop": ["<|im_end|>", "<|endoftext|>", "<end_of_turn>"],
        },
        timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    try:
        raw_output = str(response["choices"][0]["text"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Student API response has no completion text: {response}") from exc
    usage = response.get("usage")
    prompt_tokens = (
        int(usage.get("prompt_tokens", prompt_tokens_local))
        if isinstance(usage, dict)
        else prompt_tokens_local
    )
    generated_tokens = (
        int(usage.get("completion_tokens", 0))
        if isinstance(usage, dict)
        else len(tokenizer(raw_output, add_special_tokens=False)["input_ids"])
    )

    prediction: Any | None = None
    json_valid = False
    schema_valid = False
    error: str | None = None
    try:
        prediction = parse_student_json_output(raw_output)
        json_valid = True
        student_target_to_compact_document(
            prediction,
            case_id=record.case_id,
            document_id=record.document_id,
        )
        schema_valid = True
    except (json.JSONDecodeError, TypeError, ValueError) as caught:
        error = str(caught)

    expected_fields = flatten_fields(expected)
    predicted_fields = flatten_fields(prediction) if prediction is not None else {}
    correct_fields = sum(
        path in predicted_fields and predicted_fields[path] == value
        for path, value in expected_fields.items()
    )
    expected_non_null = {
        path: value for path, value in expected_fields.items() if value is not None
    }
    correct_non_null = sum(
        path in predicted_fields and predicted_fields[path] == value
        for path, value in expected_non_null.items()
    )
    applicant_count_match = bool(
        isinstance(prediction, dict)
        and isinstance(prediction.get("a"), list)
        and isinstance(expected, dict)
        and isinstance(expected.get("a"), list)
        and len(prediction["a"]) == len(expected["a"])
    )
    return PredictionResult(
        id=record.record_id,
        case_id=record.case_id,
        document_id=record.document_id,
        expected=expected,
        prediction=prediction,
        raw_output=raw_output,
        json_valid=json_valid,
        schema_valid=schema_valid,
        exact_match=schema_valid and prediction == expected,
        applicant_count_match=applicant_count_match,
        expected_fields=len(expected_fields),
        predicted_fields=len(predicted_fields),
        correct_fields=correct_fields,
        expected_non_null_fields=len(expected_non_null),
        correct_non_null_fields=correct_non_null,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        latency_ms=latency_ms,
        error=error,
    )


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be greater than zero")
    if args.max_new_tokens <= 0:
        raise RuntimeError("--max-new-tokens must be greater than zero")

    dataset_file = resolve_path(args.dataset_dir) / f"{args.split}.jsonl"
    records = read_training_records(dataset_file)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"Dataset split is empty: {dataset_file}")

    model_dir = resolve_path(args.model_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    api_url = f"{args.api_base_url.rstrip('/')}/completions"
    print(f"Evaluating {len(records)} record(s) via {api_url}; model={args.api_model}")
    results: list[PredictionResult] = []
    for index, record in enumerate(records, start=1):
        result = evaluate_record(
            record,
            tokenizer=tokenizer,
            api_url=api_url,
            api_model=args.api_model,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )
        results.append(result)
        label = "EXACT" if result.exact_match else (
            "VALID" if result.schema_valid else "INVALID"
        )
        print(
            f"[{index}/{len(records)}] {label} id={record.record_id} "
            f"fields={result.correct_fields}/{result.expected_fields} "
            f"latency={result.latency_ms:.0f}ms tokens={result.generated_tokens}"
        )
        if not result.schema_valid:
            print(f"  error={result.error}; output={result.raw_output[:300]!r}")

    # Reuse exactly the same metric calculation and file format as the HF evaluator.
    args.device = "api"
    args.gpu = None
    metrics = build_metrics(results, args)
    metrics["api_base_url"] = args.api_base_url
    metrics["api_model"] = args.api_model
    metrics["tokens_per_second"] = safe_ratio(
        sum(result.generated_tokens for result in results),
        sum(result.latency_ms for result in results) / 1000.0,
    )
    output_dir = resolve_path(args.output_dir)
    write_results(results, metrics, output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {output_dir / 'predictions.jsonl'}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    if args.fail_on_invalid and metrics["schema_valid"] != metrics["total"]:
        raise RuntimeError(
            f"{metrics['total'] - metrics['schema_valid']} prediction(s) failed schema validation"
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"Student API evaluation could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error
