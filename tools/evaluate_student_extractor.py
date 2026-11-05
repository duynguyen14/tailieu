from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.schemas.student_extraction import student_target_to_compact_document
from tools.train_student_extractor import DatasetRecord, read_training_records


DEFAULT_DATASET_DIR = Path("data/training/gemma3_270m_compact_v1")
DEFAULT_MODEL_DIR = Path("output/student_extractor_gemma3_270m_lora_v1/merged_model")
DEFAULT_OUTPUT_DIR = Path("output/student_extractor_gemma3_270m_lora_v1/test_evaluation")


@dataclass
class PredictionResult:
    id: str
    case_id: str
    document_id: str
    expected: Any
    prediction: Any | None
    raw_output: str
    json_valid: bool
    schema_valid: bool
    exact_match: bool
    applicant_count_match: bool
    expected_fields: int
    predicted_fields: int
    correct_fields: int
    expected_non_null_fields: int
    correct_non_null_fields: int
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the merged Gemma student extractor on a held-out JSONL split "
            "and report strict JSON/schema/field accuracy plus latency."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu", default="2", help="Physical GPU ID used for inference")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def flatten_fields(value: Any, prefix: str = "$") -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if isinstance(value, list):
        for index, item in enumerate(value):
            fields.update(flatten_fields(item, f"{prefix}[{index}]"))
    elif isinstance(value, dict):
        for key in sorted(value):
            fields.update(flatten_fields(value[key], f"{prefix}.{key}"))
    else:
        fields[prefix] = value
    return fields


def choose_dtype(torch: Any, *, device: str, precision: str) -> Any:
    if device == "cpu" or precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, str]:
    if args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check the CUDA-enabled PyTorch environment or "
            "run with --device cpu."
        )
    model_dir = resolve_path(args.model_dir)
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"Merged model was not found: {model_dir}")

    dtype = choose_dtype(torch, device=args.device, precision=args.precision)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    load_kwargs: dict[str, Any] = {"local_files_only": True}
    major = int(str(transformers.__version__).split(".", maxsplit=1)[0])
    load_kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    runtime_device = "cuda:0" if args.device == "gpu" else "cpu"
    model.to(runtime_device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return torch, tokenizer, model, runtime_device


def generation_stop_token_ids(tokenizer: Any, model: Any) -> list[int]:
    """Keep both Gemma EOS and end-of-turn tokens.

    Gemma 3 instruction checkpoints normally stop on either token 1 or token
    106.  ``tokenizer.eos_token_id`` only exposes token 1 in some Transformers
    versions, so using it alone makes generation run until max_new_tokens.
    """

    candidates: list[Any] = [
        getattr(model.generation_config, "eos_token_id", None),
        getattr(model.config, "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    ]
    values: list[int] = []
    for candidate in candidates:
        items = candidate if isinstance(candidate, (list, tuple)) else [candidate]
        for item in items:
            if isinstance(item, int) and item >= 0 and item not in values:
                values.append(item)
    if not values:
        raise RuntimeError("The tokenizer/model exposes no generation stop token")
    return values


def evaluate_record(
    record: DatasetRecord,
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    runtime_device: str,
    max_new_tokens: int,
) -> PredictionResult:
    expected = json.loads(record.assistant_text)
    message = {"role": "user", "content": record.user_text}
    prompt = tokenizer.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(runtime_device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    stop_token_ids = generation_stop_token_ids(tokenizer, model)

    if runtime_device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )
    if runtime_device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0

    output_ids = generated[0, prompt_tokens:]
    generated_tokens = int(output_ids.shape[-1])
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    prediction: Any | None = None
    json_valid = False
    schema_valid = False
    error: str | None = None
    try:
        prediction = json.loads(strip_markdown_fence(raw_output))
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
        1
        for path, value in expected_fields.items()
        if path in predicted_fields and predicted_fields[path] == value
    )
    expected_non_null = {
        path: value for path, value in expected_fields.items() if value is not None
    }
    correct_non_null = sum(
        1
        for path, value in expected_non_null.items()
        if path in predicted_fields and predicted_fields[path] == value
    )
    applicant_count_match = bool(
        isinstance(prediction, list)
        and len(prediction) == 2
        and isinstance(prediction[1], list)
        and len(prediction[1]) == len(expected[1])
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


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_metrics(results: list[PredictionResult], args: argparse.Namespace) -> dict[str, Any]:
    total = len(results)
    latencies = [result.latency_ms for result in results]
    correct_fields = sum(result.correct_fields for result in results)
    expected_fields = sum(result.expected_fields for result in results)
    predicted_fields = sum(result.predicted_fields for result in results)
    non_null_correct = sum(result.correct_non_null_fields for result in results)
    non_null_expected = sum(result.expected_non_null_fields for result in results)
    field_precision = safe_ratio(correct_fields, predicted_fields)
    field_recall = safe_ratio(correct_fields, expected_fields)
    field_f1 = safe_ratio(
        2 * field_precision * field_recall,
        field_precision + field_recall,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(resolve_path(args.model_dir)),
        "dataset_file": str(resolve_path(args.dataset_dir) / f"{args.split}.jsonl"),
        "split": args.split,
        "device": args.device,
        "physical_gpu": str(args.gpu) if args.device == "gpu" else None,
        "total": total,
        "json_valid": sum(result.json_valid for result in results),
        "json_valid_rate": safe_ratio(sum(result.json_valid for result in results), total),
        "schema_valid": sum(result.schema_valid for result in results),
        "schema_valid_rate": safe_ratio(sum(result.schema_valid for result in results), total),
        "exact_matches": sum(result.exact_match for result in results),
        "exact_match_rate": safe_ratio(sum(result.exact_match for result in results), total),
        "applicant_count_matches": sum(result.applicant_count_match for result in results),
        "applicant_count_accuracy": safe_ratio(
            sum(result.applicant_count_match for result in results), total
        ),
        "field_precision": field_precision,
        "field_recall": field_recall,
        "field_f1": field_f1,
        "non_null_field_recall": safe_ratio(non_null_correct, non_null_expected),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "minimum": min(latencies, default=0.0),
            "maximum": max(latencies, default=0.0),
        },
        "average_generated_tokens": (
            statistics.fmean(result.generated_tokens for result in results)
            if results
            else 0.0
        ),
    }


def write_results(
    results: list[PredictionResult], metrics: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataset_file = resolve_path(args.dataset_dir) / f"{args.split}.jsonl"
    if not dataset_file.is_file():
        raise RuntimeError(f"Dataset split was not found: {dataset_file}")
    records = read_training_records(dataset_file)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"Dataset split is empty: {dataset_file}")

    torch, tokenizer, model, runtime_device = load_runtime(args)
    stop_token_ids = generation_stop_token_ids(tokenizer, model)
    print(
        f"Evaluating {len(records)} {args.split} record(s) on {runtime_device}; "
        f"model={resolve_path(args.model_dir)}; stop_tokens={stop_token_ids}"
    )
    results: list[PredictionResult] = []
    for index, record in enumerate(records, start=1):
        result = evaluate_record(
            record,
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            runtime_device=runtime_device,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(result)
        status = "EXACT" if result.exact_match else (
            "VALID" if result.schema_valid else "INVALID"
        )
        print(
            f"[{index}/{len(records)}] {status} id={record.record_id} "
            f"fields={result.correct_fields}/{result.expected_fields} "
            f"latency={result.latency_ms:.0f}ms tokens={result.generated_tokens}"
        )
        if not result.schema_valid:
            preview = result.raw_output.replace("\n", " ")[:300]
            print(f"  error={result.error}; output={preview!r}")

    metrics = build_metrics(results, args)
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
    except (RuntimeError, ValueError) as error:
        print(f"Student extractor evaluation could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error
