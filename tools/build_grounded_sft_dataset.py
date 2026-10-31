from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.schemas.case_extraction import CASE_SCHEMA_VERSION, CaseExtractionTarget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate reviewed grounded targets and build train/val/test SFT JSONL."
    )
    parser.add_argument("--seed", type=Path, default=None)
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def collect_evidence(value: Any) -> list[str]:
    evidence: list[str] = []
    if isinstance(value, dict):
        for key in ("evidence_token_ids", "token_ids"):
            ids = value.get(key)
            if isinstance(ids, list):
                evidence.extend(str(item) for item in ids)
        for child in value.values():
            evidence.extend(collect_evidence(child))
    elif isinstance(value, list):
        for child in value:
            evidence.extend(collect_evidence(child))
    return evidence


def split_name(document_id: str) -> str:
    bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if settings.llm_schema_version != CASE_SCHEMA_VERSION:
        raise RuntimeError(
            f"LLM_SCHEMA_VERSION must be {CASE_SCHEMA_VERSION!r}; "
            f"current value is {settings.llm_schema_version!r}."
        )
    root = settings.llm_dataset_path
    seed_path = args.seed or (root / "annotation_seed.jsonl")
    target_path = args.targets or (root / "reviewed_targets.jsonl")
    output_dir = args.output_dir or (root / "sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = {item["case_id"]: item for item in read_jsonl(seed_path)}
    targets = read_jsonl(target_path)
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for target_record in targets:
        case_id = target_record.get("case_id")
        target = target_record.get("target")
        seed = seeds.get(case_id)
        if seed is None or not isinstance(target, dict):
            raise ValueError(
                f"Reviewed target has no matching case seed or target object: {case_id}"
            )
        if target.get("case_id") not in {None, case_id}:
            raise ValueError(f"Target case_id does not match its seed: {case_id}")
        if target.get("document_id") not in {None, seed["document_id"]}:
            raise ValueError(f"Target document_id does not match case {case_id}")
        canonical_target = CaseExtractionTarget.model_validate(target).model_dump(mode="json")
        allowed = set(seed.get("allowed_token_ids", []))
        invalid = sorted(set(collect_evidence(canonical_target)) - allowed)
        if invalid:
            raise ValueError(f"Case {case_id} cites unknown OCR token IDs: {invalid}")
        messages = list(seed["messages"])
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    canonical_target, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
        split = split_name(seed["document_id"])
        outputs[split].append(
            {
                "id": seed["example_id"],
                "case_id": case_id,
                "document_id": seed["document_id"],
                "pages": seed["pages"],
                "schema_version": seed["schema_version"],
                "messages": messages,
            }
        )

    for split, records in outputs.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records)} -> {path}")


if __name__ == "__main__":
    main()
