from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.core.config import PROJECT_ROOT
from tools.kie_ser_train_config import KieSerTrainSettings


@dataclass(frozen=True)
class SerRecord:
    record_id: str
    document_id: str
    page_id: str
    page_number: int
    image_path: Path
    image_width: int
    image_height: int
    words: list[str]
    boxes: list[list[int]]
    labels: list[str]


@dataclass(frozen=True)
class DatasetSummary:
    dataset_dir: Path
    labels: list[str]
    counts: dict[str, int]
    document_counts: dict[str, int]
    entity_types: list[str]
    maximum_words: int
    dataset_fingerprint: str | None

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune LayoutLMv3 for KIE semantic entity recognition (SER) "
            "using the reviewed OCR text, bounding boxes and page images."
        )
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("gpu", "cpu"))
    parser.add_argument("--gpus", help="Physical GPU IDs, for example 2,3,4")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument(
        "--allow-missing-train-label",
        action="store_true",
        help="Allow an entity type to occur only in validation/test (not recommended).",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def apply_warmup_training_argument(
    options: dict[str, Any],
    *,
    parameters: dict[str, inspect.Parameter] | Any,
    warmup_ratio: float,
    train_item_count: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    epochs: float,
) -> None:
    """Set the warmup option supported by the installed Transformers version."""
    if "warmup_ratio" in parameters:
        options["warmup_ratio"] = warmup_ratio
        return
    if "warmup_steps" not in parameters:
        if warmup_ratio:
            raise RuntimeError(
                "The installed Transformers TrainingArguments supports neither "
                "warmup_ratio nor warmup_steps."
            )
        return

    samples_per_rank = math.ceil(train_item_count / max(world_size, 1))
    batches_per_rank = math.ceil(samples_per_rank / max(per_device_batch_size, 1))
    updates_per_epoch = math.ceil(
        batches_per_rank / max(gradient_accumulation_steps, 1)
    )
    total_steps = math.ceil(updates_per_epoch * epochs)
    options["warmup_steps"] = (
        max(1, math.ceil(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0
    )


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def apply_cli_overrides(args: argparse.Namespace) -> KieSerTrainSettings:
    updates: dict[str, Any] = {}
    mapping = {
        "dataset_dir": "kie_train_dataset_dir",
        "output_dir": "kie_train_output_dir",
        "model_id": "kie_train_model_id",
        "model_dir": "kie_train_model_dir",
        "device": "kie_train_device",
        "gpus": "kie_train_gpu_device_ids",
        "epochs": "kie_train_epochs",
        "batch_size": "kie_train_batch_size",
        "eval_batch_size": "kie_train_eval_batch_size",
        "gradient_accumulation_steps": "kie_train_gradient_accumulation_steps",
        "learning_rate": "kie_train_learning_rate",
        "max_sequence_length": "kie_train_max_sequence_length",
        "precision": "kie_train_precision",
        "checkpoint": "kie_train_resume_from_checkpoint",
    }
    for argument_name, setting_name in mapping.items():
        value = getattr(args, argument_name)
        if value is not None:
            updates[setting_name] = value
    if args.overwrite_output:
        updates["kie_train_overwrite_output"] = True
    if args.offline:
        updates["kie_train_offline"] = True
    if args.allow_truncation:
        updates["kie_train_fail_on_truncation"] = False
    if args.allow_missing_train_label:
        updates["kie_train_fail_on_missing_train_label"] = False
    return KieSerTrainSettings(**updates)


def _read_label_schema(dataset_dir: Path) -> list[str]:
    path = dataset_dir / "labels.json"
    if not path.is_file():
        raise RuntimeError(f"SER label schema was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"labels.json is invalid: {error}") from error
    labels = payload.get("bio_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels.json must contain a non-empty bio_labels list")
    normalized = [str(label).strip() for label in labels]
    if normalized[0] != "O":
        raise ValueError("The first BIO label must be O")
    if any(not label for label in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("BIO labels must be non-empty and unique")
    valid_pattern = re.compile(r"^(O|[BI]-[A-Z][A-Z0-9_]*)$")
    invalid = [label for label in normalized if not valid_pattern.fullmatch(label)]
    if invalid:
        raise ValueError(f"Invalid BIO labels: {', '.join(invalid[:5])}")
    return normalized


def _resolve_image_path(dataset_dir: Path, raw_path: object) -> Path:
    value = str(raw_path or "").strip()
    if not value:
        raise ValueError("record has no image_path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (dataset_dir / path).resolve()


def _validate_box(box: object, *, context: str) -> list[int]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{context} must be [x1,y1,x2,y2]")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in box):
        raise ValueError(f"{context} coordinates must be integers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000):
        raise ValueError(f"{context} is outside LayoutLM coordinates [0,1000]: {box}")
    return [x1, y1, x2, y2]


def read_ser_records(
    path: Path,
    *,
    dataset_dir: Path,
    allowed_labels: set[str],
    validate_images: bool = True,
) -> list[SerRecord]:
    records: list[SerRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}:{line_number} is not valid JSON: {error}"
                ) from error
            context = f"{path.name}:{line_number}"
            record_id = str(payload.get("id", "")).strip()
            document_id = str(payload.get("document_id", "")).strip()
            page_id = str(payload.get("page_id", "")).strip()
            if not record_id or not document_id or not page_id:
                raise ValueError(f"{context} is missing id/document_id/page_id")
            if record_id in seen_ids:
                raise ValueError(f"{path.name} contains duplicate id {record_id}")
            seen_ids.add(record_id)
            words = payload.get("words")
            boxes = payload.get("boxes")
            labels = payload.get("labels")
            if not isinstance(words, list) or not isinstance(boxes, list) or not isinstance(labels, list):
                raise ValueError(f"{context} words/boxes/labels must be lists")
            if not words or not (len(words) == len(boxes) == len(labels)):
                raise ValueError(
                    f"{context} word/bbox/label lengths differ: "
                    f"{len(words)}/{len(boxes)}/{len(labels)}"
                )
            normalized_words = [str(word).strip() for word in words]
            if any(not word for word in normalized_words):
                raise ValueError(f"{context} contains an empty OCR segment")
            normalized_boxes = [
                _validate_box(box, context=f"{context} boxes[{index}]")
                for index, box in enumerate(boxes)
            ]
            normalized_labels = [str(label).strip() for label in labels]
            unknown = sorted(set(normalized_labels) - allowed_labels)
            if unknown:
                raise ValueError(
                    f"{context} contains labels absent from labels.json: "
                    f"{', '.join(unknown)}"
                )
            image_path = _resolve_image_path(dataset_dir, payload.get("image_path"))
            width = int(payload.get("image_width", 0))
            height = int(payload.get("image_height", 0))
            if width <= 0 or height <= 0:
                raise ValueError(f"{context} has invalid image dimensions {width}x{height}")
            if validate_images:
                if not image_path.is_file():
                    raise FileNotFoundError(f"{context} image was not found: {image_path}")
                try:
                    from PIL import Image

                    with Image.open(image_path) as image:
                        actual_size = image.size
                except ImportError as error:
                    raise RuntimeError(
                        "Pillow is required to validate KIE dataset images"
                    ) from error
                if actual_size != (width, height):
                    raise ValueError(
                        f"{context} image dimensions differ: "
                        f"file={actual_size}, record={(width, height)}"
                    )
            records.append(
                SerRecord(
                    record_id=record_id,
                    document_id=document_id,
                    page_id=page_id,
                    page_number=int(payload.get("page_number", 0)),
                    image_path=image_path,
                    image_width=width,
                    image_height=height,
                    words=normalized_words,
                    boxes=normalized_boxes,
                    labels=normalized_labels,
                )
            )
    return records


def _entity_types(labels: Iterable[str]) -> set[str]:
    return {label[2:] for label in labels if label != "O"}


def validate_dataset(
    dataset_dir: Path,
    *,
    validate_images: bool = True,
    fail_on_missing_train_label: bool = True,
) -> DatasetSummary:
    dataset_dir = _resolve(dataset_dir)
    if not dataset_dir.is_dir():
        raise RuntimeError(f"SER dataset directory was not found: {dataset_dir}")
    labels = _read_label_schema(dataset_dir)
    allowed_labels = set(labels)
    counts: dict[str, int] = {}
    document_counts: dict[str, int] = {}
    records_by_split: dict[str, list[SerRecord]] = {}
    global_ids: set[str] = set()
    documents_by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"SER dataset split was not found: {path}")
        records = read_ser_records(
            path,
            dataset_dir=dataset_dir,
            allowed_labels=allowed_labels,
            validate_images=validate_images,
        )
        duplicates = global_ids.intersection(record.record_id for record in records)
        if duplicates:
            raise RuntimeError(
                "Records occur in more than one split: "
                + ", ".join(sorted(duplicates)[:3])
            )
        global_ids.update(record.record_id for record in records)
        documents = {record.document_id for record in records}
        for previous_split, previous_documents in documents_by_split.items():
            leaked = documents.intersection(previous_documents)
            if leaked:
                raise RuntimeError(
                    f"Document leakage between {previous_split} and {split}: "
                    + ", ".join(sorted(leaked)[:3])
                )
        documents_by_split[split] = documents
        records_by_split[split] = records
        counts[split] = len(records)
        document_counts[split] = len(documents)
    if not counts["train"] or not counts["val"] or not counts["test"]:
        raise RuntimeError(f"SER train/val/test splits must be non-empty: {counts}")
    train_types = _entity_types(
        label for record in records_by_split["train"] for label in record.labels
    )
    all_types = _entity_types(labels)
    missing_train_types = sorted(all_types - train_types)
    if missing_train_types and fail_on_missing_train_label:
        raise RuntimeError(
            "Training split has no samples for entity types: "
            + ", ".join(missing_train_types)
        )
    statistics_path = dataset_dir / "dataset_statistics.json"
    fingerprint: str | None = None
    if statistics_path.is_file():
        try:
            fingerprint = json.loads(
                statistics_path.read_text(encoding="utf-8")
            ).get("dataset_fingerprint")
        except json.JSONDecodeError as error:
            raise ValueError(f"dataset_statistics.json is invalid: {error}") from error
    all_records = [
        record for records in records_by_split.values() for record in records
    ]
    return DatasetSummary(
        dataset_dir=dataset_dir,
        labels=labels,
        counts=counts,
        document_counts=document_counts,
        entity_types=sorted(all_types),
        maximum_words=max(len(record.words) for record in all_records),
        dataset_fingerprint=fingerprint,
    )


def _parse_bio(label: str) -> tuple[str, str | None]:
    if label == "O":
        return "O", None
    return label[0], label[2:]


def bio_entities(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    """Convert BIO labels into (entity_type, start, exclusive_end) spans."""

    entities: list[tuple[str, int, int]] = []
    current_type: str | None = None
    start = 0
    for index, label in enumerate([*labels, "O"]):
        prefix, entity_type = _parse_bio(label)
        continues = prefix == "I" and current_type == entity_type
        if current_type is not None and not continues:
            entities.append((current_type, start, index))
            current_type = None
        if prefix == "B" or (prefix == "I" and not continues):
            current_type = entity_type
            start = index
    return entities


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_bio_metrics(
    predictions: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    *,
    id_to_label: dict[int, str],
) -> dict[str, float]:
    predicted_entities: Counter[tuple[str, int, int, int]] = Counter()
    reference_entities: Counter[tuple[str, int, int, int]] = Counter()
    correct_tokens = 0
    total_tokens = 0
    predicted_non_o = 0
    reference_non_o = 0
    correct_non_o = 0
    for row_index, (predicted_row, reference_row) in enumerate(
        zip(predictions, references)
    ):
        predicted_labels: list[str] = []
        reference_labels: list[str] = []
        for predicted_id, reference_id in zip(predicted_row, reference_row):
            if int(reference_id) == -100:
                continue
            predicted_label = id_to_label[int(predicted_id)]
            reference_label = id_to_label[int(reference_id)]
            predicted_labels.append(predicted_label)
            reference_labels.append(reference_label)
            total_tokens += 1
            correct_tokens += predicted_label == reference_label
            predicted_non_o += predicted_label != "O"
            reference_non_o += reference_label != "O"
            correct_non_o += predicted_label == reference_label and reference_label != "O"
        predicted_entities.update(
            (entity_type, start, end, row_index)
            for entity_type, start, end in bio_entities(predicted_labels)
        )
        reference_entities.update(
            (entity_type, start, end, row_index)
            for entity_type, start, end in bio_entities(reference_labels)
        )
    matched_entities = predicted_entities & reference_entities
    true_positive = sum(matched_entities.values())
    predicted_total = sum(predicted_entities.values())
    reference_total = sum(reference_entities.values())
    entity_precision = _safe_ratio(true_positive, predicted_total)
    entity_recall = _safe_ratio(true_positive, reference_total)
    entity_f1 = _safe_ratio(
        2 * true_positive,
        predicted_total + reference_total,
    )
    metrics: dict[str, float] = {
        "token_accuracy": _safe_ratio(correct_tokens, total_tokens),
        "token_non_o_precision": _safe_ratio(correct_non_o, predicted_non_o),
        "token_non_o_recall": _safe_ratio(correct_non_o, reference_non_o),
        "token_non_o_f1": _safe_ratio(
            2 * correct_non_o,
            predicted_non_o + reference_non_o,
        ),
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
    }
    entity_types = sorted(
        {item[0] for item in predicted_entities}
        | {item[0] for item in reference_entities}
    )
    for entity_type in entity_types:
        predicted_for_type = Counter(
            {
                item: count
                for item, count in predicted_entities.items()
                if item[0] == entity_type
            }
        )
        reference_for_type = Counter(
            {
                item: count
                for item, count in reference_entities.items()
                if item[0] == entity_type
            }
        )
        matched = sum((predicted_for_type & reference_for_type).values())
        predicted_count = sum(predicted_for_type.values())
        reference_count = sum(reference_for_type.values())
        key = entity_type.lower()
        metrics[f"entity_precision_{key}"] = _safe_ratio(matched, predicted_count)
        metrics[f"entity_recall_{key}"] = _safe_ratio(matched, reference_count)
        metrics[f"entity_f1_{key}"] = _safe_ratio(
            2 * matched, predicted_count + reference_count
        )
    return metrics


def settings_environment(settings: KieSerTrainSettings) -> dict[str, str]:
    environment = os.environ.copy()
    values: dict[str, Any] = {
        "KIE_TRAIN_MODEL_ID": settings.kie_train_model_id,
        "KIE_TRAIN_MODEL_DIR": settings.model_path or "",
        "KIE_TRAIN_DATASET_DIR": settings.dataset_path,
        "KIE_TRAIN_OUTPUT_DIR": settings.output_path,
        "KIE_TRAIN_DEVICE": settings.kie_train_device,
        "KIE_TRAIN_GPU_DEVICE_IDS": settings.kie_train_gpu_device_ids,
        "KIE_TRAIN_EPOCHS": settings.kie_train_epochs,
        "KIE_TRAIN_BATCH_SIZE": settings.kie_train_batch_size,
        "KIE_TRAIN_EVAL_BATCH_SIZE": settings.kie_train_eval_batch_size,
        "KIE_TRAIN_GRADIENT_ACCUMULATION_STEPS": settings.kie_train_gradient_accumulation_steps,
        "KIE_TRAIN_LEARNING_RATE": settings.kie_train_learning_rate,
        "KIE_TRAIN_WEIGHT_DECAY": settings.kie_train_weight_decay,
        "KIE_TRAIN_WARMUP_RATIO": settings.kie_train_warmup_ratio,
        "KIE_TRAIN_MAX_SEQUENCE_LENGTH": settings.kie_train_max_sequence_length,
        "KIE_TRAIN_NUM_WORKERS": settings.kie_train_num_workers,
        "KIE_TRAIN_PRECISION": settings.kie_train_precision,
        "KIE_TRAIN_EARLY_STOPPING_PATIENCE": settings.kie_train_early_stopping_patience,
        "KIE_TRAIN_SAVE_TOTAL_LIMIT": settings.kie_train_save_total_limit,
        "KIE_TRAIN_SEED": settings.kie_train_seed,
        "KIE_TRAIN_FAIL_ON_TRUNCATION": settings.kie_train_fail_on_truncation,
        "KIE_TRAIN_FAIL_ON_MISSING_TRAIN_LABEL": settings.kie_train_fail_on_missing_train_label,
        "KIE_TRAIN_OVERWRITE_OUTPUT": settings.kie_train_overwrite_output,
        "KIE_TRAIN_RESUME_FROM_CHECKPOINT": settings.resume_checkpoint_path or "",
        "KIE_TRAIN_OFFLINE": settings.kie_train_offline,
    }
    for key, value in values.items():
        environment[key] = (
            "true" if value is True else "false" if value is False else str(value)
        )
    if settings.kie_train_device == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = settings.kie_train_gpu_device_ids
        environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if settings.kie_train_offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def build_launch_command(
    settings: KieSerTrainSettings,
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    if settings.kie_train_device == "gpu" and len(settings.gpu_device_ids) > 1:
        return [
            python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(settings.gpu_device_ids)}",
            "--module",
            "tools.train_kie_ser",
            "--worker",
        ]
    return [python_executable, "-m", "tools.train_kie_ser", "--worker"]


def _load_split(
    summary: DatasetSummary,
    split: str,
    *,
    validate_images: bool = False,
) -> list[SerRecord]:
    return read_ser_records(
        summary.dataset_dir / f"{split}.jsonl",
        dataset_dir=summary.dataset_dir,
        allowed_labels=set(summary.labels),
        validate_images=validate_images,
    )


def _encode_records(
    records: list[SerRecord],
    *,
    processor: Any,
    label_to_id: dict[str, int],
    maximum_length: int,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, Any]], list[tuple[int, list[int | None]]], int]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for LayoutLMv3 training") from error
    items: list[dict[str, Any]] = []
    chunk_metadata: list[tuple[int, list[int | None]]] = []
    overflow_pages = 0
    for record_index, record in enumerate(records):
        with Image.open(record.image_path) as image:
            encoding = processor(
                images=image.convert("RGB"),
                text=record.words,
                boxes=record.boxes,
                word_labels=[label_to_id[label] for label in record.labels],
                padding="max_length",
                truncation=True,
                max_length=maximum_length,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
        chunk_count = int(encoding["input_ids"].shape[0])
        if chunk_count > 1:
            overflow_pages += 1
        page_word_ids: set[int] = set()
        for chunk_index in range(chunk_count):
            word_ids = list(encoding.word_ids(batch_index=chunk_index))
            page_word_ids.update(
                word_id for word_id in word_ids if word_id is not None
            )
            item = {
                key: value[chunk_index]
                for key, value in encoding.items()
                if key != "overflow_to_sample_mapping"
            }
            # A window can begin in the middle of a multi-word entity. It is a
            # new sequence for the model, so normalize an orphan I-X to B-X.
            supervised_positions = [
                position
                for position, label_id in enumerate(item["labels"].tolist())
                if int(label_id) != -100
            ]
            if supervised_positions:
                first_position = supervised_positions[0]
                first_label = next(
                    label
                    for label, label_id in label_to_id.items()
                    if label_id == int(item["labels"][first_position])
                )
                if first_label.startswith("I-"):
                    beginning_label = f"B-{first_label[2:]}"
                    item["labels"][first_position] = label_to_id[beginning_label]
            items.append(item)
            chunk_metadata.append((record_index, word_ids))
        included_words = page_word_ids
        missing_words = len(record.words) - len(included_words)
        if missing_words:
            if fail_on_truncation:
                raise RuntimeError(
                    f"Record {record.record_id} still loses "
                    f"{missing_words}/{len(record.words)} words after overflow chunking."
                )
        print(
            f"Encoded {record_index + 1}/{len(records)}: id={record.record_id} "
            f"words={len(record.words)} chunks={chunk_count} missing={missing_words}"
        )
    return items, chunk_metadata, overflow_pages


def _write_test_predictions(
    path: Path,
    *,
    records: list[SerRecord],
    chunk_metadata: list[tuple[int, list[int | None]]],
    predictions: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
    id_to_label: dict[int, str],
) -> None:
    predicted_by_record: list[dict[int, str]] = [{} for _ in records]
    true_by_record: list[dict[int, str]] = [{} for _ in records]
    for (record_index, word_ids), predicted_row, reference_row in zip(
        chunk_metadata, predictions, references
    ):
        for position, word_id in enumerate(word_ids):
            if word_id is None or int(reference_row[position]) == -100:
                continue
            predicted_by_record[record_index].setdefault(
                word_id, id_to_label[int(predicted_row[position])]
            )
            true_by_record[record_index].setdefault(
                word_id, id_to_label[int(reference_row[position])]
            )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record_index, record in enumerate(records):
            predicted_by_word = predicted_by_record[record_index]
            true_by_word = true_by_record[record_index]
            included = sorted(true_by_word)
            payload = {
                "id": record.record_id,
                "document_id": record.document_id,
                "page_id": record.page_id,
                "page_number": record.page_number,
                "words": [record.words[word_id] for word_id in included],
                "boxes": [record.boxes[word_id] for word_id in included],
                "true_labels": [true_by_word[word_id] for word_id in included],
                "predicted_labels": [
                    predicted_by_word[word_id] for word_id in included
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_worker(settings: KieSerTrainSettings) -> None:
    try:
        import numpy as np
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            __version__ as transformers_version,
            EarlyStoppingCallback,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            Trainer,
            TrainingArguments,
            default_data_collator,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "KIE training dependencies are missing. Install "
            "requirements-kie-training.txt in the training venv."
        ) from error
    if settings.kie_train_device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("KIE_TRAIN_DEVICE=gpu but CUDA PyTorch is unavailable")
    summary = validate_dataset(
        settings.dataset_path,
        validate_images=True,
        fail_on_missing_train_label=settings.kie_train_fail_on_missing_train_label,
    )
    set_seed(settings.kie_train_seed)
    model_source = str(settings.model_path or settings.kie_train_model_id)
    token = os.getenv("HF_TOKEN") or None
    load_options = {
        "token": token,
        "local_files_only": settings.kie_train_offline,
    }
    processor = LayoutLMv3Processor.from_pretrained(
        model_source,
        apply_ocr=False,
        **load_options,
    )
    label_to_id = {label: index for index, label in enumerate(summary.labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    records_by_split = {
        split: _load_split(summary, split) for split in ("train", "val", "test")
    }
    items_by_split: dict[str, list[dict[str, Any]]] = {}
    chunk_metadata_by_split: dict[str, list[tuple[int, list[int | None]]]] = {}
    overflow_page_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        items, chunk_metadata, overflow_pages = _encode_records(
            records_by_split[split],
            processor=processor,
            label_to_id=label_to_id,
            maximum_length=settings.kie_train_max_sequence_length,
            fail_on_truncation=settings.kie_train_fail_on_truncation,
        )
        items_by_split[split] = items
        chunk_metadata_by_split[split] = chunk_metadata
        overflow_page_counts[split] = overflow_pages

    class EncodedDataset(Dataset):
        def __init__(self, items: list[dict[str, Any]]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.items[index]

    use_cuda = settings.kie_train_device == "gpu"
    precision = settings.kie_train_precision
    if use_cuda and precision == "bf16" and not torch.cuda.is_bf16_supported():
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print("BF16 is unsupported; falling back to FP16.")
        precision = "fp16"
    if not use_cuda:
        precision = "fp32"
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(summary.labels),
        id2label=id_to_label,
        label2id=label_to_id,
        **load_options,
    )
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    effective_batch = (
        settings.kie_train_batch_size
        * settings.kie_train_gradient_accumulation_steps
        * world_size
    )
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(
            f"KIE SER training: model={model_source}; train={summary.counts['train']}; "
            f"val={summary.counts['val']}; test={summary.counts['test']}; "
            f"labels={len(summary.labels)}; world_size={world_size}; "
            f"precision={precision}; effective_batch={effective_batch}"
        )

    def compute_metrics(evaluation: Any) -> dict[str, float]:
        logits = evaluation.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted_ids = np.argmax(logits, axis=-1)
        return calculate_bio_metrics(
            predicted_ids,
            evaluation.label_ids,
            id_to_label=id_to_label,
        )

    options: dict[str, Any] = {
        "output_dir": str(settings.output_path),
        "num_train_epochs": settings.kie_train_epochs,
        "per_device_train_batch_size": settings.kie_train_batch_size,
        "per_device_eval_batch_size": settings.kie_train_eval_batch_size,
        "gradient_accumulation_steps": settings.kie_train_gradient_accumulation_steps,
        "learning_rate": settings.kie_train_learning_rate,
        "weight_decay": settings.kie_train_weight_decay,
        "lr_scheduler_type": "linear",
        "logging_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": settings.kie_train_save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "entity_f1",
        "greater_is_better": True,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "tf32": use_cuda,
        "dataloader_num_workers": settings.kie_train_num_workers,
        # LayoutLMv3ForTokenClassification keeps a few base-model parameters
        # outside the SER loss graph. DDP must explicitly account for them;
        # otherwise the second optimizer iteration fails with an unfinished
        # reduction error. Single-GPU training does not need this setting.
        "ddp_find_unused_parameters": True if world_size > 1 else None,
        "report_to": "none",
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "seed": settings.kie_train_seed,
        "data_seed": settings.kie_train_seed,
        "optim": "adamw_torch",
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    apply_warmup_training_argument(
        options,
        parameters=parameters,
        warmup_ratio=settings.kie_train_warmup_ratio,
        train_item_count=len(items_by_split["train"]),
        per_device_batch_size=settings.kie_train_batch_size,
        gradient_accumulation_steps=settings.kie_train_gradient_accumulation_steps,
        world_size=world_size,
        epochs=settings.kie_train_epochs,
    )
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        options["evaluation_strategy"] = options.pop("eval_strategy")
    if "overwrite_output_dir" in parameters:
        options["overwrite_output_dir"] = settings.kie_train_overwrite_output
    unsupported = sorted(set(options) - set(parameters))
    if unsupported:
        raise RuntimeError(
            f"Transformers {transformers_version} has unsupported TrainingArguments: "
            + ", ".join(unsupported)
        )
    training_arguments = TrainingArguments(**options)
    callbacks = []
    if settings.kie_train_early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=settings.kie_train_early_stopping_patience
            )
        )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=EncodedDataset(items_by_split["train"]),
        eval_dataset=EncodedDataset(items_by_split["val"]),
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    checkpoint = settings.resume_checkpoint_path
    train_result = trainer.train(
        resume_from_checkpoint=str(checkpoint) if checkpoint else None
    )
    validation_metrics = trainer.evaluate()
    test_output = trainer.predict(EncodedDataset(items_by_split["test"]))
    test_logits = test_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_predictions = np.argmax(test_logits, axis=-1)
    best_model_dir = settings.output_path / "best_model"
    if trainer.is_world_process_zero():
        if best_model_dir.exists():
            shutil.rmtree(best_model_dir)
        trainer.save_model(str(best_model_dir))
        processor.save_pretrained(best_model_dir)
        (best_model_dir / "labels.json").write_text(
            json.dumps(
                {"bio_labels": summary.labels}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_metrics("validation", validation_metrics)
        trainer.save_metrics("test", test_output.metrics)
        trainer.save_state()
        _write_test_predictions(
            settings.output_path / "test_predictions.jsonl",
            records=records_by_split["test"],
            chunk_metadata=chunk_metadata_by_split["test"],
            predictions=test_predictions,
            references=test_output.label_ids,
            id_to_label=id_to_label,
        )
        training_summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_source": model_source,
            "dataset_dir": str(summary.dataset_dir),
            "dataset_fingerprint": summary.dataset_fingerprint,
            "dataset_counts": summary.counts,
            "document_counts": summary.document_counts,
            "bio_labels": summary.labels,
            "entity_types": summary.entity_types,
            "precision": precision,
            "world_size": world_size,
            "effective_batch_size": effective_batch,
            "overflow_pages": overflow_page_counts,
            "encoded_chunks": {
                split: len(items) for split, items in items_by_split.items()
            },
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_model": str(best_model_dir),
            "train_metrics": train_result.metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_output.metrics,
        }
        settings.output_path.mkdir(parents=True, exist_ok=True)
        (settings.output_path / "training_summary.json").write_text(
            json.dumps(training_summary, ensure_ascii=False, indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )
        print(f"Best KIE SER model: {best_model_dir}")
        print(json.dumps(test_output.metrics, ensure_ascii=False, indent=2))


def _assert_output_is_safe(settings: KieSerTrainSettings) -> None:
    output_path = settings.output_path
    if not output_path.exists():
        return
    has_training = any(output_path.glob("checkpoint-*")) or (
        output_path / "best_model"
    ).exists()
    if (
        has_training
        and settings.resume_checkpoint_path is None
        and not settings.kie_train_overwrite_output
    ):
        raise RuntimeError(
            f"Training output already exists: {output_path}. Use --checkpoint PATH "
            "to resume or --overwrite-output for a new run."
        )


def print_plan(
    settings: KieSerTrainSettings,
    summary: DatasetSummary,
    command: list[str],
) -> None:
    device_count = (
        len(settings.gpu_device_ids) if settings.kie_train_device == "gpu" else 1
    )
    effective_batch = (
        settings.kie_train_batch_size
        * settings.kie_train_gradient_accumulation_steps
        * device_count
    )
    device = (
        f"physical GPUs {settings.kie_train_gpu_device_ids}"
        if settings.kie_train_device == "gpu"
        else "CPU"
    )
    print(
        f"Dataset: {summary.dataset_dir}\n"
        f"Splits: {summary.counts}; documents: {summary.document_counts}\n"
        f"Entity types: {len(summary.entity_types)}; BIO labels: {len(summary.labels)}\n"
        f"Maximum OCR segments/page: {summary.maximum_words}\n"
        f"Model: {settings.model_path or settings.kie_train_model_id}\n"
        f"Device: {device}\n"
        f"Epochs: {settings.kie_train_epochs}; early stopping: "
        f"{settings.kie_train_early_stopping_patience}\n"
        f"Batch/device: {settings.kie_train_batch_size}; accumulation: "
        f"{settings.kie_train_gradient_accumulation_steps}; effective: "
        f"{effective_batch}\n"
        f"Output: {settings.output_path}\n"
        f"Launch command: {subprocess.list2cmdline(command)}"
    )


def main() -> None:
    args = parse_args()
    settings = apply_cli_overrides(args)
    if args.worker:
        run_worker(settings)
        return
    if settings.kie_train_device == "gpu" and not settings.gpu_device_ids:
        raise RuntimeError(
            "GPU training requires at least one KIE_TRAIN_GPU_DEVICE_IDS value"
        )
    summary = validate_dataset(
        settings.dataset_path,
        validate_images=True,
        fail_on_missing_train_label=settings.kie_train_fail_on_missing_train_label,
    )
    command = build_launch_command(settings)
    print_plan(settings, summary, command)
    if args.dry_run:
        print("DRY RUN: dataset/schema/image validation passed; no files were changed.")
        return
    _assert_output_is_safe(settings)
    if args.prepare_only:
        print("PREPARE ONLY: training was not started and no model was downloaded.")
        return
    subprocess.run(
        command,
        check=True,
        cwd=PROJECT_ROOT,
        env=settings_environment(settings),
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"KIE SER training could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error
