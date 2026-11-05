from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.core.config import PROJECT_ROOT
from app.schemas.student_extraction import (
    STUDENT_SCHEMA_VERSION,
    student_target_to_compact_document,
)
from tools.student_extractor_train_config import StudentExtractorTrainSettings


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    case_id: str
    document_id: str
    user_text: str
    assistant_text: str


@dataclass(frozen=True)
class DatasetSummary:
    dataset_dir: Path
    counts: dict[str, int]
    maximum_input_characters: int
    maximum_target_characters: int

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune google/gemma-3-270m-it as the compact text-to-JSON "
            "student extractor. The parent command launches torchrun when "
            "multiple GPUs are configured."
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--merge-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def apply_cli_overrides(args: argparse.Namespace) -> StudentExtractorTrainSettings:
    updates: dict[str, Any] = {}
    mapping = {
        "dataset_dir": "student_train_dataset_dir",
        "output_dir": "student_train_output_dir",
        "model_id": "student_train_model_id",
        "model_dir": "student_train_model_dir",
        "device": "student_train_device",
        "gpus": "student_train_gpu_device_ids",
        "epochs": "student_train_epochs",
        "batch_size": "student_train_batch_size",
        "eval_batch_size": "student_train_eval_batch_size",
        "gradient_accumulation_steps": "student_train_gradient_accumulation_steps",
        "learning_rate": "student_train_learning_rate",
        "max_sequence_length": "student_train_max_sequence_length",
        "checkpoint": "student_train_resume_from_checkpoint",
        "precision": "student_train_precision",
    }
    for argument_name, setting_name in mapping.items():
        value = getattr(args, argument_name)
        if value is not None:
            updates[setting_name] = value
    if args.overwrite_output:
        updates["student_train_overwrite_output"] = True
    if args.no_merge:
        updates["student_train_merge_after_train"] = False
    if args.offline:
        updates["student_train_offline"] = True
    return StudentExtractorTrainSettings(**updates)


def read_training_records(path: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
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
            record_id = str(payload.get("id", "")).strip()
            case_id = str(payload.get("case_id", "")).strip()
            document_id = str(payload.get("document_id", "")).strip()
            if not record_id or not case_id or not document_id:
                raise ValueError(
                    f"{path.name}:{line_number} is missing id/case_id/document_id"
                )
            if record_id in seen_ids:
                raise ValueError(f"{path.name} contains duplicate id {record_id}")
            seen_ids.add(record_id)
            if payload.get("schema_version") != STUDENT_SCHEMA_VERSION:
                raise ValueError(
                    f"{path.name}:{line_number} uses schema "
                    f"{payload.get('schema_version')!r}; expected "
                    f"{STUDENT_SCHEMA_VERSION!r}"
                )
            messages = payload.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(
                    f"{path.name}:{line_number} must contain exactly two messages"
                )
            if (
                not isinstance(messages[0], dict)
                or not isinstance(messages[1], dict)
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
            ):
                raise ValueError(
                    f"{path.name}:{line_number} roles must be user then assistant"
                )
            user_text = messages[0].get("content")
            assistant_text = messages[1].get("content")
            if not isinstance(user_text, str) or not user_text.strip():
                raise ValueError(f"{path.name}:{line_number} has empty user content")
            if not isinstance(assistant_text, str) or not assistant_text.strip():
                raise ValueError(
                    f"{path.name}:{line_number} has empty assistant content"
                )
            try:
                target = json.loads(assistant_text)
                student_target_to_compact_document(
                    target,
                    case_id=case_id,
                    document_id=document_id,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"{path.name}:{line_number} has an invalid student target: {error}"
                ) from error
            records.append(
                DatasetRecord(
                    record_id=record_id,
                    case_id=case_id,
                    document_id=document_id,
                    user_text=user_text.strip(),
                    assistant_text=assistant_text.strip(),
                )
            )
    return records


def validate_dataset(dataset_dir: Path) -> DatasetSummary:
    dataset_dir = _resolve(dataset_dir)
    if not dataset_dir.is_dir():
        raise RuntimeError(f"Student dataset directory was not found: {dataset_dir}")
    counts: dict[str, int] = {}
    all_records: list[DatasetRecord] = []
    global_ids: set[str] = set()
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Student dataset split was not found: {path}")
        records = read_training_records(path)
        duplicate_ids = global_ids.intersection(record.record_id for record in records)
        if duplicate_ids:
            preview = ", ".join(sorted(duplicate_ids)[:3])
            raise RuntimeError(f"Records occur in more than one split: {preview}")
        global_ids.update(record.record_id for record in records)
        counts[split] = len(records)
        all_records.extend(records)
    if not counts["train"]:
        raise RuntimeError("Student training split is empty")
    if not counts["val"]:
        raise RuntimeError("Student validation split is empty")
    return DatasetSummary(
        dataset_dir=dataset_dir,
        counts=counts,
        maximum_input_characters=max(
            (len(record.user_text) for record in all_records), default=0
        ),
        maximum_target_characters=max(
            (len(record.assistant_text) for record in all_records), default=0
        ),
    )


def settings_environment(settings: StudentExtractorTrainSettings) -> dict[str, str]:
    environment = os.environ.copy()
    values: dict[str, Any] = {
        "STUDENT_TRAIN_MODEL_ID": settings.student_train_model_id,
        "STUDENT_TRAIN_MODEL_DIR": settings.model_path or "",
        "STUDENT_TRAIN_DATASET_DIR": settings.dataset_path,
        "STUDENT_TRAIN_OUTPUT_DIR": settings.output_path,
        "STUDENT_TRAIN_DEVICE": settings.student_train_device,
        "STUDENT_TRAIN_GPU_DEVICE_IDS": settings.student_train_gpu_device_ids,
        "STUDENT_TRAIN_EPOCHS": settings.student_train_epochs,
        "STUDENT_TRAIN_BATCH_SIZE": settings.student_train_batch_size,
        "STUDENT_TRAIN_EVAL_BATCH_SIZE": settings.student_train_eval_batch_size,
        "STUDENT_TRAIN_GRADIENT_ACCUMULATION_STEPS": settings.student_train_gradient_accumulation_steps,
        "STUDENT_TRAIN_LEARNING_RATE": settings.student_train_learning_rate,
        "STUDENT_TRAIN_WEIGHT_DECAY": settings.student_train_weight_decay,
        "STUDENT_TRAIN_WARMUP_RATIO": settings.student_train_warmup_ratio,
        "STUDENT_TRAIN_MAX_SEQUENCE_LENGTH": settings.student_train_max_sequence_length,
        "STUDENT_TRAIN_NUM_WORKERS": settings.student_train_num_workers,
        "STUDENT_TRAIN_PRECISION": settings.student_train_precision,
        "STUDENT_TRAIN_LORA_RANK": settings.student_train_lora_rank,
        "STUDENT_TRAIN_LORA_ALPHA": settings.student_train_lora_alpha,
        "STUDENT_TRAIN_LORA_DROPOUT": settings.student_train_lora_dropout,
        "STUDENT_TRAIN_LORA_TARGET_MODULES": settings.student_train_lora_target_modules,
        "STUDENT_TRAIN_EARLY_STOPPING_PATIENCE": settings.student_train_early_stopping_patience,
        "STUDENT_TRAIN_SAVE_TOTAL_LIMIT": settings.student_train_save_total_limit,
        "STUDENT_TRAIN_SEED": settings.student_train_seed,
        "STUDENT_TRAIN_GRADIENT_CHECKPOINTING": settings.student_train_gradient_checkpointing,
        "STUDENT_TRAIN_FAIL_ON_TRUNCATION": settings.student_train_fail_on_truncation,
        "STUDENT_TRAIN_MERGE_AFTER_TRAIN": settings.student_train_merge_after_train,
        "STUDENT_TRAIN_OVERWRITE_OUTPUT": settings.student_train_overwrite_output,
        "STUDENT_TRAIN_RESUME_FROM_CHECKPOINT": settings.resume_checkpoint_path or "",
        "STUDENT_TRAIN_OFFLINE": settings.student_train_offline,
    }
    for key, value in values.items():
        if isinstance(value, bool):
            environment[key] = "true" if value else "false"
        else:
            environment[key] = str(value)
    if settings.student_train_device == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = settings.student_train_gpu_device_ids
    if settings.student_train_offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def build_launch_command(
    settings: StudentExtractorTrainSettings,
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    device_ids = settings.gpu_device_ids
    if settings.student_train_device == "gpu" and len(device_ids) > 1:
        return [
            python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(device_ids)}",
            "--module",
            "tools.train_student_extractor",
            "--worker",
        ]
    return [
        python_executable,
        "-m",
        "tools.train_student_extractor",
        "--worker",
    ]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        length += 1
    return length


def _normalize_token_ids(encoded: Any, *, context: str) -> list[int]:
    """Return plain token IDs across Transformers v4/v5 tokenizer outputs."""

    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise RuntimeError(f"{context} tokenizer output has no input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while (
        isinstance(encoded, (list, tuple))
        and len(encoded) == 1
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in encoded
    ):
        raise RuntimeError(
            f"{context} tokenizer output is not a one-dimensional integer sequence"
        )
    return [int(token_id) for token_id in encoded]


def tokenize_records(
    records: Iterable[DatasetRecord],
    *,
    tokenizer: Any,
    maximum_length: int,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, list[int]]], int]:
    tokenized: list[dict[str, list[int]]] = []
    truncated = 0
    for record in records:
        user_message = {"role": "user", "content": record.user_text}
        assistant_message = {"role": "assistant", "content": record.assistant_text}
        prompt_ids = _normalize_token_ids(
            tokenizer.apply_chat_template(
                [user_message], tokenize=True, add_generation_prompt=True
            ),
            context=f"Record {record.record_id} prompt",
        )
        full_ids = _normalize_token_ids(
            tokenizer.apply_chat_template(
                [user_message, assistant_message],
                tokenize=True,
                add_generation_prompt=False,
            ),
            context=f"Record {record.record_id} full conversation",
        )
        prefix_length = _common_prefix_length(prompt_ids, full_ids)
        if prefix_length < max(1, int(len(prompt_ids) * 0.80)):
            raise RuntimeError(
                f"Chat template prefix mismatch for record {record.record_id}. "
                "The installed Transformers/tokenizer version is incompatible."
            )
        prompt = full_ids[:prefix_length]
        target = full_ids[prefix_length:]
        if not target:
            # A few tokenizer/chat-template combinations can merge the boundary
            # between the generation prompt and the assistant response so that
            # the separately rendered sequences appear identical at token level.
            # The dataset reader has already validated assistant_text as a
            # non-empty JSON target.  Re-encode that target explicitly instead
            # of dropping an otherwise valid supervised example.
            prompt = prompt_ids
            target = _normalize_token_ids(
                tokenizer.encode(record.assistant_text, add_special_tokens=False),
                context=f"Record {record.record_id} explicit target",
            )
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None and (
                not target or target[-1] != eos_token_id
            ):
                target.append(eos_token_id)
            if not target:
                raise RuntimeError(
                    f"Record {record.record_id} has no supervised tokens after "
                    "explicit target encoding"
                )
        if len(target) >= maximum_length:
            raise RuntimeError(
                f"Target for record {record.record_id} uses {len(target)} tokens; "
                f"maximum sequence length is {maximum_length}."
            )
        if len(prompt) + len(target) > maximum_length:
            if fail_on_truncation:
                raise RuntimeError(
                    f"Record {record.record_id} uses {len(prompt) + len(target)} "
                    f"tokens, exceeding STUDENT_TRAIN_MAX_SEQUENCE_LENGTH="
                    f"{maximum_length}. Increase the limit or clean the input; the "
                    "tool will not silently remove source text."
                )
            truncated += 1
            prompt_budget = maximum_length - len(target)
            head_size = min(prompt_budget // 4, len(prompt))
            tail_size = prompt_budget - head_size
            prompt = prompt[:head_size] + prompt[-tail_size:]
        input_ids = prompt + target
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt) + target,
            }
        )
    return tokenized, truncated


def run_worker(settings: StudentExtractorTrainSettings) -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            __version__ as transformers_version,
            AutoModelForCausalLM,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "Student training dependencies are missing. Install "
            "requirements-student-training.txt inside the Linux training venv."
        ) from error

    if settings.student_train_device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "STUDENT_TRAIN_DEVICE=gpu but CUDA PyTorch is unavailable in this venv"
        )
    dataset_summary = validate_dataset(settings.dataset_path)
    train_records = read_training_records(settings.dataset_path / "train.jsonl")
    val_records = read_training_records(settings.dataset_path / "val.jsonl")
    set_seed(settings.student_train_seed)
    model_source = str(settings.model_path or settings.student_train_model_id)
    token = os.getenv("HF_TOKEN") or None
    common_load_options = {
        "token": token,
        "local_files_only": settings.student_train_offline,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_source, **common_load_options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_items, train_truncated = tokenize_records(
        train_records,
        tokenizer=tokenizer,
        maximum_length=settings.student_train_max_sequence_length,
        fail_on_truncation=settings.student_train_fail_on_truncation,
    )
    val_items, val_truncated = tokenize_records(
        val_records,
        tokenizer=tokenizer,
        maximum_length=settings.student_train_max_sequence_length,
        fail_on_truncation=settings.student_train_fail_on_truncation,
    )

    class TokenizedDataset(Dataset):
        def __init__(self, items: list[dict[str, list[int]]]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.items[index]

    class CompletionOnlyCollator:
        def __init__(self, pad_token_id: int) -> None:
            self.pad_token_id = pad_token_id

        def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
            maximum = max(len(feature["input_ids"]) for feature in features)
            input_ids: list[list[int]] = []
            attention_masks: list[list[int]] = []
            labels: list[list[int]] = []
            for feature in features:
                padding = maximum - len(feature["input_ids"])
                input_ids.append(
                    feature["input_ids"] + [self.pad_token_id] * padding
                )
                attention_masks.append(feature["attention_mask"] + [0] * padding)
                labels.append(feature["labels"] + [-100] * padding)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    use_cuda = settings.student_train_device == "gpu"
    precision = settings.student_train_precision
    if use_cuda and precision == "bf16" and not torch.cuda.is_bf16_supported():
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print("BF16 is unsupported by this CUDA runtime; falling back to FP16.")
        precision = "fp16"
    if not use_cuda:
        precision = "fp32"
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    transformers_major_version = int(transformers_version.split(".", maxsplit=1)[0])
    model_load_options = dict(common_load_options)
    model_load_options[
        "dtype" if transformers_major_version >= 5 else "torch_dtype"
    ] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_load_options)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=settings.student_train_lora_rank,
        lora_alpha=settings.student_train_lora_alpha,
        lora_dropout=settings.student_train_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(settings.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    effective_batch = (
        settings.student_train_batch_size
        * settings.student_train_gradient_accumulation_steps
        * world_size
    )
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(
            f"Student training: model={model_source}; train={len(train_items)}; "
            f"val={len(val_items)}; world_size={world_size}; precision={precision}; "
            f"effective_batch={effective_batch}; trainable="
            f"{trainable_parameters:,}/{total_parameters:,}"
        )
        model.print_trainable_parameters()
    training_argument_options: dict[str, Any] = {
        "output_dir": str(settings.output_path),
        "num_train_epochs": settings.student_train_epochs,
        "per_device_train_batch_size": settings.student_train_batch_size,
        "per_device_eval_batch_size": settings.student_train_eval_batch_size,
        "gradient_accumulation_steps": (
            settings.student_train_gradient_accumulation_steps
        ),
        "learning_rate": settings.student_train_learning_rate,
        "weight_decay": settings.student_train_weight_decay,
        "lr_scheduler_type": "cosine",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": settings.student_train_save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
        "tf32": use_cuda,
        "gradient_checkpointing": settings.student_train_gradient_checkpointing,
        "dataloader_num_workers": settings.student_train_num_workers,
        "ddp_find_unused_parameters": False if world_size > 1 else None,
        "report_to": "none",
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "seed": settings.student_train_seed,
        "data_seed": settings.student_train_seed,
    }
    training_argument_parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters
    if "overwrite_output_dir" in training_argument_parameters:
        training_argument_options["overwrite_output_dir"] = (
            settings.student_train_overwrite_output
        )
    if "warmup_ratio" in training_argument_parameters:
        training_argument_options["warmup_ratio"] = (
            settings.student_train_warmup_ratio
        )
    else:
        training_argument_options["warmup_steps"] = (
            settings.student_train_warmup_ratio
        )
    if "save_safetensors" in training_argument_parameters:
        training_argument_options["save_safetensors"] = True
    unsupported_arguments = sorted(
        set(training_argument_options) - set(training_argument_parameters)
    )
    if unsupported_arguments:
        raise RuntimeError(
            f"Transformers {transformers_version} has unsupported TrainingArguments: "
            f"{', '.join(unsupported_arguments)}"
        )
    training_arguments = TrainingArguments(**training_argument_options)
    callbacks = []
    if settings.student_train_early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=settings.student_train_early_stopping_patience
            )
        )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=TokenizedDataset(train_items),
        eval_dataset=TokenizedDataset(val_items),
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    resume_checkpoint = settings.resume_checkpoint_path
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    evaluation_metrics = trainer.evaluate()
    final_adapter = settings.output_path / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", evaluation_metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_source": model_source,
            "schema_version": STUDENT_SCHEMA_VERSION,
            "dataset_counts": dataset_summary.counts,
            "precision": precision,
            "world_size": world_size,
            "effective_batch_size": effective_batch,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "train_truncated": train_truncated,
            "val_truncated": val_truncated,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "train_metrics": train_result.metrics,
            "eval_metrics": evaluation_metrics,
            "final_adapter": str(final_adapter),
        }
        settings.output_path.mkdir(parents=True, exist_ok=True)
        (settings.output_path / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Final LoRA adapter: {final_adapter}")


def merge_adapter(settings: StudentExtractorTrainSettings) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            __version__ as transformers_version,
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as error:
        raise RuntimeError(
            "Student training dependencies are missing; cannot merge the adapter"
        ) from error
    adapter_path = settings.output_path / "final_adapter"
    if not adapter_path.is_dir():
        raise RuntimeError(f"Final adapter was not found: {adapter_path}")
    model_source = str(settings.model_path or settings.student_train_model_id)
    token = os.getenv("HF_TOKEN") or None
    use_cuda = settings.student_train_device == "gpu" and torch.cuda.is_available()
    transformers_major_version = int(transformers_version.split(".", maxsplit=1)[0])
    load_options: dict[str, Any] = {
        "token": token,
        "local_files_only": settings.student_train_offline,
        "trust_remote_code": False,
    }
    load_options[
        "dtype" if transformers_major_version >= 5 else "torch_dtype"
    ] = torch.float16 if use_cuda else torch.float32
    if use_cuda:
        load_options["device_map"] = {"": 0}
    base_model = AutoModelForCausalLM.from_pretrained(model_source, **load_options)
    adapter_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = adapter_model.merge_and_unload()
    merged_model.config.use_cache = True
    merged_path = settings.output_path / "merged_model"
    # Safetensors is the default in Transformers v4 and mandatory in v5.
    merged_model.save_pretrained(merged_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        token=token,
        local_files_only=settings.student_train_offline,
    )
    tokenizer.save_pretrained(merged_path)
    (merged_path / "merge_metadata.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_model": model_source,
                "adapter": str(adapter_path),
                "schema_version": STUDENT_SCHEMA_VERSION,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Merged student model: {merged_path}")


def _assert_output_is_safe(settings: StudentExtractorTrainSettings) -> None:
    output_path = settings.output_path
    if not output_path.exists():
        return
    has_previous_training = any(output_path.glob("checkpoint-*")) or (
        output_path / "final_adapter"
    ).exists()
    if (
        has_previous_training
        and settings.resume_checkpoint_path is None
        and not settings.student_train_overwrite_output
    ):
        raise RuntimeError(
            f"Training output already exists: {output_path}. Use --checkpoint PATH "
            "to resume or --overwrite-output for an intentional new run."
        )


def print_plan(
    settings: StudentExtractorTrainSettings,
    dataset_summary: DatasetSummary,
    command: list[str],
) -> None:
    device_description = (
        f"physical GPUs {settings.student_train_gpu_device_ids}"
        if settings.student_train_device == "gpu"
        else "CPU"
    )
    device_count = (
        len(settings.gpu_device_ids) if settings.student_train_device == "gpu" else 1
    )
    effective_batch = (
        settings.student_train_batch_size
        * settings.student_train_gradient_accumulation_steps
        * device_count
    )
    print(
        f"Dataset: {dataset_summary.dataset_dir}\n"
        f"Splits: {dataset_summary.counts}\n"
        f"Model: {settings.model_path or settings.student_train_model_id}\n"
        f"Device: {device_description}\n"
        f"Epochs: {settings.student_train_epochs}\n"
        f"Batch per GPU: {settings.student_train_batch_size}; "
        f"gradient accumulation: {settings.student_train_gradient_accumulation_steps}; "
        f"effective batch: {effective_batch}\n"
        f"Output: {settings.output_path}\n"
        f"Launch command: {subprocess.list2cmdline(command)}"
    )
    if settings.student_train_model_id.startswith("google/gemma") and not settings.model_path:
        print(
            "Gemma is gated: accept its Hugging Face license and run `hf auth "
            "login` (or export HF_TOKEN) before starting."
        )


def main() -> None:
    args = parse_args()
    settings = apply_cli_overrides(args)
    if args.worker:
        run_worker(settings)
        return
    if args.merge_only:
        merge_adapter(settings)
        return
    if settings.student_train_device == "gpu" and not settings.gpu_device_ids:
        raise RuntimeError(
            "GPU training requires at least one STUDENT_TRAIN_GPU_DEVICE_IDS value"
        )
    dataset_summary = validate_dataset(settings.dataset_path)
    command = build_launch_command(settings)
    print_plan(settings, dataset_summary, command)
    if args.dry_run:
        print("DRY RUN: dataset/schema validation passed; no files were changed.")
        return
    _assert_output_is_safe(settings)
    if args.prepare_only:
        print("PREPARE ONLY: training was not started.")
        return
    environment = settings_environment(settings)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)
    if settings.student_train_merge_after_train:
        merge_environment = environment.copy()
        if settings.student_train_device == "gpu":
            merge_environment["CUDA_VISIBLE_DEVICES"] = str(
                settings.gpu_device_ids[0]
            )
        merge_command = [
            sys.executable,
            "-m",
            "tools.train_student_extractor",
            "--merge-only",
        ]
        print(f"Merge command: {subprocess.list2cmdline(merge_command)}")
        subprocess.run(
            merge_command,
            check=True,
            cwd=PROJECT_ROOT,
            env=merge_environment,
        )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Student extractor training could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error
