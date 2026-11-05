from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.core.config import PROJECT_ROOT
from tools.ocr_recognition_train_config import (
    OcrRecognitionTrainSettings,
    parse_gpu_device_ids,
)


@dataclass(frozen=True)
class DatasetInfo:
    dataset_dir: Path
    train_labels: Path
    val_labels: Path
    test_labels: Path
    train_count: int
    val_count: int
    test_count: int
    recommended_max_text_length: int
    characters: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune latin_PP-OCRv5_mobile_rec with the human-approved "
            "PaddleOCR recognition dataset."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("gpu", "cpu"), default=None)
    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated physical GPU IDs. Multiple IDs use Paddle "
            "Distributed on Linux, for example: --gpus 2,3,4."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Shallow-clone the official PaddleOCR repository if it is absent.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and write the generated config without starting training.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download missing official pretrained weights.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not export best_accuracy to a Paddle inference model after training.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_label_file(path: Path, dataset_dir: Path) -> tuple[int, set[str]]:
    count = 0
    characters: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            try:
                image_name, text = line.split("\t", 1)
            except ValueError as error:
                raise RuntimeError(
                    f"Invalid Paddle label at {path}:{line_number}; expected image<TAB>text"
                ) from error
            image_path = (dataset_dir / image_name).resolve()
            if not image_path.is_relative_to(dataset_dir):
                raise RuntimeError(f"Dataset image escapes root at {path}:{line_number}")
            if not image_path.is_file():
                raise RuntimeError(
                    f"Dataset crop is missing at {path}:{line_number}: {image_name}"
                )
            if not text:
                raise RuntimeError(f"Empty OCR label at {path}:{line_number}")
            characters.update(text)
            count += 1
    if count == 0:
        raise RuntimeError(f"Paddle label file contains no records: {path}")
    return count, characters


def inspect_dataset(dataset_dir: Path) -> DatasetInfo:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise RuntimeError(f"OCR recognition dataset was not found: {dataset_dir}")
    label_paths = {
        split: dataset_dir / f"{split}.txt" for split in ("train", "val", "test")
    }
    for path in label_paths.values():
        if not path.is_file():
            raise RuntimeError(f"Dataset label file was not found: {path}")

    counts: dict[str, int] = {}
    characters: set[str] = set()
    for split, path in label_paths.items():
        count, split_characters = _read_label_file(path, dataset_dir)
        counts[split] = count
        characters.update(split_characters)

    statistics_path = dataset_dir / "dataset_statistics.json"
    recommended = 160
    if statistics_path.is_file():
        statistics = json.loads(statistics_path.read_text(encoding="utf-8-sig"))
        recommended = int(
            statistics.get("text_length", {}).get(
                "recommended_max_text_length", recommended
            )
        )
    return DatasetInfo(
        dataset_dir=dataset_dir,
        train_labels=label_paths["train"],
        val_labels=label_paths["val"],
        test_labels=label_paths["test"],
        train_count=counts["train"],
        val_count=counts["val"],
        test_count=counts["test"],
        recommended_max_text_length=recommended,
        characters=frozenset(characters),
    )


def bootstrap_paddleocr(repository: Path) -> None:
    repository.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/PaddlePaddle/PaddleOCR.git",
            str(repository),
        ],
        check=True,
    )


def ensure_paddleocr_repository(repository: Path, *, bootstrap: bool) -> None:
    train_script = repository / "tools" / "train.py"
    if train_script.is_file():
        return
    if not bootstrap:
        raise RuntimeError(
            f"PaddleOCR source checkout was not found: {repository}. "
            "Run again with --bootstrap or set OCR_TRAIN_PADDLEOCR_DIR."
        )
    if repository.exists() and any(repository.iterdir()):
        raise RuntimeError(
            f"Cannot bootstrap into a non-empty directory: {repository}"
        )
    bootstrap_paddleocr(repository)
    if not train_script.is_file():
        raise RuntimeError("PaddleOCR clone completed but tools/train.py is missing.")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    downloaded = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or 0)
        with temporary.open("wb") as target:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                target.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"Downloading pretrained weights: "
                        f"{downloaded / 1024 / 1024:.1f}/"
                        f"{total / 1024 / 1024:.1f} MB",
                        end="\r",
                        flush=True,
                    )
    print()
    if downloaded < 1024 * 1024:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Downloaded pretrained model is unexpectedly small.")
    temporary.replace(destination)


def normalize_checkpoint_prefix(path: Path) -> Path:
    path = path.resolve()
    if path.suffix in {".pdparams", ".pdopt", ".states"}:
        path = path.with_suffix("")
    if not path.with_suffix(".pdparams").is_file():
        raise RuntimeError(
            f"Paddle checkpoint was not found: {path.with_suffix('.pdparams')}"
        )
    return path


def load_character_dictionary_entries(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"PaddleOCR character dictionary was not found: {path}")
    entries = [
        line.rstrip("\r\n")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.rstrip("\r\n")
    ]
    return entries


def load_character_dictionary(path: Path) -> set[str]:
    return set(load_character_dictionary_entries(path))


def build_extended_character_dictionary(
    base_dictionary_path: Path,
    dataset_characters: frozenset[str],
    output_path: Path,
) -> list[str]:
    """Append dataset-only characters while preserving every pretrained index."""

    entries = load_character_dictionary_entries(base_dictionary_path)
    existing = set(entries)
    added = sorted(
        character
        for character in dataset_characters
        if character != " " and character not in existing
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{character}\n" for character in [*entries, *added]),
        encoding="utf-8",
    )
    return added


def validate_characters(
    dataset_characters: frozenset[str],
    dictionary_path: Path,
    *,
    fail_on_unknown: bool,
) -> list[str]:
    supported = load_character_dictionary(dictionary_path)
    unknown = sorted(
        character
        for character in dataset_characters
        if character != " " and character not in supported
    )
    if unknown and fail_on_unknown:
        preview = " ".join(f"U+{ord(char):04X}={char!r}" for char in unknown[:30])
        raise RuntimeError(
            "Dataset contains characters absent from ppocrv5_latin_dict.txt: "
            f"{preview}. Normalize/remove these labels or intentionally extend "
            "the dictionary before training."
        )
    return unknown


def _set_transform_value(transforms: list[dict], name: str, key: str, value: Any) -> None:
    for transform in transforms:
        if name in transform:
            config = transform[name]
            if config is None:
                config = {}
                transform[name] = config
            config[key] = value


def build_training_config(
    base: dict[str, Any],
    *,
    dataset: DatasetInfo,
    output_dir: Path,
    pretrained_path: Path,
    checkpoint_prefix: Path | None,
    character_dict_path: Path,
    device: str,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    distributed: bool,
    workers: int,
    learning_rate: float,
    max_text_length: int,
    image_height: int,
    image_width: int,
    eval_batch_step: int,
    save_epoch_step: int,
    use_visualdl: bool,
) -> dict[str, Any]:
    global_config = base["Global"]
    global_config.update(
        {
            "use_gpu": device == "gpu",
            "epoch_num": epochs,
            "save_model_dir": str(output_dir),
            "save_epoch_step": save_epoch_step,
            "eval_batch_step": [0, eval_batch_step],
            "pretrained_model": "" if checkpoint_prefix else str(pretrained_path),
            "checkpoints": str(checkpoint_prefix) if checkpoint_prefix else "",
            "save_inference_dir": str(output_dir / "inference"),
            "character_dict_path": str(character_dict_path),
            "max_text_length": max_text_length,
            "distributed": distributed,
            "use_visualdl": use_visualdl,
            "save_res_path": str(output_dir / "predicts.txt"),
            "d2s_train_image_shape": [3, image_height, image_width],
        }
    )
    base["Optimizer"]["lr"]["learning_rate"] = learning_rate

    for head in base["Architecture"]["Head"]["head_list"]:
        if "NRTRHead" in head:
            head["NRTRHead"]["max_text_length"] = max_text_length

    train = base["Train"]
    train["dataset"]["data_dir"] = str(dataset.dataset_dir)
    train["dataset"]["label_file_list"] = [str(dataset.train_labels)]
    train["sampler"]["scales"] = [
        [image_width, 32],
        [image_width, image_height],
        [image_width, 64],
    ]
    train["sampler"]["first_bs"] = batch_size
    train["loader"]["batch_size_per_card"] = batch_size
    train["loader"]["num_workers"] = workers
    train_transforms = train["dataset"]["transforms"]
    _set_transform_value(
        train_transforms, "RecConAug", "image_shape", [image_height, image_width, 3]
    )
    _set_transform_value(
        train_transforms, "RecConAug", "max_text_length", max_text_length
    )
    _set_transform_value(
        train_transforms, "MultiLabelEncode", "max_text_length", max_text_length
    )

    evaluation = base["Eval"]
    evaluation["dataset"]["data_dir"] = str(dataset.dataset_dir)
    evaluation["dataset"]["label_file_list"] = [str(dataset.val_labels)]
    # The official PP-OCRv5 config uses eval_mode=True, which produces
    # variable-width tensors. They cannot be stacked into batches larger than
    # one by Paddle's default collate function.
    evaluation["loader"]["batch_size_per_card"] = eval_batch_size
    evaluation["loader"]["num_workers"] = workers
    eval_transforms = evaluation["dataset"]["transforms"]
    _set_transform_value(
        eval_transforms, "MultiLabelEncode", "max_text_length", max_text_length
    )
    _set_transform_value(
        eval_transforms,
        "RecResizeImg",
        "image_shape",
        [3, image_height, image_width],
    )
    return base


def build_train_command(
    repository: Path,
    generated_config_path: Path,
    output_dir: Path,
    gpu_device_ids: tuple[int, ...],
) -> list[str]:
    paddle_script = [
        str(PROJECT_ROOT / "tools" / "paddleocr_entrypoint.py"),
        str(repository / "tools" / "train.py"),
        "-c",
        str(generated_config_path),
    ]
    if len(gpu_device_ids) <= 1:
        return [sys.executable, *paddle_script]
    return [
        sys.executable,
        "-m",
        "paddle.distributed.launch",
        f"--log_dir={output_dir / 'distributed_logs'}",
        f"--gpus={','.join(str(device_id) for device_id in gpu_device_ids)}",
        *paddle_script,
    ]


def check_paddle_runtime(device: str, gpu_device_ids: tuple[int, ...]) -> None:
    code = (
        "import paddle; "
        "print('Paddle:', paddle.__version__); "
        "print('CUDA:', paddle.is_compiled_with_cuda()); "
        "print('GPU count:', paddle.device.cuda.device_count())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Paddle runtime check failed:\n" + (result.stderr or result.stdout)
        )
    print(result.stdout.strip())
    if device == "gpu" and "CUDA: True" not in result.stdout:
        raise RuntimeError(
            "OCR_TRAIN_DEVICE=gpu but the active Paddle installation has no CUDA."
        )
    if device == "gpu":
        gpu_count_line = next(
            (line for line in result.stdout.splitlines() if line.startswith("GPU count:")),
            None,
        )
        if gpu_count_line is None:
            raise RuntimeError("Paddle runtime did not report a GPU count")
        gpu_count = int(gpu_count_line.split(":", 1)[1].strip())
        invalid = [device_id for device_id in gpu_device_ids if device_id >= gpu_count]
        if invalid:
            raise RuntimeError(
                f"Requested GPU IDs do not exist: {invalid}; detected {gpu_count} GPU(s)."
            )


def run(args: argparse.Namespace) -> None:
    settings = OcrRecognitionTrainSettings()
    dataset_dir = (
        resolve_project_path(args.dataset_dir)
        if args.dataset_dir is not None
        else settings.dataset_path
    )
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else settings.output_path
    )
    device = args.device or settings.ocr_train_device
    gpu_device_ids = (
        parse_gpu_device_ids(args.gpus)
        if args.gpus is not None
        else settings.gpu_device_ids
    )
    if device == "cpu":
        gpu_device_ids = ()
    elif not gpu_device_ids:
        raise RuntimeError("GPU training requires at least one GPU device ID")
    if len(gpu_device_ids) > 1 and os.name == "nt":
        raise RuntimeError(
            "PaddleOCR multi-GPU training is supported by this tool on Linux only."
        )
    distributed = device == "gpu" and len(gpu_device_ids) > 1
    batch_size = args.batch_size or settings.ocr_train_batch_size
    epochs = args.epochs or settings.ocr_train_epochs
    if batch_size < 1 or epochs < 1:
        raise ValueError("--batch-size and --epochs must be at least 1")

    dataset = inspect_dataset(dataset_dir)
    repository = settings.paddleocr_path
    ensure_paddleocr_repository(
        repository,
        bootstrap=args.bootstrap or settings.ocr_train_bootstrap_repo,
    )
    base_config_path = repository / settings.ocr_train_config_relative_path
    if not base_config_path.is_file():
        raise RuntimeError(f"Official PaddleOCR config was not found: {base_config_path}")
    base_character_dict_path = (
        repository / "ppocr" / "utils" / "dict" / "ppocrv5_latin_dict.txt"
    )
    unknown = validate_characters(
        dataset.characters,
        base_character_dict_path,
        fail_on_unknown=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    character_dict_path = output_dir / "character_dict.generated.txt"
    if unknown and not settings.ocr_train_extend_character_dictionary:
        if settings.ocr_train_fail_on_unknown_characters:
            preview = " ".join(
                f"U+{ord(character):04X}={character!r}" for character in unknown[:30]
            )
            raise RuntimeError(
                "Dataset contains characters absent from ppocrv5_latin_dict.txt: "
                f"{preview}. Enable OCR_TRAIN_EXTEND_CHARACTER_DICTIONARY or "
                "provide labels covered by the base dictionary."
            )
        character_dict_path = base_character_dict_path
        added_characters: list[str] = []
    else:
        added_characters = build_extended_character_dictionary(
            base_character_dict_path,
            dataset.characters,
            character_dict_path,
        )
        if added_characters:
            print(
                "Extended Latin dictionary with "
                f"{len(added_characters)} dataset character(s): "
                + " ".join(added_characters)
            )

    pretrained_path = settings.pretrained_path
    if not pretrained_path.is_file():
        if args.skip_download or not settings.ocr_train_auto_download_pretrained:
            raise RuntimeError(
                f"Pretrained model was not found: {pretrained_path}"
            )
        download_file(settings.ocr_train_pretrained_url, pretrained_path)

    checkpoint = args.checkpoint or settings.checkpoint_path
    checkpoint_prefix = (
        normalize_checkpoint_prefix(resolve_project_path(checkpoint))
        if checkpoint is not None
        else None
    )
    if checkpoint_prefix is None and (output_dir / "best_accuracy.pdparams").exists():
        raise RuntimeError(
            f"Training output already contains a model: {output_dir}. "
            "Use a new output version or pass --checkpoint to resume."
        )

    max_text_length = max(
        settings.ocr_train_max_text_length,
        dataset.recommended_max_text_length,
    )
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    generated = build_training_config(
        base_config,
        dataset=dataset,
        output_dir=output_dir,
        pretrained_path=pretrained_path,
        checkpoint_prefix=checkpoint_prefix,
        character_dict_path=character_dict_path,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=settings.ocr_train_eval_batch_size,
        distributed=distributed,
        workers=settings.ocr_train_num_workers,
        learning_rate=settings.ocr_train_learning_rate,
        max_text_length=max_text_length,
        image_height=settings.ocr_train_image_height,
        image_width=settings.ocr_train_image_width,
        eval_batch_step=settings.ocr_train_eval_batch_step,
        save_epoch_step=settings.ocr_train_save_epoch_step,
        use_visualdl=settings.ocr_train_use_visualdl,
    )
    generated_config_path = output_dir / "train_config.generated.yml"
    generated_config_path.write_text(
        yaml.safe_dump(generated, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    train_command = build_train_command(
        repository,
        generated_config_path,
        output_dir,
        gpu_device_ids,
    )
    plan = {
        "model": settings.ocr_train_model_name,
        "dataset": str(dataset.dataset_dir),
        "output": str(output_dir),
        "device": device,
        "gpu_device_ids": list(gpu_device_ids),
        "distributed": distributed,
        "world_size": len(gpu_device_ids) if device == "gpu" else 1,
        "epochs": epochs,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * max(1, len(gpu_device_ids)),
        "eval_batch_size": settings.ocr_train_eval_batch_size,
        "max_text_length": max_text_length,
        "image_shape": [3, settings.ocr_train_image_height, settings.ocr_train_image_width],
        "dataset_counts": {
            "train": dataset.train_count,
            "val": dataset.val_count,
            "test": dataset.test_count,
        },
        "unknown_characters": unknown,
        "added_characters": added_characters,
        "character_dictionary": str(character_dict_path),
        "checkpoint": str(checkpoint_prefix) if checkpoint_prefix else None,
        "train_command": train_command,
    }
    (output_dir / "training_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Prepared {settings.ocr_train_model_name}: "
        f"train={dataset.train_count}, val={dataset.val_count}, "
        f"test={dataset.test_count}, max_text_length={max_text_length}"
    )
    print(f"Generated config: {generated_config_path}")
    print(
        "Training topology: "
        f"distributed={distributed}; "
        f"gpus={','.join(str(device_id) for device_id in gpu_device_ids) or 'none'}; "
        f"batch_per_gpu={batch_size}; "
        f"effective_batch={batch_size * max(1, len(gpu_device_ids))}"
    )
    print("Command: " + subprocess.list2cmdline(train_command))
    if args.prepare_only:
        print("PREPARE ONLY: training was not started.")
        return

    check_paddle_runtime(device, gpu_device_ids)
    environment = os.environ.copy()
    if device == "gpu" and not distributed:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_device_ids[0])
    elif distributed:
        # --gpus uses physical IDs; avoid an inherited mask remapping them.
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run(train_command, cwd=repository, env=environment, check=True)

    should_export = settings.ocr_train_export_after_train and not args.no_export
    if not should_export:
        return
    best_prefix = output_dir / "best_accuracy"
    if not best_prefix.with_suffix(".pdparams").is_file():
        raise RuntimeError(
            "Training finished but best_accuracy.pdparams was not found; "
            "review the PaddleOCR training log."
        )
    export_command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "paddleocr_entrypoint.py"),
        str(repository / "tools" / "export_model.py"),
        "-c",
        str(generated_config_path),
        "-o",
        f"Global.pretrained_model={best_prefix}",
        f"Global.checkpoints=",
        f"Global.distributed=False",
        f"Global.save_inference_dir={output_dir / 'inference'}",
    ]
    print("Export command: " + subprocess.list2cmdline(export_command))
    export_environment = environment.copy()
    if device == "gpu":
        export_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_device_ids[0])
    subprocess.run(
        export_command,
        cwd=repository,
        env=export_environment,
        check=True,
    )
    print(f"Inference model exported to: {output_dir / 'inference'}")


def main() -> None:
    try:
        run(parse_args())
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"OCR recognition training could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()

