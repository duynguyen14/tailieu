from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.visa_form_fields import FIELD_CODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO visa-form field detector.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/training/visa_form_yolo_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/visa_form_yolo_v1"),
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpus", default="0", help="Physical GPU IDs, e.g. 2 or 2,3,4")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", choices=("ram", "disk", "false"), default="ram")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_classes(dataset_dir: Path) -> list[str]:
    path = dataset_dir / "classes.json"
    if not path.is_file():
        raise RuntimeError(f"Missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if labels != list(FIELD_CODES):
        raise RuntimeError(
            "Dataset class order does not match the 19 canonical visa-form fields."
        )
    return [str(item) for item in labels]


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    if not dataset_dir.is_dir():
        raise RuntimeError(f"Dataset directory does not exist: {dataset_dir}")
    labels = load_classes(dataset_dir)
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing {manifest_path}")

    counts: Counter[str] = Counter()
    box_counts: Counter[str] = Counter()
    document_splits: dict[str, str] = {}
    seen_pages: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        split = str(row.get("split", ""))
        if split not in {"train", "val", "test"}:
            raise RuntimeError(f"manifest.jsonl:{line_number}: invalid split {split!r}")
        document_id = str(row["document_id"])
        previous = document_splits.setdefault(document_id, split)
        if previous != split:
            raise RuntimeError(f"Document {document_id} leaks across dataset splits.")
        page_id = str(row["page_id"])
        if page_id in seen_pages:
            raise RuntimeError(f"Duplicate page_id in manifest: {page_id}")
        seen_pages.add(page_id)
        image_path = (dataset_dir / str(row["image_path"])).resolve()
        label_path = (dataset_dir / str(row["label_path"])).resolve()
        if not image_path.is_relative_to(dataset_dir) or not image_path.is_file():
            raise RuntimeError(f"Missing or unsafe image path: {row['image_path']}")
        if not label_path.is_relative_to(dataset_dir) or not label_path.is_file():
            raise RuntimeError(f"Missing or unsafe label path: {row['label_path']}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except OSError as error:
            raise RuntimeError(f"Invalid image {image_path}: {error}") from error
        lines = [item for item in label_path.read_text(encoding="utf-8").splitlines() if item]
        if len(lines) != len(labels):
            raise RuntimeError(
                f"{label_path}: expected {len(labels)} boxes, found {len(lines)}."
            )
        observed_classes: set[int] = set()
        for label_line in lines:
            parts = label_line.split()
            if len(parts) != 5:
                raise RuntimeError(f"Invalid YOLO row in {label_path}: {label_line!r}")
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
            if class_id < 0 or class_id >= len(labels):
                raise RuntimeError(f"Invalid class ID {class_id} in {label_path}")
            if class_id in observed_classes:
                raise RuntimeError(f"Duplicate class ID {class_id} in {label_path}")
            if any(value <= 0 or value > 1 for value in coordinates):
                raise RuntimeError(f"Invalid normalized bbox in {label_path}: {label_line!r}")
            observed_classes.add(class_id)
            box_counts[labels[class_id]] += 1
        counts[split] += 1

    if not counts["train"]:
        raise RuntimeError("Training split is empty.")
    if not counts["val"]:
        raise RuntimeError("Validation split is empty.")
    if not counts["test"]:
        raise RuntimeError("Test split is empty.")
    return {
        "counts": {split: counts[split] for split in ("train", "val", "test")},
        "documents": len(document_splits),
        "classes": labels,
        "boxes_by_class": dict(box_counts),
    }


def write_runtime_yaml(dataset_dir: Path, output_dir: Path) -> Path:
    path = output_dir / "data.generated.yaml"
    lines = [
        f"path: {json.dumps(dataset_dir.as_posix())}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {code}" for index, code in enumerate(FIELD_CODES))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def json_metrics(metrics: Any) -> dict[str, float]:
    values = getattr(metrics, "results_dict", {}) or {}
    result: dict[str, float] = {}
    for key, value in values.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def run(args: argparse.Namespace) -> None:
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 320:
        raise ValueError("epochs/batch-size must be positive and image-size must be >= 320.")
    if args.workers < 0 or args.patience < 0:
        raise ValueError("workers and patience cannot be negative.")
    summary = validate_dataset(dataset_dir)
    device = "cpu" if args.device == "cpu" else args.gpus.strip()
    if args.device == "gpu" and not device:
        raise ValueError("--gpus cannot be empty for GPU training.")
    initialization = resolve_path(args.resume) if args.resume else args.model

    print(
        f"Dataset: {dataset_dir}\n"
        f"Splits: {summary['counts']}; documents: {summary['documents']}\n"
        f"Classes: {len(summary['classes'])}; boxes: {sum(summary['boxes_by_class'].values())}\n"
        f"Initialization: {initialization}\n"
        f"Device: {device}; epochs: {args.epochs}; batch: {args.batch_size}; "
        f"imgsz: {args.image_size}; cache: {args.cache}\n"
        f"Output: {output_dir}"
    )
    if args.dry_run:
        print("DRY RUN: dataset and every YOLO label were validated; no files changed.")
        return
    if args.prepare_only:
        print("PREPARE ONLY: validation passed; training and model download were not started.")
        return
    if args.resume is not None and not initialization.is_file():
        raise RuntimeError(f"Resume checkpoint does not exist: {initialization}")
    if args.resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Fresh training output is not empty: {output_dir}. Use a new directory or --resume."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is missing. Install requirements-visa-form-yolo-training.txt."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = write_runtime_yaml(dataset_dir, output_dir)
    model = YOLO(str(initialization))
    cache: str | bool = False if args.cache == "false" else args.cache
    train_result = model.train(
        data=str(data_yaml),
        project=str(output_dir.parent),
        name=output_dir.name,
        # The directory is known to be empty for a fresh run except for the
        # generated data YAML written immediately above. Keep the exact path
        # instead of allowing Ultralytics to silently increment its name.
        exist_ok=True,
        resume=bool(args.resume),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.image_size,
        device=device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        cache=cache,
        plots=True,
        val=True,
    )
    best_path = Path(model.trainer.best).resolve()
    if not best_path.is_file():
        raise RuntimeError(f"Ultralytics did not create best.pt: {best_path}")
    best_model = YOLO(str(best_path))
    test_result = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=device,
        workers=args.workers,
        plots=True,
    )
    onnx_path: str | None = None
    if args.export_onnx:
        exported = best_model.export(format="onnx", imgsz=args.image_size, dynamic=True)
        onnx_path = str(Path(exported).resolve())
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "dataset": summary,
        "initialization": str(initialization),
        "best_checkpoint": str(best_path),
        "onnx_model": onnx_path,
        "train_metrics": json_metrics(train_result),
        "test_metrics": json_metrics(test_result),
        "settings": {
            "device": device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "patience": args.patience,
            "seed": args.seed,
            "cache": args.cache,
        },
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best YOLO checkpoint: {best_path}")
    print(f"Test metrics: {json.dumps(payload['test_metrics'], ensure_ascii=False)}")
    if onnx_path:
        print(f"ONNX model: {onnx_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Visa-form YOLO training could not start: {error}", file=sys.stderr)
        raise SystemExit(2) from error
