from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.core.config import PROJECT_ROOT
from tools.train_kie_ser import calculate_bio_metrics


DEFAULT_MODEL_DIR = Path("models/kie/ser/layoutlmv3_anotation_2_poc_v1")
DEFAULT_DATASET_DIR = Path("data/training/kie_ser_anotation_2_poc_v1")
DEFAULT_OUTPUT_FILE = Path("output/kie_ser_inference/predictions.jsonl")


@dataclass(frozen=True)
class PredictedEntity:
    label: str
    value: str
    word_start: int
    word_end: int
    word_indices: list[int]
    bbox: list[int]
    pixel_bbox: list[int] | None
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fine-tuned LayoutLMv3 SER model on exported OCR page records. "
            "This POC writes JSONL only and never changes annotation tables."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split was not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            words = record.get("words")
            boxes = record.get("boxes")
            if not isinstance(words, list) or not isinstance(boxes, list):
                raise ValueError(f"Record at {path}:{line_number} has no words/boxes arrays.")
            if len(words) != len(boxes) or not words:
                raise ValueError(
                    f"Record at {path}:{line_number} has inconsistent words/boxes."
                )
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    return records


def decode_bio_entities(
    words: Sequence[str],
    boxes: Sequence[Sequence[int]],
    labels: Sequence[str],
    confidences: Sequence[float],
    *,
    pixel_boxes: Sequence[Sequence[int]] | None = None,
) -> list[PredictedEntity]:
    if not (len(words) == len(boxes) == len(labels) == len(confidences)):
        raise ValueError("words, boxes, labels and confidences must have equal length.")

    entities: list[PredictedEntity] = []
    active_label: str | None = None
    active_indices: list[int] = []

    def flush() -> None:
        nonlocal active_label, active_indices
        if active_label is None or not active_indices:
            active_label = None
            active_indices = []
            return
        selected_boxes = [boxes[index] for index in active_indices]
        bbox = [
            min(item[0] for item in selected_boxes),
            min(item[1] for item in selected_boxes),
            max(item[2] for item in selected_boxes),
            max(item[3] for item in selected_boxes),
        ]
        pixel_bbox = None
        if pixel_boxes is not None:
            selected_pixels = [pixel_boxes[index] for index in active_indices]
            pixel_bbox = [
                min(item[0] for item in selected_pixels),
                min(item[1] for item in selected_pixels),
                max(item[2] for item in selected_pixels),
                max(item[3] for item in selected_pixels),
            ]
        entities.append(
            PredictedEntity(
                label=active_label,
                value=" ".join(words[index] for index in active_indices),
                word_start=active_indices[0],
                word_end=active_indices[-1] + 1,
                word_indices=list(active_indices),
                bbox=bbox,
                pixel_bbox=pixel_bbox,
                confidence=sum(confidences[index] for index in active_indices)
                / len(active_indices),
            )
        )
        active_label = None
        active_indices = []

    for index, bio_label in enumerate(labels):
        if bio_label == "O" or "-" not in bio_label:
            flush()
            continue
        prefix, entity_label = bio_label.split("-", 1)
        if prefix == "B" or active_label != entity_label:
            flush()
            active_label = entity_label
            active_indices = [index]
        else:
            active_indices.append(index)
    flush()
    return entities


class LayoutLmV3SerPredictor:
    def __init__(
        self,
        model_dir: Path,
        *,
        device: str,
        maximum_length: int,
    ) -> None:
        try:
            import torch
            from transformers import (
                LayoutLMv3ForTokenClassification,
                LayoutLMv3Processor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "KIE inference dependencies are missing. Install "
                "requirements-kie-inference.txt and a compatible PyTorch build."
            ) from exc

        if not model_dir.is_dir():
            raise FileNotFoundError(f"KIE model directory was not found: {model_dir}")
        self._torch = torch
        self.device = torch.device(device)
        self.maximum_length = maximum_length
        self.processor = LayoutLMv3Processor.from_pretrained(
            str(model_dir), apply_ocr=False, local_files_only=True
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.id_to_label = {
            int(index): label for index, label in self.model.config.id2label.items()
        }

    def predict(
        self,
        *,
        image_path: Path,
        words: list[str],
        boxes: list[list[int]],
    ) -> tuple[list[str], list[float], int, float]:
        from PIL import Image

        torch = self._torch
        started = time.perf_counter()
        with Image.open(image_path) as image:
            encoding = self.processor(
                images=image.convert("RGB"),
                text=words,
                boxes=boxes,
                padding="max_length",
                truncation=True,
                max_length=self.maximum_length,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
        word_predictions: dict[int, tuple[str, float]] = {}
        chunk_count = int(encoding["input_ids"].shape[0])
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if key != "overflow_to_sample_mapping"
        }
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**model_inputs).logits
            else:
                logits = self.model(**model_inputs).logits
            probabilities = torch.softmax(logits.float(), dim=-1)
            confidence_rows, predicted_rows = probabilities.max(dim=-1)

        for chunk_index in range(chunk_count):
            word_ids = encoding.word_ids(batch_index=chunk_index)
            previous_word_id = None
            for position, word_id in enumerate(word_ids):
                if word_id is None or word_id == previous_word_id:
                    previous_word_id = word_id
                    continue
                previous_word_id = word_id
                confidence = float(confidence_rows[chunk_index, position].item())
                label = self.id_to_label[
                    int(predicted_rows[chunk_index, position].item())
                ]
                current = word_predictions.get(word_id)
                if current is None or confidence > current[1]:
                    word_predictions[word_id] = (label, confidence)

        missing = [index for index in range(len(words)) if index not in word_predictions]
        if missing:
            raise RuntimeError(
                f"SER inference lost {len(missing)}/{len(words)} OCR words after "
                f"overflow chunking; first missing index={missing[0]}."
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000
        labels = [word_predictions[index][0] for index in range(len(words))]
        confidences = [word_predictions[index][1] for index in range(len(words))]
        return labels, confidences, chunk_count, latency_ms


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")
    if args.max_sequence_length < 32:
        raise ValueError("--max-sequence-length must be at least 32.")
    model_dir = _resolve(args.model_dir)
    dataset_dir = _resolve(args.dataset_dir)
    output_file = _resolve(args.output_file)
    if output_file.exists() and not args.force:
        raise RuntimeError(f"Output already exists: {output_file}. Add --force to replace it.")

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device_id)
    records = read_jsonl(dataset_dir / f"{args.split}.jsonl", args.limit)
    if not records:
        raise RuntimeError("No dataset records were found.")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for KIE inference.") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA PyTorch is unavailable.")
    effective_device = (
        "cuda:0"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    predictor = LayoutLmV3SerPredictor(
        model_dir,
        device=effective_device,
        maximum_length=args.max_sequence_length,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    true_sequences: list[list[int]] = []
    predicted_sequences: list[list[int]] = []
    label_to_id = {label: index for index, label in predictor.id_to_label.items()}
    with output_file.open("w", encoding="utf-8", newline="\n") as handle:
        for position, record in enumerate(records, start=1):
            image_path = Path(record["image_path"])
            if not image_path.is_absolute():
                image_path = dataset_dir / image_path
            words = [str(value) for value in record["words"]]
            boxes = [[int(number) for number in value] for value in record["boxes"]]
            pixel_boxes = record.get("pixel_boxes")
            labels, confidences, chunks, latency_ms = predictor.predict(
                image_path=image_path,
                words=words,
                boxes=boxes,
            )
            entities = decode_bio_entities(
                words,
                boxes,
                labels,
                confidences,
                pixel_boxes=pixel_boxes,
            )
            payload = {
                "id": record.get("id"),
                "document_id": record.get("document_id"),
                "page_id": record.get("page_id"),
                "page_number": record.get("page_number"),
                "predicted_labels": labels,
                "word_confidences": confidences,
                "entities": [entity.__dict__ for entity in entities],
                "chunks": chunks,
                "latency_ms": latency_ms,
            }
            if isinstance(record.get("labels"), list):
                payload["true_labels"] = record["labels"]
                true_sequences.append([label_to_id[label] for label in record["labels"]])
                predicted_sequences.append([label_to_id[label] for label in labels])
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            latencies.append(latency_ms)
            print(
                f"[{position}/{len(records)}] page={record.get('page_id')} "
                f"words={len(words)} entities={len(entities)} chunks={chunks} "
                f"time={latency_ms:.0f}ms"
            )

    metrics: dict[str, Any] = {}
    if true_sequences:
        metrics = calculate_bio_metrics(
            predicted_sequences,
            true_sequences,
            id_to_label=predictor.id_to_label,
        )
    summary = {
        "model_dir": str(model_dir),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "device": effective_device,
        "pages": len(records),
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": _percentile(latencies, 0.95),
            "minimum": min(latencies),
            "maximum": max(latencies),
        },
        "metrics": metrics,
        "predictions": str(output_file),
    }
    summary_file = output_file.with_suffix(".summary.json")
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("KIE SER inference finished:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    try:
        run(parse_args())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"KIE SER inference could not start: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
