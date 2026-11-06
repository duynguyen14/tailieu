from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.kie_ser_train_config import KieSerTrainSettings, parse_gpu_device_ids
from tools.train_kie_ser import (
    apply_warmup_training_argument,
    bio_entities,
    build_launch_command,
    calculate_bio_metrics,
    validate_dataset,
)


LABELS = ["O", "B-FULL_NAME", "I-FULL_NAME", "B-DATE_OF_BIRTH", "I-DATE_OF_BIRTH"]


def _write_dataset(root: Path) -> None:
    (root / "images").mkdir()
    (root / "labels.json").write_text(
        json.dumps({"bio_labels": LABELS}), encoding="utf-8"
    )
    for index, split in enumerate(("train", "val", "test"), start=1):
        image_path = root / "images" / f"{split}.png"
        Image.new("RGB", (100, 80), "white").save(image_path)
        payload = {
            "id": f"record-{split}",
            "document_id": f"document-{split}",
            "page_id": f"page-{split}",
            "page_number": index,
            "image_path": f"images/{split}.png",
            "image_width": 100,
            "image_height": 80,
            "words": ["Nguyen", "Van", "A", "01/01/2000"],
            "boxes": [
                [10, 10, 100, 100],
                [110, 10, 180, 100],
                [190, 10, 220, 100],
                [10, 150, 200, 250],
            ],
            "labels": [
                "B-FULL_NAME",
                "I-FULL_NAME",
                "I-FULL_NAME",
                "B-DATE_OF_BIRTH",
            ],
        }
        (root / f"{split}.jsonl").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )
    (root / "dataset_statistics.json").write_text(
        json.dumps({"dataset_fingerprint": "abc123"}), encoding="utf-8"
    )


def test_validate_dataset_checks_images_splits_and_labels(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    summary = validate_dataset(tmp_path)

    assert summary.counts == {"train": 1, "val": 1, "test": 1}
    assert summary.document_counts == {"train": 1, "val": 1, "test": 1}
    assert summary.entity_types == ["DATE_OF_BIRTH", "FULL_NAME"]
    assert summary.maximum_words == 4
    assert summary.dataset_fingerprint == "abc123"


def test_validate_dataset_rejects_document_leakage(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    validation_path = tmp_path / "val.jsonl"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["document_id"] = "document-train"
    validation_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Document leakage"):
        validate_dataset(tmp_path)


def test_bio_entity_metrics_use_exact_spans() -> None:
    labels = {
        0: "O",
        1: "B-FULL_NAME",
        2: "I-FULL_NAME",
        3: "B-DATE_OF_BIRTH",
    }
    references = [[1, 2, 0, 3, -100]]
    predictions = [[1, 2, 0, 0, 3]]

    metrics = calculate_bio_metrics(
        predictions,
        references,
        id_to_label=labels,
    )

    assert metrics["entity_precision"] == 1.0
    assert metrics["entity_recall"] == 0.5
    assert metrics["entity_f1"] == pytest.approx(2 / 3)
    assert metrics["entity_f1_full_name"] == 1.0
    assert metrics["entity_f1_date_of_birth"] == 0.0
    assert bio_entities(["I-FULL_NAME", "I-FULL_NAME", "O"]) == [
        ("FULL_NAME", 0, 2)
    ]


def test_three_gpu_launch_uses_torchrun() -> None:
    settings = KieSerTrainSettings(
        kie_train_device="gpu",
        kie_train_gpu_device_ids="2,3,4",
    )

    command = build_launch_command(settings, python_executable="python")

    assert command[:3] == ["python", "-m", "torch.distributed.run"]
    assert "--nproc_per_node=3" in command
    assert command[-3:] == ["--module", "tools.train_kie_ser", "--worker"]


def test_gpu_parser_rejects_duplicates() -> None:
    assert parse_gpu_device_ids("2, 3,4") == (2, 3, 4)
    with pytest.raises(ValueError, match="duplicate"):
        parse_gpu_device_ids("2,2")


def test_warmup_uses_ratio_when_supported() -> None:
    options: dict[str, object] = {}

    apply_warmup_training_argument(
        options,
        parameters={"warmup_ratio": object()},
        warmup_ratio=0.1,
        train_item_count=100,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        world_size=2,
        epochs=10,
    )

    assert options == {"warmup_ratio": 0.1}


def test_warmup_falls_back_to_steps_for_transformers_v5() -> None:
    options: dict[str, object] = {}

    apply_warmup_training_argument(
        options,
        parameters={"warmup_steps": object()},
        warmup_ratio=0.1,
        train_item_count=100,
        per_device_batch_size=2,
        gradient_accumulation_steps=2,
        world_size=2,
        epochs=10,
    )

    # 100 / 2 ranks / batch 2 / accumulation 2 = 13 updates per epoch.
    assert options == {"warmup_steps": 13}
