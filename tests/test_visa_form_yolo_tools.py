from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from app.services.visa_form_annotation import FIELD_CODES
from tools.export_visa_form_yolo_dataset import (
    pixel_bbox_to_yolo,
    split_document_ids,
)
from tools.train_visa_form_yolo import validate_dataset


def test_pixel_bbox_to_yolo() -> None:
    assert pixel_bbox_to_yolo((100, 200, 500, 600), 1000, 1000) == pytest.approx(
        (0.3, 0.4, 0.4, 0.4)
    )
    with pytest.raises(ValueError, match="exceeds image dimensions"):
        pixel_bbox_to_yolo((-1, 0, 20, 20), 1000, 1000)


def test_split_document_ids_is_deterministic_and_has_no_leakage() -> None:
    identifiers = [f"doc-{index}" for index in range(10)]
    first = split_document_ids(
        identifiers,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
    second = split_document_ids(
        list(reversed(identifiers)),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
    assert first == second
    assert set(first) == set(identifiers)
    assert list(first.values()).count("train") == 8
    assert list(first.values()).count("val") == 1
    assert list(first.values()).count("test") == 1


def _write_record(dataset: Path, split: str, index: int, document_id: str) -> dict:
    image = dataset / "images" / split / f"page-{index}.png"
    label = dataset / "labels" / split / f"page-{index}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color="white").save(image)
    label.write_text(
        "\n".join(
            f"{class_id} 0.50000000 0.50000000 0.10000000 0.10000000"
            for class_id in range(len(FIELD_CODES))
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "document_id": document_id,
        "page_id": f"page-{index}",
        "split": split,
        "image_path": image.relative_to(dataset).as_posix(),
        "label_path": label.relative_to(dataset).as_posix(),
    }


def test_validate_dataset_accepts_canonical_19_class_dataset(tmp_path: Path) -> None:
    (tmp_path / "classes.json").write_text(
        json.dumps({"labels": list(FIELD_CODES)}), encoding="utf-8"
    )
    records = [
        _write_record(tmp_path, "train", 1, "doc-train"),
        _write_record(tmp_path, "val", 2, "doc-val"),
        _write_record(tmp_path, "test", 3, "doc-test"),
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = validate_dataset(tmp_path)
    assert summary["counts"] == {"train": 1, "val": 1, "test": 1}
    assert summary["documents"] == 3
    assert len(summary["boxes_by_class"]) == 19


def test_validate_dataset_rejects_document_leakage(tmp_path: Path) -> None:
    (tmp_path / "classes.json").write_text(
        json.dumps({"labels": list(FIELD_CODES)}), encoding="utf-8"
    )
    records = [
        _write_record(tmp_path, "train", 1, "same-doc"),
        _write_record(tmp_path, "val", 2, "same-doc"),
        _write_record(tmp_path, "test", 3, "doc-test"),
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="leaks across dataset splits"):
        validate_dataset(tmp_path)
