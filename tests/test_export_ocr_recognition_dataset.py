import sys

import pytest

from tools.export_ocr_recognition_dataset import (
    _recommended_max_text_length,
    normalize_text,
    parse_args,
    padded_crop_box,
    split_document_ids,
    export_dataset,
)


def test_parse_args_accepts_dry_run_and_batch_size(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_ocr_recognition_dataset",
            "--dry-run",
            "--batch-size",
            "250",
        ],
    )

    args = parse_args()

    assert args.dry_run is True
    assert args.batch_size == 250
    assert args.resume is False


def test_export_rejects_invalid_batch_options_before_accessing_db(tmp_path) -> None:
    with pytest.raises(ValueError, match="batch-size"):
        export_dataset(output_dir=tmp_path / "out", config=None, batch_size=0)
    with pytest.raises(ValueError, match="cannot be combined"):
        export_dataset(
            output_dir=tmp_path / "out",
            config=None,
            dry_run=True,
            resume=True,
        )


def test_normalize_text_keeps_visible_unicode_and_removes_line_breaks() -> None:
    assert normalize_text("  Cộng\thòa\nViệt Nam  ") == "Cộng hòa Việt Nam"


def test_split_document_ids_is_deterministic_and_document_level() -> None:
    document_ids = [f"doc-{index}" for index in range(20)]
    first = split_document_ids(document_ids, val_ratio=0.1, test_ratio=0.1, seed=42)
    second = split_document_ids(
        list(reversed(document_ids)), val_ratio=0.1, test_ratio=0.1, seed=42
    )

    assert first == second
    assert set(first) == set(document_ids)
    assert list(first.values()).count("train") == 16
    assert list(first.values()).count("val") == 2
    assert list(first.values()).count("test") == 2


def test_padded_crop_box_uses_line_height_and_clamps_to_image() -> None:
    assert padded_crop_box(
        (1, 2, 100, 22),
        image_width=120,
        image_height=40,
        padding_ratio=0.1,
    ) == (0, 0, 102, 24)


def test_recommended_max_text_length_has_headroom() -> None:
    assert _recommended_max_text_length(138) == 160
