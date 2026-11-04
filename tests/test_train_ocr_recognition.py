from __future__ import annotations

from pathlib import Path

from tools.train_ocr_recognition import (
    DatasetInfo,
    build_extended_character_dictionary,
    build_training_config,
    normalize_checkpoint_prefix,
)


def test_paddle_entrypoint_exists() -> None:
    assert (Path(__file__).parents[1] / "tools" / "paddleocr_entrypoint.py").is_file()


def dataset(tmp_path: Path) -> DatasetInfo:
    return DatasetInfo(
        dataset_dir=tmp_path,
        train_labels=tmp_path / "train.txt",
        val_labels=tmp_path / "val.txt",
        test_labels=tmp_path / "test.txt",
        train_count=100,
        val_count=10,
        test_count=10,
        recommended_max_text_length=96,
        characters=frozenset("Việt Nam"),
    )


def base_config() -> dict:
    return {
        "Global": {},
        "Optimizer": {"lr": {"learning_rate": 0.0005}},
        "Architecture": {
            "Head": {"head_list": [{"CTCHead": {}}, {"NRTRHead": {}}]}
        },
        "Train": {
            "dataset": {
                "data_dir": "old",
                "label_file_list": ["old.txt"],
                "transforms": [
                    {"RecConAug": {"image_shape": [48, 320, 3]}},
                    {"MultiLabelEncode": {}},
                ],
            },
            "sampler": {"scales": [[320, 48]], "first_bs": 128},
            "loader": {"batch_size_per_card": 128, "num_workers": 8},
        },
        "Eval": {
            "dataset": {
                "data_dir": "old",
                "label_file_list": ["old.txt"],
                "transforms": [
                    {"MultiLabelEncode": {}},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                ],
            },
            "loader": {"batch_size_per_card": 128, "num_workers": 4},
        },
    }


def test_build_training_config_overrides_dataset_and_model_shape(tmp_path) -> None:
    result = build_training_config(
        base_config(),
        dataset=dataset(tmp_path),
        output_dir=tmp_path / "output",
        pretrained_path=tmp_path / "pretrained.pdparams",
        checkpoint_prefix=None,
        character_dict_path=tmp_path / "dict.txt",
        device="gpu",
        epochs=50,
        batch_size=64,
        workers=2,
        learning_rate=0.0001,
        max_text_length=160,
        image_height=48,
        image_width=640,
        eval_batch_step=500,
        save_epoch_step=5,
        use_visualdl=False,
    )

    assert result["Global"]["use_gpu"] is True
    assert result["Global"]["epoch_num"] == 50
    assert result["Global"]["max_text_length"] == 160
    assert result["Train"]["dataset"]["data_dir"] == str(tmp_path)
    assert result["Train"]["sampler"]["first_bs"] == 64
    assert result["Train"]["sampler"]["scales"][1] == [640, 48]
    assert result["Eval"]["dataset"]["label_file_list"] == [
        str(tmp_path / "val.txt")
    ]
    assert result["Architecture"]["Head"]["head_list"][1]["NRTRHead"][
        "max_text_length"
    ] == 160


def test_normalize_checkpoint_prefix_accepts_pdparams_path(tmp_path) -> None:
    checkpoint = tmp_path / "latest.pdparams"
    checkpoint.write_bytes(b"weights")

    assert normalize_checkpoint_prefix(checkpoint) == tmp_path / "latest"


def test_extended_dictionary_preserves_base_order_and_appends_vietnamese(
    tmp_path,
) -> None:
    base = tmp_path / "base.txt"
    generated = tmp_path / "output" / "character_dict.generated.txt"
    base.write_text("A\na\nĐ\n", encoding="utf-8")

    added = build_extended_character_dictionary(
        base,
        frozenset("A Đặng"),
        generated,
    )

    assert added == ["g", "n", "ặ"]
    assert generated.read_text(encoding="utf-8").splitlines() == [
        "A",
        "a",
        "Đ",
        "g",
        "n",
        "ặ",
    ]


def test_extended_dictionary_preserves_official_duplicate_entries(tmp_path) -> None:
    base = tmp_path / "base.txt"
    generated = tmp_path / "character_dict.generated.txt"
    base.write_text("A\nA\na\n", encoding="utf-8")

    added = build_extended_character_dictionary(
        base,
        frozenset("Aạ"),
        generated,
    )

    assert added == ["ạ"]
    assert generated.read_text(encoding="utf-8").splitlines() == [
        "A",
        "A",
        "a",
        "ạ",
    ]

