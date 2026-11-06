from __future__ import annotations

import json
from pathlib import Path

from app.schemas.student_extraction import (
    STUDENT_SCHEMA_VERSION,
)
from tools.student_extractor_train_config import (
    StudentExtractorTrainSettings,
    parse_gpu_device_ids,
)
from tools.train_student_extractor import (
    DatasetRecord,
    build_launch_command,
    tokenize_records,
    validate_dataset,
)


def write_split(path: Path, split: str, record_id: str) -> None:
    payload = {
        "id": record_id,
        "case_id": f"case-{record_id}",
        "document_id": f"document-{record_id}",
        "schema_version": STUDENT_SCHEMA_VERSION,
        "messages": [
            {"role": "user", "content": "DOCUMENT_TEXT:\nNo. 850-2026"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "n": {
                            "no": None,
                            "dt": None,
                            "pl": None,
                            "snd": None,
                        },
                        "a": [],
                    }
                ),
            },
        ],
    }
    (path / f"{split}.jsonl").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )


def test_validate_dataset_checks_all_three_splits(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        write_split(tmp_path, split, split)

    summary = validate_dataset(tmp_path)

    assert summary.counts == {"train": 1, "val": 1, "test": 1}
    assert summary.total == 3


def test_three_gpu_launch_uses_torchrun() -> None:
    settings = StudentExtractorTrainSettings(
        student_train_device="gpu",
        student_train_gpu_device_ids="2,3,4",
    )

    command = build_launch_command(settings, python_executable="python")

    assert command[:3] == ["python", "-m", "torch.distributed.run"]
    assert "--nproc_per_node=3" in command
    assert command[-3:] == [
        "--module",
        "tools.train_student_extractor",
        "--worker",
    ]


def test_completion_tokens_are_the_only_supervised_tokens() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if len(messages) == 1:
                return [10, 11, 12]
            return [10, 11, 12, 20, 21]

    records = [
        DatasetRecord(
            record_id="review-1",
            case_id="case-1",
            document_id="document-1",
            user_text="source",
            assistant_text="target",
        )
    ]

    tokenized, truncated = tokenize_records(
        records,
        tokenizer=FakeTokenizer(),
        maximum_length=32,
        fail_on_truncation=True,
    )

    assert truncated == 0
    assert tokenized[0]["input_ids"] == [10, 11, 12, 20, 21]
    assert tokenized[0]["labels"] == [-100, -100, -100, 20, 21]


def test_gpu_id_parser_rejects_duplicates() -> None:
    assert parse_gpu_device_ids("2, 3,4") == (2, 3, 4)
