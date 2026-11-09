from __future__ import annotations

import pytest

from tools.generate_student_extractions import local_model_id, normalize_student_target


def test_normalize_student_target_fills_only_known_missing_keys() -> None:
    normalized = normalize_student_target(
        {
            "n": {"no": "2026-01", "unexpected": "drop-me"},
            "a": [{"nm": "ALICE", "pp": "P123", "unexpected": "drop-me"}],
            "unexpected": "drop-me",
        }
    )

    assert normalized["n"] == {
        "no": "2026-01",
        "snd": None,
        "rt": None,
    }
    assert normalized["a"] == [
        {
            "nm": "ALICE",
            "dob": None,
            "sex": None,
            "nat": None,
            "pp": "P123",
            "pt": None,
            "pid": None,
            "pex": None,
            "vt": None,
            "ne": None,
            "in": None,
            "out": None,
        }
    ]


@pytest.mark.parametrize(
    "payload,error",
    [
        ([], "root"),
        ({"n": [], "a": []}, "n"),
        ({"n": {}, "a": {}}, "a"),
        ({"n": {}, "a": ["bad"]}, "a[0]"),
    ],
)
def test_normalize_student_target_rejects_invalid_shapes(payload, error: str) -> None:
    with pytest.raises(ValueError, match=error.replace("[", r"\[").replace("]", r"\]")):
        normalize_student_target(payload)


def test_local_model_id_uses_model_family_instead_of_hardcoded_gemma(
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3"}', encoding="utf-8"
    )

    assert local_model_id(tmp_path, "abcdef1234567890") == (
        "local-qwen3:abcdef123456"
    )
