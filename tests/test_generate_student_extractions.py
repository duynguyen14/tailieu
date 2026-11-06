from __future__ import annotations

import pytest

from tools.generate_student_extractions import normalize_student_target


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
        "dt": None,
        "pl": None,
        "snd": None,
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
