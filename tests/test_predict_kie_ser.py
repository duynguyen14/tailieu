from __future__ import annotations

import pytest

from tools.predict_kie_ser import decode_bio_entities


def test_decode_bio_entities_repairs_orphan_i_and_merges_bbox() -> None:
    entities = decode_bio_entities(
        ["NGUYEN", "VAN", "A", "Male"],
        [[10, 10, 30, 20], [31, 10, 50, 20], [51, 10, 60, 20], [10, 30, 30, 40]],
        ["I-FULL_NAME", "I-FULL_NAME", "I-FULL_NAME", "B-GENDER"],
        [0.8, 0.9, 1.0, 0.95],
        pixel_boxes=[[100, 100, 300, 200], [310, 100, 500, 200], [510, 100, 600, 200], [100, 300, 300, 400]],
    )

    assert len(entities) == 2
    assert entities[0].label == "FULL_NAME"
    assert entities[0].value == "NGUYEN VAN A"
    assert entities[0].bbox == [10, 10, 60, 20]
    assert entities[0].pixel_bbox == [100, 100, 600, 200]
    assert entities[0].confidence == pytest.approx(0.9)
    assert entities[1].label == "GENDER"


def test_decode_bio_entities_breaks_on_outside_label() -> None:
    entities = decode_bio_entities(
        ["Alice", "and", "Bob"],
        [[1, 1, 2, 2], [3, 1, 4, 2], [5, 1, 6, 2]],
        ["B-FULL_NAME", "O", "B-FULL_NAME"],
        [0.9, 0.99, 0.8],
    )

    assert [item.value for item in entities] == ["Alice", "Bob"]
