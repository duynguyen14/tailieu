from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.visa_form_fields import FIELD_CODES
from tools.generate_visa_form_yolo_annotations import (
    build_field_predictions,
    load_class_mapping,
    select_best_detections,
)


def test_load_class_mapping_requires_canonical_order(tmp_path: Path) -> None:
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(
        json.dumps(
            {
                "labels": list(FIELD_CODES),
                "class_to_index": {
                    code: index for index, code in enumerate(FIELD_CODES)
                },
            }
        ),
        encoding="utf-8",
    )
    mapping = load_class_mapping(classes_path)
    assert mapping == {index: code for index, code in enumerate(FIELD_CODES)}

    payload = json.loads(classes_path.read_text(encoding="utf-8"))
    payload["class_to_index"]["full_name"] = 18
    classes_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical 19-field class order"):
        load_class_mapping(classes_path)


def test_select_best_detection_keeps_highest_confidence_per_class() -> None:
    selected = select_best_detections(
        boxes=[
            [10.2, 20.2, 100.4, 80.4],
            [12.0, 22.0, 102.0, 82.0],
            [200.0, 210.0, 300.0, 250.0],
        ],
        class_ids=[0, 0, 1],
        confidences=[0.7, 0.95, 0.8],
        class_by_index={index: code for index, code in enumerate(FIELD_CODES)},
        width=1000,
        height=1000,
    )
    assert selected["full_name"] == ((12, 22, 102, 82), 0.95)
    assert selected["date_of_birth"] == ((200, 210, 300, 250), 0.8)
    assert len(selected) == 2


def test_build_field_predictions_falls_back_to_template_for_missing_class() -> None:
    selected = {"full_name": ((100, 100, 400, 160), 0.9)}
    predictions = build_field_predictions(selected, width=1000, height=1000)
    assert len(predictions) == 19
    assert [item.field_code for item in predictions] == list(FIELD_CODES)
    assert predictions[0].bbox == (100, 100, 400, 160)
    assert predictions[0].confidence == 0.9
    assert predictions[0].used_fallback is False
    assert predictions[1].confidence is None
    assert predictions[1].used_fallback is True
    assert predictions[1].normalized_bbox == (160, 211, 350, 235)
