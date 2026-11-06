from __future__ import annotations

from tools.export_kie_ser_dataset import ExportToken
from tools.infer_kie_ser_pages import build_inference_segments, entity_storage_payload
from tools.predict_kie_ser import PredictedEntity


def test_pending_inference_preserves_source_character_spans() -> None:
    segments = build_inference_segments(
        [
            ExportToken(
                id="token-1",
                order=1,
                text="Name: NGUYEN VAN A",
                bbox=(100, 200, 500, 240),
            )
        ]
    )
    assert [item.text for item in segments] == ["Name:", "NGUYEN", "VAN", "A"]

    entity = PredictedEntity(
        label="FULL_NAME",
        value="NGUYEN VAN A",
        word_start=1,
        word_end=4,
        word_indices=[1, 2, 3],
        bbox=[250, 100, 1000, 120],
        pixel_bbox=[200, 200, 500, 240],
        confidence=0.95,
    )
    payload = entity_storage_payload(entity, segments)

    assert payload["token_ids"] == ["token-1"]
    assert payload["value"] == "NGUYEN VAN A"
    assert [item["text"] for item in payload["spans"]] == ["NGUYEN", "VAN", "A"]
    assert payload["spans"][0]["char_start"] == 6
    assert payload["bbox"] == [200, 200, 500, 240]
