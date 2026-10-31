from app.models.paddle_ocr_engine import parse_ocr_tokens


def test_parse_ocr_tokens_filters_and_sorts_by_geometry() -> None:
    payload = {
        "rec_texts": ["second line", "first line", "too weak"],
        "rec_scores": [0.93, 0.99, 0.20],
        "rec_boxes": [[20, 80, 180, 110], [10, 20, 170, 50], [5, 5, 30, 15]],
        "rec_polys": [
            [[20, 80], [180, 80], [180, 110], [20, 110]],
            [[10, 20], [170, 20], [170, 50], [10, 50]],
            [[5, 5], [30, 5], [30, 15], [5, 15]],
        ],
    }

    tokens = parse_ocr_tokens(payload, min_confidence=0.5)

    assert [token.token_id for token in tokens] == ["t_0001", "t_0002"]
    assert [token.text for token in tokens] == ["first line", "second line"]
    assert tokens[0].bbox == (10, 20, 170, 50)


def test_parse_ocr_tokens_builds_bbox_from_polygon() -> None:
    payload = {
        "rec_texts": ["passport RF0210530"],
        "rec_scores": [0.97],
        "rec_polys": [[[10.2, 20.7], [90.8, 20.1], [91.1, 40.4], [9.9, 40.2]]],
    }

    tokens = parse_ocr_tokens(payload, min_confidence=0.5)

    assert len(tokens) == 1
    assert tokens[0].bbox == (9, 20, 92, 41)
