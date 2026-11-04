from pathlib import Path

from PIL import Image

from app.models.ocr_region_refinement import refine_ocr_tokens
from app.models.paddle_ocr_engine import OcrTokenPrediction


def token(
    text: str,
    confidence: float,
    *,
    token_id: str = "t_0001",
) -> OcrTokenPrediction:
    return OcrTokenPrediction(
        token_id=token_id,
        token_index=1,
        text=text,
        confidence=confidence,
        bbox=(20, 20, 280, 55),
        polygon=[[20, 20], [280, 20], [280, 55], [20, 55]],
    )


def candidate(text: str, confidence: float) -> list[OcrTokenPrediction]:
    return [token(text, confidence)]


def run_refinement(
    tmp_path: Path,
    original: OcrTokenPrediction,
    predictions: list[list[OcrTokenPrediction]],
):
    def predict_many(paths: list[Path]) -> list[list[OcrTokenPrediction]]:
        assert len(paths) == 2
        assert all(path.is_file() for path in paths)
        return predictions

    image = Image.new("RGB", (320, 100), "white")
    try:
        return refine_ocr_tokens(
            image,
            [original],
            temp_dir=tmp_path,
            predict_many=predict_many,
            scales=[2.0, 3.0],
            confidence_threshold=0.90,
            candidate_min_confidence=0.70,
            agreement_similarity=0.96,
            min_confidence_gain=0.05,
            max_regions=32,
            max_seconds=15.0,
            padding_ratio=0.12,
            max_crop_side=2400,
            protect_digits=True,
            version="test-v1",
        )
    finally:
        image.close()


def test_multiscale_consensus_repairs_vietnamese_accents(tmp_path: Path) -> None:
    result = run_refinement(
        tmp_path,
        token("Ban dich tieng Viet khong chinh thuc", 0.82),
        [
            candidate("Bản dịch tiếng Việt không chính thức", 0.95),
            candidate("Bản dịch tiếng Việt không chính thức", 0.96),
        ],
    )

    assert result.tokens[0].text == "Bản dịch tiếng Việt không chính thức"
    assert result.metadata["attempted"] == 1
    assert result.metadata["replaced"] == 1
    assert result.metadata["items"][0]["reason"] == "accent_consensus"


def test_digit_mismatch_never_replaces_identifier(tmp_path: Path) -> None:
    result = run_refinement(
        tmp_path,
        token("So cong ham: 850-2026", 0.60),
        [
            candidate("Số công hàm: 850-2028", 0.98),
            candidate("Số công hàm: 850-2028", 0.99),
        ],
    )

    assert result.tokens[0].text == "So cong ham: 850-2026"
    assert result.metadata["replaced"] == 0
    assert result.metadata["unresolved"] == 1
    assert result.metadata["items"][0]["reason"] == "digit_mismatch_protected"


def test_disagreeing_scales_keep_original_text(tmp_path: Path) -> None:
    result = run_refinement(
        tmp_path,
        token("Quoc tich Viet Nam", 0.71),
        [
            candidate("Quốc tịch Việt Nam", 0.97),
            candidate("Quốc tịch Việt Narn", 0.98),
        ],
    )

    assert result.tokens[0].text == "Quoc tich Viet Nam"
    assert result.metadata["replaced"] == 0
    assert result.metadata["items"][0]["reason"] == "candidates_disagree"
