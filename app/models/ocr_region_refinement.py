from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

from PIL import Image, ImageOps

from app.models.paddle_ocr_engine import OcrTokenPrediction

PredictMany = Callable[[list[Path]], list[list[OcrTokenPrediction]]]

_DIGIT_PATTERN = re.compile(r"\d")
_SPACE_PATTERN = re.compile(r"\s+")
_VIETNAMESE_MARKS = frozenset(
    "ăâđêôơư"
    "áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữự"
    "ýỳỷỹỵ"
)
_VIETNAMESE_HINTS = frozenset(
    {
        "ban",
        "bo",
        "canh",
        "chinh",
        "chu",
        "cong",
        "dai",
        "dich",
        "giao",
        "gui",
        "ho",
        "khong",
        "kinh",
        "lanh",
        "nam",
        "ngay",
        "nghi",
        "nghia",
        "ngoai",
        "nguoi",
        "nhap",
        "nuoc",
        "quan",
        "quoc",
        "su",
        "ten",
        "thang",
        "thi",
        "thuc",
        "tich",
        "tieng",
        "viet",
        "xa",
    }
)


@dataclass(frozen=True)
class RefinementCandidate:
    scale: float
    text: str
    confidence: float

    def audit_dict(self) -> dict[str, object]:
        return {
            "scale": round(self.scale, 4),
            "text": self.text,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class SelectiveRefinementResult:
    tokens: list[OcrTokenPrediction]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _RegionRequest:
    token_position: int
    scale: float
    path: Path


def _normalize_text(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", value)).strip()


def _comparison_text(value: str) -> str:
    return _normalize_text(value).casefold()


def _accent_fold(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


def _digit_signature(value: str) -> str:
    return "".join(_DIGIT_PATTERN.findall(value))


def _has_vietnamese_signal(value: str) -> bool:
    lowered = unicodedata.normalize("NFC", value).casefold()
    if any(character in _VIETNAMESE_MARKS for character in lowered):
        return True
    folded_words = re.findall(r"[a-z]+", _accent_fold(lowered))
    return sum(word in _VIETNAMESE_HINTS for word in folded_words) >= 2


def _suspicion_score(
    token: OcrTokenPrediction,
    *,
    confidence_threshold: float,
    typical_height: float,
) -> float:
    text = _normalize_text(token.text)
    if len(text) < 2 or not any(character.isalnum() for character in text):
        return -1.0
    score = max(0.0, confidence_threshold - token.confidence) * 12.0
    if token.confidence < confidence_threshold:
        score += 2.0
    if _has_vietnamese_signal(text):
        score += 1.5
    if "�" in text or "?" in text:
        score += 3.0
    height = max(1, token.bbox[3] - token.bbox[1])
    if typical_height > 0 and height < typical_height * 0.80:
        score += min(1.5, (typical_height - height) / typical_height * 2.0)
    return score


def _candidate_from_tokens(
    scale: float, tokens: list[OcrTokenPrediction]
) -> RefinementCandidate:
    ordered = sorted(tokens, key=lambda item: (item.bbox[1], item.bbox[0]))
    text = _normalize_text(" ".join(token.text for token in ordered))
    weights = [max(1, len(token.text.strip())) for token in ordered]
    confidence = (
        sum(
            token.confidence * weight
            for token, weight in zip(ordered, weights, strict=True)
        )
        / sum(weights)
        if weights
        else 0.0
    )
    return RefinementCandidate(scale=scale, text=text, confidence=confidence)


def _crop_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    scale: float,
    padding_ratio: float,
    max_side: int,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    padding_x = max(4, round(width * padding_ratio))
    padding_y = max(4, round(height * max(padding_ratio, 0.20)))
    crop_box = (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(image.width, x2 + padding_x),
        min(image.height, y2 + padding_y),
    )
    crop = image.crop(crop_box).convert("RGB")
    requested = max(1.0, scale)
    limited = min(requested, max_side / max(crop.width, crop.height))
    effective = max(1.0, limited)
    if effective > 1.001:
        crop = crop.resize(
            (
                max(1, round(crop.width * effective)),
                max(1, round(crop.height * effective)),
            ),
            Image.Resampling.LANCZOS,
        )
    # A white border prevents glyphs touching the crop edge from being lost by
    # the detector. It is added only to the temporary refinement image.
    return ImageOps.expand(crop, border=max(6, round(crop.height * 0.06)), fill="white")


def _choose_replacement(
    original: OcrTokenPrediction,
    candidates: list[RefinementCandidate],
    *,
    candidate_min_confidence: float,
    agreement_similarity: float,
    min_confidence_gain: float,
    protect_digits: bool,
) -> tuple[str, float, bool, str]:
    if len(candidates) < 2:
        return original.text, original.confidence, False, "insufficient_candidates"
    first, second = candidates[0], candidates[1]
    if not first.text or not second.text:
        return original.text, original.confidence, False, "empty_candidate"
    if min(first.confidence, second.confidence) < candidate_min_confidence:
        return original.text, original.confidence, False, "candidate_confidence_low"

    first_text = _comparison_text(first.text)
    second_text = _comparison_text(second.text)
    similarity = SequenceMatcher(None, first_text, second_text).ratio()
    if similarity < agreement_similarity:
        return original.text, original.confidence, False, "candidates_disagree"

    candidate = first if first.confidence >= second.confidence else second
    original_digits = _digit_signature(original.text)
    if protect_digits and original_digits:
        if (
            _digit_signature(first.text) != original_digits
            or _digit_signature(second.text) != original_digits
        ):
            return original.text, original.confidence, False, "digit_mismatch_protected"

    if _comparison_text(candidate.text) == _comparison_text(original.text):
        return original.text, max(original.confidence, candidate.confidence), False, "confirmed"

    accent_only_change = _accent_fold(candidate.text) == _accent_fold(original.text)
    confidence_gain = min(first.confidence, second.confidence) - original.confidence
    if accent_only_change:
        if confidence_gain < -0.02:
            return original.text, original.confidence, False, "accent_candidate_weaker"
    elif confidence_gain < min_confidence_gain:
        return original.text, original.confidence, False, "confidence_gain_too_small"

    return candidate.text, candidate.confidence, True, (
        "accent_consensus" if accent_only_change else "multiscale_consensus"
    )


def refine_ocr_tokens(
    image: Image.Image,
    tokens: list[OcrTokenPrediction],
    *,
    temp_dir: Path,
    predict_many: PredictMany,
    scales: list[float],
    confidence_threshold: float,
    candidate_min_confidence: float,
    agreement_similarity: float,
    min_confidence_gain: float,
    max_regions: int,
    max_seconds: float,
    padding_ratio: float,
    max_crop_side: int,
    protect_digits: bool,
    version: str,
) -> SelectiveRefinementResult:
    started = time.perf_counter()
    if len(scales) < 2 or not tokens or max_regions <= 0:
        return SelectiveRefinementResult(
            tokens=tokens,
            metadata={
                "version": version,
                "attempted": 0,
                "replaced": 0,
                "unresolved": 0,
                "elapsed_ms": 0,
                "items": [],
            },
        )

    heights = [max(1, token.bbox[3] - token.bbox[1]) for token in tokens]
    typical_height = float(median(heights)) if heights else 0.0
    ranked = sorted(
        (
            (
                _suspicion_score(
                    token,
                    confidence_threshold=confidence_threshold,
                    typical_height=typical_height,
                ),
                position,
            )
            for position, token in enumerate(tokens)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    # Vietnamese-looking regions are eligible even at high confidence. Other
    # regions must have a concrete low-confidence/small-text signal.
    selected_positions = [
        position for score, position in ranked if score > 0.0
    ][:max_regions]

    requests: list[_RegionRequest] = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    for position in selected_positions:
        if time.perf_counter() - started > max_seconds:
            break
        token = tokens[position]
        for scale in scales[:2]:
            crop = _crop_region(
                image,
                token.bbox,
                scale=scale,
                padding_ratio=padding_ratio,
                max_side=max_crop_side,
            )
            path = temp_dir / f"refine-{position:04d}-{scale:.3f}x.png"
            crop.save(path, format="PNG", compress_level=1)
            crop.close()
            requests.append(_RegionRequest(position, scale, path))

    predictions = predict_many([request.path for request in requests]) if requests else []
    if len(predictions) != len(requests):
        raise RuntimeError(
            "Selective OCR refinement returned a different number of results "
            f"({len(predictions)}) than requested crops ({len(requests)})."
        )

    by_position: dict[int, list[RefinementCandidate]] = {}
    for request, prediction in zip(requests, predictions, strict=True):
        by_position.setdefault(request.token_position, []).append(
            _candidate_from_tokens(request.scale, prediction)
        )

    refined = list(tokens)
    items: list[dict[str, object]] = []
    replaced = 0
    unresolved = 0
    for position in selected_positions:
        original = tokens[position]
        candidates = by_position.get(position, [])
        text, confidence, changed, reason = _choose_replacement(
            original,
            candidates,
            candidate_min_confidence=candidate_min_confidence,
            agreement_similarity=agreement_similarity,
            min_confidence_gain=min_confidence_gain,
            protect_digits=protect_digits,
        )
        if changed:
            replaced += 1
            refined[position] = OcrTokenPrediction(
                token_id=original.token_id,
                token_index=original.token_index,
                text=text,
                confidence=confidence,
                bbox=original.bbox,
                polygon=original.polygon,
            )
        elif reason not in {"confirmed"}:
            unresolved += 1
        items.append(
            {
                "token_id": original.token_id,
                "bbox": list(original.bbox),
                "original_text": original.text,
                "original_confidence": round(original.confidence, 6),
                "candidates": [candidate.audit_dict() for candidate in candidates],
                "selected_text": text,
                "selected_confidence": round(confidence, 6),
                "replaced": changed,
                "reason": reason,
            }
        )

    return SelectiveRefinementResult(
        tokens=refined,
        metadata={
            "version": version,
            "scales": [round(scale, 4) for scale in scales[:2]],
            "attempted": len(by_position),
            "replaced": replaced,
            "unresolved": unresolved,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "items": items,
        },
    )
