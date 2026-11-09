from tools.auto_review_pending_ocr import (
    CorrectionMemory,
    apply_correction_memory,
    classify_language,
    find_unresolved,
)


def memory() -> CorrectionMemory:
    return CorrectionMemory(
        line_map={"Kính gi:": ("Kính gửi:", 22)},
        word_map={"hòờa": ("hòa", 19), "tóới": ("tới", 33)},
        known_error_words=frozenset({"hòờa", "tóới", "thiị"}),
        source_review_count=367,
        source_token_count=13539,
        changed_token_count=1396,
    )


def test_high_confidence_english_page() -> None:
    metrics = classify_language(
        [
            "The Embassy presents its compliments to the Ministry of Foreign Affairs",
            "and has the honor to request that the Department issue a visa.",
        ]
    )

    assert metrics.language == "ENGLISH"


def test_vietnamese_and_mixed_pages_are_not_treated_as_english() -> None:
    vietnamese = classify_language(
        ["Đại sứ quán kính gửi Cục Lãnh sự Bộ Ngoại giao Việt Nam."]
    )
    mixed = classify_language(
        [
            "The Embassy presents its compliments to the Ministry of Foreign Affairs.",
            "Đại sứ quán kính gửi Cục Lãnh sự Bộ Ngoại giao Việt Nam.",
        ]
    )

    assert vietnamese.language == "VIETNAMESE"
    assert mixed.language != "ENGLISH"


def test_exact_reviewed_line_has_priority() -> None:
    corrected, method, support = apply_correction_memory("  Kính   gi: ", memory())

    assert corrected == "Kính gửi:"
    assert method == "EXACT_REVIEWED_LINE"
    assert support == 22


def test_safe_learned_words_preserve_sentence_case() -> None:
    corrected, method, support = apply_correction_memory(
        "Cộng hòờa xin gửi tóới Quý Cục", memory()
    )

    assert corrected == "Cộng hòa xin gửi tới Quý Cục"
    assert method == "SAFE_REVIEWED_WORD"
    assert support == 33
    assert find_unresolved(corrected, memory()) == []


def test_unresolved_known_malformed_word_is_reported() -> None:
    unresolved = find_unresolved("Cộng hòờa", memory())

    assert "known_malformed_word:hòờa" in unresolved
    assert "suspicious_doubled_base:hòờa" in unresolved


def test_known_error_not_safe_for_auto_replacement_is_still_reported() -> None:
    corrected, method, support = apply_correction_memory("thị thiị", memory())

    assert corrected == "thị thiị"
    assert method is None
    assert support == 0
    unresolved = find_unresolved(corrected, memory())
    assert "known_malformed_word:thiị" in unresolved
    assert "suspicious_doubled_base:thiị" in unresolved
