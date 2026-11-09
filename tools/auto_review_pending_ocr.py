from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy import select, true

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import (
    AnnotationDocument,
    AnnotationPage,
    OcrReview,
    OcrReviewToken,
    OcrRun,
    OcrToken,
)
from app.schemas.ocr_review import (
    OcrReviewRegionPayload,
    OcrReviewRequest,
    OcrReviewTokenPayload,
)
from app.services.ocr_review import OcrReviewService
from app.services.storage import AnnotationStorage


TOOL_VERSION = "pending-ocr-language-review-v1"
AUTO_REVIEWER_PREFIX = "auto-ocr-"
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
SPACE_PATTERN = re.compile(r"\s+")

VIETNAMESE_MARKS = frozenset(
    "ăâđêôơư"
    "áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữự"
    "ýỳỷỹỵ"
)
ENGLISH_WORDS = frozenset(
    "the of and to in for is are this that with from on at by as be has have will "
    "would embassy ministry foreign affairs department passport visa applicant date "
    "number name nationality gender entry exit proposed request presents compliments "
    "honor issue official consular attached enclosed".split()
)
VIETNAMESE_WORDS = frozenset(
    "cộng hòa xã hội chủ nghĩa việt nam đại sứ quán bộ ngoại giao cục lãnh sự kính "
    "gửi ngày tháng năm hộ chiếu thị thực nhập cảnh quốc tịch họ tên giới tính đề nghị "
    "trân trọng nhân viên bản dịch không chính thức nơi nhận".split()
)

# Only these visibly malformed OCR lexemes may be replaced globally.  Their
# target values are still learned from human reviews; keeping an allow-list is
# essential because a statistically stable replacement can still corrupt a
# proper name (for example, "Ho" must not be rewritten as Vietnamese "Họ").
SAFE_LEARNED_WORDS = frozenset(
    {
        "băc",
        "cãp",
        "cãấp",
        "câp",
        "câấp",
        "cng",
        "cộông",
        "dịich",
        "gửứi",
        "hêết",
        "hòờa",
        "hờa",
        "kyỳ",
        "kýỳ",
        "làần",
        "lòi",
        "nghịi",
        "nhâin",
        "nhâập",
        "nhâận",
        "nhâện",
        "nhậip",
        "nhiệim",
        "nhiệều",
        "nuước",
        "nưc",
        "nưóc",
        "nưóớc",
        "phòờng",
        "phờòng",
        "sế",
        "taại",
        "téới",
        "tói",
        "tóới",
        "tôội",
        "trj",
        "trjị",
        "thưực",
        "viêc",
        "viêệt",
        "viêết",
        "viêt",
        "vối",
        "đêến",
        "đếêến",
        "địinh",
        "đìinh",
        "đưc",
        "ộông",
    }
)

@dataclass(frozen=True)
class LanguageMetrics:
    language: str
    english_hits: int
    vietnamese_hits: int
    vietnamese_mark_count: int
    alphabetic_count: int
    vietnamese_mark_ratio: float


@dataclass(frozen=True)
class Correction:
    token_id: str
    token_index: int
    before: str
    after: str
    method: str
    support: int


@dataclass(frozen=True)
class PageAudit:
    run_id: str
    page_id: str
    filename: str
    page_number: int
    language: str
    english_hits: int
    vietnamese_hits: int
    vietnamese_mark_ratio: float
    token_count: int
    correction_count: int
    unresolved_count: int
    action: str
    corrections: list[dict[str, object]]
    unresolved: list[str]


@dataclass(frozen=True)
class CorrectionMemory:
    line_map: dict[str, tuple[str, int]]
    word_map: dict[str, tuple[str, int]]
    known_error_words: frozenset[str]
    source_review_count: int
    source_token_count: int
    changed_token_count: int


def normalize_text(value: str) -> str:
    return SPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", value or "")).strip()


def word_tokens(value: str) -> list[str]:
    return WORD_PATTERN.findall(normalize_text(value))


def classify_language(lines: list[str]) -> LanguageMetrics:
    text = " ".join(normalize_text(line) for line in lines if normalize_text(line))
    words = [word.casefold() for word in word_tokens(text)]
    english_hits = sum(word in ENGLISH_WORDS for word in words)
    vietnamese_hits = sum(word in VIETNAMESE_WORDS for word in words)
    vietnamese_mark_count = sum(
        character.casefold() in VIETNAMESE_MARKS for character in text
    )
    alphabetic_count = sum(character.isalpha() for character in text)
    mark_ratio = vietnamese_mark_count / max(1, alphabetic_count)

    if (
        english_hits >= 8
        and english_hits >= max(vietnamese_hits * 2, 8)
        and mark_ratio < 0.015
    ):
        language = "ENGLISH"
    elif english_hits >= 8 and vietnamese_hits >= 8:
        language = "MIXED"
    elif vietnamese_hits >= 2 or mark_ratio >= 0.025:
        language = "VIETNAMESE"
    else:
        language = "UNCERTAIN"
    return LanguageMetrics(
        language=language,
        english_hits=english_hits,
        vietnamese_hits=vietnamese_hits,
        vietnamese_mark_count=vietnamese_mark_count,
        alphabetic_count=alphabetic_count,
        vietnamese_mark_ratio=mark_ratio,
    )


def _preserve_case(source: str, replacement: str) -> str:
    lowered = replacement.casefold()
    if source.isupper():
        return lowered.upper()
    if source[:1].isupper() and source[1:].islower():
        return lowered[:1].upper() + lowered[1:]
    return lowered


def apply_correction_memory(
    text: str,
    memory: CorrectionMemory,
) -> tuple[str, str | None, int]:
    normalized = normalize_text(text)
    line_match = memory.line_map.get(normalized)
    if line_match is not None:
        replacement, support = line_match
        return replacement, "EXACT_REVIEWED_LINE", support

    applied_support = 0

    def replace_word(match: re.Match[str]) -> str:
        nonlocal applied_support
        source = match.group(0)
        learned = memory.word_map.get(source.casefold())
        if learned is None:
            return source
        replacement, support = learned
        applied_support = max(applied_support, support)
        return _preserve_case(source, replacement)

    corrected = WORD_PATTERN.sub(replace_word, normalized)
    if corrected != normalized:
        return corrected, "SAFE_REVIEWED_WORD", applied_support
    return normalized, None, 0


def find_unresolved(text: str, memory: CorrectionMemory) -> list[str]:
    issues: list[str] = []
    if "�" in text or "?" in text:
        issues.append("replacement_character")
    for word in word_tokens(text):
        folded_word = word.casefold()
        if folded_word in memory.known_error_words:
            issues.append(f"known_malformed_word:{word}")
        decomposed = unicodedata.normalize("NFD", folded_word)
        base_letters = [
            character
            for character in decomposed
            if character.isalpha() and not unicodedata.combining(character)
        ]
        if any(
            left == right
            for left, right in zip(base_letters, base_letters[1:], strict=False)
        ) and any(ord(character) > 127 for character in word):
            issues.append(f"suspicious_doubled_base:{word}")
        if "j" in folded_word and any(ord(character) > 127 for character in word):
            issues.append(f"suspicious_ocr_j:{word}")
        if any(
            character.isupper()
            for character in word[1:-1]
        ) and not word.isupper():
            issues.append(f"suspicious_internal_uppercase:{word}")
    return sorted(set(issues))


def _learn_memory(
    session,
    *,
    model_name: str,
    model_version: str,
    min_support: int,
) -> CorrectionMemory:
    rows = session.execute(
        select(
            OcrReview.id,
            OcrToken.text,
            OcrReviewToken.reviewed_text,
        )
        .join(OcrReviewToken, OcrReviewToken.source_token_id == OcrToken.id)
        .join(OcrReview, OcrReview.id == OcrReviewToken.review_id)
        .join(OcrRun, OcrRun.id == OcrReview.ocr_run_id)
        .where(
            OcrReview.is_active == true(),
            ~OcrReview.reviewed_by.like(f"{AUTO_REVIEWER_PREFIX}%"),
            OcrRun.model_name == model_name,
            OcrRun.model_version == model_version,
        )
    ).all()
    review_ids: set[str] = set()
    line_targets: dict[str, Counter[str]] = defaultdict(Counter)
    word_targets: dict[str, Counter[str]] = defaultdict(Counter)
    unchanged_words: Counter[str] = Counter()
    changed_token_count = 0
    for review_id, raw_value, reviewed_value in rows:
        review_ids.add(review_id)
        raw = normalize_text(raw_value)
        reviewed = normalize_text(reviewed_value)
        if raw == reviewed:
            continue
        changed_token_count += 1
        line_targets[raw][reviewed] += 1

        raw_words = word_tokens(raw)
        reviewed_words = word_tokens(reviewed)
        matcher = SequenceMatcher(
            None,
            [word.casefold() for word in raw_words],
            [word.casefold() for word in reviewed_words],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for source in raw_words[i1:i2]:
                    unchanged_words[source.casefold()] += 1
                continue
            if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
                continue
            source = raw_words[i1]
            target = reviewed_words[j1]
            word_targets[source.casefold()][target.casefold()] += 1

    line_map: dict[str, tuple[str, int]] = {}
    for source, targets in line_targets.items():
        total = sum(targets.values())
        target, support = targets.most_common(1)[0]
        if total >= min_support and support == total:
            line_map[source] = (target, support)

    word_map: dict[str, tuple[str, int]] = {}
    known_error_words: set[str] = set()
    for source, targets in word_targets.items():
        total = sum(targets.values())
        target, support = targets.most_common(1)[0]
        is_stable_error = (
            total >= min_support
            and support / total >= 0.90
            and unchanged_words[source] == 0
        )
        if is_stable_error:
            known_error_words.add(source)
        if is_stable_error and source in SAFE_LEARNED_WORDS:
            word_map[source] = (target, support)
    return CorrectionMemory(
        line_map=line_map,
        word_map=word_map,
        known_error_words=frozenset(known_error_words),
        source_review_count=len(review_ids),
        source_token_count=len(rows),
        changed_token_count=changed_token_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify pending OCR pages by language, approve high-confidence English "
            "pages verbatim, and propose data-driven Vietnamese corrections."
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--approve-vietnamese",
        action="store_true",
        help=(
            "Persist Vietnamese/MIXED reviews after learned corrections, but only "
            "when the post-correction unresolved count is zero."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reviewed-by", default=f"{AUTO_REVIEWER_PREFIX}language-v1")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/ocr_pending_language_audit.json"),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.min_support < 2:
        raise ValueError("--min-support must be at least 2")
    settings = get_settings()
    database = Database(settings)
    storage = AnnotationStorage(settings.annotation_storage_path)
    review_service = OcrReviewService(
        database=database,
        storage=storage,
        line_tolerance_ratio=settings.text_reconstruction_line_tolerance,
    )
    try:
        with database.session() as session:
            active_review_exists = (
                select(OcrReview.id)
                .where(
                    OcrReview.ocr_run_id == OcrRun.id,
                    OcrReview.is_active == true(),
                )
                .exists()
            )
            statement = (
                select(OcrRun, AnnotationPage, AnnotationDocument)
                .join(AnnotationPage, AnnotationPage.id == OcrRun.page_id)
                .join(
                    AnnotationDocument,
                    AnnotationDocument.id == AnnotationPage.document_id,
                )
                .where(
                    OcrRun.is_active == true(),
                    OcrRun.status == "COMPLETED",
                    AnnotationPage.review_status == "REVIEWED",
                    ~active_review_exists,
                )
                .order_by(OcrRun.created_at.asc(), OcrRun.id.asc())
            )
            if args.limit is not None:
                statement = statement.limit(args.limit)
            pending = list(session.execute(statement).all())
            memory_by_model: dict[tuple[str, str], CorrectionMemory] = {}
            prepared: list[
                tuple[
                    OcrRun,
                    AnnotationPage,
                    AnnotationDocument,
                    list[OcrToken],
                    list,
                    CorrectionMemory,
                ]
            ] = []
            for run_row, page, document in pending:
                key = (run_row.model_name, run_row.model_version)
                memory = memory_by_model.get(key)
                if memory is None:
                    memory = _learn_memory(
                        session,
                        model_name=run_row.model_name,
                        model_version=run_row.model_version,
                        min_support=args.min_support,
                    )
                    memory_by_model[key] = memory
                tokens = list(
                    session.scalars(
                        select(OcrToken)
                        .where(OcrToken.ocr_run_id == run_row.id)
                        .order_by(OcrToken.token_index.asc())
                    ).all()
                )
                regions = list(run_row.layout_regions)
                prepared.append(
                    (run_row, page, document, tokens, regions, memory)
                )

        audits: list[PageAudit] = []
        executed = 0
        action_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        for position, (run_row, page, document, tokens, regions, memory) in enumerate(
            prepared, start=1
        ):
            included_lines = [
                token.text for token in tokens if token.include_in_extraction
            ]
            metrics = classify_language(included_lines)
            language_counts[metrics.language] += 1
            corrections: list[Correction] = []
            corrected_by_id: dict[str, str] = {}
            unresolved: list[str] = []
            for token in tokens:
                corrected, method, support = apply_correction_memory(token.text, memory)
                corrected_by_id[token.id] = corrected
                if method is not None and corrected != normalize_text(token.text):
                    corrections.append(
                        Correction(
                            token_id=token.id,
                            token_index=token.token_index,
                            before=normalize_text(token.text),
                            after=corrected,
                            method=method,
                            support=support,
                        )
                    )
                if token.include_in_extraction:
                    unresolved.extend(find_unresolved(corrected, memory))
            unresolved = sorted(set(unresolved))

            should_persist = False
            if metrics.language == "ENGLISH":
                action = "APPROVE_ENGLISH" if args.execute else "WOULD_APPROVE_ENGLISH"
                should_persist = args.execute
            elif metrics.language in {"VIETNAMESE", "MIXED"}:
                if unresolved:
                    action = "HOLD_VIETNAMESE_UNRESOLVED"
                elif args.approve_vietnamese:
                    action = (
                        "APPROVE_VIETNAMESE_CORRECTED"
                        if args.execute
                        else "WOULD_APPROVE_VIETNAMESE_CORRECTED"
                    )
                    should_persist = args.execute
                else:
                    action = "HOLD_VIETNAMESE_AUDIT"
            else:
                action = "HOLD_UNCERTAIN_LANGUAGE"

            if should_persist:
                payload = OcrReviewRequest(
                    reviewed_by=args.reviewed_by,
                    review_note=(
                        f"{TOOL_VERSION}; language={metrics.language}; "
                        f"english_hits={metrics.english_hits}; "
                        f"vietnamese_hits={metrics.vietnamese_hits}; "
                        f"mark_ratio={metrics.vietnamese_mark_ratio:.6f}; "
                        f"corrections={len(corrections)}; "
                        f"human_source_reviews={memory.source_review_count}"
                    ),
                    include_in_llm=True,
                    tokens=[
                        OcrReviewTokenPayload(
                            source_token_id=token.id,
                            reviewed_text=corrected_by_id[token.id],
                            reviewed_bbox=(token.x1, token.y1, token.x2, token.y2),
                            include_in_extraction=token.include_in_extraction,
                            ignored_reason=token.ignored_reason,
                        )
                        for token in tokens
                    ],
                    exclusion_regions=[
                        OcrReviewRegionPayload(
                            source_region_id=region.id,
                            region_type=region.region_type,
                            reviewed_bbox=(
                                region.x1,
                                region.y1,
                                region.x2,
                                region.y2,
                            ),
                        )
                        for region in regions
                    ],
                )
                review_service.review(
                    run_id=run_row.id,
                    payload=payload,
                    load_detail=False,
                )
                executed += 1

            action_counts[action] += 1
            audit = PageAudit(
                run_id=run_row.id,
                page_id=page.id,
                filename=document.original_filename,
                page_number=page.page_number,
                language=metrics.language,
                english_hits=metrics.english_hits,
                vietnamese_hits=metrics.vietnamese_hits,
                vietnamese_mark_ratio=round(metrics.vietnamese_mark_ratio, 8),
                token_count=len(tokens),
                correction_count=len(corrections),
                unresolved_count=len(unresolved),
                action=action,
                corrections=[asdict(item) for item in corrections],
                unresolved=unresolved,
            )
            audits.append(audit)
            print(
                f"[{position}/{len(prepared)}] {action} "
                f"file={document.original_filename!r} page={page.page_number} "
                f"lang={metrics.language} corrections={len(corrections)} "
                f"unresolved={len(unresolved)}"
            )

        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "tool_version": TOOL_VERSION,
            "database": settings.db_name,
            "execute": bool(args.execute),
            "approve_vietnamese": bool(args.approve_vietnamese),
            "pending_discovered": len(prepared),
            "reviews_created": executed,
            "language_counts": dict(sorted(language_counts.items())),
            "action_counts": dict(sorted(action_counts.items())),
            "correction_memory": [
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "source_review_count": memory.source_review_count,
                    "source_token_count": memory.source_token_count,
                    "changed_token_count": memory.changed_token_count,
                    "exact_line_rules": len(memory.line_map),
                    "safe_word_rules": len(memory.word_map),
                    "known_error_words": len(memory.known_error_words),
                }
                for (model_name, model_version), memory in memory_by_model.items()
            ],
            "pages": [asdict(audit) for audit in audits],
        }
        report_path = args.report
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({key: value for key, value in report.items() if key != "pages"}, ensure_ascii=False, indent=2))
        print(f"Audit report: {report_path}")
        return report
    finally:
        database.close()


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        raise SystemExit(f"Pending OCR auto-review failed: {exc}") from exc


if __name__ == "__main__":
    main()
