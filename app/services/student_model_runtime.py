from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def apply_student_chat_template(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
) -> Any:
    """Render one student conversation without Qwen reasoning tokens.

    ``enable_thinking`` is a chat-template variable. Qwen3 consumes it while
    Gemma and other templates safely ignore it, so the same exported dataset
    and runtime code can be used by both model families.
    """

    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def student_training_assistant_content(tokenizer: Any, content: str) -> str:
    """Align Qwen3's supervised assistant prefix with non-thinking inference.

    Qwen3 inserts an empty ``<think>...</think>`` block only when rendering a
    generation prompt with ``enable_thinking=False``. Supplying that same empty
    block in the training assistant message makes the prefix byte-for-byte
    consistent while leaving the JSON itself as the only supervised content.
    Other chat templates receive the original assistant content unchanged.
    """

    template = str(getattr(tokenizer, "chat_template", "") or "")
    if "enable_thinking" in template and "<think>" in template:
        return f"<think>\n\n</think>\n\n{content}"
    return content


def generation_stop_token_ids(tokenizer: Any, model: Any) -> list[int]:
    """Return EOS/end-of-turn IDs without treating an unknown token as EOS."""

    candidates: list[Any] = [
        getattr(model.generation_config, "eos_token_id", None),
        getattr(model.config, "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for token in ("<end_of_turn>", "<|im_end|>"):
        if token in vocabulary:
            candidates.append(tokenizer.convert_tokens_to_ids(token))

    values: list[int] = []
    for candidate in candidates:
        items = candidate if isinstance(candidate, (list, tuple)) else [candidate]
        for item in items:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                if item not in values:
                    values.append(item)
    if not values:
        raise RuntimeError("The tokenizer/model exposes no generation stop token")
    return values


def _first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object while respecting JSON strings."""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_student_json_output(text: str) -> Any:
    """Parse strict JSON plus harmless Markdown/preamble wrappers.

    Schema validation remains the caller's responsibility. This deliberately
    does not repair values or truncated JSON; it only removes presentation
    text that some instruction checkpoints add around an otherwise valid
    object.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as original_error:
        candidate = _first_json_object(cleaned)
        if candidate is None:
            raise original_error
        return json.loads(candidate)


__all__ = [
    "apply_student_chat_template",
    "generation_stop_token_ids",
    "parse_student_json_output",
    "student_training_assistant_content",
]
