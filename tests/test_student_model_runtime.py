from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.student_model_runtime import (
    apply_student_chat_template,
    generation_stop_token_ids,
    parse_student_json_output,
    student_training_assistant_content,
)


def test_chat_template_explicitly_disables_qwen_thinking() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "source"}]
            assert kwargs["enable_thinking"] is False
            return [1, 2, 3]

    rendered = apply_student_chat_template(
        Tokenizer(),
        [{"role": "user", "content": "source"}],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert rendered == [1, 2, 3]


def test_qwen_training_content_contains_matching_empty_think_prefix() -> None:
    tokenizer = SimpleNamespace(
        chat_template="{% if enable_thinking %}<think>{% endif %}"
    )

    assert student_training_assistant_content(tokenizer, '{"n":{},"a":[]}') == (
        '<think>\n\n</think>\n\n{"n":{},"a":[]}'
    )
    assert student_training_assistant_content(
        SimpleNamespace(chat_template="gemma template"), '{"n":{},"a":[]}'
    ) == '{"n":{},"a":[]}'


def test_stop_tokens_support_qwen_without_adding_unknown_end_of_turn() -> None:
    class Tokenizer:
        eos_token_id = 151645

        @staticmethod
        def get_vocab():
            return {"<|im_end|>": 151645, "<unk>": 0}

        @staticmethod
        def convert_tokens_to_ids(token):
            return {"<|im_end|>": 151645}.get(token, 0)

    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=151645),
        config=SimpleNamespace(eos_token_id=151645),
    )

    assert generation_stop_token_ids(Tokenizer(), model) == [151645]


@pytest.mark.parametrize(
    "raw",
    [
        '{"n":{},"a":[]}',
        '```json\n{"n":{},"a":[]}\n```',
        '**{"n":{},"a":[]}**',
        'Result:\n{"n":{},"a":[]}\nDone.',
    ],
)
def test_parser_accepts_harmless_wrappers(raw: str) -> None:
    assert parse_student_json_output(raw) == {"n": {}, "a": []}


def test_parser_rejects_truncated_json_instead_of_repairing_values() -> None:
    with pytest.raises(ValueError):
        parse_student_json_output('{"n":{},"a":[{"nm":"Alice"}')
