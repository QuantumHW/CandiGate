import pytest

from jev_like.core import Example, answer_index, build_prompt, candidate_token_ids


def example(**changes):
    values = {
        "state": "A current weather lookup is required.",
        "question": "Which tool should run?",
        "options": ["weather", "calendar"],
        "answer": "weather",
        "example_id": "unit-1",
        "primitive": "choice",
    }
    values.update(changes)
    return Example(**values)


class StableTokenizer:
    marker_ids = {"A": 101, "B": 102, "C": 103}

    def encode(self, text, add_special_tokens=False):
        for marker, token_id in self.marker_ids.items():
            if text.endswith(f" {marker}"):
                return [1, 2, token_id]
        return [1, 2]


class UnstableTokenizer(StableTokenizer):
    def encode(self, text, add_special_tokens=False):
        if text.endswith(" B"):
            return [1, 2, 9, 102]
        return super().encode(text, add_special_tokens=add_special_tokens)


class DuplicateTokenizer(StableTokenizer):
    marker_ids = {"A": 101, "B": 101, "C": 103}


def test_build_prompt_contains_protocol_fields():
    prompt = build_prompt(example())
    assert "Primitive: CHOICE" in prompt
    assert "STATE:" in prompt
    assert "QUESTION:" in prompt
    assert "A = weather" in prompt
    assert prompt.endswith("ANSWER:")


def test_build_prompt_validates_primitive_and_score_values():
    with pytest.raises(ValueError, match="Unsupported primitive"):
        build_prompt(example(primitive="other"))
    with pytest.raises(ValueError, match="one numeric value"):
        build_prompt(example(primitive="score"))
    prompt = build_prompt(example(primitive="score", score_values=[0.0, 1.0]))
    assert "Primitive: SCORE" in prompt


def test_build_prompt_rejects_more_than_26_candidates():
    options = [f"option-{index}" for index in range(27)]
    with pytest.raises(ValueError, match="at most 26"):
        build_prompt(example(options=options, answer=options[0]))


def test_answer_index_requires_answer_in_options():
    assert answer_index(example()) == 0
    with pytest.raises(ValueError, match="absent from options"):
        answer_index(example(answer="email"))


def test_candidate_markers_are_unique_stable_continuations():
    assert candidate_token_ids(StableTokenizer(), "prompt", 3) == [101, 102, 103]
    with pytest.raises(ValueError, match="not one stable continuation token"):
        candidate_token_ids(UnstableTokenizer(), "prompt", 3)
    with pytest.raises(ValueError, match="unique token IDs"):
        candidate_token_ids(DuplicateTokenizer(), "prompt", 3)
