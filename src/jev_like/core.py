from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class Example:
    state: str
    question: str
    options: list[str]
    answer: str
    example_id: str
    primitive: str = "choice"
    score_values: list[float] | None = None
    metadata: dict = field(default_factory=dict)


def read_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(Example(**json.loads(line)))
    return examples


def build_prompt(example: Example) -> str:
    if len(example.options) > len(LETTERS):
        raise ValueError("This prototype supports at most 26 choices")
    if example.primitive not in {"choice", "noul", "score"}:
        raise ValueError(f"Unsupported primitive: {example.primitive}")
    if example.primitive == "score":
        if example.score_values is None or len(example.score_values) != len(example.options):
            raise ValueError("Score examples require one numeric value per option")
    choices = "\n".join(
        f"{LETTERS[index]} = {option}" for index, option in enumerate(example.options)
    )
    instruction = {
        "choice": "Select exactly one option that is the best decision.",
        "noul": "Judge the proposition and select exactly one binary option.",
        "score": "Select exactly one ordered rating that best matches the evidence.",
    }[example.primitive]
    return (
        f"You are a Jev-like decision model. Primitive: {example.primitive.upper()}.\n"
        f"{instruction}\n\nSTATE:\n{example.state}\n\nQUESTION:\n{example.question}\n\n"
        f"OPTIONS:\n{choices}\n\nANSWER:"
    )


def candidate_token_ids(tokenizer, prompt: str, count: int) -> list[int]:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids: list[int] = []
    for letter in LETTERS[:count]:
        combined = tokenizer.encode(prompt + f" {letter}", add_special_tokens=False)
        if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
            raise ValueError(f"Candidate marker {letter!r} is not one stable continuation token")
        ids.append(combined[-1])
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate markers do not map to unique token IDs")
    return ids


def answer_index(example: Example) -> int:
    try:
        return example.options.index(example.answer)
    except ValueError as exc:
        raise ValueError(f"Answer {example.answer!r} is absent from options") from exc


def selected_logits(model, tokenizer, example: Example) -> torch.Tensor:
    prompt = build_prompt(example)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    token_ids = candidate_token_ids(tokenizer, prompt, len(example.options))
    outputs = model(**encoded, use_cache=False)
    return outputs.logits[0, -1, token_ids].float()
