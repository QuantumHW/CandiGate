from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from .core import Example, build_prompt, candidate_token_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 2 or args.count > 26:
        parser.error("--count must be between 2 and 26")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    example = Example(
        state="The application crashes during export.",
        question="Which department should handle this?",
        options=[f"option {index}" for index in range(args.count)],
        answer="technical",
        example_id="token-check",
    )
    prompt = build_prompt(example)
    ids = candidate_token_ids(tokenizer, prompt, len(example.options))
    for letter, token_id in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", ids):
        print(f"{letter}: token_id={token_id} decoded={tokenizer.decode([token_id])!r}")
    print("PASS: all decision markers are stable unique single-token continuations")


if __name__ == "__main__":
    main()
