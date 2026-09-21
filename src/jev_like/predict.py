from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import Example, selected_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--input", required=True, help="JSON file containing state, question, options and primitive")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    example = Example(
        state=payload["state"], question=payload["question"], options=payload["options"],
        answer=payload.get("answer", payload["options"][0]), example_id=payload.get("example_id", "request"),
        primitive=payload.get("primitive", "choice"), score_values=payload.get("score_values"),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": "cuda"})
    if args.adapter: model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    with torch.inference_mode():
        probs = torch.softmax(selected_logits(model, tokenizer, example) / args.temperature, dim=-1).cpu().numpy()
    prediction = int(probs.argmax())
    output = {
        "primitive": example.primitive,
        "choice": example.options[prediction],
        "confidence": float(probs[prediction]),
        "probabilities": {option: float(prob) for option, prob in zip(example.options, probs)},
    }
    if example.primitive == "score":
        values = torch.tensor(example.score_values, dtype=torch.float64).numpy()
        output["expected_score"] = float((probs * values).sum())
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
