from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import Example, answer_index, build_prompt, read_jsonl


def generated_index(raw: str, option_count: int) -> int | None:
    match = re.match(r"\s*([A-Z])(?:\s|$|[.):,])", raw)
    if not match:
        return None
    index = ord(match.group(1)) - ord("A")
    return index if 0 <= index < option_count else None


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["prediction"] is not None]
    latencies = [row["latency_ms"] for row in rows]
    return {
        "samples": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "valid_rate": len(valid) / len(rows),
        "invalid_count": len(rows) - len(valid),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
    }


def semantic_metrics(rows: list[dict]) -> dict:
    semantic_rows = [row for row in rows if row["target_family"] is not None]
    targets = [row["target_family"] for row in semantic_rows]
    predictions = [row["prediction_family"] or "__invalid__" for row in semantic_rows]
    labels = sorted(set(targets))
    return {
        "samples": len(semantic_rows),
        "classes": len(labels),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "no_tool_samples": sum(target == "none" for target in targets),
        "no_tool_accuracy": sum(t == p == "none" for t, p in zip(targets, predictions)) / max(1, sum(t == "none" for t in targets)),
    }


def predict(model, tokenizer, example: Example, max_new_tokens: int) -> tuple[int | None, str, float]:
    prompt = build_prompt(example)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    raw = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_index(raw, len(example.options)), raw, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--warmup-samples", type=int, default=3)
    args = parser.parse_args()

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": "cuda"})
    model.eval()
    model_load_seconds = time.perf_counter() - load_started
    examples = read_jsonl(args.split)
    if args.max_samples:
        examples = examples[: args.max_samples]
    if not examples:
        raise ValueError("Evaluation split is empty")

    with torch.inference_mode():
        for index in range(args.warmup_samples):
            predict(model, tokenizer, examples[index % len(examples)], args.max_new_tokens)
    torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    with torch.inference_mode():
        for example in examples:
            prediction, raw, latency_ms = predict(model, tokenizer, example, args.max_new_tokens)
            target = answer_index(example)
            families = example.metadata.get("candidate_families")
            target_family = families[target] if families else None
            prediction_family = families[prediction] if families and prediction is not None else None
            rows.append({
                "example_id": example.example_id,
                "primitive": example.primitive,
                "target": target,
                "prediction": prediction,
                "correct": prediction == target,
                "target_label": example.answer,
                "prediction_label": example.options[prediction] if prediction is not None else None,
                "target_family": target_family,
                "prediction_family": prediction_family,
                "latency_ms": latency_ms,
                "raw": raw,
                "metadata": example.metadata,
            })

    by_primitive: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_primitive[row["primitive"]].append(row)
    metrics = {
        "model": args.model,
        "mode": "generate",
        "model_load_seconds": model_load_seconds,
        "max_new_tokens": args.max_new_tokens,
        "warmup_samples": args.warmup_samples,
        "overall": summarize(rows),
        "by_primitive": {name: summarize(group) for name, group in sorted(by_primitive.items())},
        "synthetic_tool_semantics": semantic_metrics(rows),
        "max_cuda_memory_mib": float(torch.cuda.max_memory_allocated() / 1024**2),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
