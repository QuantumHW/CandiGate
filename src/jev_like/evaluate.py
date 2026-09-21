from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import answer_index, read_jsonl, selected_logits


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    return float(sum(
        mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        for lower, upper in zip(edges[:-1], edges[1:])
        if (mask := (confidence > lower) & (confidence <= upper)).any()
    ))


def classification_metrics(rows: list[dict]) -> dict:
    targets = [row["target_label"] for row in rows]
    predictions = [row["prediction_label"] for row in rows]
    correct = np.asarray([a == b for a, b in zip(targets, predictions)])
    confidence = np.asarray([row["confidence"] for row in rows])
    nll = np.mean([-np.log(max(row["probabilities"][row["target"]], 1e-12)) for row in rows])
    brier_values = []
    for row in rows:
        probs = np.asarray(row["probabilities"], dtype=np.float64)
        one_hot = np.zeros(len(probs)); one_hot[row["target"]] = 1.0
        brier_values.append(np.sum((probs - one_hot) ** 2))
    return {
        "samples": len(rows),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "nll": float(nll),
        "brier": float(np.mean(brier_values)),
        "ece_10": ece(confidence, correct),
    }


def score_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    targets = np.asarray([row["target_score"] for row in rows], dtype=np.float64)
    expected = np.asarray([row["expected_score"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["prediction_score"] for row in rows], dtype=np.float64)
    correlation = spearmanr(targets, expected).statistic
    return {
        "mae_expected": float(np.mean(np.abs(targets - expected))),
        "spearman_expected": float(correlation) if np.isfinite(correlation) else None,
        "quadratic_weighted_kappa": float(cohen_kappa_score(targets, predicted, weights="quadratic")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": "cuda"})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model_load_seconds = time.perf_counter() - load_started
    examples = read_jsonl(args.split)
    if args.max_samples:
        examples = examples[: args.max_samples]
    if not examples:
        raise ValueError("Evaluation split is empty")

    with torch.inference_mode():
        for index in range(args.warmup_samples):
            selected_logits(model, tokenizer, examples[index % len(examples)])
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for example in examples:
            torch.cuda.synchronize(); started = time.perf_counter()
            probs = torch.softmax(selected_logits(model, tokenizer, example) / args.temperature, dim=-1).cpu().numpy()
            torch.cuda.synchronize(); latency = (time.perf_counter() - started) * 1000
            target = answer_index(example); prediction = int(probs.argmax())
            row = {
                "example_id": example.example_id,
                "primitive": example.primitive,
                "target": target,
                "prediction": prediction,
                "target_label": example.answer,
                "prediction_label": example.options[prediction],
                "confidence": float(probs[prediction]),
                "probabilities": probs.tolist(),
                "latency_ms": latency,
                "metadata": example.metadata,
            }
            if example.primitive == "score":
                values = np.asarray(example.score_values, dtype=np.float64)
                row.update({
                    "target_score": float(values[target]),
                    "prediction_score": float(values[prediction]),
                    "expected_score": float(np.dot(probs, values)),
                })
            rows.append(row); latencies.append(latency)

    by_primitive: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_primitive[row["primitive"]].append(row)
    metrics = {
        "model": args.model,
        "adapter": args.adapter,
        "temperature": args.temperature,
        "overall": classification_metrics(rows),
        "by_primitive": {name: classification_metrics(group) for name, group in sorted(by_primitive.items())},
        "score_ordinal": score_metrics(by_primitive.get("score", [])),
        "model_load_seconds": model_load_seconds,
        "warmup_samples": args.warmup_samples,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
        "max_cuda_memory_mib": float(torch.cuda.max_memory_allocated() / 1024**2),
    }
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
