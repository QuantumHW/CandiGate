from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest


def load_rows(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"]


def accuracy(rows: list[dict]) -> float:
    return sum(row["target_label"] == row["prediction_label"] for row in rows) / len(rows)


def breakdown(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["metadata"][key])].append(row)
    return {name: {"rows": len(group), "accuracy": accuracy(group)} for name, group in sorted(groups.items())}


def semantic_summary(rows: list[dict]) -> tuple[dict, dict[str, bool]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["metadata"]["semantic_id"]].append(row)
    majority: dict[str, bool] = {}
    all_correct = 0
    invariant = 0
    for semantic_id, group in groups.items():
        correct = [row["target_label"] == row["prediction_label"] for row in group]
        majority[semantic_id] = sum(correct) >= (len(correct) // 2 + 1)
        all_correct += all(correct)
        invariant += len({row["prediction_label"] for row in group}) == 1
    count = len(groups)
    return ({
        "independent_cases": count,
        "majority_accuracy": sum(majority.values()) / count,
        "all_orders_correct_rate": all_correct / count,
        "order_invariant_prediction_rate": invariant / count,
    }, majority)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = load_rows(args.base)
    lora = load_rows(args.lora)
    base_by_id = {row["example_id"]: row for row in base}
    lora_by_id = {row["example_id"]: row for row in lora}
    if base_by_id.keys() != lora_by_id.keys():
        raise ValueError("Base and LoRA result IDs differ")
    base_semantic, base_majority = semantic_summary(base)
    lora_semantic, lora_majority = semantic_summary(lora)
    ids = sorted(base_majority)
    wrong_to_right = sum(not base_majority[item] and lora_majority[item] for item in ids)
    right_to_wrong = sum(base_majority[item] and not lora_majority[item] for item in ids)
    discordant = wrong_to_right + right_to_wrong
    p_value = binomtest(min(wrong_to_right, right_to_wrong), discordant, 0.5).pvalue if discordant else 1.0
    result = {
        "row_level": {
            "base_accuracy": accuracy(base),
            "lora_accuracy": accuracy(lora),
            "absolute_delta": accuracy(lora) - accuracy(base),
        },
        "base_by_category": breakdown(base, "category"),
        "lora_by_category": breakdown(lora, "category"),
        "base_by_order_seed": breakdown(base, "order_seed"),
        "lora_by_order_seed": breakdown(lora, "order_seed"),
        "base_semantic": base_semantic,
        "lora_semantic": lora_semantic,
        "paired_majority_test": {
            "base_wrong_lora_right": wrong_to_right,
            "base_right_lora_wrong": right_to_wrong,
            "discordant_cases": discordant,
            "two_sided_exact_binomial_p": p_value,
        },
    }
    destination = Path(args.output)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
