from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


MULTIPLE = ["BFCL_v4_multiple.json", "BFCL_v4_live_multiple.json"]
IRRELEVANCE = ["BFCL_v4_irrelevance.json", "BFCL_v4_live_irrelevance.json"]
SIMPLE = [
    "BFCL_v4_live_simple.json",
    "BFCL_v4_simple_python.json",
    "BFCL_v4_simple_java.json",
    "BFCL_v4_simple_javascript.json",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_text(question: object) -> str:
    messages: list[str] = []
    if isinstance(question, list):
        for item in question:
            if isinstance(item, list):
                messages.extend(prompt_text(item).splitlines())
            elif isinstance(item, dict) and item.get("content"):
                messages.append(f"{item.get('role', 'user')}: {item['content']}")
    return "\n".join(part for part in messages if part).strip()


def tool_label(function: dict) -> str:
    description = " ".join(str(function.get("description", "")).split())
    return f"{function['name']} — {description}" if description else function["name"]


def answer_name(answer_row: dict) -> str | None:
    names: list[str] = []
    for call in answer_row.get("ground_truth", []):
        if isinstance(call, dict):
            names.extend(call)
    unique = list(dict.fromkeys(names))
    return unique[0] if len(unique) == 1 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfcl-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--profile", choices=("dev", "simple"), default="dev")
    parser.add_argument("--simple-candidate-count", type=int, default=8)
    parser.add_argument("--order-seeds", nargs="+", type=int, default=[101, 202, 303])
    args = parser.parse_args()
    source = Path(args.bfcl_data)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    independent = 0
    skipped = Counter()
    categories = Counter()

    multiple_files = MULTIPLE if args.profile == "dev" else SIMPLE
    irrelevance_files = IRRELEVANCE if args.profile == "dev" else []
    simple_labels: set[str] = set()
    if args.profile == "simple":
        if not 2 <= args.simple_candidate_count <= 26:
            raise ValueError("simple candidate count must be between 2 and 26")
        for filename in SIMPLE:
            for item in load_jsonl(source / filename):
                simple_labels.update(
                    tool_label(function) for function in item.get("function", []) if function.get("name")
                )

    for filename in multiple_files:
        answers = {row["id"]: row for row in load_jsonl(source / "possible_answer" / filename)}
        for item in load_jsonl(source / filename):
            name = answer_name(answers[item["id"]])
            functions = item.get("function", [])
            by_name = {function.get("name"): tool_label(function) for function in functions if function.get("name")}
            if name is None or name not in by_name:
                skipped[f"{filename}:unsupported_ground_truth"] += 1
                continue
            options = list(by_name.values())
            if args.profile == "simple" and name in by_name:
                answer = by_name[name]
                distractors = sorted(
                    (label for label in simple_labels if label != answer),
                    key=lambda label: hashlib.sha256(f"{item['id']}:{label}".encode()).hexdigest(),
                )
                options = [answer, *distractors[: args.simple_candidate_count - 1]]
            if len(options) < 2 or len(options) > 26 or len(set(options)) != len(options):
                skipped[f"{filename}:invalid_options"] += 1
                continue
            text = prompt_text(item.get("question"))
            if not text:
                skipped[f"{filename}:empty_question"] += 1
                continue
            independent += 1
            category = filename.removesuffix(".json")
            categories[category] += 1
            answer = by_name[name]
            for seed in args.order_seeds:
                variant = options.copy()
                random.Random(f"{seed}:{item['id']}").shuffle(variant)
                rows.append({
                    "state": text,
                    "question": "Which available tool is the best match for this request?",
                    "options": variant,
                    "answer": answer,
                    "example_id": f"bfcl:{item['id']}:order-{seed}",
                    "primitive": "choice",
                    "score_values": None,
                    "metadata": {"source": "BFCL-v4", "category": category,
                                 "semantic_id": item["id"], "order_seed": seed},
                })

    for filename in irrelevance_files:
        for item in load_jsonl(source / filename):
            functions = item.get("function", [])
            options = [tool_label(function) for function in functions if function.get("name")]
            answer = "none of the available tools"
            options.append(answer)
            if len(options) < 2 or len(options) > 26 or len(set(options)) != len(options):
                skipped[f"{filename}:invalid_options"] += 1
                continue
            text = prompt_text(item.get("question"))
            if not text:
                skipped[f"{filename}:empty_question"] += 1
                continue
            independent += 1
            category = filename.removesuffix(".json")
            categories[category] += 1
            for seed in args.order_seeds:
                variant = options.copy()
                random.Random(f"{seed}:{item['id']}").shuffle(variant)
                rows.append({
                    "state": text,
                    "question": "Which available tool is the best match for this request?",
                    "options": variant,
                    "answer": answer,
                    "example_id": f"bfcl:{item['id']}:order-{seed}",
                    "primitive": "choice",
                    "score_values": None,
                    "metadata": {"source": "BFCL-v4", "category": category,
                                 "semantic_id": item["id"], "order_seed": seed},
                })

    test_path = output / "test.jsonl"
    with test_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": "Berkeley Function Calling Leaderboard V4",
        "license": "Apache-2.0",
        "revision": args.revision,
        "profile": args.profile,
        "simple_candidate_count": args.simple_candidate_count if args.profile == "simple" else None,
        "role": "frozen external evaluation only; not training data",
        "independent_semantic_cases": independent,
        "order_seeds": args.order_seeds,
        "evaluation_rows": len(rows),
        "categories": dict(sorted(categories.items())),
        "skipped": dict(sorted(skipped.items())),
        "test_sha256": sha256(test_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
