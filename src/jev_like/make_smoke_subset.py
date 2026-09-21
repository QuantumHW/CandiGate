from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(rows: list[dict], pairs_per_tool: int, base_samples: int, seed: int) -> list[dict]:
    synthetic_by_tool: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    base: list[dict] = []
    for row in rows:
        metadata = row.get("metadata", {})
        if metadata.get("source") == "synthetic-hard-tools-v3":
            synthetic_by_tool[metadata["target_family"]][metadata["pair_id"]].append(row)
        else:
            base.append(row)

    selected: list[dict] = []
    for tool, pairs in sorted(synthetic_by_tool.items()):
        ordered = sorted(pairs, key=lambda pair_id: stable_seed(seed, tool, pair_id, "smoke-pair"))
        chosen = ordered[:pairs_per_tool]
        if len(chosen) != pairs_per_tool:
            raise ValueError(f"not enough pairs for {tool}")
        for pair_id in chosen:
            pair_rows = pairs[pair_id]
            if len(pair_rows) != 2 or {row["metadata"]["relevant"] for row in pair_rows} != {False, True}:
                raise ValueError(f"invalid counterfactual pair {pair_id}")
            selected.extend(pair_rows)

    ordered_base = sorted(base, key=lambda row: stable_seed(seed, row["example_id"], "smoke-base"))
    selected.extend(ordered_base[:base_samples])
    random.Random(stable_seed(seed, "smoke-shuffle")).shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-pairs-per-tool", type=int, default=5)
    parser.add_argument("--validation-pairs-per-tool", type=int, default=2)
    parser.add_argument("--train-base-samples", type=int, default=160)
    parser.add_argument("--validation-base-samples", type=int, default=56)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    source = Path(args.input_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "train": (args.train_pairs_per_tool, args.train_base_samples),
        "validation": (args.validation_pairs_per_tool, args.validation_base_samples),
    }
    manifest = {"seed": args.seed, "source_dir": str(source), "splits": {}}
    for split, (pairs_per_tool, base_samples) in settings.items():
        source_path = source / f"{split}.jsonl"
        rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line]
        selected = select_rows(rows, pairs_per_tool, base_samples, stable_seed(args.seed, split))
        destination = output / f"{split}.jsonl"
        with destination.open("x", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        synthetic = [row for row in selected if row.get("metadata", {}).get("source") == "synthetic-hard-tools-v3"]
        manifest["splits"][split] = {
            "samples": len(selected),
            "synthetic_samples": len(synthetic),
            "base_samples": len(selected) - len(synthetic),
            "tools": len({row["metadata"]["target_family"] for row in synthetic}),
            "relevance": dict(sorted(Counter(str(row["metadata"]["relevant"]) for row in synthetic).items())),
            "primitives": dict(sorted(Counter(row.get("primitive", "choice") for row in selected).items())),
            "source_sha256": sha256(source_path),
            "sha256": sha256(destination),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
