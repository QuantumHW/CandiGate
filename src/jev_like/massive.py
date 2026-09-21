from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_LOCALES = ("en-US", "zh-CN")
PARTITION_FOR_OUTPUT = {
    "train": "train",
    "validation": "dev",
    "calibration": "dev",
    "test": "test",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_locales(archive: Path, locales: tuple[str, ...]) -> list[dict]:
    wanted = {f"{locale}.jsonl": locale for locale in locales}
    rows: list[dict] = []
    found: set[str] = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            basename = Path(member.name).name
            if basename not in wanted or not member.isfile():
                continue
            locale = wanted[basename]
            found.add(locale)
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unable to read {member.name}")
            for raw_line in stream:
                if raw_line.strip():
                    row = json.loads(raw_line)
                    if row["locale"] != locale:
                        raise ValueError(f"Locale mismatch in {member.name}")
                    rows.append(row)
    missing = set(locales) - found
    if missing:
        raise ValueError(f"Archive is missing locales: {sorted(missing)}")
    return rows


def deduplicate(rows: list[dict]) -> tuple[list[dict], dict]:
    priority = {"test": 0, "dev": 1, "train": 2}
    seen: dict[tuple[str, str], dict] = {}
    kept: list[dict] = []
    removed_by_partition: Counter = Counter()
    conflicting_labels = 0
    for row in sorted(rows, key=lambda item: priority[item["partition"]]):
        key = (row["locale"], row["utt"].strip().casefold())
        previous = seen.get(key)
        if previous is None:
            seen[key] = row
            kept.append(row)
            continue
        removed_by_partition[row["partition"]] += 1
        if previous["intent"] != row["intent"]:
            conflicting_labels += 1
    return kept, {
        "removed_by_partition": dict(sorted(removed_by_partition.items())),
        "conflicting_labels": conflicting_labels,
    }


def label_text(intent: str) -> str:
    return intent.replace("_", " ")


def make_options(
    intent: str,
    scenario: str,
    intents_by_scenario: dict[str, list[str]],
    all_intents: list[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    if count < 2 or count > 26:
        raise ValueError("candidate count must be between 2 and 26")
    same_scenario = [item for item in intents_by_scenario[scenario] if item != intent]
    rng.shuffle(same_scenario)
    selected = [intent, *same_scenario[: min(3, count - 1)]]
    remaining = [item for item in all_intents if item not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise ValueError("not enough distinct intents to build candidates")
    rng.shuffle(selected)
    return [label_text(item) for item in selected]


def capped_sample(
    groups: dict[tuple[str, str], list[dict]],
    per_intent: int,
    rng: random.Random,
) -> list[dict]:
    sampled: list[dict] = []
    for key in sorted(groups):
        items = groups[key].copy()
        rng.shuffle(items)
        sampled.extend(items[:per_intent])
    rng.shuffle(sampled)
    return sampled


def split_dev(
    groups: dict[tuple[str, str], list[dict]],
    validation_cap: int,
    calibration_cap: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    validation: list[dict] = []
    calibration: list[dict] = []
    for key in sorted(groups):
        items = groups[key].copy()
        rng.shuffle(items)
        if len(items) == 1:
            validation.extend(items)
            continue
        validation_count = min(validation_cap, max(1, len(items) // 2))
        calibration_count = min(calibration_cap, len(items) - validation_count)
        validation.extend(items[:validation_count])
        calibration.extend(items[validation_count : validation_count + calibration_count])
    rng.shuffle(validation)
    rng.shuffle(calibration)
    return validation, calibration


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--locales", nargs="+", default=list(DEFAULT_LOCALES))
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--train-per-intent", type=int, default=10)
    parser.add_argument("--validation-per-intent", type=int, default=2)
    parser.add_argument("--calibration-per-intent", type=int, default=2)
    parser.add_argument("--test-per-intent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    archive = Path(args.archive)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected = [output / f"{name}.jsonl" for name in PARTITION_FOR_OUTPUT]
    expected.append(output / "manifest.json")
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    locales = tuple(args.locales)
    source_rows = read_locales(archive, locales)
    source_row_count = len(source_rows)
    source_rows, deduplication = deduplicate(source_rows)
    intents = sorted({row["intent"] for row in source_rows})
    intent_scenarios: dict[str, str] = {}
    intents_by_scenario: dict[str, list[str]] = defaultdict(list)
    for row in source_rows:
        previous = intent_scenarios.setdefault(row["intent"], row["scenario"])
        if previous != row["scenario"]:
            raise ValueError(f"Intent {row['intent']} occurs in multiple scenarios")
    for intent, scenario in intent_scenarios.items():
        intents_by_scenario[scenario].append(intent)

    grouped: dict[str, dict[tuple[str, str], list[dict]]] = {
        partition: defaultdict(list) for partition in ("train", "dev", "test")
    }
    for row in source_rows:
        grouped[row["partition"]][(row["locale"], row["intent"])].append(row)

    rng = random.Random(args.seed)
    validation_rows, calibration_rows = split_dev(
        grouped["dev"], args.validation_per_intent, args.calibration_per_intent, rng
    )
    selections = {
        "train": capped_sample(grouped["train"], args.train_per_intent, rng),
        "validation": validation_rows,
        "calibration": calibration_rows,
        "test": capped_sample(grouped["test"], args.test_per_intent, rng),
    }

    output_hashes: dict[str, str] = {}
    summary: dict[str, dict] = {}
    for split, selected in selections.items():
        converted: list[dict] = []
        for row in selected:
            options = make_options(
                row["intent"], row["scenario"], intents_by_scenario,
                intents, args.candidate_count, rng,
            )
            answer = label_text(row["intent"])
            converted.append({
                "state": row["utt"],
                "question": "Which intent best describes the user's request?",
                "options": options,
                "answer": answer,
                "example_id": f"massive-{split}-{row['locale']}-{row['id']}",
            })
        if len({row["example_id"] for row in converted}) != len(converted):
            raise ValueError(f"Duplicate example IDs in {split}")
        if any(row["answer"] not in row["options"] for row in converted):
            raise ValueError(f"Answer missing from candidate set in {split}")
        path = output / f"{split}.jsonl"
        write_jsonl(path, converted)
        output_hashes[path.name] = sha256(path)
        summary[split] = {
            "samples": len(converted),
            "locales": dict(sorted(Counter(row["locale"] for row in selected).items())),
            "intents": len({row["intent"] for row in selected}),
            "per_locale_intent_min": min(
                Counter((row["locale"], row["intent"]) for row in selected).values()
            ),
            "per_locale_intent_max": max(
                Counter((row["locale"], row["intent"]) for row in selected).values()
            ),
        }

    manifest = {
        "dataset": "Amazon MASSIVE v1.1",
        "source_url": "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
        "license": "CC BY 4.0",
        "archive_sha256": sha256(archive),
        "locales": list(locales),
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "intent_count": len(intents),
        "source_rows_before_deduplication": source_row_count,
        "source_rows_after_deduplication": len(source_rows),
        "deduplication": deduplication,
        "split_policy": "official train/test; official dev deterministically separated into validation and calibration",
        "summary": summary,
        "output_sha256": output_hashes,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
