# CandiGate

[简体中文](./README.md) | English

CandiGate (**Candidate Logit Gate**) is a toolkit for training, inference, calibration, and Agent evaluation over dynamic decision candidates. The current model, **CandiGate-Qwen3-4B v0.2.0-0921**, is trained from `Qwen/Qwen3-4B` to reproduce Jev's observable decision behavior: one forward pass produces a distribution over validated candidate markers.

Published models: [Hugging Face](https://huggingface.co/CullenYap/CandiGate-Qwen3-4B) · [ModelScope](https://modelscope.cn/models/QuantumCloud/CandiGate-Qwen3-4B)

## Current release

| Field | Value |
|---|---|
| Model version | `v0.2.0-0921` |
| Base model | `Qwen/Qwen3-4B` |
| Format | PEFT LoRA adapter |
| Training | 1 epoch, repeated with 3 random seeds |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Published decision primitives | `choice`, `noul` |
| Recommended temperature | `1.5496953009` |
| Languages | Simplified Chinese and English |
| License | Apache-2.0 |

## v0.2 results

### Same-protocol comparisons

| Model / inference mode | v0.2 validation accuracy |
|---|---:|
| Qwen3-4B Generate | 72.79% |
| Qwen3-4B candidate logits | 75.13% |
| **CandiGate v0.2 selected seed** | **97.66%** |

| Model | BFCL multiple/irrelevance | BFCL live_irrelevance |
|---|---:|---:|
| Qwen3-4B | 70.42% | 38.11% |
| CandiGate v0.1.0 | 80.29% | 60.56% |
| **CandiGate v0.2.0** | **81.57%** | **61.32%** |

| Agent v2 inference mode | Episode success | Decision accuracy | Unsafe executions |
|---|---:|---:|---:|
| Qwen3-4B Generate | 50.00% | 79.72% | 1 / 144 |
| Qwen3-4B candidate logits | 50.00% | 79.72% | 1 / 144 |
| **CandiGate v0.2 candidate logits** | **75.69%** | **92.47%** | **0 / 144** |

### v0.2 standalone results

| Evaluation | Result |
|---|---:|
| Three-seed validation accuracy | 97.26% ± 0.29% |
| Frozen test accuracy (1,560 rows) | 96.35% |
| Frozen test ECE | 0.25% |
| Hard dynamic-tool semantic accuracy | 99.77% |
| Hard dynamic-tool no-tool accuracy | 100.00% |
| BFCL simple fixed 8-candidate holdout | 99.22% |

Comparison tables use matching data and protocols. The v0.1 internal test contains 1,956 rows, while v0.2 uses a regrouped 1,560-row frozen test; each remains a standalone release result. See the [v0.2 result record](./docs/results-v0.2.0-0921.md) for the complete protocol.

## Features

- `choice` decisions over 2–26 dynamic candidates;
- `noul` binary proposition judgments;
- unique, stable single-token continuation validation;
- LoRA training, held-out temperature calibration, and frozen-test evaluation;
- MASSIVE conversion and semantic-cluster-isolated hard Agent data generation;
- BFCL V4 fixed-candidate evaluation and option-order stability analysis;
- controlled multi-step Agent evaluation for routing, authorization, retries, and completion.

The toolkit retains experimental `score` code, while the current v0.2 model publishes `choice` and `noul` capabilities.

## How it works

```text
state + question + dynamic candidates
                 ↓
single-token candidate gate
                 ↓
Qwen3-4B + CandiGate LoRA
                 ↓
restricted logits → calibration → decision distribution
```

CandiGate maps candidates to `A`–`Z`, verifies token boundaries after the complete prompt, and extracts candidate scores from the final-position logits. This release formally validates and uses `A` through `I`.

## Quick start

CandiGate requires Python 3.11 or 3.12, `uv`, and a CUDA GPU with BF16 support.

```bash
git clone https://github.com/QuantumHW/CandiGate.git
cd CandiGate
uv sync --locked
```

Download the CandiGate adapter from either model platform above and prepare the Qwen3-4B base model. Validate a new tokenizer or prompt template before first use:

```bash
uv run candigate-check-tokens \
  --model /path/to/Qwen3-4B \
  --count 9
```

Run the included request:

```bash
uv run candigate-predict \
  --model /path/to/Qwen3-4B \
  --adapter /path/to/CandiGate-Qwen3-4B \
  --input examples/request.json \
  --temperature 1.5496953009
```

The output contains the selected candidate, confidence, and the full candidate probability distribution.

## Input format

```json
{
  "state": "The user needs the current weather in Chengdu. Only the weather tool can access live weather data.",
  "question": "Which tool should be selected next?",
  "options": ["Weather lookup", "Calendar lookup", "Reject"],
  "primitive": "choice"
}
```

## Training and evaluation

Training data uses JSONL with the request fields plus `answer` and `example_id`. Split data by semantic cluster into independent `train`, `validation`, `calibration`, and `test` partitions.

```bash
uv run candigate-train \
  --model /path/to/Qwen3-4B \
  --train data/splits/train.jsonl \
  --validation data/splits/validation.jsonl \
  --output artifacts/adapters/candigate

uv run candigate-evaluate \
  --model /path/to/Qwen3-4B \
  --adapter artifacts/adapters/candigate \
  --split data/splits/calibration.jsonl \
  --output results/raw/calibration-eval.json

uv run candigate-calibrate \
  --input results/raw/calibration-eval.json \
  --output artifacts/calibration/temperature.json
```

Developer tests:

```bash
uv sync --locked --group dev
uv run pytest -q
```

## Data

v0.2 training combines Chinese and English intent examples from MASSIVE v1.1, project-generated Agent policy examples, and semantic-cluster-isolated hard positives and negatives for 36 tool families. MASSIVE v1.1 is licensed under CC BY 4.0. Synthetic examples and labels are produced by published, auditable construction rules.

## Known scope

v0.2 focuses on dynamic tool selection, no-tool rejection, and binary control decisions. Transient database retries, deterministic permission escalation, safe deferral, and Chinese missing-parameter routing are priority data-expansion areas. The BFCL simple result uses this project's fixed 8-candidate protocol for reproducible comparisons under consistent conditions.

## Repository layout

```text
src/jev_like/       core library, data construction, and command-line tools
tests/              unit tests
examples/           inference request examples
configs/            calibration values for published models
docs/               per-release results and experiment notes
```

## Version history

- `v0.2.0-0921`: Current release. Adds semantic-cluster-isolated hard examples, three-seed model selection, independent calibration and frozen testing, and expands controlled Agent evaluation to 48 scenarios.
- `v0.1.0-0920`: Initial public release establishing candidate-logit decisions, LoRA inference, calibration, and the two-platform model release workflow.

## License and attribution

The project code is released under the [Apache License 2.0](./LICENSE). Qwen3-4B, MASSIVE, and BFCL retain their respective licenses and attribution requirements; see [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md). Base-model paper: Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), 2025.
