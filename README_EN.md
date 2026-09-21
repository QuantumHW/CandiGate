# CandiGate

[中文](./README.md) | [English](./README_EN.md)

CandiGate (**Candidate Logit Gate**) is a toolkit for training, inference, calibration, and Agent evaluation over dynamic decision candidates. Built on Qwen3-4B, it reproduces Jev's observable decision behavior with a candidate-constrained LoRA: one causal-LM forward pass is followed by probability computation over validated candidate markers instead of free-form answer generation.

Published model:

- [Hugging Face: CullenYap/CandiGate-Qwen3-4B](https://huggingface.co/CullenYap/CandiGate-Qwen3-4B)
- [ModelScope: QuantumCloud/CandiGate-Qwen3-4B](https://modelscope.cn/models/QuantumCloud/CandiGate-Qwen3-4B)

Model version `v0.1.0-0920` is trained from [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B).

## Features

- `choice` for selecting one decision from dynamic candidates;
- `noul` for binary proposition judgments;
- `score` for ordered distributions and expected scores;
- stable single-token candidate validation;
- LoRA training, held-out temperature calibration, and frozen-test evaluation;
- MASSIVE conversion, BFCL V4 evaluation, and candidate-order analysis;
- controlled multi-step Agent evaluation for routing, authorization, retry, and completion decisions.

## How it works

CandiGate maps dynamic options to `A`–`Z` markers and verifies that each marker is a unique, stable single-token continuation after the complete prompt. It then extracts those token scores from the final-position logits of one forward pass and applies softmax plus temperature calibration.

```text
state + question + dynamic options
              ↓
single-token candidate gate
              ↓
Qwen3-4B + CandiGate LoRA
              ↓
restricted logits → probabilities → decision
```

## Quick start

CandiGate requires Python 3.11 or 3.12, `uv`, and a CUDA GPU with BF16 support.

```bash
git clone https://github.com/QuantumHW/CandiGate.git
cd CandiGate
uv sync --locked
```

Download the CandiGate adapter from either model platform above and prepare the Qwen3-4B base model. Validate a tokenizer and prompt template before first use:

```bash
uv run candigate-check-tokens \
  --model /path/to/Qwen3-4B \
  --count 8
```

Run the included request:

```bash
uv run candigate-predict \
  --model /path/to/Qwen3-4B \
  --adapter /path/to/CandiGate-Qwen3-4B \
  --input examples/request.json \
  --temperature 1.2637946123
```

The output contains the selected option, confidence, and the full candidate probability distribution. The `score` primitive also returns `expected_score`.

## Input format

```json
{
  "state": "The user needs the current weather in Chengdu. Only the weather tool can access live weather data.",
  "question": "Which tool should be selected next?",
  "options": ["Weather lookup", "Calendar lookup", "Reject"],
  "primitive": "choice"
}
```

`options` accepts 2–26 dynamic candidates. A `score` request also supplies one numeric `score_values` entry per candidate.

## Training and evaluation

Training data uses JSONL with the request fields plus `answer` and `example_id`. Keep independent `train`, `validation`, `calibration`, and `test` splits.

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

uv run candigate-evaluate \
  --model /path/to/Qwen3-4B \
  --adapter artifacts/adapters/candigate \
  --temperature 1.2637946123 \
  --split data/splits/test.jsonl \
  --output results/raw/test-eval.json
```

Developer tests:

```bash
uv sync --locked --group dev
uv run pytest
```

## Published evaluation

The frozen BFCL V4 evaluation contains 2,371 independent semantic cases, each evaluated under three candidate orderings:

| Model | Accuracy | Macro-F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 70.42% | 0.528 | 1.324 | 0.498 | 0.229 |
| CandiGate v1 | 74.20% | 0.572 | 1.336 | 0.455 | 0.203 |
| **CandiGate-Qwen3-4B** | **80.29%** | **0.654** | **1.375** | **0.367** | **0.174** |
| **CandiGate + calibration** | **80.29%** | **0.654** | **1.104** | **0.361** | **0.166** |

Additional results:

- 60.56% accuracy on BFCL `live_irrelevance`;
- 93.76% prediction invariance across three candidate orders;
- 96.27% accuracy on the 1,956-example frozen internal test;
- 79.17% trajectory success and zero unsafe executions across 16 independent multi-step Agent scenarios under three candidate orderings.

BFCL is evaluation-only. The published model uses temperature `1.2637946123` for `choice`/`noul` and `1.0` for `score`.

## Evaluation scope

The current release has a 39.44% error rate on BFCL `live_irrelevance`; the multi-step Agent evaluation contains 16 independent semantic scenarios; and `score` exhibits measurable multi-task interference with the other decision primitives. Agent actions with external side effects should use independent authorization controls and argument validation.

## Repository layout

```text
src/jev_like/       core library and command-line tools
tests/              unit tests
examples/           inference request examples
configs/            published calibration parameters
docs/               result summaries and data notes
```

## License and attribution

The project code is released under the [Apache License 2.0](./LICENSE). Qwen3-4B, MASSIVE, and BFCL retain their respective licenses and attribution requirements; see [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
