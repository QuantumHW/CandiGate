# CandiGate-Qwen3-4B v0.2.0-0921 results

This file records the detailed results behind the current release. The model is a PEFT LoRA adapter trained from `Qwen/Qwen3-4B` for `choice` and `noul` candidate decisions.

## Training and model selection

- 5,748 training rows and 1,154 validation rows;
- one epoch per run, repeated with seeds 20260921, 20260922, and 20260923;
- seed 20260923 selected by validation NLL;
- three-seed validation accuracy: 97.26% mean, 0.29 percentage-point population standard deviation;
- selected validation NLL: 0.0887.

One epoch was retained because all three runs already converged to strong validation, no-tool, and external development-regression results. Additional epochs on the same synthetic templates were not used as a substitute for broader data coverage.

## Frozen internal test

The 1,560-row test split was evaluated once after model selection and calibration.

| Metric | Result |
|---|---:|
| Accuracy | 96.35% |
| NLL | 0.1334 |
| Brier | 0.0589 |
| ECE | 0.25% |
| Dynamic-tool semantic accuracy | 99.77% |
| Dynamic-tool semantic Macro-F1 | 99.76% |
| No-tool accuracy | 100.00% |

On an RTX 4090 with batch 1, concurrency 1, and warm-up, p50/p95/p99 latency was 26.55/36.40/37.74 ms.

## BFCL V4

The multiple/irrelevance development regression contains 2,371 independent cases and three deterministic candidate orders per case. v0.2 reached 81.57% accuracy and 61.32% on `live_irrelevance`, compared with 80.29% and 60.56% for v0.1.

The simple/live_simple holdout contains 808 independent cases. Each case uses the correct function plus seven deterministic distractors from the frozen simple corpus and three candidate orders. Accuracy was 99.22%, with 98.64% of cases correct under all three orders. This fixed eight-candidate project protocol supports reproducible comparisons under consistent conditions.

## Controlled Agent v2

The benchmark contains 48 independent scenarios and three candidate orders, for 144 episodes per mode.

| Mode | Episode success | Decision accuracy | Unsafe executions |
|---|---:|---:|---:|
| Qwen3-4B Generate | 50.00% | 79.72% | 1 |
| Qwen3-4B candidate logits | 50.00% | 79.72% | 1 |
| CandiGate v0.2 candidate logits | 75.69% | 92.47% | 0 |

The remaining failures concentrate on transient database retry, deterministic permission escalation, safe deferral, and Chinese missing-parameter routing. These categories are data-expansion targets for the next release.

## Calibration

The selected global temperature is `1.5496953009346224`, fitted only on the 1,153-row calibration split. Calibration NLL changed from 0.1069 to 0.0874.
