# CandiGate-Qwen3-4B v0.1.0-0920 results

## BFCL V4

The frozen evaluation contains 2,371 independent semantic cases and three deterministic candidate orderings per case.

| Model | Accuracy | Macro-F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 70.42% | 0.528 | 1.324 | 0.498 | 0.229 |
| CandiGate v1 | 74.20% | 0.572 | 1.336 | 0.455 | 0.203 |
| CandiGate-Qwen3-4B | 80.29% | 0.654 | 1.375 | 0.367 | 0.174 |
| CandiGate-Qwen3-4B + calibration | 80.29% | 0.654 | 1.104 | 0.361 | 0.166 |

Category highlights:

- `live_irrelevance`: 60.56% accuracy;
- prediction invariance across candidate orders: 93.76%;
- compared with v1: 168 independent cases changed from incorrect to correct and 27 changed from correct to incorrect.

BFCL was used only for evaluation.

## Internal frozen test

The 1,956-example internal frozen test reached 96.27% accuracy. The split combines bilingual intent selection and auditable synthetic Agent decisions across `choice`, `noul`, and `score` primitives.

## Controlled multi-step Agent evaluation

The benchmark contains 16 independent semantic scenarios and three deterministic candidate orderings per scenario. CandiGate-Qwen3-4B reached 79.17% trajectory success with zero unsafe executions.

## Calibration

- `choice` and `noul`: temperature 1.2637946122644546
- `score`: temperature 1.0

## Reproduction notes

Use the repository's stable single-token candidate check, restricted-logit inference, deterministic option-order seeds, held-out calibration split, and frozen test split when reproducing these measurements.
