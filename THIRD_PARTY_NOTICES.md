# Third-party notices

CandiGate source code is licensed under the Apache License 2.0. The following third-party models and datasets are referenced by the training and evaluation tools and remain governed by their own licenses.

## Qwen3-4B

- Project: Qwen3-4B
- Provider: Qwen Team, Alibaba Cloud
- Source: https://huggingface.co/Qwen/Qwen3-4B
- License: Apache License 2.0

Qwen3-4B is the base model used to train CandiGate-Qwen3-4B. Base model weights are distributed separately.

## MASSIVE

- Project: MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
- Provider: Amazon Science
- Source: https://github.com/alexa/massive
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

CandiGate's data preparation tool converts selected MASSIVE utterances into dynamic intent-choice examples. The dataset itself is not included in this repository.

## Berkeley Function Calling Leaderboard

- Project: Berkeley Function Calling Leaderboard (BFCL)
- Provider: Gorilla LLM, UC Berkeley
- Source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- License: Apache License 2.0

BFCL V4 is used as a frozen external evaluation set and is not included in this repository.
