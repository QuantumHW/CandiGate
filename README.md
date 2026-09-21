# CandiGate

[中文](./README.md) | [English](./README_EN.md)

CandiGate（**Candidate Logit Gate**）是一套面向动态候选决策的训练、推理、校准与 Agent 评测工具。项目以 Qwen3-4B 为基座，通过候选约束 LoRA 复现 Jev 的可观察决策行为：一次因果语言模型前向计算后，只在经过验证的候选标记上计算概率，而不是生成自由文本答案。

已发布模型：

- [Hugging Face：CullenYap/CandiGate-Qwen3-4B](https://huggingface.co/CullenYap/CandiGate-Qwen3-4B)
- [ModelScope：QuantumCloud/CandiGate-Qwen3-4B](https://modelscope.cn/models/QuantumCloud/CandiGate-Qwen3-4B)

模型版本为 `v0.1.0-0920`，以 [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) 为基座训练。

## 核心能力

- `choice`：从动态候选集合中选择一个决策；
- `noul`：二元命题判断；
- `score`：输出有序等级分布与期望分数；
- 候选标记单 token 稳定性检查；
- LoRA 训练、独立温度校准和冻结测试评测；
- MASSIVE 数据转换、BFCL V4 外部评测与候选顺序重排分析；
- 可控多步 Agent 的路由、授权、重试和完成判断评测。

## 工作原理

给定状态、问题和动态候选项，CandiGate 将候选映射到 `A`–`Z` 标记，并验证每个标记在完整提示词后都是唯一且稳定的单 token continuation。模型随后完成一次前向计算，从末位 logits 中提取候选 token 的分数，再经过 softmax 与温度校准得到概率分布。

```text
状态 + 问题 + 动态候选
        ↓
单 token 候选门槛
        ↓
Qwen3-4B + CandiGate LoRA
        ↓
restricted logits → 概率分布 → 决策
```

## 快速开始

需要 Python 3.11 或 3.12、`uv` 和支持 BF16 的 CUDA GPU。

```bash
git clone https://github.com/QuantumHW/CandiGate.git
cd CandiGate
uv sync --locked
```

从上方任一模型平台下载 CandiGate adapter，并准备 Qwen3-4B 基座模型。首次使用一个 tokenizer 或提示模板时，先运行候选门槛检查：

```bash
uv run candigate-check-tokens \
  --model /path/to/Qwen3-4B \
  --count 8
```

运行仓库内置请求：

```bash
uv run candigate-predict \
  --model /path/to/Qwen3-4B \
  --adapter /path/to/CandiGate-Qwen3-4B \
  --input examples/request.json \
  --temperature 1.2637946123
```

输出包含最终选择、置信度以及所有候选的概率。`score` 原语还会返回 `expected_score`。

## 输入格式

```json
{
  "state": "用户需要查询成都当前天气。可用工具中只有天气查询能够访问实时天气数据。",
  "question": "下一步应选择哪个工具？",
  "options": ["天气查询", "日历查询", "拒绝执行"],
  "primitive": "choice"
}
```

`options` 支持 2–26 个动态候选。`score` 请求还需提供与候选一一对应的 `score_values`。

## 训练与评测

训练数据采用 JSONL，每行字段与推理请求一致，并额外包含 `answer` 和 `example_id`。建议保持 `train`、`validation`、`calibration`、`test` 四个独立划分。

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

开发测试：

```bash
uv sync --locked --group dev
uv run pytest
```

## 已发布评测

BFCL V4 冻结评测包含 2,371 个独立语义案例，每个案例使用三种候选顺序：

| 模型 | Accuracy | Macro-F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 70.42% | 0.528 | 1.324 | 0.498 | 0.229 |
| CandiGate v1 | 74.20% | 0.572 | 1.336 | 0.455 | 0.203 |
| **CandiGate-Qwen3-4B** | **80.29%** | **0.654** | **1.375** | **0.367** | **0.174** |
| **CandiGate + 校准** | **80.29%** | **0.654** | **1.104** | **0.361** | **0.166** |

另外：

- BFCL `live_irrelevance` 准确率为 60.56%；
- 三种候选顺序下预测不变率为 93.76%；
- 1,956 条内部冻结测试准确率为 96.27%；
- 16 个独立多步 Agent 场景、三次候选重排的轨迹成功率为 79.17%，未发生不安全执行。

BFCL 只用于评测。发布模型使用的 `choice`/`noul` 温度为 `1.2637946123`，`score` 温度为 `1.0`。

## 评测范围

当前版本在 BFCL `live_irrelevance` 上仍有 39.44% 的错误；多步 Agent 评测包含 16 个独立语义场景；`score` 与其他决策原语之间可观察到多任务干扰。涉及外部副作用的 Agent 操作应结合独立的权限控制和参数校验。

## 项目结构

```text
src/jev_like/       核心库与命令行工具
tests/              单元测试
examples/           推理请求示例
configs/            发布模型的校准参数
docs/               结果摘要与数据说明
```

## 许可证与引用

项目代码以 [Apache License 2.0](./LICENSE) 发布。Qwen3-4B、MASSIVE 与 BFCL 适用各自的许可证和署名要求，详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。
