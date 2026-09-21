# CandiGate

简体中文 | [English](./README_EN.md)

CandiGate（**Candidate Logit Gate**）是一套面向动态候选决策的训练、推理、校准与 Agent 评测工具。当前模型 **CandiGate-Qwen3-4B v0.2.0-0921** 基于 `Qwen/Qwen3-4B` 训练，用于复现 Jev 的可观察决策行为：模型通过一次前向计算，在经过验证的候选标记上输出决策分布。

已发布模型：[Hugging Face](https://huggingface.co/CullenYap/CandiGate-Qwen3-4B) · [ModelScope](https://modelscope.cn/models/QuantumCloud/CandiGate-Qwen3-4B)

## 当前版本

| 项目 | 内容 |
|---|---|
| 模型版本 | `v0.2.0-0921` |
| 基座模型 | `Qwen/Qwen3-4B` |
| 模型形式 | PEFT LoRA adapter |
| 训练 | 1 epoch，3 个随机种子重复 |
| LoRA | rank 16，alpha 32，dropout 0.05 |
| 当前模型决策原语 | `choice`、`noul` |
| 推荐温度 | `1.5496953009` |
| 语言 | 简体中文、英文 |
| 许可证 | Apache-2.0 |

## v0.2 结果

| 评测 | 结果 |
|---|---:|
| 三种子 validation Accuracy | 97.26% ± 0.29% |
| 冻结 test Accuracy（1,560 条） | 96.35% |
| 冻结 test ECE | 0.25% |
| 困难动态工具语义 Accuracy | 99.77% |
| 困难动态工具 no-tool Accuracy | 100.00% |
| BFCL V4 multiple/irrelevance 开发回归 | 81.57% |
| BFCL V4 simple 固定 8 候选留出 | 99.22% |
| 受控 Agent v2 episode 成功率 | 75.69% |
| 受控 Agent v2 决策准确率 | 92.47% |
| 受控 Agent v2 不安全执行 | 0 / 144 |

内部冻结 test 在 RTX 4090、batch 1、concurrency 1、预热后测得 p50/p95/p99 为 26.55/36.40/37.74 ms。完整协议与限制见 [v0.2 结果记录](./docs/results-v0.2.0-0921.md)。

## 核心能力

- 在 2–26 个动态候选中执行 `choice` 决策；
- 执行 `noul` 二元命题判断；
- 验证候选标记是否为唯一、稳定的单 token continuation；
- 进行 LoRA 训练、独立温度校准和冻结测试评测；
- 转换 MASSIVE 数据并构造语义簇隔离的 Agent 困难正负例；
- 运行 BFCL V4 固定候选评测和候选顺序稳定性分析；
- 对受控多步 Agent 的路由、授权、重试和完成判断进行评测。

工具包仍保留 `score` 实验代码，当前 v0.2 模型只发布 `choice` 与 `noul` 能力。

## 工作原理

```text
状态 + 问题 + 动态候选
        ↓
单 token 候选门槛
        ↓
Qwen3-4B + CandiGate LoRA
        ↓
restricted logits → 温度校准 → 决策分布
```

CandiGate 将候选映射到 `A`–`Z` 标记，验证标记在完整提示词后的 token 边界，再从末位 logits 中提取候选分数。本版本正式验证并使用 `A`–`I`。

## 快速开始

需要 Python 3.11 或 3.12、`uv` 和支持 BF16 的 CUDA GPU。

```bash
git clone https://github.com/QuantumHW/CandiGate.git
cd CandiGate
uv sync --locked
```

从上方任一模型平台下载 CandiGate adapter，并准备 Qwen3-4B 基座模型。首次使用新的 tokenizer 或提示模板时，先运行候选门槛检查：

```bash
uv run candigate-check-tokens \
  --model /path/to/Qwen3-4B \
  --count 9
```

运行仓库内置请求：

```bash
uv run candigate-predict \
  --model /path/to/Qwen3-4B \
  --adapter /path/to/CandiGate-Qwen3-4B \
  --input examples/request.json \
  --temperature 1.5496953009
```

输出包含所选候选、置信度和完整候选概率分布。

## 输入格式

```json
{
  "state": "用户需要查询成都当前天气。可用工具中只有天气查询能够访问实时天气数据。",
  "question": "下一步应选择哪个工具？",
  "options": ["天气查询", "日历查询", "拒绝执行"],
  "primitive": "choice"
}
```

## 训练与评测

训练数据采用 JSONL，每行在推理请求字段外增加 `answer` 和 `example_id`。数据应按语义簇划分为独立的 `train`、`validation`、`calibration` 与 `test`。

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

开发测试：

```bash
uv sync --locked --group dev
uv run pytest -q
```

## 数据

v0.2 训练数据由 MASSIVE v1.1 的中英文意图样本、项目生成的 Agent 策略样本，以及 36 类工具的语义簇隔离困难正负例组成。MASSIVE v1.1 采用 CC BY 4.0；合成样本及标签由公开、可审计的规则生成。

## 已知范围

v0.2 聚焦动态工具选择、无合适工具拒绝和二元控制判断。Agent v2 中的临时数据库重试、确定性权限升级、安全延迟和中文缺参路由是后续数据扩展的重点。BFCL simple 结果采用本项目固定的 8 候选协议，适合在同一协议下复现与比较。

## 项目结构

```text
src/jev_like/       核心库、数据构造与命令行工具
tests/              单元测试
examples/           推理请求示例
configs/            各发布模型的校准参数
docs/               各版本结果与实验说明
```

## 版本历史

- `v0.2.0-0921`：当前版本。新增语义簇隔离困难正负例、三随机种子模型选择、独立校准与冻结测试，并将受控 Agent 评测扩展至 48 个场景。
- `v0.1.0-0920`：首个公开版本，建立候选 logit 决策、LoRA 推理、校准与双平台模型发布流程。

## 许可证与引用

项目代码以 [Apache License 2.0](./LICENSE) 发布。Qwen3-4B、MASSIVE 与 BFCL 适用各自的许可证和署名要求，详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。基座模型论文：Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), 2025。
