from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


INTENTS = {
    "billing": [
        "I was charged twice for the same order.",
        "The amount on my invoice is incorrect.",
        "Why did my subscription renewal cost more?",
        "Please explain this unexpected card charge.",
        "My refund has not reached my bank account.",
        "同一个订单扣了我两次款。",
        "账单金额和约定价格不一致。",
        "退款已经批准，但银行卡一直没有到账。",
    ],
    "technical": [
        "The application crashes when I export a report.",
        "I cannot sign in after installing the update.",
        "The dashboard stays blank in every browser.",
        "File upload fails at ninety percent.",
        "The API returns a server error for valid requests.",
        "升级以后应用一打开就闪退。",
        "导出按钮点击后没有任何反应。",
        "接口对合法请求一直返回服务器错误。",
    ],
    "sales": [
        "Can you prepare a quote for fifty team seats?",
        "I want to compare the enterprise and business plans.",
        "Does the annual plan include volume discounts?",
        "We would like a product demonstration next week.",
        "Can someone discuss a company-wide license with us?",
        "我们想采购五十个团队账号，请提供报价。",
        "企业版是否支持批量折扣？",
        "下周可以给我们安排一次产品演示吗？",
    ],
    "cancellation": [
        "Please cancel my subscription at the end of this month.",
        "I no longer need the service and want to close my account.",
        "Stop the automatic renewal for my plan.",
        "How can I terminate the contract before renewal?",
        "Please deactivate my paid membership.",
        "请在本月底取消我的订阅。",
        "我不再需要这个服务，请关闭账号。",
        "请停止套餐的自动续费。",
    ],
}

WRAPPERS = [
    "Customer message: {text}",
    "A support ticket says: {text}",
    "Route the following request. {text}",
    "用户请求如下：{text}",
]

QUESTION = "Which department should handle the primary request?"


def write_split(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {
        "train": [], "validation": [], "calibration": [], "test": []
    }
    split_for_phrase = ["train"] * 5 + ["validation", "calibration", "test"]
    labels = list(INTENTS)
    for label, phrases in INTENTS.items():
        for phrase_index, (phrase, split) in enumerate(zip(phrases, split_for_phrase)):
            wrappers = WRAPPERS if split == "train" else WRAPPERS[:2]
            for wrapper_index, wrapper in enumerate(wrappers):
                options = labels.copy()
                rng.shuffle(options)
                splits[split].append({
                    "state": wrapper.format(text=phrase),
                    "question": QUESTION,
                    "options": options,
                    "answer": label,
                    "example_id": f"{split}-{label}-{phrase_index}-{wrapper_index}",
                })
    for split, rows in splits.items():
        rng.shuffle(rows)
        write_split(output / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)}")


if __name__ == "__main__":
    main()
