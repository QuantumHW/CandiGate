from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import answer_index, read_jsonl, selected_logits


def evaluate_loss(model, tokenizer, examples) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for example in examples:
            logits = selected_logits(model, tokenizer, example)
            target = torch.tensor([answer_index(example)], device=logits.device)
            losses.append(float(F.cross_entropy(logits.unsqueeze(0), target).item()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda"}
    )
    model.config.use_cache = False
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    train_examples = read_jsonl(args.train)
    validation_examples = read_jsonl(args.validation)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    history: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        random.shuffle(train_examples)
        running = 0.0
        for step, example in enumerate(train_examples, start=1):
            logits = selected_logits(model, tokenizer, example)
            target = torch.tensor([answer_index(example)], device=logits.device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            (loss / args.gradient_accumulation).backward()
            running += float(loss.detach().item())
            if step % args.gradient_accumulation == 0 or step == len(train_examples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_loss = evaluate_loss(model, tokenizer, validation_examples)
        record = {
            "epoch": epoch + 1,
            "train_loss": running / len(train_examples),
            "validation_loss": validation_loss,
        }
        history.append(record)
        print(json.dumps(record))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
