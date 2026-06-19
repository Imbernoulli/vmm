#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen_path_sweep import (
    default_device,
    load_safetensor_state,
    parse_lambdas,
    set_interpolated_weights,
)


LETTERS = ["A", "B", "C", "D"]


def normalize_subjects(raw: str) -> list[str] | None:
    subjects = [part.strip() for part in raw.split(",") if part.strip()]
    return None if not subjects or subjects == ["all"] else subjects


def load_mmlu_examples(max_examples: int, seed: int, subjects: str) -> list[dict[str, object]]:
    dataset = load_dataset("cais/mmlu", "all", split="test", download_mode="reuse_dataset_if_exists")
    selected_subjects = normalize_subjects(subjects)
    if selected_subjects is not None:
        wanted = set(selected_subjects)
        dataset = dataset.filter(lambda row: row["subject"] in wanted)
    dataset = dataset.shuffle(seed=seed)
    if max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    examples = []
    for row in dataset:
        choices = list(row["choices"])
        if len(choices) != 4:
            continue
        answer_index = int(row["answer"])
        if not 0 <= answer_index < 4:
            continue
        examples.append(
            {
                "question": row["question"],
                "subject": row["subject"],
                "choices": choices,
                "answer_index": answer_index,
                "answer_letter": LETTERS[answer_index],
            }
        )
    return examples


def build_prompt(tokenizer: AutoTokenizer, example: dict[str, object], use_chat_template: bool) -> str:
    choices = example["choices"]
    assert isinstance(choices, list)
    prompt = (
        "Answer the following multiple-choice question. Reply with only the letter A, B, C, or D.\n\n"
        f"Question: {example['question']}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def encode_choice(
    tokenizer: AutoTokenizer,
    prompt: str,
    letter: str,
    max_length: int,
    use_chat_template: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = letter if use_chat_template and getattr(tokenizer, "chat_template", None) else f" {letter}"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    token_ids = (prompt_ids + target_ids)[:max_length]
    target_start = min(len(prompt_ids), len(token_ids))
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    next_positions = torch.arange(max(0, len(token_ids) - 1)) + 1
    mask = next_positions >= target_start
    return input_ids, mask


@torch.no_grad()
def target_nll(model: torch.nn.Module, input_ids: torch.Tensor, target_mask: torch.Tensor, device: torch.device) -> tuple[float, int]:
    if input_ids.shape[1] < 2 or int(target_mask.sum()) == 0:
        return float("inf"), 0
    input_ids = input_ids.to(device)
    target_mask = target_mask.to(device)
    logits = model(input_ids=input_ids).logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    losses = F.cross_entropy(
        logits.view(-1, logits.shape[-1]).to(torch.float32),
        targets.view(-1),
        reduction="none",
    ).view_as(targets)
    selected = losses[:, target_mask]
    return float(selected.sum().detach().cpu()), int(target_mask.sum().item())


def evaluate_lambda(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    examples: list[dict[str, object]],
    lam: float,
    device: torch.device,
    max_length: int,
    use_chat_template: bool,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    rows = []
    for index, example in enumerate(tqdm(examples, desc=f"mmlu lambda={lam:.3f}", leave=False)):
        prompt = build_prompt(tokenizer, example, use_chat_template)
        choice_nlls = []
        choice_tokens = []
        for letter in LETTERS:
            input_ids, mask = encode_choice(tokenizer, prompt, letter, max_length, use_chat_template)
            nll, tokens = target_nll(model, input_ids, mask, device)
            choice_nlls.append(nll)
            choice_tokens.append(tokens)
        prediction_index = min(range(len(choice_nlls)), key=lambda idx: choice_nlls[idx])
        gold_index = int(example["answer_index"])
        sorted_nlls = sorted(choice_nlls)
        margin = sorted_nlls[1] - sorted_nlls[0] if len(sorted_nlls) > 1 else 0.0
        rows.append(
            {
                "lambda": lam,
                "index": index,
                "subject": example["subject"],
                "question": example["question"],
                "gold": example["answer_letter"],
                "prediction": LETTERS[prediction_index],
                "correct": prediction_index == gold_index,
                "gold_nll": choice_nlls[gold_index],
                "predicted_nll": choice_nlls[prediction_index],
                "margin": margin,
                "choice_a_nll": choice_nlls[0],
                "choice_b_nll": choice_nlls[1],
                "choice_c_nll": choice_nlls[2],
                "choice_d_nll": choice_nlls[3],
                "choice_tokens": max(choice_tokens),
            }
        )
    correct = sum(int(row["correct"]) for row in rows)
    metrics = {
        "lambda": lam,
        "examples": len(rows),
        "accuracy_count": correct,
        "accuracy": correct / max(1, len(rows)),
        "avg_gold_nll": sum(float(row["gold_nll"]) for row in rows) / max(1, len(rows)),
        "avg_predicted_nll": sum(float(row["predicted_nll"]) for row in rows) / max(1, len(rows)),
        "avg_margin": sum(float(row["margin"]) for row in rows) / max(1, len(rows)),
    }
    return metrics, rows


def plot_metrics(metrics_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    ax = axes[0]
    ax.plot(metrics_df["lambda"], metrics_df["accuracy"], marker="o", color="#2a9d8f", linewidth=2)
    for _, row in metrics_df.iterrows():
        ax.annotate(
            f"{int(row['accuracy_count'])}/{int(row['examples'])}",
            (row["lambda"], row["accuracy"]),
            xytext=(4, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("lambda: base + lambda * (instruct - base)")
    ax.set_ylabel("MMLU accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Qwen MMLU multiple-choice slice")

    ax = axes[1]
    ax.plot(metrics_df["lambda"], metrics_df["avg_gold_nll"], marker="o", label="gold answer NLL", color="#e76f51")
    ax.plot(metrics_df["lambda"], metrics_df["avg_predicted_nll"], marker="o", label="predicted answer NLL", color="#264653")
    ax.plot(metrics_df["lambda"], metrics_df["avg_margin"], marker="o", label="choice margin", color="#6d597a")
    ax.set_xlabel("lambda")
    ax.set_ylabel("NLL / margin")
    ax.set_title("Multiple-choice log-likelihood")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, args: argparse.Namespace, metrics_df: pd.DataFrame) -> None:
    best = metrics_df.sort_values(["accuracy", "avg_gold_nll"], ascending=[False, True]).iloc[0]
    lines = [
        "# Qwen MMLU Benchmark Slice",
        "",
        "This run evaluates interpolated Qwen2.5-1.5B weights on a small MMLU test slice. It is a benchmark-slice diagnostic, not a full MMLU run.",
        "",
        "The path is `theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)`. Each multiple-choice question is scored by the log-likelihood of answer letters A-D; the predicted answer is the lowest-NLL letter.",
        "",
        "## Key Results",
        "",
        f"- Best lambda by accuracy: `{best['lambda']:.3f}` with accuracy {best['accuracy']:.3f} ({int(best['accuracy_count'])}/{int(best['examples'])}).",
        "",
        "| lambda | accuracy | correct / total | avg gold NLL | avg predicted NLL | avg margin |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['lambda']:.3f} | {row['accuracy']:.3f} | {int(row['accuracy_count'])}/{int(row['examples'])} | {row['avg_gold_nll']:.3f} | {row['avg_predicted_nll']:.3f} | {row['avg_margin']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics.csv`: per-lambda multiple-choice metrics.",
            "- `predictions.csv`: per-example answer-letter NLLs and predictions.",
            "- `mmlu_accuracy.png`: accuracy and NLL path plot.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(
                {
                    "base": args.base,
                    "expert": args.expert,
                    "tokenizer": args.tokenizer or args.expert,
                    "lambdas": parse_lambdas(args.lambdas),
                    "max_examples": args.max_examples,
                    "seed": args.seed,
                    "subjects": args.subjects,
                    "max_length": args.max_length,
                    "device": args.device,
                    "dtype": args.dtype,
                    "use_chat_template": not args.no_chat_template,
                },
                indent=2,
            ),
            "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen base-to-instruct interpolation path on a small MMLU slice.")
    parser.add_argument("--base", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-1.5B")
    parser.add_argument("--expert", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/qwen2.5-1.5b-instruct")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to expert tokenizer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_mmlu_slice"))
    parser.add_argument("--lambdas", default="0.0,0.75,1.0")
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--no-chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer_path = args.tokenizer or args.expert
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=Path(tokenizer_path).exists(), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_mmlu_examples(args.max_examples, args.seed, args.subjects)
    base_state = load_safetensor_state(args.base)
    expert_state = load_safetensor_state(args.expert)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        local_files_only=Path(args.base).exists(),
        trust_remote_code=True,
    ).to(device)
    model.eval()

    metric_rows = []
    prediction_rows = []
    use_chat_template = not args.no_chat_template
    for lam in parse_lambdas(args.lambdas):
        set_interpolated_weights(model, base_state, expert_state, lam, device)
        metrics, rows = evaluate_lambda(model, tokenizer, examples, lam, device, args.max_length, use_chat_template)
        metric_rows.append(metrics)
        prediction_rows.extend(rows)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(args.output_dir / "predictions.csv", index=False)
    plot_metrics(metrics_df, args.output_dir / "mmlu_accuracy.png")
    write_report(args.output_dir, args, metrics_df)
    best = metrics_df.sort_values(["accuracy", "avg_gold_nll"], ascending=[False, True]).iloc[0]
    summary = {
        "base": args.base,
        "expert": args.expert,
        "tokenizer": tokenizer_path,
        "lambdas": parse_lambdas(args.lambdas),
        "max_examples": args.max_examples,
        "evaluated_examples": len(examples),
        "seed": args.seed,
        "subjects": normalize_subjects(args.subjects) or "all",
        "device": str(device),
        "dtype": args.dtype,
        "best_lambda": float(best["lambda"]),
        "best_accuracy": float(best["accuracy"]),
        "best_accuracy_count": int(best["accuracy_count"]),
        "best_avg_gold_nll": float(best["avg_gold_nll"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Qwen MMLU slice artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
