#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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


def load_humaneval_examples(max_examples: int, seed: int) -> list[dict[str, str]]:
    dataset = load_dataset("openai/openai_humaneval", split="test", download_mode="reuse_dataset_if_exists")
    dataset = dataset.shuffle(seed=seed)
    if max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return [
        {
            "task_id": row["task_id"],
            "entry_point": row["entry_point"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
        }
        for row in dataset
    ]


def build_prompt(tokenizer: AutoTokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        instruction = (
            "Complete the following Python function. Return only the code continuation.\n\n"
            f"```python\n{prompt}\n```"
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def encode_solution(
    tokenizer: AutoTokenizer,
    prompt: str,
    solution: str,
    max_length: int,
    add_eos: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    solution_ids = tokenizer.encode(solution, add_special_tokens=False)
    if add_eos and tokenizer.eos_token_id is not None:
        solution_ids.append(tokenizer.eos_token_id)
    token_ids = (prompt_ids + solution_ids)[:max_length]
    target_start = min(len(prompt_ids), len(token_ids))
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    next_positions = torch.arange(max(0, len(token_ids) - 1)) + 1
    mask = next_positions >= target_start
    return input_ids, mask


@torch.no_grad()
def sequence_nll(model: torch.nn.Module, input_ids: torch.Tensor, target_mask: torch.Tensor, device: torch.device) -> tuple[float, int]:
    if input_ids.shape[1] < 2 or int(target_mask.sum()) == 0:
        return 0.0, 0
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
    examples: list[dict[str, str]],
    lam: float,
    device: torch.device,
    max_length: int,
    use_chat_template: bool,
    add_eos: bool,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    rows = []
    total_loss = 0.0
    total_tokens = 0
    for index, example in enumerate(tqdm(examples, desc=f"humaneval lambda={lam:.3f}", leave=False)):
        prompt = build_prompt(tokenizer, example["prompt"], use_chat_template)
        input_ids, mask = encode_solution(tokenizer, prompt, example["canonical_solution"], max_length, add_eos)
        loss, tokens = sequence_nll(model, input_ids, mask, device)
        nll = loss / max(1, tokens)
        rows.append(
            {
                "lambda": lam,
                "index": index,
                "task_id": example["task_id"],
                "entry_point": example["entry_point"],
                "solution_tokens": tokens,
                "solution_nll": nll,
                "solution_loss": loss,
                "prompt_chars": len(example["prompt"]),
                "solution_chars": len(example["canonical_solution"]),
            }
        )
        total_loss += loss
        total_tokens += tokens
    avg_nll = total_loss / max(1, total_tokens)
    metrics = {
        "lambda": lam,
        "examples": len(rows),
        "solution_tokens": total_tokens,
        "avg_solution_nll": avg_nll,
        "solution_ppl": math.exp(min(20.0, avg_nll)),
        "mean_task_nll": sum(float(row["solution_nll"]) for row in rows) / max(1, len(rows)),
        "median_task_nll": float(pd.Series([row["solution_nll"] for row in rows]).median()) if rows else 0.0,
    }
    return metrics, rows


def plot_metrics(metrics_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.3), constrained_layout=True)
    ax.plot(metrics_df["lambda"], metrics_df["avg_solution_nll"], marker="o", color="#2a9d8f", linewidth=2, label="token-weighted NLL")
    ax.plot(metrics_df["lambda"], metrics_df["mean_task_nll"], marker="o", color="#e76f51", linewidth=1.8, label="mean task NLL")
    ax.plot(metrics_df["lambda"], metrics_df["median_task_nll"], marker="o", color="#6d597a", linewidth=1.8, label="median task NLL")
    ax.set_xlabel("lambda: base + lambda * (instruct - base)")
    ax.set_ylabel("canonical solution NLL")
    ax.set_title("Qwen HumanEval NLL slice")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, args: argparse.Namespace, metrics_df: pd.DataFrame) -> None:
    best = metrics_df.sort_values("avg_solution_nll").iloc[0]
    lines = [
        "# Qwen HumanEval NLL Slice",
        "",
        "This run evaluates interpolated Qwen2.5-1.5B weights on a small HumanEval code-completion slice. It scores the canonical solutions by token-level negative log-likelihood; it does not execute generated code or report pass@k.",
        "",
        "The path is `theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)`. Lower NLL is better.",
        "",
        "## Key Results",
        "",
        f"- Best lambda by token-weighted solution NLL: `{best['lambda']:.3f}` with NLL {best['avg_solution_nll']:.3f}.",
        "",
        "| lambda | examples | solution tokens | avg solution NLL | mean task NLL | median task NLL |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['lambda']:.3f} | {int(row['examples'])} | {int(row['solution_tokens'])} | {row['avg_solution_nll']:.3f} | {row['mean_task_nll']:.3f} | {row['median_task_nll']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics.csv`: per-lambda HumanEval NLL metrics.",
            "- `predictions.csv`: per-task canonical-solution NLLs.",
            "- `humaneval_nll.png`: NLL path plot.",
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
                    "max_length": args.max_length,
                    "device": args.device,
                    "dtype": args.dtype,
                    "use_chat_template": args.use_chat_template,
                    "add_eos": not args.no_eos,
                },
                indent=2,
            ),
            "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen base-to-instruct interpolation path on HumanEval canonical-solution NLL.")
    parser.add_argument("--base", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-1.5B")
    parser.add_argument("--expert", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/qwen2.5-1.5b-instruct")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to expert tokenizer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_humaneval_nll_slice"))
    parser.add_argument("--lambdas", default="0.0,0.75,1.0")
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
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

    examples = load_humaneval_examples(args.max_examples, args.seed)
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
    for lam in parse_lambdas(args.lambdas):
        set_interpolated_weights(model, base_state, expert_state, lam, device)
        metrics, rows = evaluate_lambda(
            model,
            tokenizer,
            examples,
            lam,
            device,
            args.max_length,
            args.use_chat_template,
            not args.no_eos,
        )
        metric_rows.append(metrics)
        prediction_rows.extend(rows)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(args.output_dir / "predictions.csv", index=False)
    plot_metrics(metrics_df, args.output_dir / "humaneval_nll.png")
    write_report(args.output_dir, args, metrics_df)
    best = metrics_df.sort_values("avg_solution_nll").iloc[0]
    summary = {
        "base": args.base,
        "expert": args.expert,
        "tokenizer": tokenizer_path,
        "lambdas": parse_lambdas(args.lambdas),
        "max_examples": args.max_examples,
        "evaluated_examples": len(examples),
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "use_chat_template": args.use_chat_template,
        "best_lambda": float(best["lambda"]),
        "best_avg_solution_nll": float(best["avg_solution_nll"]),
        "best_mean_task_nll": float(best["mean_task_nll"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Qwen HumanEval NLL slice artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
