#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
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


ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_number(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def extract_gold(answer: str) -> str | None:
    match = ANSWER_RE.search(answer)
    if match:
        return normalize_number(match.group(1))
    matches = NUMBER_RE.findall(answer)
    return normalize_number(matches[-1]) if matches else None


def extract_prediction(text: str) -> tuple[str | None, str]:
    match = ANSWER_RE.search(text)
    if match:
        return normalize_number(match.group(1)), "hash"
    matches = NUMBER_RE.findall(text)
    return (normalize_number(matches[-1]), "last_number") if matches else (None, "none")


def build_prompt(tokenizer: AutoTokenizer, question: str, use_chat_template: bool) -> str:
    instruction = (
        "Solve the grade-school math problem. Show concise reasoning, then put the final answer "
        "on its own last line in the exact format #### <number>.\n\n"
        f"Question: {question}"
    )
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction + "\n\nAnswer:"


def load_gsm8k(max_examples: int, seed: int) -> list[dict[str, str]]:
    dataset = load_dataset("openai/gsm8k", "main", split="test", download_mode="reuse_dataset_if_exists")
    if max_examples > 0:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_examples, len(dataset))))
    rows = []
    for item in dataset:
        gold = extract_gold(item["answer"])
        if gold is None:
            continue
        rows.append({"question": item["question"], "answer": item["answer"], "gold": gold})
    return rows


@torch.no_grad()
def generate_answer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_lambda(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    examples: list[dict[str, str]],
    lam: float,
    device: torch.device,
    max_prompt_tokens: int,
    max_new_tokens: int,
    use_chat_template: bool,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    rows = []
    correct = 0
    for index, example in enumerate(tqdm(examples, desc=f"gsm8k lambda={lam:.3f}", leave=False)):
        prompt = build_prompt(tokenizer, example["question"], use_chat_template)
        generated = generate_answer(model, tokenizer, prompt, device, max_prompt_tokens, max_new_tokens)
        prediction, prediction_source = extract_prediction(generated)
        loose_exact = prediction == example["gold"]
        exact = prediction_source == "hash" and loose_exact
        correct += int(exact)
        rows.append(
            {
                "lambda": lam,
                "index": index,
                "question": example["question"],
                "gold": example["gold"],
                "prediction": prediction,
                "prediction_source": prediction_source,
                "exact": exact,
                "loose_exact": loose_exact,
                "generated_text": generated,
            }
        )
    loose_correct = sum(int(row["loose_exact"]) for row in rows)
    hash_count = sum(int(row["prediction_source"] == "hash") for row in rows)
    metrics = {
        "lambda": lam,
        "examples": len(examples),
        "exact_count": correct,
        "exact_match": correct / max(1, len(examples)),
        "loose_exact_count": loose_correct,
        "loose_exact_match": loose_correct / max(1, len(examples)),
        "hash_format_count": hash_count,
        "hash_format_rate": hash_count / max(1, len(examples)),
        "avg_generated_chars": sum(len(str(row["generated_text"])) for row in rows) / max(1, len(rows)),
    }
    return metrics, rows


def plot_metrics(metrics_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(metrics_df["lambda"], metrics_df["exact_match"], marker="o", color="#2a9d8f", linewidth=2, label="strict #### exact")
    ax.plot(metrics_df["lambda"], metrics_df["loose_exact_match"], marker="o", color="#e76f51", linewidth=1.8, linestyle="--", label="loose last-number exact")
    for _, row in metrics_df.iterrows():
        ax.annotate(f"{int(row['exact_count'])}/{int(row['examples'])}", (row["lambda"], row["exact_match"]), xytext=(4, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("lambda: base + lambda * (instruct - base)")
    ax.set_ylabel("GSM8K exact match")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Qwen GSM8K benchmark slice")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, args: argparse.Namespace, metrics_df: pd.DataFrame) -> None:
    best = metrics_df.sort_values(["exact_match", "loose_exact_match", "exact_count"], ascending=False).iloc[0]
    lines = [
        "# Qwen GSM8K Benchmark Slice",
        "",
        "This run evaluates interpolated Qwen2.5-1.5B weights on a small cached GSM8K test slice. It is a benchmark-slice diagnostic, not a full GSM8K run.",
        "",
        "The path is `theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)`. Each model generates an answer. The strict score requires the model to emit the GSM8K `#### <number>` format; the loose score falls back to the last generated number when that marker is missing.",
        "",
        "## Key Results",
        "",
        f"- Best lambda by strict exact match: `{best['lambda']:.3f}` with exact match {best['exact_match']:.3f} ({int(best['exact_count'])}/{int(best['examples'])}).",
        "",
        "| lambda | strict exact | loose exact | hash format | strict correct / total | avg generated chars |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['lambda']:.3f} | {row['exact_match']:.3f} | {row['loose_exact_match']:.3f} | {row['hash_format_rate']:.3f} | {int(row['exact_count'])}/{int(row['examples'])} | {row['avg_generated_chars']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
        "- `metrics.csv`: per-lambda strict and loose exact-match metrics.",
            "- `predictions.csv`: per-example generations and extracted answers.",
            "- `gsm8k_exact_match.png`: exact-match path plot.",
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
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "max_new_tokens": args.max_new_tokens,
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
    parser = argparse.ArgumentParser(description="Evaluate a Qwen base-to-instruct interpolation path on a cached GSM8K slice.")
    parser.add_argument("--base", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-1.5B")
    parser.add_argument("--expert", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/qwen2.5-1.5b-instruct")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to expert tokenizer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_gsm8k_slice"))
    parser.add_argument("--lambdas", default="0.0,0.75,1.0")
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=160)
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

    examples = load_gsm8k(args.max_examples, args.seed)
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
            args.max_prompt_tokens,
            args.max_new_tokens,
            not args.no_chat_template,
        )
        metric_rows.append(metrics)
        prediction_rows.extend(rows)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(args.output_dir / "predictions.csv", index=False)
    plot_metrics(metrics_df, args.output_dir / "gsm8k_exact_match.png")
    write_report(args.output_dir, args, metrics_df)
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
        "best_lambda": float(metrics_df.sort_values(["exact_match", "loose_exact_match", "exact_count"], ascending=False).iloc[0]["lambda"]),
        "best_exact_match": float(metrics_df["exact_match"].max()),
        "best_loose_exact_match": float(metrics_df["loose_exact_match"].max()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Qwen GSM8K slice artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
