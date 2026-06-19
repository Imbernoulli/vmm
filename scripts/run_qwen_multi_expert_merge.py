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
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen_path_sweep import default_device, load_safetensor_state, resolve_safetensors


GENERAL_TEXTS = [
    "Model merging combines neural networks in weight space and evaluates whether capabilities survive in one checkpoint.",
    "The task-vector plane is a diagnostic view of where fine-tuned experts and merged models land.",
    "A small validation slice can reveal whether an interpolation path preserves general language modeling.",
]

INSTRUCTION_EXAMPLES = [
    {
        "prompt": "Answer with only the number: what is 18 + 27?",
        "answer": "45",
    },
    {
        "prompt": "Rewrite the sentence to be concise: The experiment produced a large number of diagnostic artifacts that can be inspected later.",
        "answer": "The experiment produced many inspectable diagnostic artifacts.",
    },
    {
        "prompt": "Translate to Chinese: Weight-space merging can preserve capabilities when task vectors are compatible.",
        "answer": "当任务向量兼容时，权重空间合并可以保留能力。",
    },
]

CODE_EXAMPLES = [
    {
        "prompt": "Write a Python function add_one(x) that returns x plus one.",
        "answer": "```python\ndef add_one(x):\n    return x + 1\n```",
    },
    {
        "prompt": "Write a Python function is_even(n) that returns True for even integers and False otherwise.",
        "answer": "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
    },
    {
        "prompt": "Write a Python list comprehension that squares the numbers in nums.",
        "answer": "```python\nsquares = [x * x for x in nums]\n```",
    },
]


def chat_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def encode_general(tokenizer: AutoTokenizer, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    token_ids = token_ids[:max_length]
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    mask = torch.ones(max(0, len(token_ids) - 1), dtype=torch.bool)
    return input_ids, mask


def encode_response_only(tokenizer: AutoTokenizer, prompt: str, answer: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer.encode(chat_prompt(tokenizer, prompt), add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)
    token_ids = (prompt_ids + answer_ids)[:max_length]
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


@torch.no_grad()
def evaluate_sets(model: torch.nn.Module, tokenizer: AutoTokenizer, device: torch.device, max_length: int) -> dict[str, float]:
    totals: dict[str, tuple[float, int]] = {}
    general_loss = 0.0
    general_tokens = 0
    for text in GENERAL_TEXTS:
        input_ids, mask = encode_general(tokenizer, text, max_length)
        loss, tokens = sequence_nll(model, input_ids, mask, device)
        general_loss += loss
        general_tokens += tokens
    totals["general"] = (general_loss, general_tokens)

    instruction_loss = 0.0
    instruction_tokens = 0
    for example in INSTRUCTION_EXAMPLES:
        input_ids, mask = encode_response_only(tokenizer, example["prompt"], example["answer"], max_length)
        loss, tokens = sequence_nll(model, input_ids, mask, device)
        instruction_loss += loss
        instruction_tokens += tokens
    totals["instruction"] = (instruction_loss, instruction_tokens)

    code_loss = 0.0
    code_tokens = 0
    for example in CODE_EXAMPLES:
        input_ids, mask = encode_response_only(tokenizer, example["prompt"], example["answer"], max_length)
        loss, tokens = sequence_nll(model, input_ids, mask, device)
        code_loss += loss
        code_tokens += tokens
    totals["code"] = (code_loss, code_tokens)

    metrics: dict[str, float] = {}
    nlls = []
    for name, (loss, tokens) in totals.items():
        nll = loss / max(1, tokens)
        metrics[f"{name}_nll"] = nll
        metrics[f"{name}_ppl"] = math.exp(min(20.0, nll))
        metrics[f"{name}_tokens"] = tokens
        nlls.append(nll)
    metrics["avg_nll"] = float(sum(nlls) / len(nlls))
    metrics["worst_nll"] = float(max(nlls))
    return metrics


@torch.no_grad()
def set_merged_weights(
    model: torch.nn.Module,
    base: dict[str, torch.Tensor],
    experts: dict[str, dict[str, torch.Tensor]],
    weights: dict[str, float],
    device: torch.device,
) -> None:
    missing = []
    for name, param in tqdm(list(model.named_parameters()), desc="set merged weights", leave=False):
        base_tensor = base.get(name)
        if base_tensor is None or base_tensor.shape != param.shape:
            missing.append(name)
            continue
        merged = base_tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        for expert_name, expert_state in experts.items():
            expert_tensor = expert_state.get(name)
            if expert_tensor is None or expert_tensor.shape != base_tensor.shape:
                missing.append(f"{expert_name}:{name}")
                continue
            weight = float(weights.get(expert_name, 0.0))
            if weight != 0.0:
                merged = merged + weight * (expert_tensor.to(device=device, dtype=torch.float32, non_blocking=True) - base_tensor.to(device=device, dtype=torch.float32, non_blocking=True))
        param.copy_(merged.to(dtype=param.dtype))
    if missing:
        raise ValueError(f"Missing or incompatible parameters: {missing[:5]} ... total={len(missing)}")
    if device.type == "cuda":
        torch.cuda.empty_cache()


def pairwise_conflict(base: dict[str, torch.Tensor], experts: dict[str, dict[str, torch.Tensor]]) -> pd.DataFrame:
    names = list(experts)
    rows = []
    for left_idx, left_name in enumerate(names):
        for right_name in names[left_idx + 1 :]:
            dot = 0.0
            left_norm2 = 0.0
            right_norm2 = 0.0
            conflict_weight = 0.0
            total_weight = 0.0
            sign_total = 0
            sign_conflicts = 0
            shared_tensors = 0
            for tensor_name, base_tensor in base.items():
                left_tensor = experts[left_name].get(tensor_name)
                right_tensor = experts[right_name].get(tensor_name)
                if left_tensor is None or right_tensor is None or left_tensor.shape != base_tensor.shape or right_tensor.shape != base_tensor.shape:
                    continue
                if not torch.is_floating_point(base_tensor):
                    continue
                left_delta = (left_tensor.to(torch.float32) - base_tensor.to(torch.float32)).reshape(-1)
                right_delta = (right_tensor.to(torch.float32) - base_tensor.to(torch.float32)).reshape(-1)
                dot += float(left_delta @ right_delta)
                left_norm2 += float(left_delta @ left_delta)
                right_norm2 += float(right_delta @ right_delta)
                active = (left_delta.abs() > 1e-10) & (right_delta.abs() > 1e-10)
                if int(active.sum()) > 0:
                    conflict = torch.sign(left_delta[active]) != torch.sign(right_delta[active])
                    weights = (left_delta[active].abs() * right_delta[active].abs()).to(torch.float64)
                    conflict_weight += float((weights * conflict.to(torch.float64)).sum())
                    total_weight += float(weights.sum())
                    sign_conflicts += int(conflict.sum().item())
                    sign_total += int(active.sum().item())
                shared_tensors += 1
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "shared_tensors": shared_tensors,
                    "cosine": dot / max(1e-12, math.sqrt(left_norm2) * math.sqrt(right_norm2)),
                    "sign_conflict": sign_conflicts / max(1, sign_total),
                    "weighted_conflict": conflict_weight / max(1e-12, total_weight),
                    "left_norm": math.sqrt(left_norm2),
                    "right_norm": math.sqrt(right_norm2),
                }
            )
    return pd.DataFrame(rows)


def method_rows(grid_df: pd.DataFrame) -> list[dict[str, float | str]]:
    best_grid = grid_df.sort_values(["avg_nll", "worst_nll"], ascending=True).iloc[0]
    best_worst = grid_df.sort_values(["worst_nll", "avg_nll"], ascending=True).iloc[0]
    rows: list[dict[str, float | str]] = [
        {"method": "base", "alpha": 0.0, "beta": 0.0},
        {"method": "instruct_expert", "alpha": 1.0, "beta": 0.0},
        {"method": "coder_expert", "alpha": 0.0, "beta": 1.0},
        {"method": "linear_average", "alpha": 0.5, "beta": 0.5},
        {"method": "task_arithmetic_0.25", "alpha": 0.25, "beta": 0.25},
        {"method": "task_arithmetic_0.75", "alpha": 0.75, "beta": 0.75},
        {"method": "validation_grid_best_avg", "alpha": float(best_grid["alpha"]), "beta": float(best_grid["beta"])},
        {"method": "validation_grid_best_worst", "alpha": float(best_worst["alpha"]), "beta": float(best_worst["beta"])},
    ]
    return rows


def plot_grid(grid_df: pd.DataFrame, methods_df: pd.DataFrame, out: Path) -> None:
    pivot = grid_df.pivot(index="beta", columns="alpha", values="avg_nll").sort_index()
    xs = pivot.columns.to_numpy(dtype=float)
    ys = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    contour = axes[0].contourf(xs, ys, z, levels=18, cmap="viridis_r")
    axes[0].set_title("Qwen multi-expert average NLL")
    axes[0].set_xlabel("alpha: instruct delta")
    axes[0].set_ylabel("beta: coder delta")
    fig.colorbar(contour, ax=axes[0], label="avg NLL")
    for _, row in methods_df.iterrows():
        axes[0].scatter(row["alpha"], row["beta"], c="white", edgecolors="black", s=55)
        axes[0].annotate(str(row["method"]).replace("_", "\n"), (row["alpha"], row["beta"]), fontsize=7)

    plot_df = methods_df.sort_values("avg_nll", ascending=False)
    y = np.arange(len(plot_df))
    axes[1].barh(y - 0.22, plot_df["general_nll"], height=0.22, color="#2a9d8f", label="general")
    axes[1].barh(y, plot_df["instruction_nll"], height=0.22, color="#e76f51", label="instruction")
    axes[1].barh(y + 0.22, plot_df["code_nll"], height=0.22, color="#457b9d", label="code")
    axes[1].scatter(plot_df["avg_nll"], y, color="black", s=20, label="avg")
    axes[1].set_yticks(y, labels=plot_df["method"])
    axes[1].set_xlabel("NLL lower is better")
    axes[1].set_title("Method tradeoff")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_paths(grid_df: pd.DataFrame, out: Path) -> None:
    diag = grid_df[np.isclose(grid_df["alpha"], grid_df["beta"])].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(diag["alpha"], diag["general_nll"], marker="o", label="general", color="#2a9d8f")
    ax.plot(diag["alpha"], diag["instruction_nll"], marker="o", label="instruction", color="#e76f51")
    ax.plot(diag["alpha"], diag["code_nll"], marker="o", label="code", color="#457b9d")
    ax.plot(diag["alpha"], diag["avg_nll"], marker="o", label="average", color="#111827")
    ax.set_xlabel("shared lambda on instruct + coder deltas")
    ax.set_ylabel("NLL")
    ax.set_title("Diagonal multi-expert task-arithmetic path")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_conflict(conflict_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    labels = [f"{row.left}/{row.right}" for row in conflict_df.itertuples()]
    x = np.arange(len(conflict_df))
    axes[0].bar(x - 0.18, conflict_df["cosine"], width=0.36, label="cosine", color="#457b9d")
    axes[0].bar(x + 0.18, conflict_df["weighted_conflict"], width=0.36, label="weighted conflict", color="#e76f51")
    axes[0].set_xticks(x, labels=labels, rotation=20, ha="right")
    axes[0].set_title("Expert delta conflict")
    axes[0].legend(fontsize=8)
    axes[1].bar(x - 0.18, conflict_df["left_norm"], width=0.36, label="left norm", color="#2a9d8f")
    axes[1].bar(x + 0.18, conflict_df["right_norm"], width=0.36, label="right norm", color="#6d597a")
    axes[1].set_xticks(x, labels=labels, rotation=20, ha="right")
    axes[1].set_title("Task-vector norms")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, args: argparse.Namespace, grid_df: pd.DataFrame, methods_df: pd.DataFrame, conflict_df: pd.DataFrame) -> None:
    best_avg = methods_df.sort_values(["avg_nll", "worst_nll"], ascending=True).iloc[0]
    best_worst = methods_df.sort_values(["worst_nll", "avg_nll"], ascending=True).iloc[0]
    lines = [
        "# Qwen Multi-Expert Merge",
        "",
        "This run evaluates a real multi-expert Qwen merge plane. The base is Qwen2.5-0.5B; the two experts are Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct. The merge plane is `base + alpha * instruct_delta + beta * coder_delta`.",
        "",
        "Metrics are token-level NLLs on small general, instruction-response, and code-response slices. Lower is better.",
        "",
        "## Key Results",
        "",
        f"- Best method by average NLL: `{best_avg['method']}` with avg NLL {best_avg['avg_nll']:.3f}, worst NLL {best_avg['worst_nll']:.3f}.",
        f"- Best method by worst NLL: `{best_worst['method']}` with avg NLL {best_worst['avg_nll']:.3f}, worst NLL {best_worst['worst_nll']:.3f}.",
        "",
        "| method | alpha | beta | general NLL | instruction NLL | code NLL | avg NLL | worst NLL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in methods_df.sort_values("avg_nll").iterrows():
        lines.append(
            f"| {row['method']} | {row['alpha']:.2f} | {row['beta']:.2f} | {row['general_nll']:.3f} | {row['instruction_nll']:.3f} | {row['code_nll']:.3f} | {row['avg_nll']:.3f} | {row['worst_nll']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Expert Conflict",
            "",
        ]
    )
    for _, row in conflict_df.iterrows():
        lines.append(
            f"- `{row['left']}` vs `{row['right']}`: cosine {row['cosine']:.3f}, sign conflict {row['sign_conflict']:.3f}, weighted conflict {row['weighted_conflict']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `grid_metrics.csv`: alpha/beta multi-expert grid.",
            "- `method_metrics.csv`: named endpoints and merge methods.",
            "- `pairwise_conflict.csv`: instruct/coder delta conflict.",
            "- `figures/*.png`: grid, diagonal path, and conflict plots.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(
                {
                    "base": args.base,
                    "instruct": args.instruct,
                    "coder": args.coder,
                    "tokenizer": args.tokenizer or args.instruct,
                    "grid_values": args.grid_values,
                    "dtype": args.dtype,
                    "device": args.device,
                    "max_length": args.max_length,
                },
                indent=2,
            ),
            "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_grid_values(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--grid-values must contain at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a two-expert Qwen merge plane with instruct and coder experts.")
    parser.add_argument("--base", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-0.5B")
    parser.add_argument("--instruct", default="/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--coder", default="/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_multi_expert_merge"))
    parser.add_argument("--grid-values", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer_path = args.tokenizer or args.instruct
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=Path(tokenizer_path).exists(), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_state = load_safetensor_state(args.base)
    expert_states = {
        "instruct": load_safetensor_state(args.instruct),
        "coder": load_safetensor_state(args.coder),
    }
    conflict_df = pairwise_conflict(base_state, expert_states)
    conflict_df.to_csv(args.output_dir / "pairwise_conflict.csv", index=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        local_files_only=Path(args.base).exists(),
        trust_remote_code=True,
    ).to(device)
    model.eval()

    grid_values = parse_grid_values(args.grid_values)
    rows = []
    for alpha in grid_values:
        for beta in grid_values:
            set_merged_weights(model, base_state, expert_states, {"instruct": alpha, "coder": beta}, device)
            rows.append({"alpha": alpha, "beta": beta, **evaluate_sets(model, tokenizer, device, args.max_length)})
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(args.output_dir / "grid_metrics.csv", index=False)

    method_metrics = []
    for row in method_rows(grid_df):
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        found = grid_df[(grid_df["alpha"] == alpha) & (grid_df["beta"] == beta)]
        if found.empty:
            set_merged_weights(model, base_state, expert_states, {"instruct": alpha, "coder": beta}, device)
            metrics = evaluate_sets(model, tokenizer, device, args.max_length)
        else:
            metrics = found.iloc[0].drop(labels=["alpha", "beta"]).to_dict()
        method_metrics.append({**row, **metrics})
    methods_df = pd.DataFrame(method_metrics)
    methods_df.to_csv(args.output_dir / "method_metrics.csv", index=False)

    plot_grid(grid_df, methods_df, fig_dir / "merge_grid.png")
    plot_paths(grid_df, fig_dir / "diagonal_path.png")
    plot_conflict(conflict_df, fig_dir / "pairwise_conflict.png")
    write_report(args.output_dir, args, grid_df, methods_df, conflict_df)

    summary = {
        "base": args.base,
        "experts": {"instruct": args.instruct, "coder": args.coder},
        "tokenizer": tokenizer_path,
        "grid_values": grid_values,
        "device": str(device),
        "dtype": args.dtype,
        "general_examples": len(GENERAL_TEXTS),
        "instruction_examples": len(INSTRUCTION_EXAMPLES),
        "code_examples": len(CODE_EXAMPLES),
        "best_avg_method": str(methods_df.sort_values(["avg_nll", "worst_nll"], ascending=True).iloc[0]["method"]),
        "best_avg_nll": float(methods_df["avg_nll"].min()),
        "best_worst_method": str(methods_df.sort_values(["worst_nll", "avg_nll"], ascending=True).iloc[0]["method"]),
        "best_worst_nll": float(methods_df["worst_nll"].min()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Qwen multi-expert merge artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
