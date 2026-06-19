#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file


GENERAL_TEXTS = [
    (
        "Model merging studies how two or more neural networks can be combined in weight space. "
        "A useful diagnostic is to evaluate losses along the path between a base model and a fine-tuned model."
    ),
    (
        "The digits experiment is a small image-classification surrogate. "
        "It shows that a merged model succeeds when it remains inside the overlap of two task-compatible regions."
    ),
    (
        "In mathematics, a vector space contains points that can be added together and multiplied by scalars. "
        "Task vectors use this idea to describe how fine-tuning changes a model."
    ),
    (
        "模型合并可以看作在参数空间中寻找多个任务都能接受的位置。"
        "如果两个任务向量互相冲突，简单平均可能会落在高损失区域。"
    ),
]


INSTRUCTION_EXAMPLES = [
    {
        "prompt": "Answer with only the number: what is 17 + 28?",
        "answer": "45",
    },
    {
        "prompt": "Translate to Chinese: model merging helps combine capabilities.",
        "answer": "模型合并有助于结合能力。",
    },
    {
        "prompt": "Write a Python function named square that returns x squared.",
        "answer": "```python\ndef square(x):\n    return x * x\n```",
    },
    {
        "prompt": "Summarize in one sentence: A base model is fine-tuned on two tasks. The two task vectors may point in compatible or conflicting directions.",
        "answer": "Task-vector compatibility determines whether the merged model is likely to preserve both tasks.",
    },
    {
        "prompt": "Classify the sentiment as positive, negative, or neutral: The experiment is small, but the diagnostic is clear.",
        "answer": "positive",
    },
]


def resolve_safetensors(model_id_or_path: str) -> list[Path]:
    path = Path(model_id_or_path)
    if path.exists():
        if path.is_file() and path.name.endswith(".safetensors"):
            return [path]
        index = path / SAFE_WEIGHTS_INDEX_NAME
        single = path / SAFE_WEIGHTS_NAME
        if index.exists():
            payload = json.loads(index.read_text(encoding="utf-8"))
            return sorted({path / shard for shard in payload["weight_map"].values()})
        if single.exists():
            return [single]
        raise FileNotFoundError(f"No safetensors weights found under {path}")
    try:
        index_file = cached_file(model_id_or_path, SAFE_WEIGHTS_INDEX_NAME)
        payload = json.loads(Path(index_file).read_text(encoding="utf-8"))
        return sorted({Path(index_file).parent / shard for shard in payload["weight_map"].values()})
    except Exception:
        single_file = cached_file(model_id_or_path, SAFE_WEIGHTS_NAME)
        return [Path(single_file)]


def load_safetensor_state(model_id_or_path: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for shard in tqdm(resolve_safetensors(model_id_or_path), desc=f"load weights {model_id_or_path}"):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                value = handle.get_tensor(name)
                if torch.is_floating_point(value):
                    state[name] = value.cpu()
    if not state:
        raise ValueError(f"No floating tensors loaded from {model_id_or_path}")
    return state


def layer_group(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def summarize_delta(base: dict[str, torch.Tensor], expert: dict[str, torch.Tensor]) -> pd.DataFrame:
    rows = []
    for name, base_tensor in tqdm(base.items(), desc="summarize deltas"):
        expert_tensor = expert.get(name)
        if expert_tensor is None or expert_tensor.shape != base_tensor.shape:
            continue
        delta = expert_tensor.to(torch.float32) - base_tensor.to(torch.float32)
        base_norm = torch.linalg.norm(base_tensor.to(torch.float32).reshape(-1)).clamp_min(1e-12)
        rows.append(
            {
                "tensor": name,
                "group": layer_group(name),
                "numel": int(delta.numel()),
                "delta_norm": float(torch.linalg.norm(delta.reshape(-1))),
                "base_norm": float(base_norm),
                "relative_norm": float(torch.linalg.norm(delta.reshape(-1)) / base_norm),
                "mean_abs_delta": float(delta.abs().mean()),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def set_interpolated_weights(
    model: torch.nn.Module,
    base: dict[str, torch.Tensor],
    expert: dict[str, torch.Tensor],
    lam: float,
    device: torch.device,
) -> None:
    missing = []
    for name, param in tqdm(list(model.named_parameters()), desc=f"set lambda={lam:.3f}", leave=False):
        base_tensor = base.get(name)
        expert_tensor = expert.get(name)
        if base_tensor is None or expert_tensor is None or base_tensor.shape != param.shape or expert_tensor.shape != param.shape:
            missing.append(name)
            continue
        b = base_tensor.to(device=device, dtype=param.dtype, non_blocking=True)
        if lam == 0.0:
            param.copy_(b)
        else:
            e = expert_tensor.to(device=device, dtype=param.dtype, non_blocking=True)
            param.copy_(b + lam * (e - b))
    if missing:
        raise ValueError(f"Missing or incompatible model parameters: {missing[:5]} ... total={len(missing)}")
    if device.type == "cuda":
        torch.cuda.empty_cache()


def encode_general(tokenizer: AutoTokenizer, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    token_ids = token_ids[:max_length]
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    mask = torch.ones(max(0, len(token_ids) - 1), dtype=torch.bool)
    return input_ids, mask


def chat_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def encode_instruction(
    tokenizer: AutoTokenizer,
    prompt: str,
    answer: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_text = chat_prompt(tokenizer, prompt)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)
    token_ids = (prompt_ids + answer_ids)[:max_length]
    target_start = min(len(prompt_ids), len(token_ids))
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    next_token_positions = torch.arange(max(0, len(token_ids) - 1)) + 1
    mask = next_token_positions >= target_start
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
def evaluate_sets(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
) -> dict[str, float]:
    general_loss = 0.0
    general_tokens = 0
    for text in GENERAL_TEXTS:
        input_ids, mask = encode_general(tokenizer, text, max_length)
        loss, tokens = sequence_nll(model, input_ids, mask, device)
        general_loss += loss
        general_tokens += tokens

    instruction_loss = 0.0
    instruction_tokens = 0
    for example in INSTRUCTION_EXAMPLES:
        input_ids, mask = encode_instruction(tokenizer, example["prompt"], example["answer"], max_length)
        loss, tokens = sequence_nll(model, input_ids, mask, device)
        instruction_loss += loss
        instruction_tokens += tokens

    general_nll = general_loss / max(1, general_tokens)
    instruction_nll = instruction_loss / max(1, instruction_tokens)
    return {
        "general_nll": general_nll,
        "instruction_nll": instruction_nll,
        "avg_nll": 0.5 * (general_nll + instruction_nll),
        "worst_nll": max(general_nll, instruction_nll),
        "general_ppl": math.exp(min(20.0, general_nll)),
        "instruction_ppl": math.exp(min(20.0, instruction_nll)),
        "general_tokens": general_tokens,
        "instruction_tokens": instruction_tokens,
    }


def plot_path(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(df["lambda"], df["general_nll"], marker="o", label="general text", color="#2a9d8f")
    ax.plot(df["lambda"], df["instruction_nll"], marker="o", label="instruction response", color="#e76f51")
    ax.plot(df["lambda"], df["avg_nll"], marker="o", label="average", color="#264653")
    ax.plot(df["lambda"], df["worst_nll"], marker="o", label="worst", color="#6d597a")
    ax.set_xlabel("lambda: base + lambda * (instruct - base)")
    ax.set_ylabel("NLL")
    ax.set_title("Qwen2.5-1.5B base to instruct path")
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.03 if len(df) > 8 else 0.045
    ax.bar(df["lambda"] - width / 2, df["general_nll"], width=width, label="general", color="#2a9d8f", alpha=0.78)
    ax.bar(df["lambda"] + width / 2, df["instruction_nll"], width=width, label="instruction", color="#e76f51", alpha=0.65)
    ax.scatter(df["lambda"], df["worst_nll"], color="black", label="worst", zorder=4)
    ax.set_xlabel("lambda")
    ax.set_ylabel("NLL")
    ax.set_title("Task tradeoff")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_delta_norms(df: pd.DataFrame, out: Path, top_k: int = 35) -> None:
    grouped = (
        df.groupby("group", as_index=False)
        .agg(numel=("numel", "sum"), delta_norm=("delta_norm", "sum"), relative_norm=("relative_norm", "mean"))
        .sort_values("delta_norm", ascending=False)
        .head(top_k)
    )
    fig, ax = plt.subplots(figsize=(12, max(5, 0.22 * len(grouped))), constrained_layout=True)
    ax.barh(grouped["group"][::-1], grouped["delta_norm"][::-1], color="#457b9d")
    ax.set_xlabel("sum of tensor delta norms")
    ax.set_title("Largest base-to-instruct parameter changes")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    tokenizer_path: str,
    path_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    best_avg = path_df.sort_values("avg_nll").iloc[0]
    best_worst = path_df.sort_values("worst_nll").iloc[0]
    base = path_df[path_df["lambda"] == 0.0].iloc[0] if (path_df["lambda"] == 0.0).any() else path_df.iloc[0]
    expert = path_df[path_df["lambda"] == 1.0].iloc[0] if (path_df["lambda"] == 1.0).any() else path_df.iloc[-1]
    top_groups = (
        delta_df.groupby("group", as_index=False)
        .agg(delta_norm=("delta_norm", "sum"), relative_norm=("relative_norm", "mean"))
        .sort_values("delta_norm", ascending=False)
        .head(5)
    )

    lines = [
        "# Qwen Base-to-Instruct Path Sweep",
        "",
        "This run evaluates a real LLM weight-space path using local Qwen2.5-1.5B base and Qwen2.5-1.5B-Instruct checkpoints.",
        "",
        "The path is `theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)`. Metrics are token-level negative log-likelihoods on a small fixed prompt slice, not full benchmark scores.",
        "",
        "## Key Results",
        "",
        f"- Base lambda 0.0: general NLL {base['general_nll']:.3f}, instruction NLL {base['instruction_nll']:.3f}, worst NLL {base['worst_nll']:.3f}.",
        f"- Instruct lambda 1.0: general NLL {expert['general_nll']:.3f}, instruction NLL {expert['instruction_nll']:.3f}, worst NLL {expert['worst_nll']:.3f}.",
        f"- Best average NLL lambda: {best_avg['lambda']:.3f}, avg NLL {best_avg['avg_nll']:.3f}.",
        f"- Best worst-task NLL lambda: {best_worst['lambda']:.3f}, worst NLL {best_worst['worst_nll']:.3f}.",
        "",
        "## Largest Delta Groups",
        "",
    ]
    for _, row in top_groups.iterrows():
        lines.append(
            f"- `{row['group']}`: delta norm {row['delta_norm']:.2f}, mean relative norm {row['relative_norm']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the LLM analogue of the lambda-sweep plot in the image experiment. Instead of a dense 2D plane, it samples a one-dimensional task-vector path from a base model to an instruction-tuned model. A lambda with lower average or worst NLL than both endpoints would indicate a useful intermediate merge point; a monotonic tradeoff indicates that the instruction delta mainly moves the model from one behavior regime toward another.",
            "",
            "Because this is a tiny fixed prompt slice, it should be treated as a diagnostic, not as an MMLU/GSM8K/HumanEval claim.",
            "",
            "## Files",
            "",
            "- `path_metrics.csv`: NLL/PPL metrics for every lambda.",
            "- `delta_summary.csv`: per-tensor base-to-instruct delta magnitudes.",
            "- `qwen_path_sweep.png`: path and tradeoff plot.",
            "- `delta_norms.png`: largest parameter-change groups.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(
                {
                    "base": args.base,
                    "expert": args.expert,
                    "tokenizer": tokenizer_path,
                    "lambdas": args.lambdas,
                    "device": args.device,
                    "dtype": args.dtype,
                    "max_length": args.max_length,
                },
                indent=2,
            ),
            "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_lambdas(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    best_device = None
    best_free = -1
    for index in range(torch.cuda.device_count()):
        name = f"cuda:{index}"
        try:
            _ = torch.empty(1, device=name)
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:
            continue
        if free > best_free:
            best_free = int(free)
            best_device = name
    return best_device or "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Qwen base-to-instruct weight-space path sweep.")
    parser.add_argument("--base", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-1.5B")
    parser.add_argument("--expert", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/qwen2.5-1.5b-instruct")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to expert tokenizer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_path_sweep"))
    parser.add_argument("--lambdas", default="-0.25,0.0,0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer_path = args.tokenizer or args.expert
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=Path(tokenizer_path).exists(), trust_remote_code=True)

    base_state = load_safetensor_state(args.base)
    expert_state = load_safetensor_state(args.expert)
    delta_df = summarize_delta(base_state, expert_state)
    delta_df.to_csv(args.output_dir / "delta_summary.csv", index=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        local_files_only=Path(args.base).exists(),
        trust_remote_code=True,
    ).to(device)
    model.eval()

    rows = []
    for lam in parse_lambdas(args.lambdas):
        set_interpolated_weights(model, base_state, expert_state, lam, device)
        metrics = evaluate_sets(model, tokenizer, device, args.max_length)
        rows.append({"lambda": lam, **metrics})
    path_df = pd.DataFrame(rows)
    path_df.to_csv(args.output_dir / "path_metrics.csv", index=False)

    plot_path(path_df, args.output_dir / "qwen_path_sweep.png")
    plot_delta_norms(delta_df, args.output_dir / "delta_norms.png")
    write_report(args.output_dir, args, tokenizer_path, path_df, delta_df)

    summary = {
        "base": args.base,
        "expert": args.expert,
        "tokenizer": tokenizer_path,
        "lambdas": parse_lambdas(args.lambdas),
        "device": str(device),
        "dtype": args.dtype,
        "general_examples": len(GENERAL_TEXTS),
        "instruction_examples": len(INSTRUCTION_EXAMPLES),
        "best_avg_lambda": float(path_df.sort_values("avg_nll").iloc[0]["lambda"]),
        "best_worst_lambda": float(path_df.sort_values("worst_nll").iloc[0]["lambda"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote Qwen path-sweep artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
