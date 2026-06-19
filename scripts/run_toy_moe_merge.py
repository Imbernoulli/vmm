#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


CATEGORY_TO_SOURCE = {"general": "general", "code": "code"}


class TinyMoEClassifier(nn.Module):
    def __init__(self, input_dim: int = 4, hidden: int = 24, n_experts: int = 4, n_classes: int = 2) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(input_dim, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, n_classes),
                )
                for _ in range(n_experts)
            ]
        )

    def expert_logits(self, x: Tensor) -> Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=1)

    def router_probs(self, x: Tensor) -> Tensor:
        return F.softmax(self.router(x), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        probs = self.router_probs(x)
        expert_logits = self.expert_logits(x)
        return torch.einsum("be,bec->bc", probs, expert_logits)


@dataclass(frozen=True)
class MethodState:
    name: str
    state: dict[str, Tensor]
    description: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_category_data(category: str, n: int, seed: int) -> TensorDataset:
    rng = np.random.default_rng(seed)
    xy = rng.normal(size=(n, 2)).astype("float32")
    if category == "general":
        domain = np.tile(np.array([1.0, 0.0], dtype="float32"), (n, 1))
        score = xy[:, 0] * xy[:, 1] + 0.35 * xy[:, 0] - 0.15 * xy[:, 1]
        y = (score > 0.0).astype("int64")
    elif category == "code":
        domain = np.tile(np.array([0.0, 1.0], dtype="float32"), (n, 1))
        radius = xy[:, 0] ** 2 + 0.7 * xy[:, 1] ** 2 + 0.25 * xy[:, 0]
        y = (radius > 1.15).astype("int64")
    else:
        raise ValueError(f"Unknown category: {category}")
    x = np.concatenate([xy, domain], axis=1).astype("float32")
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def concat_datasets(left: TensorDataset, right: TensorDataset) -> TensorDataset:
    lx, ly = left.tensors
    rx, ry = right.tensors
    return TensorDataset(torch.cat([lx, rx], dim=0), torch.cat([ly, ry], dim=0))


def prepare_data(seed: int, n_train_per_category: int, n_test_per_category: int, batch_size: int) -> dict[str, Any]:
    general_train = make_category_data("general", n_train_per_category, seed + 1)
    code_train = make_category_data("code", n_train_per_category, seed + 2)
    general_test = make_category_data("general", n_test_per_category, seed + 3)
    code_test = make_category_data("code", n_test_per_category, seed + 4)
    mixed_train = concat_datasets(general_train, code_train)
    mixed_test = concat_datasets(general_test, code_test)

    def loader(dataset: TensorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return {
        "general_train": loader(general_train, True),
        "code_train": loader(code_train, True),
        "mixed_train": loader(mixed_train, True),
        "general_test": loader(general_test, False),
        "code_test": loader(code_test, False),
        "mixed_test": loader(mixed_test, False),
    }


def load_balance_loss(model: TinyMoEClassifier, x: Tensor) -> Tensor:
    probs = model.router_probs(x)
    mean_probs = probs.mean(dim=0)
    return model.n_experts * mean_probs.pow(2).sum()


def train_model(
    model: TinyMoEClassifier,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    aux_coef: float,
    device: torch.device,
    desc: str,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + aux_coef * load_balance_loss(model, x)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model: TinyMoEClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.to(device)
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").detach().cpu())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return {"loss": loss_sum / total, "acc": correct / total, "n": total}


def cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def average_states(*states: dict[str, Tensor], weights: list[float] | None = None) -> dict[str, Tensor]:
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    out: dict[str, Tensor] = {}
    for name in states[0]:
        value = torch.zeros_like(states[0][name], dtype=torch.float32)
        for state, weight in zip(states, weights):
            value = value + float(weight) * state[name].to(torch.float32)
        out[name] = value.to(dtype=states[0][name].dtype)
    return out


def task_vector_average(base: dict[str, Tensor], sources: list[dict[str, Tensor]], weights: list[float]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name in base:
        value = base[name].to(torch.float32)
        for source, weight in zip(sources, weights):
            value = value + float(weight) * (source[name].to(torch.float32) - base[name].to(torch.float32))
        out[name] = value.to(dtype=base[name].dtype)
    return out


def permute_experts_and_router(model: TinyMoEClassifier, order: list[int]) -> TinyMoEClassifier:
    permuted = deepcopy(model).cpu()
    with torch.no_grad():
        for new_idx, old_idx in enumerate(order):
            permuted.experts[new_idx].load_state_dict(model.experts[old_idx].state_dict())
        permuted.router.weight.copy_(model.router.weight.detach().cpu()[order])
        permuted.router.bias.copy_(model.router.bias.detach().cpu()[order])
    return permuted


@torch.no_grad()
def expert_output_features(model: TinyMoEClassifier, loader: DataLoader, device: torch.device, max_batches: int) -> Tensor:
    model.to(device)
    model.eval()
    chunks: list[Tensor] = []
    batches = 0
    for x, _ in loader:
        x = x.to(device)
        chunks.append(model.expert_logits(x).detach().cpu())
        batches += 1
        if batches >= max_batches:
            break
    logits = torch.cat(chunks, dim=0)
    features = logits.transpose(0, 1).reshape(model.n_experts, -1)
    return F.normalize(features, dim=1)


def match_experts(
    reference: TinyMoEClassifier,
    target: TinyMoEClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[TinyMoEClassifier, pd.DataFrame]:
    ref_features = expert_output_features(reference, loader, device, max_batches)
    target_features = expert_output_features(target, loader, device, max_batches)
    similarity = ref_features @ target_features.T
    rows, cols = linear_sum_assignment((-similarity).numpy())
    order = [0] * target.n_experts
    match_rows = []
    for ref_idx, target_idx in zip(rows, cols):
        order[int(ref_idx)] = int(target_idx)
        match_rows.append(
            {
                "reference_expert": int(ref_idx),
                "target_expert_before_alignment": int(target_idx),
                "output_cosine": float(similarity[ref_idx, target_idx].item()),
            }
        )
    return permute_experts_and_router(target, order), pd.DataFrame(match_rows)


def route_mass_weights(
    base_model: TinyMoEClassifier,
    loaders: dict[str, DataLoader],
    device: torch.device,
    anchor_floor: float,
) -> pd.DataFrame:
    rows = []
    per_expert: dict[int, dict[str, float]] = {
        expert_id: {"general": 0.0, "code": 0.0} for expert_id in range(base_model.n_experts)
    }
    for category in ("general", "code"):
        stats = router_stats(base_model, loaders[f"{category}_test"], device, method="base", category=category)
        for row in stats["expert_rows"]:
            per_expert[int(row["expert_id"])][CATEGORY_TO_SOURCE[category]] += float(row["topk_fraction"])
    for expert_id, masses in per_expert.items():
        total = masses["general"] + masses["code"]
        if total <= 0:
            general_weight = 0.0
            code_weight = 0.0
            action = "anchor_heavy_or_freeze"
        else:
            scale = 1.0 - anchor_floor
            general_weight = scale * masses["general"] / total
            code_weight = scale * masses["code"] / total
            action = "route_frequency_weighted_average"
        rows.append(
            {
                "expert_id": expert_id,
                "route_mass_general": masses["general"],
                "route_mass_code": masses["code"],
                "weight_general": general_weight,
                "weight_code": code_weight,
                "anchor_floor": anchor_floor,
                "same_shape_action": action,
            }
        )
    return pd.DataFrame(rows)


def route_aware_state(
    base: dict[str, Tensor],
    general: dict[str, Tensor],
    code: dict[str, Tensor],
    weights: pd.DataFrame,
    n_experts: int,
) -> dict[str, Tensor]:
    out = {name: value.clone() for name, value in base.items()}
    weight_by_expert = {
        int(row["expert_id"]): (float(row["weight_general"]), float(row["weight_code"]))
        for _, row in weights.iterrows()
    }
    for name in base:
        if name.startswith("router."):
            out[name] = base[name].clone()
            continue
        expert_id = None
        for idx in range(n_experts):
            if name.startswith(f"experts.{idx}."):
                expert_id = idx
                break
        if expert_id is None:
            out[name] = task_vector_average(base, [general, code], [0.5, 0.5])[name]
            continue
        general_weight, code_weight = weight_by_expert.get(expert_id, (0.0, 0.0))
        value = base[name].to(torch.float32)
        value = value + general_weight * (general[name].to(torch.float32) - base[name].to(torch.float32))
        value = value + code_weight * (code[name].to(torch.float32) - base[name].to(torch.float32))
        out[name] = value.to(dtype=base[name].dtype)
    return out


@torch.no_grad()
def router_stats(
    model: TinyMoEClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    method: str,
    category: str,
    top_k: int = 2,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    top1_values: list[Tensor] = []
    topk_values: list[Tensor] = []
    entropy_values: list[Tensor] = []
    margin_values: list[Tensor] = []
    for x, _ in loader:
        x = x.to(device)
        probs = model.router_probs(x)
        k = min(top_k, model.n_experts)
        top_values, top_indices = torch.topk(probs, k=k, dim=-1)
        top1_values.append(top_indices[:, 0].cpu())
        topk_values.append(top_indices.cpu())
        entropy_values.append((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)).cpu())
        if k > 1:
            margin_values.append((top_values[:, 0] - top_values[:, 1]).cpu())
        else:
            margin_values.append(torch.ones_like(top_values[:, 0]).cpu())
    top1 = torch.cat(top1_values)
    topk = torch.cat(topk_values, dim=0)
    entropy = torch.cat(entropy_values)
    margin = torch.cat(margin_values)
    total = int(top1.numel())
    top1_counts = torch.bincount(top1, minlength=model.n_experts).to(torch.float64)
    topk_counts = torch.bincount(topk.reshape(-1), minlength=model.n_experts).to(torch.float64)
    top1_probs = top1_counts / max(1, total)
    positive = top1_probs[top1_probs > 0]
    top1_dist_entropy = -float((positive * torch.log(positive)).sum().item()) if len(positive) else 0.0
    summary_row = {
        "model": method,
        "method": method,
        "category": category,
        "prompt_idx": 0,
        "router": "toy_router",
        "num_experts": model.n_experts,
        "tokens": total,
        "top_k": min(top_k, model.n_experts),
        "router_entropy_mean": float(entropy.mean().item()),
        "top1_margin_mean": float(margin.mean().item()),
        "unique_top1_experts": int((top1_counts > 0).sum().item()),
        "unique_topk_experts": int((topk_counts > 0).sum().item()),
        "max_top1_fraction": float((top1_counts.max() / max(1, total)).item()),
        "effective_top1_experts": math.exp(top1_dist_entropy),
    }
    expert_rows = []
    for expert_id in range(model.n_experts):
        expert_rows.append(
            {
                "model": method,
                "method": method,
                "category": category,
                "prompt_idx": 0,
                "router": "toy_router",
                "expert_id": expert_id,
                "top1_count": int(top1_counts[expert_id].item()),
                "top1_fraction": float((top1_counts[expert_id] / max(1, total)).item()),
                "topk_count": int(topk_counts[expert_id].item()),
                "topk_fraction": float((topk_counts[expert_id] / max(1, total * min(top_k, model.n_experts))).item()),
            }
        )
    return {"summary_row": summary_row, "expert_rows": expert_rows, "top1": top1, "topk": topk}


def route_overlap(
    left: TinyMoEClassifier,
    right: TinyMoEClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    left_name: str,
    right_name: str,
    category: str,
) -> dict[str, Any]:
    left_stats = router_stats(left, loader, device, method=left_name, category=category)
    right_stats = router_stats(right, loader, device, method=right_name, category=category)
    left_top1 = left_stats["top1"]
    right_top1 = right_stats["top1"]
    left_topk = left_stats["topk"]
    right_topk = right_stats["topk"]
    n = min(left_top1.numel(), right_top1.numel())
    top1_agreement = float((left_top1[:n] == right_top1[:n]).to(torch.float32).mean().item())
    jaccards = []
    for idx in range(n):
        a = set(left_topk[idx].tolist())
        b = set(right_topk[idx].tolist())
        jaccards.append(len(a & b) / max(1, len(a | b)))
    return {
        "left_model": left_name,
        "right_model": right_name,
        "category": category,
        "prompt_idx": 0,
        "router": "toy_router",
        "tokens_compared": n,
        "top1_agreement": top1_agreement,
        "topk_jaccard": float(sum(jaccards) / len(jaccards)),
    }


def evaluate_method(
    template: TinyMoEClassifier,
    method: MethodState,
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model = deepcopy(template)
    model.load_state_dict(method.state)
    general = evaluate(model, loaders["general_test"], device)
    code = evaluate(model, loaders["code_test"], device)
    method_row = {
        "method": method.name,
        "description": method.description,
        "general_loss": general["loss"],
        "code_loss": code["loss"],
        "general_acc": general["acc"],
        "code_acc": code["acc"],
        "avg_loss": 0.5 * (general["loss"] + code["loss"]),
        "worst_loss": max(general["loss"], code["loss"]),
        "avg_acc": 0.5 * (general["acc"] + code["acc"]),
        "worst_acc": min(general["acc"], code["acc"]),
    }
    router_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    for category in ("general", "code"):
        stats = router_stats(model, loaders[f"{category}_test"], device, method=method.name, category=category)
        router_rows.append(stats["summary_row"])
        expert_rows.extend(stats["expert_rows"])
    return method_row, router_rows, expert_rows


def plot_results(method_metrics: pd.DataFrame, router_summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    order = method_metrics.sort_values("worst_acc", ascending=False)["method"].tolist()
    ax = axes[0]
    values = method_metrics.set_index("method").loc[order, "worst_acc"]
    ax.barh(range(len(order)), values, color="#2a9d8f")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("worst-category accuracy")
    ax.set_title("MoE average methods")

    ax = axes[1]
    pivot = router_summary.pivot_table(index="method", columns="category", values="max_top1_fraction", aggfunc="mean")
    pivot = pivot.reindex(order)
    pivot.plot(kind="barh", ax=ax, color=["#264653", "#e76f51"])
    ax.invert_yaxis()
    ax.set_xlabel("max top-1 expert fraction")
    ax.set_title("Router load concentration")
    ax.legend(title="category", fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, summary: dict[str, Any], method_metrics: pd.DataFrame) -> None:
    best = method_metrics.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    all_avg = method_metrics[method_metrics["method"] == "all_weight_average"].iloc[0]
    matched = method_metrics[method_metrics["method"] == "expert_matched_average"].iloc[0]
    matched_router_frozen = method_metrics[method_metrics["method"] == "matched_router_frozen_average"].iloc[0]
    route_aware = method_metrics[method_metrics["method"] == "route_aware_expert_average"].iloc[0]
    lines = [
        "# Toy MoE Route-Aware Merge",
        "",
        "这个实验用一个很小的 soft-router MoE 做可控验证：base 先在 general/code 两类合成任务上训练，然后从同一 base fine-tune 两个同构 source。为了模拟 MoE 中常见的 expert-index 语义漂移，code source 在保持函数等价的前提下被 permute experts 和 router rows。",
        "",
        "它验证的点很具体：直接 all-weight average 会把不同语义的 expert index 相加；expert matching 和 route-frequency expert weights 可以缓解这个问题；router 是否能开放平均要看 route overlap、load concentration 和 top-k margin。",
        "",
        "## 关键结果",
        "",
        f"- Best method by worst accuracy: `{best['method']}` = `{best['worst_acc']:.3f}`.",
        f"- All-weight average worst accuracy: `{all_avg['worst_acc']:.3f}`.",
        f"- Expert-matched average worst accuracy: `{matched['worst_acc']:.3f}`.",
        f"- Matched + router-frozen average worst accuracy: `{matched_router_frozen['worst_acc']:.3f}`.",
        f"- Route-aware expert average worst accuracy: `{route_aware['worst_acc']:.3f}`.",
        f"- Recovered expert matching mean cosine: `{summary['expert_match_mean_cosine']:.3f}`.",
        f"- Code source permutation: `{summary['code_source_permutation']}`.",
        "",
        "## Method Table",
        "",
        "| method | general acc | code acc | worst acc | avg loss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in method_metrics.sort_values("worst_acc", ascending=False).iterrows():
        lines.append(
            f"| {row['method']} | {row['general_acc']:.3f} | {row['code_acc']:.3f} | "
            f"{row['worst_acc']:.3f} | {row['avg_loss']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `all_weight_average` 是朴素 baseline：router 和 expert tensors 都按同名 index 平均，因此在 expert permutation 后会暴露 MoE index-alignment 风险。",
            "- `expert_matched_average` 先用 unlabeled calibration input 的 expert-output cosine 做 Hungarian matching，再平均；这对应 Sub-MoE / Expert Merging 里强调的 function-aware expert alignment。",
            "- `matched_router_frozen_average` 直接验证 MoE 特有假设：先对齐 expert 功能，再固定 token-to-expert dispatch，只平均非 router 权重。",
            "- `route_aware_expert_average` 冻结 base router，并按 base router 在 general/code prompt 上的 route mass 给每个 expert 设置 source delta 权重；这对应 route-weight recipes 的 toy 版本。",
            "- 这个实验不是 Qwen3 结果，但它把 MoE merging 的特质从报告落成了可跑的 probe：expert index、router overlap、expert load 和 category route mass 都会影响 average 是否安全。",
            "",
            "## Files",
            "",
            "- `method_metrics.csv`",
            "- `router_summary.csv`",
            "- `expert_load.csv`",
            "- `route_overlap.csv`",
            "- `expert_match.csv`",
            "- `route_weights_by_expert.csv`",
            "- `toy_moe_merge.png`",
            "- `summary.json`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small route-aware MoE model averaging experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_moe_merge"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--train-per-category", type=int, default=500)
    parser.add_argument("--test-per-category", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-coef", type=float, default=0.02)
    parser.add_argument("--anchor-floor", type=float, default=0.15)
    parser.add_argument("--match-batches", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    loaders = prepare_data(args.seed, args.train_per_category, args.test_per_category, args.batch_size)
    template = TinyMoEClassifier(hidden=args.hidden, n_experts=args.experts)

    base = deepcopy(template)
    train_model(
        base,
        loaders["mixed_train"],
        epochs=args.base_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aux_coef=args.aux_coef,
        device=device,
        desc="train toy MoE base",
    )
    base.cpu()
    base_state = cpu_state(base)

    general_model = deepcopy(base)
    train_model(
        general_model,
        loaders["general_train"],
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        aux_coef=args.aux_coef,
        device=device,
        desc="fine-tune general source",
    )
    general_model.cpu()
    general_state = cpu_state(general_model)

    code_model = deepcopy(base)
    train_model(
        code_model,
        loaders["code_train"],
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        aux_coef=args.aux_coef,
        device=device,
        desc="fine-tune code source",
    )
    code_model.cpu()
    permutation = list(range(args.experts))
    if args.experts >= 4:
        permutation = [2, 0, 3, 1] + list(range(4, args.experts))
    else:
        permutation = list(reversed(permutation))
    code_permuted = permute_experts_and_router(code_model, permutation)
    code_permuted_state = cpu_state(code_permuted)

    matched_code, expert_match = match_experts(general_model, code_permuted, loaders["mixed_test"], device, args.match_batches)
    matched_code.cpu()
    matched_code_state = cpu_state(matched_code)

    route_weights = route_mass_weights(base, loaders, device, args.anchor_floor)
    route_aware = route_aware_state(base_state, general_state, matched_code_state, route_weights, args.experts)

    all_average = task_vector_average(base_state, [general_state, code_permuted_state], [0.5, 0.5])
    router_frozen = {name: value.clone() for name, value in all_average.items()}
    for name in router_frozen:
        if name.startswith("router."):
            router_frozen[name] = base_state[name].clone()
    expert_matched = task_vector_average(base_state, [general_state, matched_code_state], [0.5, 0.5])
    matched_router_frozen = {name: value.clone() for name, value in expert_matched.items()}
    for name in matched_router_frozen:
        if name.startswith("router."):
            matched_router_frozen[name] = base_state[name].clone()

    methods = [
        MethodState("base", base_state, "mixed-task base before fine-tuning"),
        MethodState("general_endpoint", general_state, "general source endpoint"),
        MethodState("code_endpoint_permuted", code_permuted_state, "code source endpoint with function-preserving expert permutation"),
        MethodState("all_weight_average", all_average, "average same-name router and expert tensors without expert matching"),
        MethodState("router_frozen_average", router_frozen, "all-weight average but router tensors reset to base"),
        MethodState("expert_matched_average", expert_matched, "align code experts to general source by output cosine before averaging"),
        MethodState(
            "matched_router_frozen_average",
            matched_router_frozen,
            "align code experts by output cosine and keep the base router fixed",
        ),
        MethodState("route_aware_expert_average", route_aware, "freeze base router and use route-frequency expert source weights"),
    ]

    method_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    models_for_overlap: dict[str, TinyMoEClassifier] = {}
    for method in methods:
        row, router_stats_rows, expert_stats_rows = evaluate_method(template, method, loaders, device)
        method_rows.append(row)
        router_rows.extend(router_stats_rows)
        expert_rows.extend(expert_stats_rows)
        model = deepcopy(template)
        model.load_state_dict(method.state)
        models_for_overlap[method.name] = model

    overlap_rows = []
    for method_name, model in models_for_overlap.items():
        if method_name == "base":
            continue
        for category in ("general", "code"):
            overlap_rows.append(
                route_overlap(
                    models_for_overlap["base"],
                    model,
                    loaders[f"{category}_test"],
                    device,
                    left_name="base",
                    right_name=method_name,
                    category=category,
                )
            )

    method_metrics = pd.DataFrame(method_rows)
    router_summary = pd.DataFrame(router_rows)
    expert_load = pd.DataFrame(expert_rows)
    route_overlap_df = pd.DataFrame(overlap_rows)

    method_metrics.to_csv(args.output_dir / "method_metrics.csv", index=False)
    router_summary.to_csv(args.output_dir / "router_summary.csv", index=False)
    expert_load.to_csv(args.output_dir / "expert_load.csv", index=False)
    route_overlap_df.to_csv(args.output_dir / "route_overlap.csv", index=False)
    expert_match.to_csv(args.output_dir / "expert_match.csv", index=False)
    route_weights.to_csv(args.output_dir / "route_weights_by_expert.csv", index=False)
    plot_results(method_metrics, router_summary, args.output_dir / "toy_moe_merge.png")

    best = method_metrics.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    all_avg = method_metrics[method_metrics["method"] == "all_weight_average"].iloc[0]
    matched_router_frozen_row = method_metrics[method_metrics["method"] == "matched_router_frozen_average"].iloc[0]
    route_aware_row = method_metrics[method_metrics["method"] == "route_aware_expert_average"].iloc[0]
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "n_experts": args.experts,
        "code_source_permutation": permutation,
        "expert_match_mean_cosine": float(expert_match["output_cosine"].mean()),
        "best_method": str(best["method"]),
        "best_worst_acc": float(best["worst_acc"]),
        "all_weight_average_worst_acc": float(all_avg["worst_acc"]),
        "matched_router_frozen_worst_acc": float(matched_router_frozen_row["worst_acc"]),
        "matched_router_frozen_minus_all_weight_worst_acc": float(
            matched_router_frozen_row["worst_acc"] - all_avg["worst_acc"]
        ),
        "route_aware_worst_acc": float(route_aware_row["worst_acc"]),
        "route_aware_minus_all_weight_worst_acc": float(route_aware_row["worst_acc"] - all_avg["worst_acc"]),
        "same_shape_constraint": "All methods keep the same TinyMoEClassifier architecture, expert count, router shape, and output classes.",
        "outputs": {
            "method_metrics": "method_metrics.csv",
            "router_summary": "router_summary.csv",
            "expert_load": "expert_load.csv",
            "route_overlap": "route_overlap.csv",
            "expert_match": "expert_match.csv",
            "route_weights_by_expert": "route_weights_by_expert.csv",
            "figure": "toy_moe_merge.png",
            "report": "report.md",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_dir, summary, method_metrics)
    print(f"Wrote toy MoE merge results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
