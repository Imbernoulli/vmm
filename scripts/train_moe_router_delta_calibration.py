#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from write_same_shape_average_checkpoint import discover_safetensors, load_tensors


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_cache(cache_path: Path) -> list[dict[str, Any]]:
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or "routers" not in payload:
        raise ValueError("Cache must be a torch file containing a {'routers': ...} payload.")
    routers = payload["routers"]
    records: list[dict[str, Any]] = []
    if isinstance(routers, dict):
        iterator = routers.items()
    elif isinstance(routers, list):
        iterator = []
        for item in routers:
            if not isinstance(item, dict) or "tensor" not in item:
                raise ValueError("Each router cache row must contain a 'tensor' field.")
            iterator.append((item["tensor"], item))
    else:
        raise ValueError("'routers' must be either a dict keyed by tensor name or a list of records.")

    for tensor_name, record in iterator:
        if not isinstance(record, dict):
            raise ValueError(f"Router record for {tensor_name!r} must be a dict.")
        hidden = record.get("hidden")
        teacher_logits = record.get("teacher_logits")
        if not torch.is_tensor(hidden) or not torch.is_tensor(teacher_logits):
            raise ValueError(f"Router record for {tensor_name!r} must contain tensor hidden and teacher_logits.")
        if hidden.ndim != 2 or teacher_logits.ndim != 2:
            raise ValueError(f"Router record for {tensor_name!r} expects 2D hidden and teacher_logits tensors.")
        if hidden.shape[0] != teacher_logits.shape[0]:
            raise ValueError(
                f"Router record for {tensor_name!r} has hidden rows {hidden.shape[0]} "
                f"but teacher rows {teacher_logits.shape[0]}."
            )
        records.append(
            {
                "tensor": str(tensor_name),
                "bias_tensor": record.get("bias_tensor"),
                "hidden": hidden.to(torch.float32),
                "teacher_logits": teacher_logits.to(torch.float32),
            }
        )
    if not records:
        raise ValueError("Cache contains no router records.")
    return records


def load_base_tensors(base_path: Path, tensor_names: list[str]) -> dict[str, torch.Tensor]:
    index = discover_safetensors(base_path)
    missing = [name for name in tensor_names if name not in index.tensor_info]
    if missing:
        raise KeyError(f"Base checkpoint is missing router tensors: {missing[:5]}")
    return load_tensors(index, tensor_names)


def orientation_for(weight: torch.Tensor, hidden_dim: int, expert_dim: int, tensor_name: str) -> str:
    if weight.ndim != 2:
        raise ValueError(f"Router tensor {tensor_name!r} must be 2D, got shape {tuple(weight.shape)}")
    if weight.shape == (expert_dim, hidden_dim):
        return "out_in"
    if weight.shape == (hidden_dim, expert_dim):
        return "in_out"
    raise ValueError(
        f"Router tensor {tensor_name!r} shape {tuple(weight.shape)} is incompatible with "
        f"hidden_dim={hidden_dim}, num_experts={expert_dim}."
    )


def router_logits(hidden: torch.Tensor, weight: torch.Tensor, orientation: str, bias: torch.Tensor | None) -> torch.Tensor:
    if orientation == "out_in":
        logits = hidden @ weight.t()
    elif orientation == "in_out":
        logits = hidden @ weight
    else:
        raise ValueError(f"Unknown router orientation: {orientation}")
    if bias is not None:
        logits = logits + bias
    return logits


def topk_jaccard(student_logits: torch.Tensor, teacher_logits: torch.Tensor, top_k: int) -> float:
    k = min(top_k, student_logits.shape[-1], teacher_logits.shape[-1])
    student = torch.topk(student_logits, k=k, dim=-1).indices
    teacher = torch.topk(teacher_logits, k=k, dim=-1).indices
    scores = []
    for row in range(student.shape[0]):
        left = set(int(item) for item in student[row].tolist())
        right = set(int(item) for item in teacher[row].tolist())
        scores.append(len(left & right) / max(1, len(left | right)))
    return float(sum(scores) / max(1, len(scores)))


def capacity_overflow_fraction(logits: torch.Tensor, capacity_factor: float) -> float:
    probs = F.softmax(logits, dim=-1)
    mean_load = probs.mean(dim=0)
    capacity = float(capacity_factor) / max(1, probs.shape[-1])
    overflow = F.relu(mean_load - capacity).sum()
    return float(overflow.item())


def _load_entropy(load: torch.Tensor) -> float:
    if load.numel() <= 1:
        return 1.0
    total = load.sum().clamp_min(1e-12)
    probs = load / total
    positive = probs[probs > 0]
    entropy = -(positive * torch.log(positive)).sum()
    return float((entropy / math.log(load.numel())).item())


def _load_cv(load: torch.Tensor) -> float:
    if load.numel() <= 1:
        return 0.0
    return float((load.std(unbiased=False) / load.mean().clamp_min(1e-12)).item())


def route_load_stats(logits: torch.Tensor, top_k: int, capacity_factor: float) -> dict[str, float]:
    num_experts = int(logits.shape[-1])
    if num_experts <= 0:
        raise ValueError("Router logits must have at least one expert dimension.")
    token_count = int(logits.shape[0])
    capacity = float(capacity_factor) / max(1, num_experts)

    top1_indices = logits.argmax(dim=-1)
    top1_counts = torch.bincount(top1_indices, minlength=num_experts).to(torch.float32)
    top1_load = top1_counts / max(1, token_count)
    top1_overflow = F.relu(top1_load - capacity).sum()

    k = min(max(1, int(top_k)), num_experts)
    topk_indices = torch.topk(logits, k=k, dim=-1).indices.reshape(-1)
    topk_counts = torch.bincount(topk_indices, minlength=num_experts).to(torch.float32)
    topk_load = topk_counts / max(1, token_count * k)
    topk_overflow = F.relu(topk_load - capacity).sum()

    return {
        "top1_max_load_fraction": float(top1_load.max().item()),
        "top1_capacity_overflow_fraction": float(top1_overflow.item()),
        "top1_load_entropy": _load_entropy(top1_load),
        "top1_load_cv": _load_cv(top1_load),
        "topk_max_load_fraction": float(topk_load.max().item()),
        "topk_capacity_overflow_fraction": float(topk_overflow.item()),
        "topk_load_entropy": _load_entropy(topk_load),
        "topk_load_cv": _load_cv(topk_load),
    }


def metric_row(
    *,
    tensor_name: str,
    stage: str,
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    top_k: int,
    capacity_factor: float,
    selection_split: str,
    train_samples: int,
    selection_samples: int,
    validation_fraction: float,
) -> dict[str, Any]:
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    route_kl = F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs, reduction="batchmean")
    teacher_top1 = teacher_logits.argmax(dim=-1)
    student_top1 = logits.argmax(dim=-1)
    delta_norm = float(torch.linalg.vector_norm(delta_weight).item())
    base_norm = float(torch.linalg.vector_norm(base_weight.to(torch.float32)).item())
    load_stats = route_load_stats(logits, top_k=top_k, capacity_factor=capacity_factor)
    return {
        "tensor": tensor_name,
        "stage": stage,
        "selection_split": selection_split,
        "train_samples": int(train_samples),
        "selection_samples": int(selection_samples),
        "validation_fraction": float(validation_fraction),
        "route_kl": float(route_kl.item()),
        "top1_agreement": float((student_top1 == teacher_top1).to(torch.float32).mean().item()),
        "topk_jaccard": topk_jaccard(logits, teacher_logits, top_k=top_k),
        "capacity_overflow_fraction": capacity_overflow_fraction(logits, capacity_factor),
        **load_stats,
        "delta_norm": delta_norm,
        "base_norm": base_norm,
        "relative_delta_norm": delta_norm / max(1e-12, base_norm),
        "max_abs_delta": float(delta_weight.abs().max().item()) if delta_weight.numel() else 0.0,
    }


def project_relative_norm(delta: torch.Tensor, base: torch.Tensor, max_relative_norm: float | None) -> None:
    if max_relative_norm is None or max_relative_norm <= 0:
        return
    base_norm = torch.linalg.vector_norm(base.to(torch.float32)).clamp_min(1e-12)
    delta_norm = torch.linalg.vector_norm(delta.data).clamp_min(1e-12)
    max_norm = float(max_relative_norm) * base_norm
    if delta_norm > max_norm:
        delta.data.mul_(max_norm / delta_norm)


def selection_metrics(
    *,
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    top_k: int,
    capacity_factor: float,
) -> dict[str, float]:
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    route_kl = F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs, reduction="batchmean")
    top1_agreement = (logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).to(torch.float32).mean()
    delta_norm = torch.linalg.vector_norm(delta_weight).item()
    base_norm = torch.linalg.vector_norm(base_weight.to(torch.float32)).item()
    return {
        "route_kl": float(route_kl.item()),
        "top1_agreement": float(top1_agreement.item()),
        "relative_delta_norm": float(delta_norm / max(1e-12, base_norm)),
        **route_load_stats(logits, top_k=top_k, capacity_factor=capacity_factor),
    }


def router_selection_score(
    candidate: dict[str, float],
    initial: dict[str, float],
    args: argparse.Namespace,
) -> float:
    top1_overflow = float(candidate["top1_capacity_overflow_fraction"])
    topk_overflow = float(candidate["topk_capacity_overflow_fraction"])
    top1_increase = max(0.0, top1_overflow - float(initial["top1_capacity_overflow_fraction"]))
    topk_increase = max(0.0, topk_overflow - float(initial["topk_capacity_overflow_fraction"]))
    top1_regression = max(0.0, float(initial["top1_agreement"]) - float(candidate["top1_agreement"]))
    return (
        float(candidate["route_kl"])
        + args.capacity_overflow_score_penalty * (top1_overflow + 2.0 * topk_overflow)
        + args.capacity_increase_score_penalty * (top1_increase + 2.0 * topk_increase)
        + args.top1_regression_score_penalty * top1_regression
        + args.relative_norm_score_penalty * float(candidate["relative_delta_norm"])
    )


def stable_name_seed(name: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name)) % 1_000_003


def split_router_cache(
    *,
    hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    tensor_name: str,
    validation_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    sample_count = int(hidden.shape[0])
    if sample_count < 2 or validation_fraction <= 0:
        return {
            "train_hidden": hidden,
            "train_teacher_logits": teacher_logits,
            "selection_hidden": hidden,
            "selection_teacher_logits": teacher_logits,
            "train_samples": sample_count,
            "selection_samples": sample_count,
            "selection_split": "train",
            "validation_fraction": 0.0,
        }

    validation_count = int(round(sample_count * float(validation_fraction)))
    validation_count = min(sample_count - 1, max(1, validation_count))
    generator = torch.Generator().manual_seed(int(split_seed) + stable_name_seed(tensor_name))
    permutation = torch.randperm(sample_count, generator=generator)
    validation_idx = permutation[:validation_count]
    train_idx = permutation[validation_count:]
    return {
        "train_hidden": hidden.index_select(0, train_idx),
        "train_teacher_logits": teacher_logits.index_select(0, train_idx),
        "selection_hidden": hidden.index_select(0, validation_idx),
        "selection_teacher_logits": teacher_logits.index_select(0, validation_idx),
        "train_samples": int(train_idx.numel()),
        "selection_samples": int(validation_idx.numel()),
        "selection_split": "validation",
        "validation_fraction": float(validation_count / sample_count),
    }


def train_one_router(
    record: dict[str, Any],
    base_values: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    tensor_name = record["tensor"]
    hidden = record["hidden"]
    teacher_logits = record["teacher_logits"]
    split = split_router_cache(
        hidden=hidden,
        teacher_logits=teacher_logits,
        tensor_name=tensor_name,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    train_hidden = split["train_hidden"]
    train_teacher_logits = split["train_teacher_logits"]
    selection_hidden = split["selection_hidden"]
    selection_teacher_logits = split["selection_teacher_logits"]
    base_weight = base_values[tensor_name].to(torch.float32)
    orientation = orientation_for(base_weight, hidden.shape[-1], teacher_logits.shape[-1], tensor_name)
    bias_tensor = record.get("bias_tensor")
    base_bias = None
    delta_bias = None
    if bias_tensor:
        base_bias = base_values[str(bias_tensor)].to(torch.float32)
        if base_bias.shape != (teacher_logits.shape[-1],):
            raise ValueError(f"Bias tensor {bias_tensor!r} has incompatible shape {tuple(base_bias.shape)}")
        delta_bias = torch.nn.Parameter(torch.zeros_like(base_bias))

    delta_weight = torch.nn.Parameter(torch.zeros_like(base_weight))
    parameters = [delta_weight]
    if delta_bias is not None:
        parameters.append(delta_bias)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    teacher_probs = F.softmax(train_teacher_logits / args.temperature, dim=-1)
    teacher_top1 = train_teacher_logits.argmax(dim=-1)
    selection_teacher_top1 = selection_teacher_logits.argmax(dim=-1)
    trace_rows: list[dict[str, Any]] = []
    initial_logits = router_logits(selection_hidden, base_weight, orientation, base_bias)
    initial_selection = selection_metrics(
        logits=initial_logits,
        teacher_logits=selection_teacher_logits,
        base_weight=base_weight,
        delta_weight=delta_weight.detach(),
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
    )
    best_epoch = 0
    best_score = router_selection_score(initial_selection, initial_selection, args)
    best_weight = delta_weight.detach().clone()
    best_bias = None if delta_bias is None else delta_bias.detach().clone()
    best_selection = dict(initial_selection)
    metric_rows = [
        metric_row(
            tensor_name=tensor_name,
            stage="initial",
            logits=initial_logits,
            teacher_logits=selection_teacher_logits,
            base_weight=base_weight,
            delta_weight=delta_weight.detach(),
            top_k=args.top_k,
            capacity_factor=args.capacity_factor,
            selection_split=split["selection_split"],
            train_samples=split["train_samples"],
            selection_samples=split["selection_samples"],
            validation_fraction=split["validation_fraction"],
        )
    ]
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        bias = None if base_bias is None else base_bias + delta_bias
        logits = router_logits(train_hidden, base_weight + delta_weight, orientation, bias)
        route_kl = F.kl_div(
            F.log_softmax(logits / args.temperature, dim=-1),
            teacher_probs,
            reduction="batchmean",
        ) * (args.temperature**2)
        top1_loss = F.cross_entropy(logits, teacher_top1)
        probs = F.softmax(logits, dim=-1)
        mean_load = probs.mean(dim=0)
        load_balance = probs.shape[-1] * mean_load.pow(2).sum()
        capacity = float(args.capacity_factor) / max(1, probs.shape[-1])
        capacity_loss = probs.shape[-1] * F.relu(mean_load - capacity).pow(2).sum()
        trust_loss = delta_weight.pow(2).sum() / base_weight.pow(2).sum().clamp_min(1e-12)
        if delta_bias is not None and base_bias is not None:
            trust_loss = trust_loss + delta_bias.pow(2).sum() / base_bias.pow(2).sum().clamp_min(1e-12)
        loss = (
            route_kl
            + args.top1_loss_coef * top1_loss
            + args.load_balance_coef * load_balance
            + args.capacity_loss_coef * capacity_loss
            + args.trust_l2_coef * trust_loss
        )
        loss.backward()
        optimizer.step()
        project_relative_norm(delta_weight, base_weight, args.max_relative_norm)
        if delta_bias is not None and base_bias is not None:
            project_relative_norm(delta_bias, base_bias, args.max_relative_norm)
        with torch.no_grad():
            eval_logits = router_logits(
                selection_hidden,
                base_weight + delta_weight,
                orientation,
                None if base_bias is None else base_bias + delta_bias,
            )
            current_selection = selection_metrics(
                logits=eval_logits,
                teacher_logits=selection_teacher_logits,
                base_weight=base_weight,
                delta_weight=delta_weight.detach(),
                top_k=args.top_k,
                capacity_factor=args.capacity_factor,
            )
            current_score = router_selection_score(current_selection, initial_selection, args)
            if args.selection_policy == "final" or current_score < best_score:
                best_epoch = epoch + 1
                best_score = current_score
                best_weight = delta_weight.detach().clone()
                best_bias = None if delta_bias is None else delta_bias.detach().clone()
                best_selection = dict(current_selection)
        if epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % max(1, args.trace_every) == 0:
            with torch.no_grad():
                load_stats = route_load_stats(eval_logits, top_k=args.top_k, capacity_factor=args.capacity_factor)
                trace_rows.append(
                    {
                        "tensor": tensor_name,
                        "epoch": epoch + 1,
                        "loss": float(loss.item()),
                        "route_kl": float(route_kl.item()),
                        "top1_loss": float(top1_loss.item()),
                        "load_balance": float(load_balance.item()),
                        "capacity_loss": float(capacity_loss.item()),
                        "trust_l2": float(trust_loss.item()),
                        "top1_agreement": float((eval_logits.argmax(dim=-1) == selection_teacher_top1).float().mean().item()),
                        "topk_jaccard": topk_jaccard(eval_logits, selection_teacher_logits, top_k=args.top_k),
                        "capacity_overflow_fraction": capacity_overflow_fraction(eval_logits, args.capacity_factor),
                        "selection_split": split["selection_split"],
                        "selection_score": current_score,
                        "selection_route_kl": current_selection["route_kl"],
                        "selection_top1_agreement": current_selection["top1_agreement"],
                        **load_stats,
                    }
                )

    with torch.no_grad():
        delta_weight.data.copy_(best_weight)
        if delta_bias is not None and best_bias is not None:
            delta_bias.data.copy_(best_bias)
        final_logits = router_logits(
            selection_hidden,
            base_weight + delta_weight,
            orientation,
            None if base_bias is None else base_bias + delta_bias,
        )
        metric_rows.append(
            metric_row(
                tensor_name=tensor_name,
                stage="final",
                logits=final_logits,
                teacher_logits=selection_teacher_logits,
                base_weight=base_weight,
                delta_weight=delta_weight.detach(),
                top_k=args.top_k,
                capacity_factor=args.capacity_factor,
                selection_split=split["selection_split"],
                train_samples=split["train_samples"],
                selection_samples=split["selection_samples"],
                validation_fraction=split["validation_fraction"],
            )
        )
        metric_rows[-1]["selected_epoch"] = best_epoch
        metric_rows[-1]["selection_score"] = best_score
        metric_rows[-1]["selection_policy"] = args.selection_policy
        metric_rows[-1]["selected_route_kl"] = best_selection["route_kl"]
        metric_rows[-1]["selected_top1_agreement"] = best_selection["top1_agreement"]
    delta_tensors = {tensor_name: delta_weight.detach().cpu()}
    if delta_bias is not None and bias_tensor:
        delta_tensors[str(bias_tensor)] = delta_bias.detach().cpu()
    return delta_tensors, metric_rows, trace_rows


def build_report(summary: dict[str, Any], metric_df: pd.DataFrame) -> str:
    lines = [
        "# MoE Router Delta Calibration",
        "",
        "这个脚本把离线 route probe 得到的 hidden states 和 teacher router logits 转成同构 checkpoint writer 可写入的 safetensors delta。它不是直接平均 router weight；目标函数是 route-KD/top-1 route imitation，并用 capacity、load-balance 和 delta trust term 约束移动幅度。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Routers: `{summary['router_count']}`",
        f"- Delta tensors: `{summary['delta_tensor_count']}`",
        f"- Mean initial KL: `{summary['mean_initial_route_kl']:.6f}`",
        f"- Mean final KL: `{summary['mean_final_route_kl']:.6f}`",
        f"- Mean initial top-1 agreement: `{summary['mean_initial_top1_agreement']:.4f}`",
        f"- Mean final top-1 agreement: `{summary['mean_final_top1_agreement']:.4f}`",
        f"- Max final hard top-1 capacity overflow: `{summary['max_final_top1_capacity_overflow_fraction']:.6f}`",
        f"- Max final hard top-k capacity overflow: `{summary['max_final_topk_capacity_overflow_fraction']:.6f}`",
        f"- Max router hard top-1 overflow increase: `{summary['max_router_top1_capacity_overflow_increase']:.6f}`",
        f"- Max router hard top-k overflow increase: `{summary['max_router_topk_capacity_overflow_increase']:.6f}`",
        f"- Selection policy: `{summary['selection_policy']}`",
        f"- Selection split: `{summary['selection_split']}`",
        f"- Mean selected epoch: `{summary['mean_selected_epoch']:.2f}`",
        f"- Mean train/selection samples: `{summary['mean_train_samples']:.1f}` / `{summary['mean_selection_samples']:.1f}`",
        "",
        "## Writer",
        "",
        "```bash",
        summary["writer_command"],
        "```",
        "",
        "## Router Metrics",
        "",
        "| tensor | selected epoch | initial KL | final KL | initial top1 | final top1 | final rel delta | top1 overflow initial-final | top-k overflow initial-final | top1 load initial-final | top-k load initial-final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tensor, group in metric_df.groupby("tensor", sort=True):
        initial = group[group["stage"] == "initial"].iloc[0]
        final = group[group["stage"] == "final"].iloc[0]
        lines.append(
            f"| `{tensor}` | {int(final.get('selected_epoch', summary['mean_selected_epoch']))} | "
            f"{float(initial['route_kl']):.6f} | {float(final['route_kl']):.6f} | "
            f"{float(initial['top1_agreement']):.4f} | {float(final['top1_agreement']):.4f} | "
            f"{float(final['relative_delta_norm']):.4f} | "
            f"{float(initial['top1_capacity_overflow_fraction']):.4f}-{float(final['top1_capacity_overflow_fraction']):.4f} | "
            f"{float(initial['topk_capacity_overflow_fraction']):.4f}-{float(final['topk_capacity_overflow_fraction']):.4f} | "
            f"{float(initial['top1_max_load_fraction']):.4f}-{float(final['top1_max_load_fraction']):.4f} | "
            f"{float(initial['topk_max_load_fraction']):.4f}-{float(final['topk_max_load_fraction']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['router_delta_safetensors']}`",
            f"- `{summary['outputs']['router_delta_summary']}`",
            f"- `{summary['outputs']['training_trace']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def display_writer_command(args: argparse.Namespace, delta_path: Path) -> str:
    base_arg = str(args.base or "")
    if "moe_router_delta_calibration_" in base_arg:
        base = "SMOKE_BASE_CHECKPOINT"
    else:
        base = rel(args.base) if args.base else "BASE_CHECKPOINT"
    output = rel(Path(args.output_dir) / "checkpoint_with_calibrated_router")
    return (
        "python scripts/write_same_shape_average_checkpoint.py "
        f"--base {base} --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 "
        f"--freeze-router --tensor-delta-safetensors {rel(delta_path)} --output-dir {output}"
    )


def calibrate_from_cache(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = parse_cache(repo_path(args.cache))
    tensor_names = [record["tensor"] for record in records]
    tensor_names.extend(str(record["bias_tensor"]) for record in records if record.get("bias_tensor"))
    base_values = load_base_tensors(repo_path(args.base), sorted(set(tensor_names)))
    all_deltas: dict[str, torch.Tensor] = {}
    all_metric_rows: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    for record in records:
        deltas, metric_rows, trace_rows = train_one_router(record, base_values, args)
        all_deltas.update(deltas)
        all_metric_rows.extend(metric_rows)
        all_trace_rows.extend(trace_rows)

    delta_path = output_dir / "router_delta.safetensors"
    save_file(all_deltas, str(delta_path), metadata={"format": "pt"})
    metric_df = pd.DataFrame(all_metric_rows)
    trace_df = pd.DataFrame(all_trace_rows)
    metric_path = output_dir / "router_delta_summary.csv"
    trace_path = output_dir / "training_trace.csv"
    metric_df.to_csv(metric_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    initial = metric_df[metric_df["stage"] == "initial"]
    final = metric_df[metric_df["stage"] == "final"]
    kl_improved = float(final["route_kl"].mean()) < float(initial["route_kl"].mean())
    top1_not_worse = float(final["top1_agreement"].mean()) >= float(initial["top1_agreement"].mean())
    status = "passed" if kl_improved and top1_not_worse else "no_improvement"

    def final_mean(column: str) -> float:
        return float(final[column].mean())

    def final_max(column: str) -> float:
        return float(final[column].max())

    def initial_mean(column: str) -> float:
        return float(initial[column].mean())

    def initial_max(column: str) -> float:
        return float(initial[column].max())

    def max_paired_increase(column: str) -> float:
        pairs = initial[["tensor", column]].merge(
            final[["tensor", column]],
            on="tensor",
            suffixes=("_initial", "_final"),
        )
        return float((pairs[f"{column}_final"] - pairs[f"{column}_initial"]).max())

    summary = {
        "schema_version": 2,
        "status": status,
        "kl_improved": kl_improved,
        "top1_not_worse": top1_not_worse,
        "router_count": int(metric_df["tensor"].nunique()),
        "delta_tensor_count": int(len(all_deltas)),
        "mean_initial_route_kl": float(initial["route_kl"].mean()),
        "mean_final_route_kl": float(final["route_kl"].mean()),
        "mean_initial_top1_agreement": float(initial["top1_agreement"].mean()),
        "mean_final_top1_agreement": float(final["top1_agreement"].mean()),
        "mean_final_relative_delta_norm": float(final["relative_delta_norm"].mean()),
        "max_final_relative_delta_norm": float(final["relative_delta_norm"].max()),
        "selection_policy": args.selection_policy,
        "selection_split": ",".join(sorted(str(item) for item in final["selection_split"].dropna().unique())),
        "mean_train_samples": float(final["train_samples"].mean()),
        "mean_selection_samples": float(final["selection_samples"].mean()),
        "mean_validation_fraction": float(final["validation_fraction"].mean()),
        "mean_selected_epoch": float(final["selected_epoch"].mean()),
        "min_selected_epoch": int(final["selected_epoch"].min()),
        "max_selected_epoch": int(final["selected_epoch"].max()),
        "mean_selection_score": float(final["selection_score"].mean()),
        "mean_initial_capacity_overflow_fraction": initial_mean("capacity_overflow_fraction"),
        "max_initial_capacity_overflow_fraction": initial_max("capacity_overflow_fraction"),
        "mean_final_capacity_overflow_fraction": final_mean("capacity_overflow_fraction"),
        "max_final_capacity_overflow_fraction": final_max("capacity_overflow_fraction"),
        "mean_initial_top1_capacity_overflow_fraction": initial_mean("top1_capacity_overflow_fraction"),
        "max_initial_top1_capacity_overflow_fraction": initial_max("top1_capacity_overflow_fraction"),
        "mean_final_top1_capacity_overflow_fraction": final_mean("top1_capacity_overflow_fraction"),
        "max_final_top1_capacity_overflow_fraction": final_max("top1_capacity_overflow_fraction"),
        "max_router_top1_capacity_overflow_increase": max_paired_increase("top1_capacity_overflow_fraction"),
        "mean_initial_topk_capacity_overflow_fraction": initial_mean("topk_capacity_overflow_fraction"),
        "max_initial_topk_capacity_overflow_fraction": initial_max("topk_capacity_overflow_fraction"),
        "mean_final_topk_capacity_overflow_fraction": final_mean("topk_capacity_overflow_fraction"),
        "max_final_topk_capacity_overflow_fraction": final_max("topk_capacity_overflow_fraction"),
        "max_router_topk_capacity_overflow_increase": max_paired_increase("topk_capacity_overflow_fraction"),
        "max_initial_top1_load_fraction": initial_max("top1_max_load_fraction"),
        "max_final_top1_load_fraction": final_max("top1_max_load_fraction"),
        "max_initial_topk_load_fraction": initial_max("topk_max_load_fraction"),
        "max_final_topk_load_fraction": final_max("topk_max_load_fraction"),
        "mean_initial_top1_load_entropy": initial_mean("top1_load_entropy"),
        "mean_final_top1_load_entropy": final_mean("top1_load_entropy"),
        "mean_initial_topk_load_entropy": initial_mean("topk_load_entropy"),
        "mean_final_topk_load_entropy": final_mean("topk_load_entropy"),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "capacity_factor": float(args.capacity_factor),
        "top1_loss_coef": float(args.top1_loss_coef),
        "capacity_loss_coef": float(args.capacity_loss_coef),
        "load_balance_coef": float(args.load_balance_coef),
        "trust_l2_coef": float(args.trust_l2_coef),
        "max_relative_norm": args.max_relative_norm,
        "writer_command": display_writer_command(args, delta_path),
        "outputs": {
            "router_delta_safetensors": rel(delta_path),
            "router_delta_summary": rel(metric_path),
            "training_trace": rel(trace_path),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, metric_df), encoding="utf-8")
    return summary


def build_smoke_inputs(root: Path, seed: int) -> tuple[Path, Path]:
    generator = torch.Generator().manual_seed(seed)
    hidden_dim = 6
    num_experts = 4
    samples = 384
    hidden = torch.randn(samples, hidden_dim, generator=generator)
    base_weight = 0.35 * torch.randn(num_experts, hidden_dim, generator=generator)
    target_delta = torch.zeros_like(base_weight)
    target_delta[0, 0] = 0.9
    target_delta[1, 1] = -0.8
    target_delta[2, 2] = 0.7
    target_delta[3, 3] = -0.6
    target_delta[:, 4] = torch.tensor([0.25, -0.25, 0.15, -0.15])
    teacher_weight = base_weight + target_delta
    teacher_logits = hidden @ teacher_weight.t()
    base_dir = root / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    save_file({"model.layers.0.router.weight": base_weight}, str(base_dir / "model.safetensors"), metadata={"format": "pt"})
    cache_path = root / "router_cache.pt"
    torch.save(
        {
            "routers": {
                "model.layers.0.router.weight": {
                    "hidden": hidden,
                    "teacher_logits": teacher_logits,
                }
            }
        },
        cache_path,
    )
    return base_dir, cache_path


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="moe_router_delta_calibration_") as tmp_raw:
        base_dir, cache_path = build_smoke_inputs(Path(tmp_raw), args.seed)
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.base = str(base_dir)
        smoke_args.cache = str(cache_path)
        summary = calibrate_from_cache(smoke_args)
    if summary["status"] != "passed":
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train same-shape MoE router additive deltas from hidden/router-logit caches.")
    parser.add_argument("--base", default=None, help="Base checkpoint directory or safetensors file.")
    parser.add_argument("--cache", default=None, help="Torch cache with hidden states and teacher router logits.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_router_delta_calibration_smoke"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--top1-loss-coef", type=float, default=0.25)
    parser.add_argument("--capacity-loss-coef", type=float, default=0.1)
    parser.add_argument("--load-balance-coef", type=float, default=0.0)
    parser.add_argument("--trust-l2-coef", type=float, default=0.001)
    parser.add_argument(
        "--selection-policy",
        choices=["capacity_aware", "final"],
        default="capacity_aware",
        help="Which trained router delta to write: capacity_aware selects the best epoch by mechanism score.",
    )
    parser.add_argument("--capacity-overflow-score-penalty", type=float, default=0.25)
    parser.add_argument("--capacity-increase-score-penalty", type=float, default=2.0)
    parser.add_argument("--top1-regression-score-penalty", type=float, default=1.0)
    parser.add_argument("--relative-norm-score-penalty", type=float, default=0.0)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Held-out fraction of each router cache used for epoch selection metrics. Use 0 to select on train.",
    )
    parser.add_argument("--split-seed", type=int, default=97)
    parser.add_argument(
        "--max-relative-norm",
        type=float,
        default=0.5,
        help="Project each trained router delta to this relative Frobenius norm cap. Use 0 to disable.",
    )
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true", help="Generate a synthetic router cache and verify training locally.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        summary = run_smoke(args)
    else:
        if not args.base or not args.cache:
            raise SystemExit("--base and --cache are required unless --smoke is set")
        summary = calibrate_from_cache(args)
    print(f"Wrote MoE router delta calibration to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; KL {summary['mean_initial_route_kl']:.6f} -> {summary['mean_final_route_kl']:.6f}")


if __name__ == "__main__":
    main()
