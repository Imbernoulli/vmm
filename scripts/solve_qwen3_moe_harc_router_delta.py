#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from train_moe_router_delta_calibration import (
    capacity_overflow_fraction,
    load_base_tensors,
    load_router_cap_table,
    max_relative_norm_for_tensor,
    orientation_for,
    parse_cache,
    project_relative_norm,
    route_load_stats,
    router_logits,
    topk_jaccard,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


METRIC_COLUMNS = [
    "tensor",
    "stage",
    "hidden_rows",
    "hidden_dim",
    "num_experts",
    "route_kl",
    "top1_agreement",
    "topk_jaccard",
    "capacity_overflow_fraction",
    "top1_capacity_overflow_fraction",
    "topk_capacity_overflow_fraction",
    "top1_max_load_fraction",
    "topk_max_load_fraction",
    "quadratic_proxy",
    "delta_norm",
    "base_norm",
    "relative_delta_norm",
    "max_relative_norm_cap",
    "cap_utilization",
    "cg_iterations",
    "cg_relative_residual",
    "cg_converged",
]


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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 6) -> str:
    value = maybe_float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def softmax_hessian_vec(probs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    centered = value - (probs * value).sum(dim=-1, keepdim=True)
    return probs * centered


def cg_solve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iters: int,
    tol: float,
) -> tuple[torch.Tensor, int, float, bool]:
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    rhs_norm = float(torch.linalg.vector_norm(rhs).item())
    if rhs_norm <= EPS:
        return solution, 0, 0.0, True
    residual_dot = torch.dot(residual, residual)
    target = float(tol) * rhs_norm
    last_relative = math.sqrt(float(residual_dot.item())) / max(rhs_norm, EPS)
    converged = last_relative <= tol
    iterations = 0
    for iterations in range(1, max_iters + 1):
        product = matvec(direction)
        denom = torch.dot(direction, product)
        if abs(float(denom.item())) <= EPS:
            break
        alpha = residual_dot / denom
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_residual_dot = torch.dot(residual, residual)
        residual_norm = math.sqrt(max(0.0, float(next_residual_dot.item())))
        last_relative = residual_norm / max(rhs_norm, EPS)
        if residual_norm <= target:
            converged = True
            residual_dot = next_residual_dot
            break
        beta = next_residual_dot / residual_dot.clamp_min(EPS)
        direction = residual + beta * direction
        residual_dot = next_residual_dot
    return solution, iterations, last_relative, converged


def apply_delta_logits(hidden: torch.Tensor, delta: torch.Tensor, orientation: str) -> torch.Tensor:
    if orientation == "out_in":
        return hidden @ delta.t()
    if orientation == "in_out":
        return hidden @ delta
    raise ValueError(f"Unknown router orientation: {orientation}")


def weighted_outer(hidden: torch.Tensor, expert_value: torch.Tensor, orientation: str) -> torch.Tensor:
    if orientation == "out_in":
        return expert_value.t().matmul(hidden) / max(1, hidden.shape[0])
    if orientation == "in_out":
        return hidden.t().matmul(expert_value) / max(1, hidden.shape[0])
    raise ValueError(f"Unknown router orientation: {orientation}")


def quadratic_proxy(logits: torch.Tensor, teacher_logits: torch.Tensor, probs: torch.Tensor) -> float:
    residual = logits - teacher_logits
    hv = softmax_hessian_vec(probs, residual)
    return float((0.5 * (residual * hv).sum(dim=-1)).mean().item())


def metric_row(
    *,
    tensor_name: str,
    stage: str,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    base_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    top_k: int,
    capacity_factor: float,
    max_relative_norm_cap: float | None,
    cg_iterations: int | None,
    cg_relative_residual: float | None,
    cg_converged: bool | None,
) -> dict[str, Any]:
    teacher_probs_for_kl = F.softmax(teacher_logits, dim=-1)
    route_kl = F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs_for_kl, reduction="batchmean")
    delta_norm = float(torch.linalg.vector_norm(delta_weight).item())
    base_norm = float(torch.linalg.vector_norm(base_weight.to(torch.float32)).item())
    load_stats = route_load_stats(logits, top_k=top_k, capacity_factor=capacity_factor)
    relative_delta = delta_norm / max(base_norm, EPS)
    return {
        "tensor": tensor_name,
        "stage": stage,
        "hidden_rows": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]),
        "num_experts": int(teacher_logits.shape[1]),
        "route_kl": float(route_kl.item()),
        "top1_agreement": float((logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).to(torch.float32).mean().item()),
        "topk_jaccard": topk_jaccard(logits, teacher_logits, top_k=top_k),
        "capacity_overflow_fraction": capacity_overflow_fraction(logits, capacity_factor),
        **load_stats,
        "quadratic_proxy": quadratic_proxy(logits, teacher_logits, teacher_probs),
        "delta_norm": delta_norm,
        "base_norm": base_norm,
        "relative_delta_norm": relative_delta,
        "max_relative_norm_cap": max_relative_norm_cap,
        "cap_utilization": None
        if max_relative_norm_cap is None or max_relative_norm_cap <= 0
        else relative_delta / max_relative_norm_cap,
        "cg_iterations": cg_iterations,
        "cg_relative_residual": cg_relative_residual,
        "cg_converged": cg_converged,
    }


def solve_one_router(
    record: dict[str, Any],
    base_values: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    tensor_name = str(record["tensor"])
    hidden = record["hidden"].to(torch.float32)
    teacher_logits = record["teacher_logits"].to(torch.float32)
    base_weight = base_values[tensor_name].to(torch.float32)
    orientation = orientation_for(base_weight, hidden.shape[-1], teacher_logits.shape[-1], tensor_name)
    bias_tensor = record.get("bias_tensor")
    base_bias = None
    if bias_tensor:
        base_bias = base_values[str(bias_tensor)].to(torch.float32)
    base_logits = router_logits(hidden, base_weight, orientation, base_bias)
    teacher_probs = F.softmax(teacher_logits / args.temperature, dim=-1)
    residual_target = teacher_logits - base_logits
    max_relative_norm_cap = max_relative_norm_for_tensor(args, tensor_name)

    def matvec(flat_delta: torch.Tensor) -> torch.Tensor:
        delta = flat_delta.reshape(base_weight.shape)
        logits_delta = apply_delta_logits(hidden, delta, orientation)
        h_delta = softmax_hessian_vec(teacher_probs, logits_delta)
        gradient = weighted_outer(hidden, h_delta, orientation)
        return (gradient + args.ridge * delta).reshape(-1)

    rhs_tensor = weighted_outer(hidden, softmax_hessian_vec(teacher_probs, residual_target), orientation)
    rhs = rhs_tensor.reshape(-1)
    solution, cg_iterations, cg_relative_residual, cg_converged = cg_solve(
        matvec,
        rhs,
        max_iters=args.cg_max_iters,
        tol=args.cg_tol,
    )
    delta_weight = solution.reshape(base_weight.shape).to(torch.float32)
    project_relative_norm(delta_weight, base_weight, max_relative_norm_cap)
    initial_logits = base_logits
    final_logits = router_logits(hidden, base_weight + delta_weight, orientation, base_bias)
    metrics = [
        metric_row(
            tensor_name=tensor_name,
            stage="initial",
            hidden=hidden,
            logits=initial_logits,
            teacher_logits=teacher_logits,
            teacher_probs=teacher_probs,
            base_weight=base_weight,
            delta_weight=torch.zeros_like(base_weight),
            top_k=args.top_k,
            capacity_factor=args.capacity_factor,
            max_relative_norm_cap=max_relative_norm_cap,
            cg_iterations=None,
            cg_relative_residual=None,
            cg_converged=None,
        ),
        metric_row(
            tensor_name=tensor_name,
            stage="final",
            hidden=hidden,
            logits=final_logits,
            teacher_logits=teacher_logits,
            teacher_probs=teacher_probs,
            base_weight=base_weight,
            delta_weight=delta_weight,
            top_k=args.top_k,
            capacity_factor=args.capacity_factor,
            max_relative_norm_cap=max_relative_norm_cap,
            cg_iterations=cg_iterations,
            cg_relative_residual=cg_relative_residual,
            cg_converged=cg_converged,
        ),
    ]
    solver_row = {
        "tensor": tensor_name,
        "cg_iterations": cg_iterations,
        "cg_relative_residual": cg_relative_residual,
        "cg_converged": cg_converged,
        "ridge": float(args.ridge),
        "rhs_norm": float(torch.linalg.vector_norm(rhs).item()),
        "cap_projected": bool(
            max_relative_norm_cap is not None
            and max_relative_norm_cap > 0
            and metrics[-1]["cap_utilization"] is not None
            and float(metrics[-1]["cap_utilization"]) >= 0.999
        ),
    }
    return {tensor_name: delta_weight.detach().cpu()}, metrics, solver_row


def output_paths(output_dir: Path) -> dict[str, str | None]:
    return {
        "router_delta_safetensors": rel(output_dir / "router_delta.safetensors"),
        "router_delta_summary": rel(output_dir / "router_delta_summary.csv"),
        "solver_trace": rel(output_dir / "solver_trace.csv"),
        "requirements": rel(output_dir / "harc_solver_requirements.csv"),
        "summary": rel(output_dir / "summary.json"),
        "report": rel(output_dir / "report.md"),
    }


def display_input_path(value: Any, smoke_label: str) -> str:
    text = str(value)
    if "harc_router_solver_" in text:
        return smoke_label
    return text


def build_requirements(summary: dict[str, Any]) -> pd.DataFrame:
    cache_path = display_input_path(summary.get("cache_path"), "SMOKE_ROUTER_CACHE")
    base_path = display_input_path(summary.get("base_path"), "SMOKE_BASE_CHECKPOINT")
    rows = [
        {
            "requirement": "router_cache_exists",
            "passed": bool(summary.get("cache_exists", False)),
            "evidence": f"cache={cache_path}; exists={summary.get('cache_exists')}",
            "next_action": "collect router calibration cache before solving HARC deltas",
        },
        {
            "requirement": "base_checkpoint_exists",
            "passed": bool(summary.get("base_exists", False)),
            "evidence": f"base={base_path}; exists={summary.get('base_exists')}",
            "next_action": "point --base to the same-shape checkpoint whose routers will receive the delta",
        },
        {
            "requirement": "router_tensors_loaded",
            "passed": int(summary.get("router_count") or 0) > 0 and not summary.get("load_error"),
            "evidence": f"routers={summary.get('router_count')}; load_error={summary.get('load_error')}",
            "next_action": "verify cache tensor names match the base checkpoint safetensors index",
        },
        {
            "requirement": "cg_converged",
            "passed": bool(summary.get("all_cg_converged", False)),
            "evidence": f"max relative residual={fmt(summary.get('max_cg_relative_residual'))}",
            "next_action": "increase --cg-max-iters or --ridge if residual is high",
        },
        {
            "requirement": "kl_improved",
            "passed": bool(summary.get("kl_improved", False)),
            "evidence": f"KL {fmt(summary.get('mean_initial_route_kl'))} -> {fmt(summary.get('mean_final_route_kl'))}",
            "next_action": "check cache/source mismatch if HARC does not improve route KL",
        },
        {
            "requirement": "relative_norm_cap_respected",
            "passed": bool(summary.get("relative_norm_cap_respected", False)),
            "evidence": f"max cap utilization={fmt(summary.get('max_cap_utilization'))}",
            "next_action": "lower --max-relative-norm or use a router cap table if needed",
        },
    ]
    return pd.DataFrame(rows)


def missing_summary(args: argparse.Namespace, status: str, load_error: str | None = None) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = repo_path(args.cache)
    base_path = repo_path(args.base)
    summary = {
        "schema_version": 1,
        "status": status,
        "cache_path": rel(cache_path),
        "cache_exists": cache_path.exists(),
        "base_path": rel(base_path),
        "base_exists": base_path.exists(),
        "load_error": load_error,
        "router_count": 0,
        "delta_tensor_count": 0,
        "mean_initial_route_kl": None,
        "mean_final_route_kl": None,
        "kl_improved": False,
        "all_cg_converged": False,
        "relative_norm_cap_respected": False,
        "recommended_action": "collect_cache_or_fix_base_before_harc_solver",
        "outputs": output_paths(output_dir),
    }
    metrics = pd.DataFrame(columns=METRIC_COLUMNS)
    trace = pd.DataFrame(columns=["tensor", "cg_iterations", "cg_relative_residual", "cg_converged", "ridge", "rhs_norm", "cap_projected"])
    requirements = build_requirements(summary)
    metrics.to_csv(output_dir / "router_delta_summary.csv", index=False)
    trace.to_csv(output_dir / "solver_trace.csv", index=False)
    requirements.to_csv(output_dir / "harc_solver_requirements.csv", index=False)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(build_report(summary, metrics, requirements), encoding="utf-8")
    return summary


def display_writer_command(args: argparse.Namespace, delta_path: Path) -> str:
    base_arg = str(args.base or "")
    base = "SMOKE_BASE_CHECKPOINT" if "harc_router_solver_" in base_arg else rel(args.base)
    output = rel(Path(args.output_dir) / "checkpoint_with_harc_router")
    return (
        "python scripts/write_same_shape_average_checkpoint.py "
        f"--base {base} --source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 "
        f"--freeze-router --tensor-delta-safetensors {rel(delta_path)} --output-dir {output}"
    )


def build_report(summary: dict[str, Any], metrics: pd.DataFrame, requirements: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE HARC Router Delta Solver",
        "",
        "This script solves a training-free second-order router calibration problem from a router hidden/logit cache. It uses the softmax Hessian as the local KL curvature and a matrix-free conjugate-gradient solve, then writes same-shape router delta tensors.",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Routers: `{summary.get('router_count')}`",
        f"- Delta tensors: `{summary.get('delta_tensor_count')}`",
        f"- Mean route KL: `{fmt(summary.get('mean_initial_route_kl'))}` -> `{fmt(summary.get('mean_final_route_kl'))}`",
        f"- Mean top-1 agreement: `{fmt(summary.get('mean_initial_top1_agreement'))}` -> `{fmt(summary.get('mean_final_top1_agreement'))}`",
        f"- Max CG relative residual: `{fmt(summary.get('max_cg_relative_residual'))}`",
        f"- Max relative delta norm: `{fmt(summary.get('max_final_relative_delta_norm'))}`",
        f"- Max cap utilization: `{fmt(summary.get('max_cap_utilization'))}`",
        f"- Recommended action: `{summary.get('recommended_action')}`",
        "",
        "## Requirements",
        "",
        "| requirement | passed | evidence | next action |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in requirements.iterrows():
        lines.append(f"| `{row['requirement']}` | `{bool(row['passed'])}` | {row['evidence']} | {row['next_action']} |")
    lines.extend(
        [
            "",
            "## Router Metrics",
            "",
            "| tensor | KL initial-final | top1 initial-final | rel delta | CG residual | cap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not metrics.empty:
        for tensor, group in metrics.groupby("tensor", sort=True):
            initial = group[group["stage"] == "initial"].iloc[0]
            final = group[group["stage"] == "final"].iloc[0]
            lines.append(
                f"| `{tensor}` | {fmt(initial['route_kl'])}-{fmt(final['route_kl'])} | "
                f"{fmt(initial['top1_agreement'], 4)}-{fmt(final['top1_agreement'], 4)} | "
                f"{fmt(final['relative_delta_norm'], 4)} | {fmt(final.get('cg_relative_residual'))} | "
                f"{fmt(final.get('max_relative_norm_cap'), 4)} |"
            )
    lines.extend(
        [
            "",
            "## Writer",
            "",
            "```bash",
            summary.get("writer_command") or "HARC solver has not written router deltas yet.",
            "```",
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['router_delta_safetensors']}`",
            f"- `{summary['outputs']['router_delta_summary']}`",
            f"- `{summary['outputs']['solver_trace']}`",
            f"- `{summary['outputs']['requirements']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def solve_from_cache(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = repo_path(args.cache)
    base_path = repo_path(args.base)
    if not cache_path.exists():
        return missing_summary(args, "harc_solver_missing_cache")
    if not base_path.exists():
        return missing_summary(args, "harc_solver_missing_base")

    try:
        args.router_cap_by_tensor = load_router_cap_table(args.router_cap_table, args.router_cap_column)
        records = parse_cache(cache_path)
        selected_records = records if args.max_routers is None else records[: args.max_routers]
        tensor_names = [str(record["tensor"]) for record in selected_records]
        tensor_names.extend(str(record["bias_tensor"]) for record in selected_records if record.get("bias_tensor"))
        base_values = load_base_tensors(base_path, sorted(set(tensor_names)))
    except Exception as exc:
        return missing_summary(args, "harc_solver_input_error", load_error=str(exc))

    all_deltas: dict[str, torch.Tensor] = {}
    metric_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    for record in selected_records:
        deltas, metrics, solver_row = solve_one_router(record, base_values, args)
        all_deltas.update(deltas)
        metric_rows.extend(metrics)
        solver_rows.append(solver_row)

    delta_path = output_dir / "router_delta.safetensors"
    save_file(all_deltas, str(delta_path), metadata={"format": "pt"})
    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    trace = pd.DataFrame(solver_rows)
    metrics.to_csv(output_dir / "router_delta_summary.csv", index=False)
    trace.to_csv(output_dir / "solver_trace.csv", index=False)
    initial = metrics[metrics["stage"] == "initial"]
    final = metrics[metrics["stage"] == "final"]
    cap_utilization = pd.to_numeric(final["cap_utilization"], errors="coerce").dropna()
    relative_norm_cap_respected = cap_utilization.empty or bool((cap_utilization <= 1.0001).all())
    kl_improved = float(final["route_kl"].mean()) < float(initial["route_kl"].mean())
    top1_not_worse = float(final["top1_agreement"].mean()) >= float(initial["top1_agreement"].mean())
    all_cg_converged = bool(trace["cg_converged"].astype(bool).all()) if not trace.empty else False
    status = "harc_solver_ready" if kl_improved and relative_norm_cap_respected else "harc_solver_no_improvement"
    summary = {
        "schema_version": 1,
        "status": status,
        "cache_path": rel(cache_path),
        "cache_exists": True,
        "base_path": rel(base_path),
        "base_exists": True,
        "load_error": None,
        "router_count": int(final["tensor"].nunique()),
        "delta_tensor_count": int(len(all_deltas)),
        "kl_improved": kl_improved,
        "top1_not_worse": top1_not_worse,
        "all_cg_converged": all_cg_converged,
        "relative_norm_cap_respected": relative_norm_cap_respected,
        "mean_initial_route_kl": float(initial["route_kl"].mean()),
        "mean_final_route_kl": float(final["route_kl"].mean()),
        "mean_initial_top1_agreement": float(initial["top1_agreement"].mean()),
        "mean_final_top1_agreement": float(final["top1_agreement"].mean()),
        "mean_initial_quadratic_proxy": float(initial["quadratic_proxy"].mean()),
        "mean_final_quadratic_proxy": float(final["quadratic_proxy"].mean()),
        "mean_final_relative_delta_norm": float(final["relative_delta_norm"].mean()),
        "max_final_relative_delta_norm": float(final["relative_delta_norm"].max()),
        "mean_cg_iterations": float(trace["cg_iterations"].mean()) if not trace.empty else None,
        "max_cg_iterations": int(trace["cg_iterations"].max()) if not trace.empty else None,
        "max_cg_relative_residual": float(trace["cg_relative_residual"].max()) if not trace.empty else None,
        "ridge": float(args.ridge),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "capacity_factor": float(args.capacity_factor),
        "max_relative_norm": args.max_relative_norm,
        "router_cap_mode": "per_router_table" if args.router_cap_by_tensor else "global",
        "max_cap_utilization": float(cap_utilization.max()) if not cap_utilization.empty else None,
        "writer_command": display_writer_command(args, delta_path),
        "recommended_action": "materialize_harc_router_delta_then_run_matched_vllm_gate",
        "outputs": output_paths(output_dir),
    }
    requirements = build_requirements(summary)
    requirements.to_csv(output_dir / "harc_solver_requirements.csv", index=False)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(build_report(summary, metrics, requirements), encoding="utf-8")
    return summary


def build_smoke_inputs(root: Path, seed: int) -> tuple[Path, Path]:
    generator = torch.Generator().manual_seed(seed)
    hidden_dim = 6
    num_experts = 12
    samples = 256
    tensors: dict[str, torch.Tensor] = {}
    routers: dict[str, dict[str, torch.Tensor]] = {}
    for layer_idx, scale in [(0, 0.10), (1, 0.14)]:
        hidden = torch.randn(samples, hidden_dim, generator=generator)
        base_weight = 0.25 * torch.randn(num_experts, hidden_dim, generator=generator)
        target_delta = scale * torch.randn(num_experts, hidden_dim, generator=generator)
        target_delta[layer_idx, layer_idx] += 0.18
        teacher_weight = base_weight + target_delta
        tensor_name = f"model.layers.{layer_idx}.mlp.gate.weight"
        tensors[tensor_name] = base_weight
        routers[tensor_name] = {
            "hidden": hidden,
            "teacher_logits": hidden @ teacher_weight.t(),
            "sample_groups": torch.arange(samples, dtype=torch.long) // 32,
        }
    base_dir = root / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(base_dir / "model.safetensors"), metadata={"format": "pt"})
    cache_path = root / "router_cache.torch"
    torch.save({"routers": routers, "metadata": {"source": "synthetic_harc_solver_smoke"}}, cache_path)
    return base_dir, cache_path


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="harc_router_solver_") as tmp_raw:
        smoke_args = argparse.Namespace(**vars(args))
        base_dir, cache_path = build_smoke_inputs(Path(tmp_raw), args.seed)
        smoke_args.base = base_dir
        smoke_args.cache = cache_path
        smoke_args.max_relative_norm = 0.50
        smoke_args.router_cap_by_tensor = {}
        summary = solve_from_cache(smoke_args)
    metrics = pd.read_csv(repo_path(summary["outputs"]["router_delta_summary"]))
    initial = metrics[metrics["stage"] == "initial"]
    final = metrics[metrics["stage"] == "final"]
    checks = [
        {
            "check": "status_ready",
            "passed": summary["status"] == "harc_solver_ready",
            "evidence": summary["status"],
        },
        {
            "check": "two_router_deltas",
            "passed": int(summary["delta_tensor_count"]) == 2,
            "evidence": str(summary["delta_tensor_count"]),
        },
        {
            "check": "route_kl_improved",
            "passed": float(final["route_kl"].mean()) < float(initial["route_kl"].mean()),
            "evidence": f"{float(initial['route_kl'].mean()):.6f}->{float(final['route_kl'].mean()):.6f}",
        },
        {
            "check": "quadratic_proxy_improved",
            "passed": float(final["quadratic_proxy"].mean()) < float(initial["quadratic_proxy"].mean()),
            "evidence": f"{float(initial['quadratic_proxy'].mean()):.6f}->{float(final['quadratic_proxy'].mean()):.6f}",
        },
        {
            "check": "cap_respected",
            "passed": bool(summary["relative_norm_cap_respected"]),
            "evidence": fmt(summary.get("max_cap_utilization")),
        },
        {
            "check": "cg_converged",
            "passed": bool(summary["all_cg_converged"]),
            "evidence": fmt(summary.get("max_cg_relative_residual")),
        },
    ]
    passed = all(bool(row["passed"]) for row in checks)
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(repo_path(args.output_dir) / "smoke_checks.csv", index=False)
    summary["smoke_status"] = "smoke_passed" if passed else "smoke_failed"
    summary["smoke_checks"] = {"passed": int(sum(bool(row["passed"]) for row in checks)), "total": int(len(checks))}
    summary["outputs"]["smoke_checks"] = rel(repo_path(args.output_dir) / "smoke_checks.csv")
    write_json(repo_path(args.output_dir) / "summary.json", summary)
    if not passed:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a HARC-style same-shape MoE router delta from hidden/logit cache.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("results/checkpoints/qwen3_moe_unified_mechanism_candidate"),
        help="Same-shape checkpoint whose router tensors receive the HARC delta.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_cache/router_calibration_cache.pt"),
        help="Torch cache with {'routers': {tensor: {'hidden', 'teacher_logits'}}}.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_harc_router_solver"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--cg-max-iters", type=int, default=120)
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--max-relative-norm", type=float, default=0.01)
    parser.add_argument("--router-cap-table", type=Path, default=None)
    parser.add_argument("--router-cap-column", default="max_relative_norm_cap")
    parser.add_argument("--max-routers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(args) if args.smoke_matrix else solve_from_cache(args)
    print(f"Wrote Qwen3 MoE HARC router solver to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; routers={summary['router_count']}; "
        f"KL={fmt(summary.get('mean_initial_route_kl'))}->{fmt(summary.get('mean_final_route_kl'))}"
    )


if __name__ == "__main__":
    main()
