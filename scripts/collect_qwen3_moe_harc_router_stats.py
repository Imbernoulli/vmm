#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


LITERATURE_SOURCES = [
    {
        "key": "harc",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Match source router distributions through a second-order KL proxy with softmax Hessian and hidden-state covariance.",
    },
    {
        "key": "regmean",
        "title": "RegMean: A Simple and Effective Method for Model Merging",
        "url": "https://arxiv.org/abs/2212.09849",
        "mechanism": "Activation covariance turns weight averaging into a layer-local least-squares merge; HARC applies the same statistics-aware logic to MoE routers.",
    },
    {
        "key": "model_connectivity",
        "title": "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs",
        "url": "https://arxiv.org/abs/1802.10026",
        "mechanism": "Average candidates need connectivity and barrier checks; low router margin means linear interpolation can cross top-k assignment boundaries.",
    },
]


CSV_COLUMNS = [
    "tensor",
    "layer",
    "hidden_rows",
    "hidden_dim",
    "num_experts",
    "top_k",
    "sample_group_count",
    "mean_router_entropy",
    "mean_hessian_trace",
    "mean_hessian_diag",
    "max_hessian_diag",
    "mean_hessian_offdiag_abs",
    "mean_topk_probability_mass",
    "mean_topk_hessian_diag_fraction",
    "mean_boundary_pair_hessian",
    "mean_topk_logit_margin",
    "min_topk_logit_margin",
    "hidden_cov_trace",
    "hidden_cov_diag_mean",
    "hidden_cov_diag_max",
    "hidden_cov_frobenius",
    "hidden_cov_stable_rank",
    "hidden_cov_top1_energy",
    "hidden_cov_top4_energy",
    "hidden_cov_eig_computed",
    "student_teacher_top1_agreement",
    "student_teacher_topk_jaccard",
    "student_teacher_kl",
    "harc_precision_proxy",
    "harc_boundary_cov_proxy",
    "harc_priority_score",
    "harc_layer_role",
    "first_stage_required",
    "solver_input_ready",
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


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    value = clean_value(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fnum(value: Any, default: float | None = None) -> float | None:
    value = clean_value(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def parse_layer_id(tensor_name: str) -> int | None:
    for pattern in (r"(?:^|\.)(?:layers|blocks)\.(\d+)\.", r"(?:^|_)layer[_\.-]?(\d+)(?:\.|_|$)"):
        match = re.search(pattern, tensor_name)
        if match:
            return int(match.group(1))
    return None


def load_priority(path: Path) -> pd.DataFrame:
    priority = read_csv(path)
    if priority.empty or "layer" not in priority:
        return pd.DataFrame()
    priority = priority.copy()
    priority["layer"] = pd.to_numeric(priority["layer"], errors="coerce").astype("Int64")
    priority = priority.dropna(subset=["layer"])
    priority["layer"] = priority["layer"].astype(int)
    if "harc_layer_role" not in priority:
        priority["harc_layer_role"] = "track_in_margin_profile"
    if "harc_priority_score" not in priority:
        priority["harc_priority_score"] = 0.0
    return priority


def load_cache_records(cache_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or "routers" not in payload:
        raise ValueError("HARC stats cache must be a torch payload containing {'routers': ...}.")
    routers = payload["routers"]
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if isinstance(routers, dict):
        items = list(routers.items())
    elif isinstance(routers, list):
        items = []
        for item in routers:
            if not isinstance(item, dict) or "tensor" not in item:
                raise ValueError("Each router list record must be a dict containing a 'tensor' field.")
            items.append((str(item["tensor"]), item))
    else:
        raise ValueError("The 'routers' cache entry must be a dict or list.")

    records = []
    for tensor_name, record in items:
        if not isinstance(record, dict):
            continue
        hidden = record.get("hidden")
        teacher_logits = record.get("teacher_logits")
        if not torch.is_tensor(hidden) or not torch.is_tensor(teacher_logits):
            continue
        if hidden.ndim != 2 or teacher_logits.ndim != 2 or hidden.shape[0] != teacher_logits.shape[0]:
            continue
        sample_groups = record.get("sample_groups")
        if sample_groups is not None and not torch.is_tensor(sample_groups):
            sample_groups = torch.as_tensor(sample_groups)
        candidate_logits = record.get("student_logits")
        if candidate_logits is None:
            candidate_logits = record.get("candidate_logits")
        if candidate_logits is not None and not torch.is_tensor(candidate_logits):
            candidate_logits = None
        if candidate_logits is not None:
            if candidate_logits.ndim != 2 or candidate_logits.shape != teacher_logits.shape:
                candidate_logits = None
        records.append(
            {
                "tensor": str(tensor_name),
                "hidden": hidden.to(torch.float32).cpu(),
                "teacher_logits": teacher_logits.to(torch.float32).cpu(),
                "candidate_logits": None if candidate_logits is None else candidate_logits.to(torch.float32).cpu(),
                "sample_groups": None if sample_groups is None else sample_groups.to(torch.long).cpu(),
            }
        )
    return records, metadata


def mean_or_none(tensor: torch.Tensor) -> float | None:
    if tensor.numel() == 0:
        return None
    return float(tensor.to(torch.float64).mean().item())


def topk_jaccard(left_logits: torch.Tensor, right_logits: torch.Tensor, top_k: int) -> float | None:
    if left_logits.numel() == 0 or right_logits.numel() == 0:
        return None
    k = min(top_k, left_logits.shape[-1], right_logits.shape[-1])
    left = torch.topk(left_logits, k=k, dim=-1).indices
    right = torch.topk(right_logits, k=k, dim=-1).indices
    scores = []
    for row_idx in range(left.shape[0]):
        left_set = set(int(item) for item in left[row_idx].tolist())
        right_set = set(int(item) for item in right[row_idx].tolist())
        scores.append(len(left_set & right_set) / max(1, len(left_set | right_set)))
    return float(sum(scores) / max(1, len(scores)))


def logit_margin(logits: torch.Tensor, top_k: int) -> tuple[float | None, float | None]:
    experts = logits.shape[-1]
    if experts <= 1 or top_k >= experts:
        return None, None
    sorted_logits = torch.sort(logits, descending=True, dim=-1).values
    margin = sorted_logits[:, top_k - 1] - sorted_logits[:, top_k]
    return float(margin.mean().item()), float(margin.min().item())


def softmax_hessian_stats(logits: torch.Tensor, top_k: int) -> dict[str, float | None]:
    probs = F.softmax(logits, dim=-1)
    experts = probs.shape[-1]
    diag = probs * (1.0 - probs)
    trace = diag.sum(dim=-1)
    entropy = -(probs.clamp_min(EPS) * probs.clamp_min(EPS).log()).sum(dim=-1)
    sorted_probs = torch.sort(probs, descending=True, dim=-1).values
    k = min(top_k, experts)
    topk_mass = sorted_probs[:, :k].sum(dim=-1)
    topk_diag_mass = (sorted_probs[:, :k] * (1.0 - sorted_probs[:, :k])).sum(dim=-1)
    if experts > 1:
        offdiag_abs = trace / (experts * (experts - 1))
    else:
        offdiag_abs = torch.zeros_like(trace)
    if k < experts:
        boundary_pair = sorted_probs[:, k - 1] * sorted_probs[:, k]
    else:
        boundary_pair = torch.zeros_like(trace)
    topk_diag_fraction = topk_diag_mass / trace.clamp_min(EPS)
    mean_margin, min_margin = logit_margin(logits, top_k)
    return {
        "mean_router_entropy": float(entropy.mean().item()),
        "mean_hessian_trace": float(trace.mean().item()),
        "mean_hessian_diag": float(diag.mean().item()),
        "max_hessian_diag": float(diag.max().item()),
        "mean_hessian_offdiag_abs": float(offdiag_abs.mean().item()),
        "mean_topk_probability_mass": float(topk_mass.mean().item()),
        "mean_topk_hessian_diag_fraction": float(topk_diag_fraction.mean().item()),
        "mean_boundary_pair_hessian": float(boundary_pair.mean().item()),
        "mean_topk_logit_margin": mean_margin,
        "min_topk_logit_margin": min_margin,
    }


def covariance_stats(hidden: torch.Tensor, max_cov_dim_for_eig: int) -> dict[str, float | bool | None]:
    rows, dim = hidden.shape
    if rows <= 1 or dim == 0:
        return {
            "hidden_cov_trace": 0.0,
            "hidden_cov_diag_mean": 0.0,
            "hidden_cov_diag_max": 0.0,
            "hidden_cov_frobenius": None,
            "hidden_cov_stable_rank": None,
            "hidden_cov_top1_energy": None,
            "hidden_cov_top4_energy": None,
            "hidden_cov_eig_computed": False,
        }
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    diag = centered.square().sum(dim=0) / max(1, rows - 1)
    trace = float(diag.sum().item())
    cov = centered.t().matmul(centered) / max(1, rows - 1)
    frob = float(cov.square().sum().sqrt().item())
    stable_rank = None if frob <= EPS else float((trace * trace) / (frob * frob + EPS))
    eig_computed = dim <= max_cov_dim_for_eig
    top1_energy = None
    top4_energy = None
    if eig_computed:
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0).sort(descending=True).values
        total = float(eigvals.sum().item())
        if total > EPS:
            top1_energy = float(eigvals[:1].sum().item() / total)
            top4_energy = float(eigvals[: min(4, eigvals.numel())].sum().item() / total)
    return {
        "hidden_cov_trace": trace,
        "hidden_cov_diag_mean": float(diag.mean().item()),
        "hidden_cov_diag_max": float(diag.max().item()),
        "hidden_cov_frobenius": frob,
        "hidden_cov_stable_rank": stable_rank,
        "hidden_cov_top1_energy": top1_energy,
        "hidden_cov_top4_energy": top4_energy,
        "hidden_cov_eig_computed": eig_computed,
    }


def candidate_alignment_stats(candidate_logits: torch.Tensor | None, teacher_logits: torch.Tensor, top_k: int) -> dict[str, float | None]:
    if candidate_logits is None:
        return {
            "student_teacher_top1_agreement": None,
            "student_teacher_topk_jaccard": None,
            "student_teacher_kl": None,
        }
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    kl = F.kl_div(F.log_softmax(candidate_logits, dim=-1), teacher_probs, reduction="batchmean")
    return {
        "student_teacher_top1_agreement": float(
            (candidate_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).to(torch.float32).mean().item()
        ),
        "student_teacher_topk_jaccard": topk_jaccard(candidate_logits, teacher_logits, top_k),
        "student_teacher_kl": float(kl.item()),
    }


def summarize_router(
    record: dict[str, Any],
    priority_by_layer: dict[int, dict[str, Any]],
    *,
    top_k: int,
    max_cov_dim_for_eig: int,
    min_rows: int,
) -> dict[str, Any]:
    tensor = str(record["tensor"])
    hidden = record["hidden"]
    logits = record["teacher_logits"]
    groups = record.get("sample_groups")
    layer = parse_layer_id(tensor)
    priority = priority_by_layer.get(layer or -1, {})
    hessian = softmax_hessian_stats(logits, top_k)
    cov = covariance_stats(hidden, max_cov_dim_for_eig)
    align = candidate_alignment_stats(record.get("candidate_logits"), logits, top_k)
    cov_trace = fnum(cov.get("hidden_cov_trace"), 0.0) or 0.0
    hessian_trace = fnum(hessian.get("mean_hessian_trace"), 0.0) or 0.0
    boundary_hessian = fnum(hessian.get("mean_boundary_pair_hessian"), 0.0) or 0.0
    role = priority.get("harc_layer_role", "not_in_priority_table")
    first_stage = role in {"critical_topk_boundary_layer", "collect_hessian_covariance_first"}
    solver_ready = bool(hidden.shape[0] >= min_rows and cov_trace > EPS and hessian_trace > EPS)
    return {
        "tensor": tensor,
        "layer": layer,
        "hidden_rows": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[1]),
        "num_experts": int(logits.shape[1]),
        "top_k": int(min(top_k, logits.shape[1])),
        "sample_group_count": int(groups.unique().numel()) if torch.is_tensor(groups) and groups.numel() else 0,
        **hessian,
        **cov,
        **align,
        "harc_precision_proxy": float(hessian_trace * cov_trace),
        "harc_boundary_cov_proxy": float(boundary_hessian * cov_trace),
        "harc_priority_score": fnum(priority.get("harc_priority_score"), 0.0),
        "harc_layer_role": role,
        "first_stage_required": first_stage,
        "solver_input_ready": solver_ready,
    }


def first_stage_coverage(stats: pd.DataFrame, priority: pd.DataFrame) -> dict[str, Any]:
    if priority.empty:
        return {
            "first_stage_required_layer_count": 0,
            "first_stage_covered_layer_count": 0,
            "first_stage_missing_layers": [],
            "first_stage_coverage_status": "priority_table_missing",
        }
    first = priority[
        priority["harc_layer_role"].isin(["critical_topk_boundary_layer", "collect_hessian_covariance_first"])
    ]
    required_layers = sorted(int(layer) for layer in first["layer"].tolist())
    covered_layers = sorted(
        int(layer)
        for layer in stats.loc[stats["solver_input_ready"].astype(bool), "layer"].dropna().astype(int).unique().tolist()
    ) if not stats.empty else []
    missing = [layer for layer in required_layers if layer not in set(covered_layers)]
    if not required_layers:
        status = "no_first_stage_layers_required"
    elif not missing:
        status = "complete"
    elif covered_layers:
        status = "partial"
    else:
        status = "missing"
    return {
        "first_stage_required_layer_count": len(required_layers),
        "first_stage_covered_layer_count": len([layer for layer in required_layers if layer in set(covered_layers)]),
        "first_stage_missing_layers": missing,
        "first_stage_coverage_status": status,
    }


def build_requirements(summary: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "requirement": "router_calibration_cache_exists",
            "passed": bool(summary["cache_exists"]),
            "evidence": f"cache={summary['cache_path']}; exists={summary['cache_exists']}",
            "next_action": "run router calibration cache collection on the GPU host",
        },
        {
            "requirement": "router_records_present",
            "passed": int(summary["router_count"]) > 0,
            "evidence": f"routers={summary['router_count']}",
            "next_action": "verify router hooks matched model.layers.*.mlp.gate/router modules",
        },
        {
            "requirement": "hidden_logits_shape_valid",
            "passed": int(summary["valid_router_count"]) > 0,
            "evidence": f"valid routers={summary['valid_router_count']}; total rows={summary['total_hidden_rows']}",
            "next_action": "collect 2D hidden states and matching teacher logits for each router",
        },
        {
            "requirement": "softmax_hessian_positive",
            "passed": (fnum(summary.get("mean_hessian_trace"), 0.0) or 0.0) > 0.0,
            "evidence": f"mean Hessian trace={fmt(summary.get('mean_hessian_trace'))}",
            "next_action": "check teacher logits are not all degenerate one-hot or empty",
        },
        {
            "requirement": "hidden_covariance_positive",
            "passed": (fnum(summary.get("mean_hidden_cov_trace"), 0.0) or 0.0) > 0.0,
            "evidence": f"mean covariance trace={fmt(summary.get('mean_hidden_cov_trace'))}",
            "next_action": "increase prompt/token coverage if hidden covariance is zero",
        },
        {
            "requirement": "first_stage_priority_covered",
            "passed": summary.get("first_stage_coverage_status") in {
                "complete",
                "no_first_stage_layers_required",
                "priority_table_missing",
            },
            "evidence": (
                f"coverage={summary.get('first_stage_coverage_status')}; "
                f"{summary.get('first_stage_covered_layer_count')}/{summary.get('first_stage_required_layer_count')} layers"
            ),
            "next_action": "collect all high-priority HARC layers before solving the matrix-free system",
        },
    ]
    return pd.DataFrame(rows)


def decide_status(summary: dict[str, Any], requirements: pd.DataFrame) -> str:
    if not summary["cache_exists"]:
        return "harc_router_stats_missing_cache"
    if int(summary["router_count"]) == 0:
        return "harc_router_stats_no_router_records"
    if int(summary["valid_router_count"]) == 0:
        return "harc_router_stats_invalid_cache"
    if summary.get("first_stage_coverage_status") == "partial":
        return "harc_router_stats_partial_first_stage"
    if bool(requirements["passed"].all()):
        return "harc_router_stats_ready"
    return "harc_router_stats_needs_more_coverage"


def build_report(summary: dict[str, Any], stats: pd.DataFrame, requirements: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE HARC Router Stats",
        "",
        "This collector converts router hidden states and teacher logits into the statistics needed by the HARC-style router objective: softmax Hessian summaries, hidden covariance summaries, top-k boundary margin, and a per-router precision proxy.",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cache: `{summary['cache_path']}`",
        f"- Routers: `{summary['valid_router_count']}/{summary['router_count']}` valid",
        f"- First-stage coverage: `{summary['first_stage_covered_layer_count']}/{summary['first_stage_required_layer_count']}` (`{summary['first_stage_coverage_status']}`)",
        f"- Mean Hessian trace: `{fmt(summary.get('mean_hessian_trace'))}`",
        f"- Mean hidden covariance trace: `{fmt(summary.get('mean_hidden_cov_trace'))}`",
        f"- Top router by precision proxy: `{summary.get('top_precision_proxy_tensor')}`",
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
            "## Top Router Stats",
            "",
            "| layer | tensor | role | rows | experts | Hessian trace | cov trace | boundary proxy | precision proxy |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not stats.empty:
        top = stats.sort_values(["harc_precision_proxy", "harc_boundary_cov_proxy"], ascending=[False, False]).head(12)
        for _, row in top.iterrows():
            lines.append(
                f"| {'' if pd.isna(row.get('layer')) else int(row['layer'])} | `{row['tensor']}` | "
                f"`{row['harc_layer_role']}` | {int(row['hidden_rows'])} | {int(row['num_experts'])} | "
                f"{fmt(row['mean_hessian_trace'])} | {fmt(row['hidden_cov_trace'])} | "
                f"{fmt(row['harc_boundary_cov_proxy'])} | {fmt(row['harc_precision_proxy'])} |"
            )
    lines.extend(
        [
            "",
            "## Solver Input",
            "",
            f"- Objective: `{summary['solver_inputs']['objective']}`",
            f"- Matrix-free matvec: `{summary['solver_inputs']['matrix_free_matvec']}`",
            f"- Ready router tensors: `{summary['solver_inputs']['ready_router_count']}`",
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['router_stats']}`",
            f"- `{summary['outputs']['requirements']}`",
            f"- `{summary['outputs']['solver_inputs']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def missing_cache_outputs(args: argparse.Namespace, output_dir: Path, priority: pd.DataFrame) -> dict[str, Any]:
    stats = pd.DataFrame(columns=CSV_COLUMNS)
    coverage = first_stage_coverage(stats, priority)
    summary = {
        "schema_version": 1,
        "status": "harc_router_stats_missing_cache",
        "cache_path": rel(args.cache),
        "cache_exists": False,
        "router_count": 0,
        "valid_router_count": 0,
        "total_hidden_rows": 0,
        "mean_hessian_trace": None,
        "mean_hidden_cov_trace": None,
        "top_precision_proxy_tensor": None,
        **coverage,
        "recommended_action": "collect_router_calibration_cache_on_gpu_host",
        "cache_collection_command": (
            "PYTHONPATH=scripts python scripts/collect_moe_router_calibration_cache.py "
            "--student-model STUDENT_OR_MERGED_CHECKPOINT --teacher-model TEACHER_SOURCE_CHECKPOINT "
            f"--output-dir {rel(Path(args.cache).parent)}"
        ),
        "solver_inputs": solver_inputs(stats),
        "literature_sources": LITERATURE_SOURCES,
        "outputs": output_paths(output_dir),
    }
    requirements = build_requirements(summary)
    stats.to_csv(output_dir / "router_harc_stats.csv", index=False)
    requirements.to_csv(output_dir / "harc_stats_requirements.csv", index=False)
    write_json(output_dir / "harc_solver_inputs.json", summary["solver_inputs"])
    write_json(output_dir / "literature_sources.json", {"sources": LITERATURE_SOURCES})
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(build_report(summary, stats, requirements), encoding="utf-8")
    return summary


def solver_inputs(stats: pd.DataFrame) -> dict[str, Any]:
    ready = stats[stats["solver_input_ready"].astype(bool)] if not stats.empty else stats
    router_rows = []
    if not ready.empty:
        for _, row in ready.sort_values("harc_precision_proxy", ascending=False).iterrows():
            router_rows.append(
                {
                    "tensor": row["tensor"],
                    "layer": None if pd.isna(row.get("layer")) else int(row["layer"]),
                    "hidden_rows": int(row["hidden_rows"]),
                    "hidden_dim": int(row["hidden_dim"]),
                    "num_experts": int(row["num_experts"]),
                    "mean_hessian_trace": fnum(row.get("mean_hessian_trace")),
                    "hidden_cov_trace": fnum(row.get("hidden_cov_trace")),
                    "harc_precision_proxy": fnum(row.get("harc_precision_proxy")),
                    "harc_boundary_cov_proxy": fnum(row.get("harc_boundary_cov_proxy")),
                }
            )
    return {
        "objective": "min_Wm sum_i E_x KL(softmax(W_i x) || softmax(W_m x))",
        "quadratic_proxy": "0.5 * (W_m x - W_i x)^T (diag(r_i)-r_i r_i^T) (W_m x - W_i x)",
        "matrix_free_matvec": "sum_i E_x [x x^T dW^T H_i] without materializing kron(H_i, xx^T)",
        "same_shape_constraint": "solve only for existing router weight tensors; checkpoint architecture and tensor shapes stay unchanged",
        "ready_router_count": int(len(ready)),
        "router_rows": router_rows,
    }


def output_paths(output_dir: Path) -> dict[str, str]:
    return {
        "router_stats": rel(output_dir / "router_harc_stats.csv"),
        "requirements": rel(output_dir / "harc_stats_requirements.csv"),
        "solver_inputs": rel(output_dir / "harc_solver_inputs.json"),
        "literature": rel(output_dir / "literature_sources.json"),
        "summary": rel(output_dir / "summary.json"),
        "report": rel(output_dir / "report.md"),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = repo_path(args.cache)
    priority = load_priority(repo_path(args.layer_priority))
    if not cache_path.exists():
        return missing_cache_outputs(args, output_dir, priority)

    records, cache_metadata = load_cache_records(cache_path)
    priority_by_layer = {
        int(row["layer"]): {str(key): clean_value(value) for key, value in row.items()}
        for _, row in priority.iterrows()
    }
    rows = [
        summarize_router(
            record,
            priority_by_layer,
            top_k=args.top_k,
            max_cov_dim_for_eig=args.max_cov_dim_for_eig,
            min_rows=args.min_rows_per_router,
        )
        for record in records[: args.max_routers] if args.max_routers is not None
    ]
    if args.max_routers is None:
        rows = [
            summarize_router(
                record,
                priority_by_layer,
                top_k=args.top_k,
                max_cov_dim_for_eig=args.max_cov_dim_for_eig,
                min_rows=args.min_rows_per_router,
            )
            for record in records
        ]
    stats = pd.DataFrame(rows, columns=CSV_COLUMNS)
    coverage = first_stage_coverage(stats, priority)
    valid = stats[stats["solver_input_ready"].astype(bool)] if not stats.empty else stats
    top_row = clean_value(valid.sort_values("harc_precision_proxy", ascending=False).iloc[0]["tensor"]) if not valid.empty else None
    summary = {
        "schema_version": 1,
        "status": "pending",
        "cache_path": rel(cache_path),
        "cache_exists": True,
        "cache_metadata": cache_metadata,
        "router_count": int(len(stats)),
        "valid_router_count": int(len(valid)),
        "total_hidden_rows": int(stats["hidden_rows"].sum()) if not stats.empty else 0,
        "mean_hessian_trace": float(valid["mean_hessian_trace"].mean()) if not valid.empty else None,
        "mean_hidden_cov_trace": float(valid["hidden_cov_trace"].mean()) if not valid.empty else None,
        "mean_boundary_pair_hessian": float(valid["mean_boundary_pair_hessian"].mean()) if not valid.empty else None,
        "mean_topk_logit_margin": float(valid["mean_topk_logit_margin"].dropna().mean())
        if not valid.empty and valid["mean_topk_logit_margin"].notna().any()
        else None,
        "top_precision_proxy_tensor": top_row,
        **coverage,
        "recommended_action": "run_matrix_free_harc_router_solver" if not valid.empty else "repair_or_recollect_router_cache",
        "solver_inputs": solver_inputs(stats),
        "literature_sources": LITERATURE_SOURCES,
        "outputs": output_paths(output_dir),
    }
    requirements = build_requirements(summary)
    summary["status"] = decide_status(summary, requirements)
    if summary["status"] != "harc_router_stats_ready":
        summary["recommended_action"] = "complete_missing_harc_router_stats_before_solver"
    stats.to_csv(output_dir / "router_harc_stats.csv", index=False)
    requirements.to_csv(output_dir / "harc_stats_requirements.csv", index=False)
    write_json(output_dir / "harc_solver_inputs.json", summary["solver_inputs"])
    write_json(output_dir / "literature_sources.json", {"sources": LITERATURE_SOURCES})
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(build_report(summary, stats, requirements), encoding="utf-8")
    return summary


def make_smoke_cache(root: Path, *, seed: int) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    routers: dict[str, dict[str, torch.Tensor]] = {}
    for layer in (0, 1):
        hidden = torch.randn(96, 6, generator=generator)
        weight = torch.randn(12, 6, generator=generator) * 0.35
        weight[layer, layer] += 1.25
        teacher_logits = hidden @ weight.t()
        candidate_logits = teacher_logits + 0.15 * torch.randn(96, 12, generator=generator)
        groups = torch.arange(96, dtype=torch.long) // 16
        routers[f"model.layers.{layer}.mlp.gate.weight"] = {
            "hidden": hidden,
            "teacher_logits": teacher_logits,
            "candidate_logits": candidate_logits,
            "sample_groups": groups,
        }
    cache = root / "router_calibration_cache.torch"
    torch.save({"schema_version": 1, "routers": routers, "metadata": {"source": "synthetic_harc_stats_smoke"}}, cache)
    priority = pd.DataFrame(
        [
            {
                "layer": 0,
                "router": "model.layers.0.mlp.gate",
                "harc_priority_score": 0.75,
                "harc_layer_role": "critical_topk_boundary_layer",
            },
            {
                "layer": 1,
                "router": "model.layers.1.mlp.gate",
                "harc_priority_score": 0.62,
                "harc_layer_role": "collect_hessian_covariance_first",
            },
        ]
    )
    priority_path = root / "layer_harc_priority.csv"
    priority.to_csv(priority_path, index=False)
    return cache, priority_path


def build_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache, priority = make_smoke_cache(output_dir / "mock_inputs", seed=args.seed)
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.cache = cache
    smoke_args.layer_priority = priority
    summary = build(smoke_args)
    stats = read_csv(output_dir / "router_harc_stats.csv")
    checks = [
        {
            "check": "status_ready",
            "passed": summary["status"] == "harc_router_stats_ready",
            "evidence": summary["status"],
        },
        {
            "check": "two_router_records",
            "passed": int(summary["valid_router_count"]) == 2,
            "evidence": str(summary["valid_router_count"]),
        },
        {
            "check": "positive_hessian_trace",
            "passed": bool((stats["mean_hessian_trace"] > 0).all()) if not stats.empty else False,
            "evidence": fmt(stats["mean_hessian_trace"].mean() if not stats.empty else None),
        },
        {
            "check": "positive_covariance_trace",
            "passed": bool((stats["hidden_cov_trace"] > 0).all()) if not stats.empty else False,
            "evidence": fmt(stats["hidden_cov_trace"].mean() if not stats.empty else None),
        },
        {
            "check": "positive_boundary_hessian",
            "passed": bool((stats["mean_boundary_pair_hessian"] > 0).all()) if not stats.empty else False,
            "evidence": fmt(stats["mean_boundary_pair_hessian"].mean() if not stats.empty else None),
        },
        {
            "check": "first_stage_complete",
            "passed": summary["first_stage_coverage_status"] == "complete",
            "evidence": (
                f"{summary['first_stage_covered_layer_count']}/"
                f"{summary['first_stage_required_layer_count']}"
            ),
        },
    ]
    passed = all(bool(row["passed"]) for row in checks)
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(output_dir / "smoke_checks.csv", index=False)
    summary["smoke_status"] = "smoke_passed" if passed else "smoke_failed"
    summary["smoke_checks"] = {"passed": int(sum(bool(row["passed"]) for row in checks)), "total": int(len(checks))}
    summary["outputs"]["smoke_checks"] = rel(output_dir / "smoke_checks.csv")
    write_json(output_dir / "summary.json", summary)
    lines = [
        "# Qwen3 MoE HARC Router Stats Smoke",
        "",
        f"- Status: `{summary['smoke_status']}`",
        f"- Checks: `{summary['smoke_checks']['passed']}/{summary['smoke_checks']['total']}`",
        f"- Stats status: `{summary['status']}`",
        "",
        "| check | passed | evidence |",
        "| --- | --- | --- |",
    ]
    for row in checks:
        lines.append(f"| `{row['check']}` | `{row['passed']}` | {row['evidence']} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect HARC router Hessian/covariance stats from a router calibration cache.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_harc_router_stats"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("results/qwen3_moe_router_calibration_cache/router_calibration_cache.pt"),
    )
    parser.add_argument(
        "--layer-priority",
        type=Path,
        default=Path("results/qwen3_moe_harc_readiness_gate/layer_harc_priority.csv"),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-rows-per-router", type=int, default=8)
    parser.add_argument("--max-routers", type=int, default=None)
    parser.add_argument("--max-cov-dim-for-eig", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_smoke(args) if args.smoke_matrix else build(args)
    print(f"Wrote Qwen3 MoE HARC router stats to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; routers={summary['valid_router_count']}/{summary['router_count']}; "
        f"first_stage={summary['first_stage_covered_layer_count']}/{summary['first_stage_required_layer_count']}"
    )


if __name__ == "__main__":
    main()
