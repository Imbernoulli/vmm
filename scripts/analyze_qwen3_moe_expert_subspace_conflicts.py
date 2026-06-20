#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


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
    return value


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def robust01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.quantile(0.05))
    hi = float(values.quantile(0.95))
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    total = float(weights.sum())
    if total <= EPS:
        return float(values.mean()) if len(values) else 0.0
    return float((values * weights).sum() / total)


def build_projection_scores(projection_geometry: pd.DataFrame) -> pd.DataFrame:
    if projection_geometry.empty:
        return pd.DataFrame()
    out = projection_geometry.copy()
    out["relative_delta_pressure"] = robust01(numeric(out, "relative_delta"))
    out["angle_pressure"] = robust01(numeric(out, "one_minus_cosine"))
    out["channel_angle_pressure"] = robust01(1.0 - numeric(out, "channel_cosine_p05", 1.0))
    out["chunk_delta_pressure"] = robust01(numeric(out, "chunk_relative_delta_p95"))
    out["chunk_angle_pressure"] = robust01(1.0 - numeric(out, "chunk_cosine_min", 1.0))
    out["concentration_pressure"] = robust01(numeric(out, "top_10pct_channel_delta_energy_fraction"))
    out["chunk_concentration_pressure"] = robust01(numeric(out, "chunk_delta_energy_max_share"))
    out["subspace_projection_conflict_score"] = (
        0.24 * out["relative_delta_pressure"]
        + 0.18 * out["angle_pressure"]
        + 0.16 * out["channel_angle_pressure"]
        + 0.16 * out["chunk_delta_pressure"]
        + 0.10 * out["chunk_angle_pressure"]
        + 0.10 * out["concentration_pressure"]
        + 0.06 * out["chunk_concentration_pressure"]
    ).clip(0.0, 1.0)
    out["subspace_conflict_driver"] = "diffuse_delta"
    out.loc[
        (out["concentration_pressure"] >= out["angle_pressure"])
        & (out["concentration_pressure"] >= out["relative_delta_pressure"]),
        "subspace_conflict_driver",
    ] = "localized_channel_delta"
    out.loc[
        (out["channel_angle_pressure"] >= out["concentration_pressure"])
        & (out["channel_angle_pressure"] >= out["relative_delta_pressure"]),
        "subspace_conflict_driver",
    ] = "low_channel_cosine"
    out.loc[
        (out["chunk_delta_pressure"] >= out["concentration_pressure"])
        & (out["chunk_delta_pressure"] >= out["channel_angle_pressure"]),
        "subspace_conflict_driver",
    ] = "chunk_delta_spike"
    keep = [
        "tensor",
        "layer_id",
        "expert_id",
        "projection",
        "relative_delta",
        "cosine",
        "channel_cosine_p05",
        "chunk_relative_delta_p95",
        "chunk_cosine_min",
        "top_10pct_channel_delta_energy_fraction",
        "chunk_delta_energy_max_share",
        "relative_delta_pressure",
        "angle_pressure",
        "channel_angle_pressure",
        "chunk_delta_pressure",
        "chunk_angle_pressure",
        "concentration_pressure",
        "chunk_concentration_pressure",
        "subspace_projection_conflict_score",
        "subspace_conflict_driver",
    ]
    return out[[column for column in keep if column in out.columns]].sort_values(
        ["subspace_projection_conflict_score", "layer_id", "expert_id"],
        ascending=[False, True, True],
    )


def merge_optional_context(
    experts: pd.DataFrame,
    unified_rules: pd.DataFrame,
    layer_chunk_rules: pd.DataFrame,
) -> pd.DataFrame:
    out = experts.copy()
    if not unified_rules.empty:
        keep = [
            "layer_id",
            "expert_id",
            "selected_weight_instruct",
            "selected_weight_coder",
            "selected_scale",
            "selected_expected_max_relative_delta_norm",
            "mechanism_risk_score",
            "tensor_pattern",
            "tensor_rule",
            "geometry_action",
            "trust_risk_flags",
        ]
        present = [column for column in keep if column in unified_rules.columns]
        renamed = unified_rules[present].rename(
            columns={
                "selected_weight_instruct": "unified_weight_instruct",
                "selected_weight_coder": "unified_weight_coder",
                "selected_scale": "unified_scale",
                "selected_expected_max_relative_delta_norm": "unified_expected_max_relative_delta_norm",
                "mechanism_risk_score": "unified_mechanism_risk_score",
                "geometry_action": "unified_geometry_action",
                "trust_risk_flags": "unified_trust_risk_flags",
            }
        )
        out = out.merge(renamed, on=["layer_id", "expert_id"], how="left")
    if not layer_chunk_rules.empty:
        keep = [
            "layer_id",
            "expert_id",
            "layer_chunk_coder_coefficient",
            "layer_chunk_expected_relative_delta",
            "mechanism_risk_score",
            "layer_importance_score",
            "chunk_policy",
        ]
        present = [column for column in keep if column in layer_chunk_rules.columns]
        renamed = layer_chunk_rules[present].rename(
            columns={
                "mechanism_risk_score": "layer_chunk_mechanism_risk_score",
                "chunk_policy": "layer_chunk_policy",
            }
        )
        out = out.merge(renamed, on=["layer_id", "expert_id"], how="left")
    defaults = {
        "total_topk_fraction": 0.0,
        "route_geometry_risk_score": 0.0,
        "internal_geometry_risk_score": 0.0,
        "combined_relative_delta": 0.0,
        "combined_cosine": 1.0,
        "unified_weight_instruct": 0.0,
        "unified_weight_coder": 0.0,
        "unified_scale": 1.0,
        "unified_expected_max_relative_delta_norm": 0.0,
        "unified_mechanism_risk_score": 0.0,
        "layer_chunk_coder_coefficient": 1.0,
        "layer_chunk_expected_relative_delta": 0.0,
        "layer_chunk_mechanism_risk_score": 0.0,
        "layer_importance_score": 0.0,
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
    for column in ["tensor_pattern", "tensor_rule", "unified_geometry_action", "unified_trust_risk_flags", "layer_chunk_policy"]:
        if column not in out:
            out[column] = ""
        out[column] = out[column].fillna("")
    return out


def build_expert_conflicts(
    projection_scores: pd.DataFrame,
    expert_geometry: pd.DataFrame,
    unified_rules: pd.DataFrame,
    layer_chunk_rules: pd.DataFrame,
    high_threshold: float,
    medium_threshold: float,
) -> pd.DataFrame:
    if projection_scores.empty:
        return pd.DataFrame()
    grouped = projection_scores.groupby(["layer_id", "expert_id"], as_index=False).agg(
        projection_count=("projection", "nunique"),
        mean_projection_subspace_conflict_score=("subspace_projection_conflict_score", "mean"),
        max_projection_subspace_conflict_score=("subspace_projection_conflict_score", "max"),
        mean_projection_relative_delta=("relative_delta", "mean"),
        min_projection_cosine=("cosine", "min"),
        min_channel_cosine_p05=("channel_cosine_p05", "min"),
        max_chunk_relative_delta_p95=("chunk_relative_delta_p95", "max"),
        max_top_10pct_channel_delta_energy_fraction=("top_10pct_channel_delta_energy_fraction", "max"),
    )
    worst = projection_scores.sort_values("subspace_projection_conflict_score", ascending=False).drop_duplicates(
        ["layer_id", "expert_id"]
    )
    grouped = grouped.merge(
        worst[["layer_id", "expert_id", "projection", "subspace_conflict_driver"]].rename(
            columns={"projection": "worst_projection", "subspace_conflict_driver": "worst_subspace_conflict_driver"}
        ),
        on=["layer_id", "expert_id"],
        how="left",
    )
    if not expert_geometry.empty:
        keep = [
            "layer_id",
            "expert_id",
            "combined_relative_delta",
            "combined_cosine",
            "internal_geometry_risk_score",
            "route_geometry_risk_score",
            "total_topk_fraction",
            "dominant_source",
            "dominant_weight",
            "weight_instruct",
            "weight_coder",
            "trust_risk_flags",
            "geometry_action",
        ]
        present = [column for column in keep if column in expert_geometry.columns]
        grouped = grouped.merge(expert_geometry[present], on=["layer_id", "expert_id"], how="left")
    grouped = merge_optional_context(grouped, unified_rules, layer_chunk_rules)
    grouped["route_importance_pressure"] = robust01(grouped["total_topk_fraction"])
    grouped["route_geometry_pressure"] = numeric(grouped, "route_geometry_risk_score").clip(0.0, 1.0)
    grouped["internal_geometry_pressure"] = numeric(grouped, "internal_geometry_risk_score").clip(0.0, 1.0)
    grouped["subspace_conflict_score"] = (
        0.40 * grouped["max_projection_subspace_conflict_score"]
        + 0.25 * grouped["mean_projection_subspace_conflict_score"]
        + 0.15 * grouped["internal_geometry_pressure"]
        + 0.10 * robust01(grouped["combined_relative_delta"])
        + 0.10 * robust01(1.0 - grouped["combined_cosine"])
    ).clip(0.0, 1.0)
    route_cut = float(grouped["total_topk_fraction"].quantile(0.75))
    grouped["route_important"] = (
        (grouped["total_topk_fraction"] >= route_cut)
        | (grouped["route_geometry_pressure"] >= 0.75)
    )
    grouped["high_subspace_conflict"] = grouped["subspace_conflict_score"] >= high_threshold
    grouped["medium_subspace_conflict"] = grouped["subspace_conflict_score"] >= medium_threshold
    grouped["route_weighted_subspace_conflict_score"] = grouped["subspace_conflict_score"] * (
        0.5 + 0.5 * grouped["route_importance_pressure"]
    )
    grouped["subspace_recommended_relative_cap"] = 0.65
    grouped.loc[grouped["medium_subspace_conflict"] & grouped["route_important"], "subspace_recommended_relative_cap"] = 0.62
    grouped.loc[grouped["high_subspace_conflict"], "subspace_recommended_relative_cap"] = 0.60
    grouped.loc[grouped["high_subspace_conflict"] & grouped["route_important"], "subspace_recommended_relative_cap"] = 0.55
    grouped.loc[
        (grouped["subspace_conflict_score"] >= 0.90) & grouped["route_important"],
        "subspace_recommended_relative_cap",
    ] = 0.50
    expected = grouped["unified_expected_max_relative_delta_norm"].clip(lower=0.0)
    cap = grouped["subspace_recommended_relative_cap"].clip(lower=EPS)
    grouped["subspace_extra_scale"] = 1.0
    needs_scale = expected > cap
    grouped.loc[needs_scale, "subspace_extra_scale"] = cap.loc[needs_scale] / expected.loc[needs_scale]
    grouped["subspace_extra_scale"] = grouped["subspace_extra_scale"].clip(0.0, 1.0)
    grouped["subspace_adjusted_weight_coder"] = grouped["unified_weight_coder"] * grouped["subspace_extra_scale"]
    grouped["subspace_coder_weight_reduction"] = grouped["unified_weight_coder"] - grouped["subspace_adjusted_weight_coder"]
    grouped["current_unified_subspace_covered"] = grouped["subspace_extra_scale"] >= 0.999
    grouped["subspace_action"] = "identity_average_subspace_ok"
    grouped.loc[grouped["medium_subspace_conflict"], "subspace_action"] = "monitor_subspace_conflict_in_vllm_gate"
    grouped.loc[grouped["high_subspace_conflict"], "subspace_action"] = "lower_nonbase_weight_for_subspace_conflict"
    grouped.loc[
        grouped["high_subspace_conflict"] & grouped["route_important"],
        "subspace_action",
    ] = "route_important_subspace_conflict_lower_cap_or_chunk"
    grouped.loc[
        grouped["high_subspace_conflict"] & grouped["route_important"] & grouped["current_unified_subspace_covered"],
        "subspace_action",
    ] = "current_unified_cap_covers_route_subspace_conflict"
    grouped["subspace_scaled_tensor_rule"] = ""
    has_pattern = grouped["tensor_pattern"].astype(str).str.len() > 0
    grouped.loc[has_pattern, "subspace_scaled_tensor_rule"] = grouped.loc[has_pattern].apply(
        lambda row: (
            f"{row['tensor_pattern']}::coder={row['subspace_adjusted_weight_coder']:.6f},"
            f"instruct={row['unified_weight_instruct']:.6f}"
        ),
        axis=1,
    )
    columns = [
        "layer_id",
        "expert_id",
        "projection_count",
        "worst_projection",
        "worst_subspace_conflict_driver",
        "subspace_conflict_score",
        "route_weighted_subspace_conflict_score",
        "high_subspace_conflict",
        "route_important",
        "total_topk_fraction",
        "route_geometry_risk_score",
        "internal_geometry_risk_score",
        "combined_relative_delta",
        "combined_cosine",
        "mean_projection_subspace_conflict_score",
        "max_projection_subspace_conflict_score",
        "min_channel_cosine_p05",
        "max_chunk_relative_delta_p95",
        "max_top_10pct_channel_delta_energy_fraction",
        "unified_weight_instruct",
        "unified_weight_coder",
        "unified_scale",
        "unified_expected_max_relative_delta_norm",
        "subspace_recommended_relative_cap",
        "subspace_extra_scale",
        "subspace_adjusted_weight_coder",
        "subspace_coder_weight_reduction",
        "current_unified_subspace_covered",
        "subspace_action",
        "dominant_source",
        "dominant_weight",
        "trust_risk_flags",
        "geometry_action",
        "unified_geometry_action",
        "layer_chunk_policy",
        "layer_chunk_coder_coefficient",
        "layer_chunk_expected_relative_delta",
        "tensor_pattern",
        "subspace_scaled_tensor_rule",
    ]
    return grouped[[column for column in columns if column in grouped.columns]].sort_values(
        ["route_weighted_subspace_conflict_score", "subspace_conflict_score", "layer_id", "expert_id"],
        ascending=[False, False, True, True],
    )


def build_layer_conflicts(expert_conflicts: pd.DataFrame) -> pd.DataFrame:
    if expert_conflicts.empty:
        return pd.DataFrame()
    rows = []
    for layer_id, group in expert_conflicts.groupby("layer_id", sort=True):
        route_weights = numeric(group, "total_topk_fraction")
        rows.append(
            {
                "layer_id": int(layer_id),
                "expert_count": int(len(group)),
                "high_subspace_conflict_experts": int(group["high_subspace_conflict"].sum()),
                "route_important_high_subspace_conflict_experts": int(
                    (group["high_subspace_conflict"] & group["route_important"]).sum()
                ),
                "subspace_extra_scaled_experts": int((group["subspace_extra_scale"] < 0.999).sum()),
                "mean_subspace_conflict_score": float(group["subspace_conflict_score"].mean()),
                "max_subspace_conflict_score": float(group["subspace_conflict_score"].max()),
                "route_mass_weighted_subspace_conflict_score": weighted_mean(
                    group["subspace_conflict_score"],
                    route_weights,
                ),
                "mean_unified_expected_max_relative_delta_norm": float(
                    group["unified_expected_max_relative_delta_norm"].mean()
                ),
                "mean_subspace_extra_scale": float(group["subspace_extra_scale"].mean()),
                "coder_weight_reduction_sum": float(group["subspace_coder_weight_reduction"].sum()),
                "route_mass_sum": float(route_weights.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["route_mass_weighted_subspace_conflict_score", "route_important_high_subspace_conflict_experts"],
        ascending=[False, False],
    )


def build_action_summary(expert_conflicts: pd.DataFrame) -> pd.DataFrame:
    if expert_conflicts.empty:
        return pd.DataFrame()
    return (
        expert_conflicts.groupby("subspace_action", as_index=False)
        .agg(
            expert_count=("expert_id", "count"),
            route_mass_sum=("total_topk_fraction", "sum"),
            mean_subspace_conflict_score=("subspace_conflict_score", "mean"),
            mean_subspace_extra_scale=("subspace_extra_scale", "mean"),
            coder_weight_reduction_sum=("subspace_coder_weight_reduction", "sum"),
        )
        .sort_values(["coder_weight_reduction_sum", "expert_count"], ascending=[False, False])
    )


def write_adjusted_rules(expert_conflicts: pd.DataFrame, path: Path) -> int:
    lines = []
    if not expert_conflicts.empty and "subspace_scaled_tensor_rule" in expert_conflicts:
        for rule in expert_conflicts["subspace_scaled_tensor_rule"].dropna().astype(str):
            if rule:
                lines.append(rule)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def build_writer_args(
    args: argparse.Namespace,
    tensor_rules_path: Path,
    output_dir: Path,
    *,
    dry_run: bool,
) -> list[str]:
    parts = [
        "python",
        "scripts/write_same_shape_average_checkpoint.py",
        "--base",
        str(args.base),
        "--source",
        f"instruct={args.instruct_source}",
        "--source",
        f"coder={args.coder_source}",
        "--source-weight",
        "instruct=0.0",
        "--source-weight",
        "coder=0.0",
        "--freeze-router",
        "--tensor-rule-file",
        rel(tensor_rules_path),
        "--output-dir",
        rel(output_dir),
    ]
    if dry_run:
        parts.append("--dry-run")
    return parts


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def summarize_dry_run_manifest(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {
            "dry_run_validated": False,
            "dry_run_manifest": rel(path),
            "dry_run_floating_tensors": 0,
            "dry_run_frozen_tensors": 0,
            "dry_run_tensor_rule_count": 0,
            "dry_run_tensor_rule_hit_count": 0,
            "dry_run_default_tensor_count": 0,
            "dry_run_freeze_router_hits": 0,
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rule_counts = manifest.get("rule_counts") or {}
    tensor_rule_hits = int(
        sum(int(value) for key, value in rule_counts.items() if str(key).startswith("tensor_rule:"))
    )
    return {
        "dry_run_validated": bool(manifest.get("dry_run")) and tensor_rule_hits > 0,
        "dry_run_manifest": rel(path),
        "dry_run_floating_tensors": int(manifest.get("floating_tensors", 0)),
        "dry_run_frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        "dry_run_tensor_rule_count": int(len(manifest.get("tensor_rules") or [])),
        "dry_run_tensor_rule_hit_count": tensor_rule_hits,
        "dry_run_default_tensor_count": int(rule_counts.get("default", 0)),
        "dry_run_freeze_router_hits": int(rule_counts.get("freeze_router", 0)),
    }


def build_report(
    summary: dict[str, Any],
    layer_conflicts: pd.DataFrame,
    action_summary: pd.DataFrame,
    expert_conflicts: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen3 MoE Expert Subspace Conflict Probe",
        "",
        "这个 probe 不再只问 expert index 是否对齐，而是读取真实 routed expert projection 的 channel/chunk 几何，检查同名 expert 内部是否存在局部子空间冲突。它的作用是给 MoE average 增加一个更细的 gate：identity mapping 通过以后，仍要判断哪些高路由质量 expert 需要更低非 base 权重、layer/chunk 系数或额外 output-space probe。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Projection tensors: `{summary['projection_tensor_count']}`",
        f"- Experts: `{summary['expert_count']}`",
        f"- Layers: `{summary['layer_count']}`",
        f"- High subspace-conflict experts: `{summary['high_subspace_conflict_expert_count']}`",
        f"- Route-important high subspace-conflict experts: `{summary['route_important_high_subspace_conflict_expert_count']}`",
        f"- Experts needing extra subspace scale beyond current unified rule: `{summary['subspace_extra_scaled_expert_count']}`",
        f"- Mean coder weight reduction if this ablation is materialized: `{summary['mean_coder_weight_reduction']:.6f}`",
        f"- Total coder weight reduction if this ablation is materialized: `{summary['total_coder_weight_reduction']:.6f}`",
        f"- Dry-run validated: `{summary['dry_run_validated']}`",
        f"- Dry-run tensor-rule hits: `{summary['dry_run_tensor_rule_hit_count']}`",
        f"- Dry-run freeze-router hits: `{summary['dry_run_freeze_router_hits']}`",
        f"- Top layer by route-weighted subspace conflict: `L{summary['top_subspace_conflict_layer']}`",
        f"- Next action: `{summary['next_action']}`",
        "",
        "## Layer Risk",
        "",
        "| layer | high experts | route-high experts | extra-scaled experts | weighted score | mean scale | coder weight reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in layer_conflicts.head(12).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['high_subspace_conflict_experts'])} | "
            f"{int(row['route_important_high_subspace_conflict_experts'])} | "
            f"{int(row['subspace_extra_scaled_experts'])} | "
            f"{float(row['route_mass_weighted_subspace_conflict_score']):.4f} | "
            f"{float(row['mean_subspace_extra_scale']):.4f} | "
            f"{float(row['coder_weight_reduction_sum']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Actions",
            "",
            "| action | experts | route mass | mean conflict | mean scale | coder weight reduction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in action_summary.iterrows():
        lines.append(
            f"| `{row['subspace_action']}` | {int(row['expert_count'])} | "
            f"{float(row['route_mass_sum']):.4f} | {float(row['mean_subspace_conflict_score']):.4f} | "
            f"{float(row['mean_subspace_extra_scale']):.4f} | "
            f"{float(row['coder_weight_reduction_sum']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Top Expert Conflicts",
            "",
            "| layer | expert | projection | driver | conflict | route mass | expected delta | cap | scale | action |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in expert_conflicts.head(12).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | `{row['worst_projection']}` | "
            f"`{row['worst_subspace_conflict_driver']}` | {float(row['subspace_conflict_score']):.4f} | "
            f"{float(row['total_topk_fraction']):.4f} | "
            f"{float(row['unified_expected_max_relative_delta_norm']):.4f} | "
            f"{float(row['subspace_recommended_relative_cap']):.4f} | "
            f"{float(row['subspace_extra_scale']):.4f} | `{row['subspace_action']}` |"
        )
    scaled = expert_conflicts[expert_conflicts["subspace_extra_scale"] < 0.999].sort_values("subspace_extra_scale")
    lines.extend(
        [
            "",
            "## Extra Scale Targets",
            "",
            "| layer | expert | projection | conflict | route mass | expected delta | cap | scale | coder before-after | action |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in scaled.head(20).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | `{row['worst_projection']}` | "
            f"{float(row['subspace_conflict_score']):.4f} | {float(row['total_topk_fraction']):.4f} | "
            f"{float(row['unified_expected_max_relative_delta_norm']):.4f} | "
            f"{float(row['subspace_recommended_relative_cap']):.4f} | "
            f"{float(row['subspace_extra_scale']):.4f} | "
            f"{float(row['unified_weight_coder']):.4f}-{float(row['subspace_adjusted_weight_coder']):.4f} | "
            f"`{row['subspace_action']}` |"
        )
    lines.extend(
        [
            "",
            "## Candidate Recipe",
            "",
            "```bash",
            summary["writer_command"],
            "```",
            "",
            "## Dry-Run",
            "",
            "```bash",
            summary["dry_run_command"],
            "```",
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['projection_scores']}`",
            f"- `{summary['outputs']['expert_conflicts']}`",
            f"- `{summary['outputs']['layer_conflicts']}`",
            f"- `{summary['outputs']['action_summary']}`",
            f"- `{summary['outputs']['subspace_adjusted_group_rules']}`",
            f"- `{summary['outputs']['subspace_adjusted_tensor_rules']}`",
            f"- `{summary['outputs']['dry_run_command']}`",
            f"- `{summary['outputs']['dry_run_manifest']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    args: argparse.Namespace,
    projection_scores: pd.DataFrame,
    expert_conflicts: pd.DataFrame,
    layer_conflicts: pd.DataFrame,
    action_summary: pd.DataFrame,
) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    projection_path = output_dir / "projection_subspace_scores.csv"
    expert_path = output_dir / "expert_subspace_conflicts.csv"
    layer_path = output_dir / "layer_subspace_conflicts.csv"
    action_path = output_dir / "subspace_action_summary.csv"
    adjusted_group_path = output_dir / "subspace_adjusted_group_rules.csv"
    adjusted_rules_path = output_dir / "subspace_adjusted_tensor_rules.txt"
    dry_run_command_path = output_dir / "dry_run_command.txt"
    dry_run_output_dir = output_dir / "dry_run"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    projection_scores.to_csv(projection_path, index=False)
    expert_conflicts.to_csv(expert_path, index=False)
    layer_conflicts.to_csv(layer_path, index=False)
    action_summary.to_csv(action_path, index=False)
    adjusted_columns = [
        "layer_id",
        "expert_id",
        "tensor_pattern",
        "unified_weight_instruct",
        "unified_weight_coder",
        "subspace_adjusted_weight_coder",
        "subspace_extra_scale",
        "subspace_recommended_relative_cap",
        "subspace_action",
        "subspace_scaled_tensor_rule",
    ]
    expert_conflicts[[column for column in adjusted_columns if column in expert_conflicts.columns]].to_csv(
        adjusted_group_path,
        index=False,
    )
    rule_count = write_adjusted_rules(expert_conflicts, adjusted_rules_path)
    writer_command = format_command(
        build_writer_args(args, adjusted_rules_path, args.checkpoint_output_dir, dry_run=False)
    )
    dry_run_command_parts = build_writer_args(args, adjusted_rules_path, dry_run_output_dir, dry_run=True)
    dry_run_command = format_command(dry_run_command_parts)
    dry_run_command_path.write_text(dry_run_command + "\n", encoding="utf-8")
    if args.validate_dry_run:
        subprocess.run(dry_run_command_parts, cwd=REPO_ROOT, check=True)
    dry_run_summary = summarize_dry_run_manifest(dry_run_output_dir / "merge_manifest.json")
    high_count = int(expert_conflicts["high_subspace_conflict"].sum()) if not expert_conflicts.empty else 0
    route_high_count = int(
        (expert_conflicts["high_subspace_conflict"] & expert_conflicts["route_important"]).sum()
    ) if not expert_conflicts.empty else 0
    extra_scaled_count = int((expert_conflicts["subspace_extra_scale"] < 0.999).sum()) if not expert_conflicts.empty else 0
    total_reduction = float(expert_conflicts["subspace_coder_weight_reduction"].sum()) if not expert_conflicts.empty else 0.0
    mean_reduction = float(expert_conflicts["subspace_coder_weight_reduction"].mean()) if not expert_conflicts.empty else 0.0
    top_layer = int(layer_conflicts.iloc[0]["layer_id"]) if not layer_conflicts.empty else None
    status = "ready_for_subspace_ablation" if extra_scaled_count else "current_unified_covers_subspace_gate"
    next_action = (
        "materialize_subspace_scaled_ablation_after_source_eval_budget"
        if extra_scaled_count
        else "keep_current_unified_and_use_subspace_probe_as_gate_for_third_party_moe"
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "high_threshold": float(args.high_threshold),
        "medium_threshold": float(args.medium_threshold),
        "projection_tensor_count": int(len(projection_scores)),
        "expert_count": int(expert_conflicts[["layer_id", "expert_id"]].drop_duplicates().shape[0])
        if not expert_conflicts.empty
        else 0,
        "layer_count": int(expert_conflicts["layer_id"].nunique()) if not expert_conflicts.empty else 0,
        "high_subspace_conflict_expert_count": high_count,
        "route_important_high_subspace_conflict_expert_count": route_high_count,
        "subspace_extra_scaled_expert_count": extra_scaled_count,
        "subspace_adjusted_tensor_rule_count": int(rule_count),
        "mean_subspace_conflict_score": float(expert_conflicts["subspace_conflict_score"].mean())
        if not expert_conflicts.empty
        else 0.0,
        "max_subspace_conflict_score": float(expert_conflicts["subspace_conflict_score"].max())
        if not expert_conflicts.empty
        else 0.0,
        "mean_coder_weight_reduction": mean_reduction,
        "total_coder_weight_reduction": total_reduction,
        **dry_run_summary,
        "top_subspace_conflict_layer": top_layer,
        "next_action": next_action,
        "writer_command": writer_command,
        "dry_run_command": dry_run_command,
        "outputs": {
            "projection_scores": rel(projection_path),
            "expert_conflicts": rel(expert_path),
            "layer_conflicts": rel(layer_path),
            "action_summary": rel(action_path),
            "subspace_adjusted_group_rules": rel(adjusted_group_path),
            "subspace_adjusted_tensor_rules": rel(adjusted_rules_path),
            "dry_run_command": rel(dry_run_command_path),
            "dry_run_manifest": dry_run_summary["dry_run_manifest"],
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, layer_conflicts, action_summary, expert_conflicts), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen3 MoE expert subspace conflicts from real parameter geometry.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_expert_subspace_conflict_probe"))
    parser.add_argument(
        "--projection-geometry",
        type=Path,
        default=Path("results/qwen3_moe_expert_geometry_probe/projection_geometry.csv"),
    )
    parser.add_argument(
        "--expert-geometry",
        type=Path,
        default=Path("results/qwen3_moe_expert_geometry_probe/expert_geometry.csv"),
    )
    parser.add_argument(
        "--unified-rules",
        type=Path,
        default=Path("results/qwen3_moe_unified_mechanism_candidate/unified_group_rules.csv"),
    )
    parser.add_argument(
        "--layer-chunk-rules",
        type=Path,
        default=Path("results/qwen3_moe_layer_chunk_candidate/selected_group_rules.csv"),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/"
            "snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
        ),
    )
    parser.add_argument(
        "--instruct-source",
        type=Path,
        default=Path(
            "/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/"
            "snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
        ),
    )
    parser.add_argument(
        "--coder-source",
        type=Path,
        default=Path(
            "/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/"
            "snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
        ),
    )
    parser.add_argument(
        "--checkpoint-output-dir",
        type=Path,
        default=Path("results/checkpoints/qwen3_moe_subspace_scaled_candidate"),
    )
    parser.add_argument("--high-threshold", type=float, default=0.72)
    parser.add_argument("--medium-threshold", type=float, default=0.55)
    parser.add_argument(
        "--validate-dry-run",
        action="store_true",
        help="Run the same-shape checkpoint writer in dry-run mode and summarize the emitted merge manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    projection_geometry = read_csv(args.projection_geometry)
    expert_geometry = read_csv(args.expert_geometry)
    unified_rules = read_csv(args.unified_rules)
    layer_chunk_rules = read_csv(args.layer_chunk_rules)
    if projection_geometry.empty:
        raise SystemExit(f"Missing projection geometry: {repo_path(args.projection_geometry)}")
    projection_scores = build_projection_scores(projection_geometry)
    expert_conflicts = build_expert_conflicts(
        projection_scores,
        expert_geometry,
        unified_rules,
        layer_chunk_rules,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
    )
    layer_conflicts = build_layer_conflicts(expert_conflicts)
    action_summary = build_action_summary(expert_conflicts)
    summary = write_outputs(args, projection_scores, expert_conflicts, layer_conflicts, action_summary)
    print(f"Wrote expert subspace conflict probe to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; high conflicts {summary['high_subspace_conflict_expert_count']}; "
        f"extra-scaled {summary['subspace_extra_scaled_expert_count']}"
    )


if __name__ == "__main__":
    main()
