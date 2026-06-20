#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_qwen3_moe_router_coupled_candidate import (
    build_group_rules,
    build_joined,
    fnum,
    json_safe,
    metrics_from_scale,
    numeric,
    read_json,
    rel,
    repo_path,
    write_rules_and_commands,
)


EPS = 1e-12


def fmt(value: Any, digits: int = 6) -> str:
    value = fnum(value)
    return f"{value:.{digits}g}"


def build_frontier(
    frame: pd.DataFrame,
    *,
    hard_cap: float,
    min_retention: float,
    pressure_quantiles: tuple[float, ...],
    strengths: tuple[float, ...],
    caps: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, Any]]:
    base_scale = frame["base_scale"].copy()
    base_metrics = metrics_from_scale(frame, base_scale, hard_cap)
    rows: list[dict[str, Any]] = []
    scales: dict[str, pd.Series] = {}
    for quantile in pressure_quantiles:
        threshold = float(frame["router_coupled_pressure_norm"].quantile(quantile))
        eligible = (
            (frame["router_coupled_pressure_norm"] >= threshold)
            & (numeric(frame, "boundary_fragility_score", 0.0) >= 0.60)
            & (numeric(frame, "feature_router_instability", 0.0) >= 0.65)
        )
        for strength in strengths:
            for cap in caps:
                extra = (
                    (strength * frame["router_coupled_pressure_norm"])
                    .clip(upper=cap)
                    .where(eligible, 0.0)
                )
                scale = (base_scale - extra).clip(lower=0.0, upper=1.0)
                candidate_id = f"router_q{quantile:.2f}_s{strength:.5f}_cap{cap:.5f}"
                metrics = metrics_from_scale(frame, scale, hard_cap)
                retention_delta = (
                    metrics["nonbase_mass_retention"] - base_metrics["nonbase_mass_retention"]
                )
                risk_reduction = (
                    base_metrics["risk_weighted_predicted_delta"]
                    - metrics["risk_weighted_predicted_delta"]
                )
                coupled_reduction = (
                    base_metrics["router_coupled_weighted_predicted_delta"]
                    - metrics["router_coupled_weighted_predicted_delta"]
                )
                row = {
                    **metrics,
                    "candidate_id": candidate_id,
                    "pressure_quantile": quantile,
                    "extra_shrink_strength": strength,
                    "extra_shrink_cap": cap,
                    "retention_delta_vs_base": retention_delta,
                    "risk_delta_reduction_vs_base": risk_reduction,
                    "router_coupled_delta_reduction_vs_base": coupled_reduction,
                    "passes_hard_cap": metrics["hard_cap_violation_count"] == 0,
                    "passes_retention_gate": metrics["nonbase_mass_retention"] >= min_retention,
                }
                row["passes_default_gate"] = row["passes_hard_cap"] and row["passes_retention_gate"]
                rows.append(row)
                scales[candidate_id] = scale
    frontier = pd.DataFrame(rows)
    frontier = frontier.sort_values(
        [
            "passes_default_gate",
            "router_coupled_delta_reduction_vs_base",
            "risk_delta_reduction_vs_base",
            "nonbase_mass_retention",
        ],
        ascending=[False, False, False, False],
    )
    return frontier, scales, base_metrics


def select_rows(frontier: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    hard_pass = frontier[frontier["passes_hard_cap"].astype(bool)].copy()
    if hard_pass.empty:
        hard_pass = frontier.copy()
    stress = hard_pass.sort_values(
        ["router_coupled_delta_reduction_vs_base", "risk_delta_reduction_vs_base"],
        ascending=[False, False],
    ).iloc[0]
    default_pass = hard_pass[hard_pass["passes_retention_gate"].astype(bool)].copy()
    if default_pass.empty:
        constrained = hard_pass.sort_values(
            ["nonbase_mass_retention", "router_coupled_delta_reduction_vs_base"],
            ascending=[False, False],
        ).iloc[0]
    else:
        constrained = default_pass.sort_values(
            [
                "router_coupled_delta_reduction_vs_base",
                "risk_delta_reduction_vs_base",
                "nonbase_mass_retention",
            ],
            ascending=[False, False, False],
        ).iloc[0]
    return constrained, stress


def build_report(
    summary: dict[str, Any],
    frontier: pd.DataFrame,
    constrained_rules: pd.DataFrame,
) -> str:
    constrained = summary["constrained"]
    stress = summary["stress"]
    lines = [
        "# Qwen3 MoE Router-Coupled Retention Frontier",
        "",
        "This probe answers whether the router-boundary term should become part of the default unified average, not just whether an aggressive ablation can reduce a proxy. It reruns the router-coupled search with a much finer near-zero shrink grid and explicitly enforces the mechanistic retention gate.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Recommended action: `{summary['recommended_unified_action']}`",
        f"- Base retention: `{fmt(summary['base_nonbase_mass_retention'])}`",
        f"- Min retention gate: `{fmt(summary['min_retention'])}`",
        f"- Passing default-gate candidates: `{summary['default_gate_candidate_count']}/{summary['candidate_count']}`",
        f"- Best constrained candidate: `{constrained['candidate_id']}`",
        f"- Constrained retention / delta: `{fmt(constrained['nonbase_mass_retention'])}` / `{fmt(constrained['retention_delta_vs_base'])}`",
        f"- Constrained router-coupled reduction: `{fmt(constrained['router_coupled_delta_reduction_vs_base'])}`",
        f"- Stress candidate: `{stress['candidate_id']}`",
        f"- Stress retention / delta: `{fmt(stress['nonbase_mass_retention'])}` / `{fmt(stress['retention_delta_vs_base'])}`",
        f"- Stress router-coupled reduction: `{fmt(stress['router_coupled_delta_reduction_vs_base'])}`",
        f"- Constrained/stress effect fraction: `{fmt(summary['constrained_effect_fraction_vs_stress'])}`",
        "",
        "## Interpretation",
        "",
        (
            "The router-boundary signal is real, but the default mechanistic solution is already sitting almost exactly on the retention boundary. "
            "The strongest default-gate-safe direct router shrink has only a tiny proxy effect, while the first materially visible direct shrink violates the retention gate. "
            "So the default algorithm should keep router fragility inside B/H/I as an interference feature and leave direct extra shrink as an ablation unless downstream vLLM evidence proves the retention trade-off is worth it."
        ),
        "",
        "## Frontier",
        "",
        "| candidate | pass default | retention | retention delta | changed | coupled reduction | risk reduction | strength | cap | q |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    top = pd.concat(
        [
            frontier[frontier["passes_default_gate"].astype(bool)].head(10),
            frontier[~frontier["passes_default_gate"].astype(bool)].head(10),
        ],
        ignore_index=True,
    )
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['candidate_id']}` | `{bool(row['passes_default_gate'])}` | "
            f"{fmt(row['nonbase_mass_retention'])} | {fmt(row['retention_delta_vs_base'])} | "
            f"{int(row['changed_group_count'])} | {fmt(row['router_coupled_delta_reduction_vs_base'])} | "
            f"{fmt(row['risk_delta_reduction_vs_base'])} | {fmt(row['extra_shrink_strength'])} | "
            f"{fmt(row['extra_shrink_cap'])} | {fmt(row['pressure_quantile'])} |"
        )
    lines.extend(
        [
            "",
            "## Constrained Changed Experts",
            "",
            "| layer | expert | category | route mass | base scale | constrained scale | extra shrink | pressure |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    changed = constrained_rules[constrained_rules["router_coupled_extra_shrink"] > 1e-12]
    changed = changed.sort_values("router_coupled_pressure", ascending=False)
    for _, row in changed.head(12).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | `{row['dominant_category']}` | "
            f"{fmt(row['total_topk_fraction'])} | {fmt(row['mechanistic_selected_scale'])} | "
            f"{fmt(row['router_coupled_selected_scale'])} | {fmt(row['router_coupled_extra_shrink'])} | "
            f"{fmt(row['router_coupled_pressure'])} |"
        )
    lines.extend(["", "## Outputs", ""])
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the retention frontier for direct router-coupled expert shrink."
    )
    parser.add_argument(
        "--mechanistic-group-rules",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate/mechanistic_group_rules.csv"),
    )
    parser.add_argument(
        "--mechanistic-summary",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate/summary.json"),
    )
    parser.add_argument(
        "--router-layer-fragility",
        type=Path,
        default=Path("results/qwen3_moe_router_margin_fragility/layer_margin_fragility.csv"),
    )
    parser.add_argument(
        "--writer-context-command",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate/writer_command.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_moe_router_coupled_retention_frontier"),
    )
    parser.add_argument(
        "--checkpoint-output-dir",
        default="results/checkpoints/qwen3_moe_router_coupled_retention_constrained_candidate",
    )
    parser.add_argument("--minimum-effective-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_rules = pd.read_csv(repo_path(args.mechanistic_group_rules))
    layer_fragility = pd.read_csv(repo_path(args.router_layer_fragility))
    mechanistic_summary = read_json(args.mechanistic_summary)
    hard_cap = fnum(mechanistic_summary.get("effective_hard_cap"), 0.649)
    min_retention = fnum(mechanistic_summary.get("min_retention"), 0.965)

    joined = build_joined(group_rules, layer_fragility)
    frontier, scales, base_metrics = build_frontier(
        joined,
        hard_cap=hard_cap,
        min_retention=min_retention,
        pressure_quantiles=(0.75, 0.80, 0.85, 0.90, 0.95),
        strengths=(
            0.00005,
            0.00010,
            0.00020,
            0.00030,
            0.00040,
            0.00050,
            0.00075,
            0.00100,
            0.00200,
            0.00300,
            0.00400,
            0.00500,
            0.00750,
            0.01000,
        ),
        caps=(
            0.00010,
            0.00020,
            0.00030,
            0.00050,
            0.00075,
            0.00100,
            0.00200,
            0.00300,
            0.00500,
            0.00750,
            0.01000,
        ),
    )
    constrained, stress = select_rows(frontier)
    constrained_id = str(constrained["candidate_id"])
    constrained_scale = scales[constrained_id]
    constrained_rules = build_group_rules(joined, constrained_scale, constrained)
    effect_fraction = (
        fnum(constrained["router_coupled_delta_reduction_vs_base"])
        / max(EPS, fnum(stress["router_coupled_delta_reduction_vs_base"]))
    )
    default_safe = bool(constrained["passes_default_gate"])
    meaningful = effect_fraction >= args.minimum_effective_fraction
    if default_safe and meaningful:
        gate = "direct_router_boundary_term_default_candidate"
        action = "materialize_and_delta_audit_retention_constrained_router_coupled_candidate"
    else:
        gate = "direct_router_boundary_term_not_default"
        action = "keep_router_fragility_inside_BHI_and_keep_direct_extra_shrink_as_ablation"

    rules_outputs = write_rules_and_commands(
        constrained_rules,
        output_dir,
        args.writer_context_command,
        args.checkpoint_output_dir,
    )
    frontier_path = output_dir / "retention_frontier.csv"
    constrained_rules_path = output_dir / "retention_constrained_group_rules.csv"
    constrained_path = output_dir / "selected_retention_constrained_candidate.json"
    stress_path = output_dir / "selected_stress_candidate.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    frontier.to_csv(frontier_path, index=False)
    constrained_rules.to_csv(constrained_rules_path, index=False)
    constrained_path.write_text(
        json.dumps(json_safe(constrained.to_dict()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stress_path.write_text(
        json.dumps(json_safe(stress.to_dict()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "status": "router_coupled_retention_frontier_ready",
        "gate": gate,
        "recommended_unified_action": action,
        "base_mechanistic_candidate_id": mechanistic_summary.get("selected_candidate_id"),
        "effective_hard_cap": hard_cap,
        "min_retention": min_retention,
        "base_nonbase_mass_retention": base_metrics["nonbase_mass_retention"],
        "base_router_coupled_weighted_predicted_delta": base_metrics[
            "router_coupled_weighted_predicted_delta"
        ],
        "candidate_count": int(len(frontier)),
        "default_gate_candidate_count": int(frontier["passes_default_gate"].astype(bool).sum()),
        "minimum_effective_fraction": args.minimum_effective_fraction,
        "constrained_effect_fraction_vs_stress": effect_fraction,
        "constrained": json_safe(constrained.to_dict()),
        "stress": json_safe(stress.to_dict()),
        "outputs": {
            "retention_frontier": rel(frontier_path),
            "retention_constrained_group_rules": rel(constrained_rules_path),
            "selected_retention_constrained_candidate": rel(constrained_path),
            "selected_stress_candidate": rel(stress_path),
            "tensor_rules": rules_outputs["tensor_rules"],
            "writer_command": rules_outputs["writer_command"],
            "dry_run_command": rules_outputs["dry_run_command"],
            "checkpoint_output_dir": rules_outputs["checkpoint_output_dir"],
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, frontier, constrained_rules), encoding="utf-8")
    print(f"Wrote Qwen3 MoE router-coupled retention frontier to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; gate={gate}; constrained={constrained_id}")


if __name__ == "__main__":
    main()
