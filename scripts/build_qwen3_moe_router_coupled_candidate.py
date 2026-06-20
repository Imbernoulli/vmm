#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = "instruct"
NONBASE_SOURCE = "coder"
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


def fnum(value: Any, default: float = 0.0) -> float:
    value = clean_value(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    return f"{fnum(value):.{digits}f}"


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def robust01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.quantile(0.05))
    hi = float(values.quantile(0.95))
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    denom = float(weights.sum())
    if denom <= EPS:
        return float(values.mean())
    return float((values * weights).sum() / denom)


def extract_writer_context(command_path: Path) -> tuple[str, dict[str, str]]:
    command = repo_path(command_path).read_text(encoding="utf-8").strip()
    parts = shlex.split(command)
    base = ""
    sources: dict[str, str] = {}
    idx = 0
    while idx < len(parts):
        if parts[idx] == "--base" and idx + 1 < len(parts):
            base = parts[idx + 1]
            idx += 2
            continue
        if parts[idx] == "--source" and idx + 1 < len(parts):
            raw = parts[idx + 1]
            if "=" in raw:
                name, path = raw.split("=", 1)
                sources[name] = path
            idx += 2
            continue
        idx += 1
    if not base or BASE_SOURCE not in sources or NONBASE_SOURCE not in sources:
        raise ValueError(f"Could not recover base/source paths from {command_path}")
    return base, sources


def build_joined(group_rules: pd.DataFrame, layer_fragility: pd.DataFrame) -> pd.DataFrame:
    layer = layer_fragility[
        [
            "layer",
            "boundary_fragility_score",
            "safe_lambda_proxy",
            "unsafe_fraction",
            "router_margin_action",
        ]
    ].copy()
    out = group_rules.merge(layer, left_on="layer_id", right_on="layer", how="left")
    out["base_scale"] = numeric(out, "mechanistic_selected_scale", 1.0).clip(0.0, 1.0)
    out["route_mass"] = numeric(out, "total_topk_fraction", 0.0).clip(lower=0.0)
    out["router_coupled_pressure"] = (
        out["route_mass"]
        * numeric(out, "boundary_fragility_score", 0.0).clip(0.0, 1.0)
        * numeric(out, "feature_router_instability", 0.0).clip(0.0, 1.0)
        * (
            0.50
            + 0.25 * numeric(out, "curvature_score", 0.0).clip(0.0, 1.0)
            + 0.25 * numeric(out, "interference_score", 0.0).clip(0.0, 1.0)
        )
    )
    out["router_coupled_pressure_norm"] = robust01(out["router_coupled_pressure"])
    return out


def metrics_from_scale(frame: pd.DataFrame, scale: pd.Series, hard_cap: float) -> dict[str, Any]:
    route = frame["route_mass"]
    original_nonbase = numeric(frame, "original_weight_coder", 0.0).clip(lower=0.0)
    delta = numeric(frame, "audit_max_relative_delta_norm", 0.0).clip(lower=0.0)
    predicted_delta = delta * scale
    risk_weight = route * (
        0.50 * numeric(frame, "curvature_score", 0.0) + 0.50 * numeric(frame, "interference_score", 0.0)
    )
    coupled_weight = numeric(frame, "router_coupled_pressure", 0.0).clip(lower=0.0)
    original_mass = float((route * original_nonbase).sum())
    selected_mass = float((route * original_nonbase * scale).sum())
    hard_violation = predicted_delta > hard_cap + 1e-9
    return {
        "nonbase_mass_retention": selected_mass / max(EPS, original_mass),
        "mean_scale": float(scale.mean()),
        "min_scale": float(scale.min()),
        "max_predicted_relative_delta": float(predicted_delta.max()),
        "hard_cap_violation_count": int(hard_violation.sum()),
        "risk_weighted_predicted_delta": weighted_mean(predicted_delta, risk_weight),
        "router_coupled_weighted_scale": weighted_mean(scale, coupled_weight),
        "router_coupled_weighted_predicted_delta": weighted_mean(predicted_delta, coupled_weight),
        "changed_group_count": int((scale < frame["base_scale"] - 1e-9).sum()),
        "route_mass_weighted_scale": weighted_mean(scale, route),
    }


def build_candidate_search(
    frame: pd.DataFrame,
    *,
    hard_cap: float,
    min_retention: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows = []
    scales: dict[str, pd.Series] = {}
    base_scale = frame["base_scale"].copy()
    base_metrics = metrics_from_scale(frame, base_scale, hard_cap)
    for quantile in (0.75, 0.80, 0.85, 0.90, 0.95):
        threshold = float(frame["router_coupled_pressure_norm"].quantile(quantile))
        eligible = (
            (frame["router_coupled_pressure_norm"] >= threshold)
            & (numeric(frame, "boundary_fragility_score", 0.0) >= 0.60)
            & (numeric(frame, "feature_router_instability", 0.0) >= 0.65)
        )
        for strength in (0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0075, 0.0100):
            for cap in (0.0030, 0.0050, 0.0075, 0.0100):
                extra = (strength * frame["router_coupled_pressure_norm"]).clip(upper=cap).where(eligible, 0.0)
                scale = (base_scale - extra).clip(lower=0.0, upper=1.0)
                candidate_id = f"router_q{quantile:.2f}_s{strength:.4f}_cap{cap:.4f}"
                metrics = metrics_from_scale(frame, scale, hard_cap)
                metrics.update(
                    {
                        "candidate_id": candidate_id,
                        "pressure_quantile": quantile,
                        "extra_shrink_strength": strength,
                        "extra_shrink_cap": cap,
                        "retention_delta_vs_base": metrics["nonbase_mass_retention"]
                        - base_metrics["nonbase_mass_retention"],
                        "risk_delta_reduction_vs_base": base_metrics["risk_weighted_predicted_delta"]
                        - metrics["risk_weighted_predicted_delta"],
                        "router_coupled_delta_reduction_vs_base": base_metrics[
                            "router_coupled_weighted_predicted_delta"
                        ]
                        - metrics["router_coupled_weighted_predicted_delta"],
                        "passes_hard_cap": metrics["hard_cap_violation_count"] == 0,
                        "passes_retention_gate": metrics["nonbase_mass_retention"] >= min_retention,
                        "ablation_only": metrics["nonbase_mass_retention"] < min_retention,
                    }
                )
                rows.append(metrics)
                scales[candidate_id] = scale
    search = pd.DataFrame(rows).sort_values(
        [
            "passes_hard_cap",
            "router_coupled_delta_reduction_vs_base",
            "risk_delta_reduction_vs_base",
            "nonbase_mass_retention",
        ],
        ascending=[False, False, False, False],
    )
    return search, scales


def select_candidate(search: pd.DataFrame) -> pd.Series:
    valid = search[search["passes_hard_cap"]].copy()
    if valid.empty:
        return search.sort_values(["hard_cap_violation_count", "router_coupled_delta_reduction_vs_base"]).iloc[0]
    # This candidate is intentionally allowed to be ablation-only: its job is to test whether
    # extra router-coupled shrink buys downstream robustness beyond the current retention gate.
    return valid.sort_values(
        ["router_coupled_delta_reduction_vs_base", "risk_delta_reduction_vs_base", "nonbase_mass_retention"],
        ascending=[False, False, False],
    ).iloc[0]


def build_group_rules(frame: pd.DataFrame, scale: pd.Series, selected: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    out["router_coupled_candidate_id"] = selected["candidate_id"]
    out["router_coupled_selected_scale"] = scale
    out["router_coupled_extra_shrink"] = (out["base_scale"] - scale).clip(lower=0.0)
    out["router_coupled_weight_instruct"] = numeric(out, "mechanistic_weight_instruct", 0.0)
    out["router_coupled_weight_coder"] = numeric(out, "original_weight_coder", 0.0) * scale
    out["router_coupled_expected_max_relative_delta_norm"] = (
        numeric(out, "audit_max_relative_delta_norm", 0.0) * scale
    )
    out["router_coupled_action"] = "keep_mechanistic"
    out.loc[out["router_coupled_extra_shrink"] > 1e-9, "router_coupled_action"] = "extra_router_coupled_shrink"
    columns = [
        "layer_id",
        "expert_id",
        "dominant_source",
        "dominant_category",
        "total_topk_fraction",
        "boundary_fragility_score",
        "safe_lambda_proxy",
        "feature_router_instability",
        "router_coupled_pressure",
        "router_coupled_pressure_norm",
        "mechanistic_selected_scale",
        "router_coupled_selected_scale",
        "router_coupled_extra_shrink",
        "mechanistic_weight_instruct",
        "mechanistic_weight_coder",
        "router_coupled_weight_instruct",
        "router_coupled_weight_coder",
        "audit_max_relative_delta_norm",
        "router_coupled_expected_max_relative_delta_norm",
        "benefit_score",
        "curvature_score",
        "interference_score",
        "mechanistic_reason",
        "router_coupled_action",
        "trust_risk_flags",
        "tensor_pattern",
    ]
    return out[[column for column in columns if column in out.columns]].sort_values(["layer_id", "expert_id"])


def write_rules_and_commands(
    group_rules: pd.DataFrame,
    output_dir: Path,
    writer_context_command: Path,
    checkpoint_output_dir: str,
) -> dict[str, str]:
    base_path, sources = extract_writer_context(writer_context_command)
    rules_path = output_dir / "tensor_rules.txt"
    lines = [
        "# Router-coupled Qwen3 MoE expert ablation rules.",
        "# Starts from mechanistic B/H/I scales and applies extra shrink to high router-coupled-risk experts.",
    ]
    for _, row in group_rules.sort_values(["layer_id", "expert_id"]).iterrows():
        pattern = str(row["tensor_pattern"])
        if not pattern:
            continue
        lines.append(
            f"{pattern}::{BASE_SOURCE}={float(row['router_coupled_weight_instruct']):.6g},"
            f"{NONBASE_SOURCE}={float(row['router_coupled_weight_coder']):.6g}"
        )
    rules_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = (
        "python scripts/write_same_shape_average_checkpoint.py "
        f"--base {shlex.quote(base_path)} "
        f"--source {BASE_SOURCE}={shlex.quote(sources[BASE_SOURCE])} "
        f"--source {NONBASE_SOURCE}={shlex.quote(sources[NONBASE_SOURCE])} "
        f"--source-weight {BASE_SOURCE}=0.0 --source-weight {NONBASE_SOURCE}=0.0 "
        f"--freeze-router --tensor-rule-file {rel(rules_path)} "
        f"--output-dir {checkpoint_output_dir}"
    )
    writer_command_path = output_dir / "writer_command.txt"
    dry_run_command_path = output_dir / "dry_run_command.txt"
    writer_command_path.write_text(command + "\n", encoding="utf-8")
    dry_run_command_path.write_text(command + " --dry-run\n", encoding="utf-8")
    return {
        "tensor_rules": rel(rules_path),
        "writer_command": rel(writer_command_path),
        "dry_run_command": rel(dry_run_command_path),
        "checkpoint_output_dir": checkpoint_output_dir,
    }


def build_report(summary: dict[str, Any], search: pd.DataFrame, group_rules: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Router-Coupled Candidate",
        "",
        "这个候选不是替换当前 mechanistic unified；它是一个显式 ablation target。做法是从 `s0.08_b1.65_h0.75_i0.75` 的 B/H/I expert scale 出发，只对 router-coupled risk 最高的 experts 施加额外小幅 shrink，用来验证 MoE 的离散 dispatch 边界是否需要比当前默认规则更保守。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Selection gate: `{summary['selection_gate']}`",
        f"- Selected candidate: `{summary['selected_candidate_id']}`",
        f"- Changed groups: `{summary['selected_changed_group_count']}`",
        f"- Nonbase retention: `{fmt(summary['selected_nonbase_mass_retention'])}` (`delta={fmt(summary['selected_retention_delta_vs_mechanistic'])}`)",
        f"- Max predicted relative delta: `{fmt(summary['selected_max_predicted_relative_delta'])}`",
        f"- Router-coupled delta reduction: `{fmt(summary['selected_router_coupled_delta_reduction_vs_mechanistic'])}`",
        f"- Risk delta reduction: `{fmt(summary['selected_risk_delta_reduction_vs_mechanistic'])}`",
        f"- Mean extra shrink on changed groups: `{fmt(summary['selected_changed_group_mean_extra_shrink'])}`",
        f"- Writer candidate kind: `{summary['writer_candidate_kind']}`",
        "",
        "## Candidate Search",
        "",
        "| candidate | retention | retention delta | changed | coupled delta reduction | risk reduction | pass retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in search.head(14).iterrows():
        lines.append(
            f"| `{row['candidate_id']}` | {fmt(row['nonbase_mass_retention'])} | "
            f"{fmt(row['retention_delta_vs_base'])} | {int(row['changed_group_count'])} | "
            f"{fmt(row['router_coupled_delta_reduction_vs_base'])} | "
            f"{fmt(row['risk_delta_reduction_vs_base'])} | `{bool(row['passes_retention_gate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Top Changed Experts",
            "",
            "| layer | expert | category | route mass | base scale | coupled scale | extra shrink | pressure | reason |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    changed = group_rules[group_rules["router_coupled_extra_shrink"] > 1e-9].sort_values(
        "router_coupled_pressure", ascending=False
    )
    for _, row in changed.head(16).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | `{row['dominant_category']}` | "
            f"{fmt(row['total_topk_fraction'])} | {fmt(row['mechanistic_selected_scale'])} | "
            f"{fmt(row['router_coupled_selected_scale'])} | {fmt(row['router_coupled_extra_shrink'])} | "
            f"{fmt(row['router_coupled_pressure'])} | `{row['mechanistic_reason']}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "这个 ablation 降低了 router-coupled risk proxy，但 retention 低于当前 mechanistic solver 的 `0.965` gate，因此不应成为默认 unified candidate。它的用途是进入 mechanism/ablation eval queue，检验额外 router-boundary 保守性是否能换来真实下游稳健性。"
    )
    lines.append(
        "如果 vLLM 下游评测没有收益，应保持当前 B/H/I；如果收益显著，再把 router-coupled shrink 作为可条件开启的 MoE-specific term。"
    )
    lines.extend(["", "## Outputs", ""])
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a router-coupled Qwen3 MoE same-shape ablation candidate.")
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_router_coupled_candidate"))
    parser.add_argument(
        "--checkpoint-output-dir",
        default="results/checkpoints/qwen3_moe_router_coupled_candidate",
    )
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
    search, scales = build_candidate_search(joined, hard_cap=hard_cap, min_retention=min_retention)
    selected = select_candidate(search)
    selected_id = str(selected["candidate_id"])
    selected_scale = scales[selected_id]
    candidate_rules = build_group_rules(joined, selected_scale, selected)
    base_metrics = metrics_from_scale(joined, joined["base_scale"], hard_cap)
    selected_metrics = metrics_from_scale(joined, selected_scale, hard_cap)
    changed = candidate_rules[candidate_rules["router_coupled_extra_shrink"] > 1e-9]
    selection_gate = "ablation_only_waiting_vllm"
    if bool(selected["passes_retention_gate"]):
        selection_gate = "candidate_waiting_vllm"

    rules_outputs = write_rules_and_commands(
        candidate_rules,
        output_dir,
        args.writer_context_command,
        args.checkpoint_output_dir,
    )
    search_path = output_dir / "candidate_search.csv"
    group_rules_path = output_dir / "router_coupled_group_rules.csv"
    selected_path = output_dir / "selected_candidate.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    search.to_csv(search_path, index=False)
    candidate_rules.to_csv(group_rules_path, index=False)

    summary = {
        "schema_version": 1,
        "status": "router_coupled_candidate_ready",
        "selection_gate": selection_gate,
        "writer_candidate_kind": "same_shape_router_frozen_expert_ablation",
        "base_mechanistic_candidate_id": mechanistic_summary.get("selected_candidate_id"),
        "selected_candidate_id": selected_id,
        "candidate_count": int(len(search)),
        "expert_group_count": int(len(candidate_rules)),
        "effective_hard_cap": hard_cap,
        "min_retention": min_retention,
        "base_nonbase_mass_retention": base_metrics["nonbase_mass_retention"],
        "selected_nonbase_mass_retention": selected_metrics["nonbase_mass_retention"],
        "selected_retention_delta_vs_mechanistic": selected_metrics["nonbase_mass_retention"]
        - base_metrics["nonbase_mass_retention"],
        "selected_changed_group_count": int(len(changed)),
        "selected_changed_group_mean_extra_shrink": float(changed["router_coupled_extra_shrink"].mean())
        if len(changed)
        else 0.0,
        "selected_max_extra_shrink": float(candidate_rules["router_coupled_extra_shrink"].max()),
        "selected_max_predicted_relative_delta": selected_metrics["max_predicted_relative_delta"],
        "selected_hard_cap_violation_count": selected_metrics["hard_cap_violation_count"],
        "base_router_coupled_weighted_predicted_delta": base_metrics[
            "router_coupled_weighted_predicted_delta"
        ],
        "selected_router_coupled_weighted_predicted_delta": selected_metrics[
            "router_coupled_weighted_predicted_delta"
        ],
        "selected_router_coupled_delta_reduction_vs_mechanistic": base_metrics[
            "router_coupled_weighted_predicted_delta"
        ]
        - selected_metrics["router_coupled_weighted_predicted_delta"],
        "base_risk_weighted_predicted_delta": base_metrics["risk_weighted_predicted_delta"],
        "selected_risk_weighted_predicted_delta": selected_metrics["risk_weighted_predicted_delta"],
        "selected_risk_delta_reduction_vs_mechanistic": base_metrics["risk_weighted_predicted_delta"]
        - selected_metrics["risk_weighted_predicted_delta"],
        "outputs": {
            "candidate_search": rel(search_path),
            "router_coupled_group_rules": rel(group_rules_path),
            "selected_candidate": rel(selected_path),
            "tensor_rules": rules_outputs["tensor_rules"],
            "writer_command": rules_outputs["writer_command"],
            "dry_run_command": rules_outputs["dry_run_command"],
            "checkpoint_output_dir": rules_outputs["checkpoint_output_dir"],
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    selected_path.write_text(json.dumps(json_safe(selected.to_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, search, candidate_rules), encoding="utf-8")

    print(f"Wrote Qwen3 MoE router-coupled candidate to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; gate={selection_gate}; selected={selected_id}")


if __name__ == "__main__":
    main()
