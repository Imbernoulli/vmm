#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

import build_qwen3_moe_mechanistic_unified_candidate as mechanistic


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


FEATURES: list[tuple[str, str]] = [
    ("delta_pressure", "feature_delta_pressure"),
    ("router_boundary", "feature_router_instability"),
    ("load_pressure", "feature_load_pressure"),
    ("source_conflict", "feature_source_conflict"),
    ("low_route_mass", "feature_low_route_mass"),
    ("expert_geometry", "feature_expert_route_geometry"),
    ("expert_geometry", "feature_expert_internal_geometry"),
    ("layer_geometry", "feature_layer_geometry"),
    ("subspace_conflict", "feature_subspace_conflict"),
    ("subspace_conflict", "feature_subspace_route_conflict"),
    ("benefit", "route_mass_pressure"),
    ("benefit", "coder_route_share"),
    ("benefit", "category_coder_prior"),
    ("benefit", "original_weight_coder"),
]


ABLATIONS: list[tuple[str, str]] = [
    ("baseline", "no features removed"),
    ("no_delta_pressure", "remove tensor-delta pressure from curvature"),
    ("no_router_boundary", "remove router fragility and fragile-router flags"),
    ("no_load_pressure", "remove high-load pressure and high-load flags"),
    ("no_source_conflict", "remove source-conflict pressure and mixed/category mismatch flags"),
    ("no_category_prior", "neutralize task/category prior in benefit"),
    ("no_expert_geometry", "remove route/internal/layer expert geometry risks"),
    ("no_subspace_conflict", "remove expert channel/chunk subspace conflict risks"),
    ("no_feedback_prior", "ignore feedback prior scale/action and solve from pre-feedback rules"),
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


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def fnum(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return float(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def parse_candidate_id(candidate_id: str) -> dict[str, float]:
    match = re.fullmatch(r"s(?P<step>[0-9.]+)_b(?P<benefit>[0-9.]+)_h(?P<curvature>[0-9.]+)_i(?P<interference>[0-9.]+)", candidate_id)
    if not match:
        raise ValueError(f"Cannot parse mechanistic candidate id: {candidate_id}")
    return {
        "step_size": float(match.group("step")),
        "benefit_gain": float(match.group("benefit")),
        "curvature_gain": float(match.group("curvature")),
        "interference_gain": float(match.group("interference")),
    }


def split_flags(raw: Any) -> list[str]:
    if raw is None:
        return []
    try:
        if pd.isna(raw):
            return []
    except (TypeError, ValueError):
        pass
    return [part.strip() for part in str(raw).split("|") if part.strip()]


def remove_flags(raw: Any, blocked: set[str]) -> str:
    return "|".join(flag for flag in split_flags(raw) if flag not in blocked)


def numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    denom = float(weights.sum())
    if denom <= EPS:
        return float(values.mean())
    return float((values * weights).sum() / denom)


def weighted_corr(x: pd.Series, y: pd.Series, weights: pd.Series) -> float | None:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0).astype(float)
    y = pd.to_numeric(y, errors="coerce").fillna(0.0).astype(float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    if float(weights.sum()) <= EPS:
        weights = pd.Series(1.0, index=x.index)
    mx = weighted_mean(x, weights)
    my = weighted_mean(y, weights)
    vx = weighted_mean((x - mx) ** 2, weights)
    vy = weighted_mean((y - my) ** 2, weights)
    if vx <= EPS or vy <= EPS:
        return None
    cov = weighted_mean((x - mx) * (y - my), weights)
    return float(cov / ((vx * vy) ** 0.5))


def apply_ablation(df: pd.DataFrame, ablation: str) -> pd.DataFrame:
    out = df.copy()
    if ablation == "baseline":
        return out
    if ablation == "no_delta_pressure":
        out["feature_delta_pressure"] = 0.0
    elif ablation == "no_router_boundary":
        out["feature_router_instability"] = 0.0
        out["trust_risk_flags"] = out.get("trust_risk_flags", "").map(
            lambda raw: remove_flags(raw, {"fragile_router_layer"})
        )
    elif ablation == "no_load_pressure":
        out["feature_load_pressure"] = 0.0
        out["max_topk_over_uniform"] = 0.0
        out["mean_topk_over_uniform"] = 0.0
        out["trust_risk_flags"] = out.get("trust_risk_flags", "").map(
            lambda raw: remove_flags(raw, {"high_load_expert", "high_load_weight_limit"})
        )
    elif ablation == "no_source_conflict":
        out["feature_source_conflict"] = 0.0
        out["trust_risk_flags"] = out.get("trust_risk_flags", "").map(
            lambda raw: remove_flags(raw, {"category_source_mismatch", "shared_mixed_expert"})
        )
    elif ablation == "no_category_prior":
        out["dominant_category"] = ""
        out["max_dominant_category_share"] = 0.0
        out["trust_risk_flags"] = out.get("trust_risk_flags", "").map(
            lambda raw: remove_flags(raw, {"category_source_mismatch"})
        )
    elif ablation == "no_expert_geometry":
        for column in [
            "feature_expert_route_geometry",
            "feature_expert_internal_geometry",
            "feature_layer_geometry",
            "route_geometry_risk_score",
            "internal_geometry_risk_score",
            "layer_route_geometry_risk_score",
            "geometry_combined_relative_delta",
        ]:
            out[column] = 0.0
        out["geometry_combined_cosine"] = 1.0
        out["geometry_action"] = "geometry_ablation_removed"
    elif ablation == "no_subspace_conflict":
        for column in [
            "feature_subspace_conflict",
            "feature_subspace_route_conflict",
            "feature_subspace_channel_angle",
            "feature_subspace_chunk_delta",
            "feature_subspace_uncovered_flag",
            "subspace_conflict_score",
            "subspace_route_weighted_conflict_score",
        ]:
            out[column] = 0.0
        out["subspace_recommended_relative_cap"] = 0.65
        out["subspace_extra_scale"] = 1.0
        out["subspace_action"] = "subspace_ablation_removed"
    elif ablation == "no_feedback_prior":
        out["feedback_selected_scale"] = pd.NA
        out["feedback_scale_delta"] = 0.0
        out["feedback_expected_max_relative_delta_norm"] = pd.NA
        out["feedback_action"] = ""
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    return out


def solve_fixed(
    terms: pd.DataFrame,
    *,
    candidate_id: str,
    gains: dict[str, float],
    hard_cap: float,
    min_scale: float,
) -> tuple[pd.Series, dict[str, Any]]:
    scale = mechanistic.solve_scale(
        terms,
        benefit_gain=gains["benefit_gain"],
        curvature_gain=gains["curvature_gain"],
        interference_gain=gains["interference_gain"],
        step_size=gains["step_size"],
        hard_cap=hard_cap,
        min_scale=min_scale,
    )
    metrics = mechanistic.metrics_from_scale(
        terms,
        scale,
        candidate_id=candidate_id,
        benefit_gain=gains["benefit_gain"],
        curvature_gain=gains["curvature_gain"],
        interference_gain=gains["interference_gain"],
        step_size=gains["step_size"],
        hard_cap=hard_cap,
    )
    return scale, metrics


def solve_reselected(
    terms: pd.DataFrame,
    *,
    hard_cap: float,
    min_scale: float,
    min_retention: float,
) -> tuple[pd.Series, dict[str, Any]]:
    search, scales = mechanistic.candidate_grid(terms, hard_cap=hard_cap, min_scale=min_scale)
    selected = mechanistic.select_candidate(search, min_retention=min_retention)
    candidate_id = str(selected["candidate_id"])
    return scales[candidate_id], selected.to_dict()


def summarize_scale_shift(
    *,
    terms: pd.DataFrame,
    scale: pd.Series,
    baseline_scale: pd.Series,
    hard_cap: float,
) -> dict[str, Any]:
    route_mass = numeric_column(terms, "total_topk_fraction", 0.0).clip(lower=0.0)
    delta = numeric_column(terms, "audit_max_relative_delta_norm", 0.0).clip(lower=0.0)
    shift = scale - baseline_scale
    predicted_delta = delta * scale
    cap_bound = predicted_delta >= hard_cap - 1e-6
    high_subspace = numeric_column(terms, "feature_subspace_conflict", 0.0) >= 0.72
    high_benefit_low_risk = (numeric_column(terms, "benefit_score", 0.0) >= 0.62) & (
        numeric_column(terms, "curvature_score", 0.0) <= 0.55
    )
    high_interference_low_benefit = (numeric_column(terms, "interference_score", 0.0) >= 0.62) & (
        numeric_column(terms, "benefit_score", 0.0) < 0.50
    )
    return {
        "mean_abs_scale_shift": float(shift.abs().mean()),
        "route_mass_weighted_abs_scale_shift": weighted_mean(shift.abs(), route_mass),
        "p95_abs_scale_shift": float(shift.abs().quantile(0.95)),
        "max_abs_scale_shift": float(shift.abs().max()),
        "groups_changed_gt_0_001": int((shift.abs() > 0.001).sum()),
        "groups_changed_gt_0_01": int((shift.abs() > 0.01).sum()),
        "cap_bound_group_count": int(cap_bound.sum()),
        "cap_bound_route_mass": float(route_mass.loc[cap_bound].sum()),
        "high_subspace_mean_scale": float(scale.loc[high_subspace].mean()) if bool(high_subspace.any()) else 1.0,
        "high_benefit_low_risk_mean_scale": float(scale.loc[high_benefit_low_risk].mean())
        if bool(high_benefit_low_risk.any())
        else 1.0,
        "high_interference_low_benefit_mean_scale": float(scale.loc[high_interference_low_benefit].mean())
        if bool(high_interference_low_benefit.any())
        else 1.0,
    }


def build_feature_correlations(terms: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    route_mass = numeric_column(terms, "total_topk_fraction", 0.0).clip(lower=0.0)
    prior = numeric_column(terms, "prior_scale", 1.0).clip(0.0, 1.0)
    shrink = (prior - scale).clip(lower=0.0)
    restore = (scale - prior).clip(lower=0.0)
    predicted_delta = numeric_column(terms, "audit_max_relative_delta_norm", 0.0) * scale
    rows = []
    for family, column in FEATURES:
        values = numeric_column(terms, column, 0.0)
        shrink_corr = weighted_corr(values, shrink, route_mass)
        restore_corr = weighted_corr(values, restore, route_mass)
        delta_corr = weighted_corr(values, predicted_delta, route_mass)
        rows.append(
            {
                "feature_family": family,
                "feature": column,
                "weighted_corr_with_shrink": shrink_corr,
                "weighted_corr_with_restore": restore_corr,
                "weighted_corr_with_predicted_delta": delta_corr,
                "mean_value": float(values.mean()),
                "route_mass_weighted_value": weighted_mean(values, route_mass),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_shrink_corr"] = out["weighted_corr_with_shrink"].abs()
    return out.sort_values(["abs_shrink_corr", "feature_family", "feature"], ascending=[False, True, True])


def build_affected_groups(
    terms: pd.DataFrame,
    baseline_scale: pd.Series,
    scale_by_ablation: dict[str, pd.Series],
    *,
    per_ablation: int,
) -> pd.DataFrame:
    rows = []
    for ablation, scale in scale_by_ablation.items():
        if ablation == "baseline":
            continue
        shift = scale - baseline_scale
        top_indices = shift.abs().sort_values(ascending=False).head(per_ablation).index
        for idx in top_indices:
            row = terms.loc[idx]
            rows.append(
                {
                    "ablation": ablation,
                    "layer_id": int(row["layer_id"]) if "layer_id" in terms else None,
                    "expert_id": int(row["expert_id"]) if "expert_id" in terms else None,
                    "dominant_source": row.get("dominant_source"),
                    "dominant_category": row.get("dominant_category"),
                    "route_mass": fnum(row.get("total_topk_fraction"), 0.0),
                    "baseline_scale": float(baseline_scale.loc[idx]),
                    "ablated_scale": float(scale.loc[idx]),
                    "scale_shift": float(shift.loc[idx]),
                    "abs_scale_shift": float(abs(shift.loc[idx])),
                    "audit_max_relative_delta_norm": fnum(row.get("audit_max_relative_delta_norm"), 0.0),
                    "benefit_score": fnum(row.get("benefit_score"), 0.0),
                    "curvature_score": fnum(row.get("curvature_score"), 0.0),
                    "interference_score": fnum(row.get("interference_score"), 0.0),
                    "feature_subspace_conflict": fnum(row.get("feature_subspace_conflict"), 0.0),
                    "feature_expert_internal_geometry": fnum(row.get("feature_expert_internal_geometry"), 0.0),
                    "feature_router_instability": fnum(row.get("feature_router_instability"), 0.0),
                    "trust_risk_flags": row.get("trust_risk_flags"),
                    "mechanistic_reason": row.get("mechanistic_reason"),
                }
            )
    return pd.DataFrame(rows).sort_values(["ablation", "abs_scale_shift"], ascending=[True, False])


def build_report(summary: dict[str, Any], ablations: pd.DataFrame, correlations: pd.DataFrame) -> str:
    strongest = summary.get("strongest_fixed_objective_regression") or {}
    scale = summary.get("strongest_scale_sensitivity") or {}
    lines = [
        "# Qwen3 MoE Mechanistic Sensitivity",
        "",
        "这个分析重放 mechanistic unified candidate 的同一套 `B/H/I` scale law，并逐个移除内部特征族，观察 expert scale、hard-cap 绑定、retention 和 objective proxy 如何变化。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Baseline candidate: `{summary['baseline_candidate_id']}`",
        f"- Effective hard cap: `{fmt(summary['effective_hard_cap'])}`",
        f"- Baseline fixed objective: `{fmt(summary['baseline_fixed_objective'])}`",
        f"- Baseline reselected objective: `{fmt(summary['baseline_reselected_objective'])}`",
        f"- Strongest fixed objective regression: `{strongest.get('ablation')}` (`delta={fmt(strongest.get('fixed_objective_delta'))}`)",
        f"- Strongest scale sensitivity: `{scale.get('ablation')}` (`route-mass weighted abs shift={fmt(scale.get('route_mass_weighted_abs_scale_shift'))}`)",
        "",
        "## Feature-Family Ablations",
        "",
        "| ablation | fixed obj delta | fixed retention delta | fixed risk-delta delta | changed >0.01 | cap-bound groups | reselected candidate | reselected obj delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in ablations.iterrows():
        lines.append(
            f"| `{row['ablation']}` | {fmt(row['fixed_objective_delta'], 6)} | "
            f"{fmt(row['fixed_nonbase_mass_retention_delta'], 6)} | "
            f"{fmt(row['fixed_risk_weighted_predicted_delta_delta'], 6)} | "
            f"{int(row['groups_changed_gt_0_01'])} | {int(row['cap_bound_group_count'])} | "
            f"`{row['reselected_candidate_id']}` | {fmt(row['reselected_objective_delta'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## Top Shrink Correlations",
            "",
            "| feature | family | corr with shrink | corr with restore | corr with predicted delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in correlations.head(12).iterrows():
        lines.append(
            f"| `{row['feature']}` | `{row['feature_family']}` | "
            f"{fmt(row['weighted_corr_with_shrink'], 4)} | "
            f"{fmt(row['weighted_corr_with_restore'], 4)} | "
            f"{fmt(row['weighted_corr_with_predicted_delta'], 4)} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "这里的 objective delta 是 counterfactual full-score：先移除某类特征求 scale，再放回完整 B/H/I objective 里打分。正值表示该特征族保护了当前 proxy；负值表示移除后完整 proxy 反而更好，应作为下一轮 ablation hypothesis，不能直接当成下游结论。"
    )
    lines.append(
        f"`{strongest.get('ablation')}` 是当前最大的完整目标退化来源，含义是这类信号不是装饰性规则；它改变了同构 expert average 在 benefit、curvature、interference 三项之间的可行折中。"
    )
    lines.append(
        f"`{scale.get('ablation')}` 对 scale 分布最敏感，说明 unified 算法不能只保留全局 cap，还要让局部 expert/subspace 结构进入 trust-region。"
    )
    lines.append(
        "这个结果不替代最终 vLLM selector；它只说明 B/H/I 内部哪些信号真正改变了 same-shape expert scales。"
    )
    lines.extend(["", "## Outputs", ""])
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature-family sensitivity of the Qwen3 MoE mechanistic scale law.")
    parser.add_argument(
        "--group-rules",
        type=Path,
        default=Path("results/qwen3_moe_feedback_optimizer/feedback_group_rules.csv"),
    )
    parser.add_argument(
        "--fallback-group-rules",
        type=Path,
        default=Path("results/qwen3_moe_unified_mechanism_candidate/unified_group_rules.csv"),
    )
    parser.add_argument(
        "--mechanistic-summary",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_mechanistic_sensitivity"))
    parser.add_argument("--top-groups-per-ablation", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_in = read_json(args.mechanistic_summary)
    if not summary_in:
        raise SystemExit(f"Missing mechanistic summary: {args.mechanistic_summary}")
    selected_id = str(summary_in["selected_candidate_id"])
    gains = parse_candidate_id(selected_id)
    hard_cap = float(summary_in.get("effective_hard_cap") or summary_in.get("hard_cap"))
    min_scale = float(summary_in.get("min_scale", 0.0))
    min_retention = float(summary_in.get("min_retention", 0.0))
    raw = mechanistic.read_group_rules(args.group_rules, args.fallback_group_rules)

    baseline_terms = mechanistic.add_mechanistic_terms(apply_ablation(raw, "baseline"))
    baseline_fixed_scale, baseline_fixed = solve_fixed(
        baseline_terms,
        candidate_id=selected_id,
        gains=gains,
        hard_cap=hard_cap,
        min_scale=min_scale,
    )
    baseline_reselected_scale, baseline_reselected = solve_reselected(
        baseline_terms,
        hard_cap=hard_cap,
        min_scale=min_scale,
        min_retention=min_retention,
    )

    scale_by_ablation: dict[str, pd.Series] = {"baseline": baseline_fixed_scale}
    rows = []
    for ablation, description in ABLATIONS:
        terms = mechanistic.add_mechanistic_terms(apply_ablation(raw, ablation))
        fixed_scale, fixed_ablation = solve_fixed(
            terms,
            candidate_id=selected_id,
            gains=gains,
            hard_cap=hard_cap,
            min_scale=min_scale,
        )
        fixed = mechanistic.metrics_from_scale(
            baseline_terms,
            fixed_scale,
            candidate_id=f"{ablation}:{selected_id}:full_score",
            benefit_gain=gains["benefit_gain"],
            curvature_gain=gains["curvature_gain"],
            interference_gain=gains["interference_gain"],
            step_size=gains["step_size"],
            hard_cap=hard_cap,
        )
        reselected_scale, reselected_ablation = solve_reselected(
            terms,
            hard_cap=hard_cap,
            min_scale=min_scale,
            min_retention=min_retention,
        )
        reselected_id = str(reselected_ablation["candidate_id"])
        reselected_gains = parse_candidate_id(reselected_id)
        reselected = mechanistic.metrics_from_scale(
            baseline_terms,
            reselected_scale,
            candidate_id=f"{ablation}:{reselected_id}:full_score",
            benefit_gain=reselected_gains["benefit_gain"],
            curvature_gain=reselected_gains["curvature_gain"],
            interference_gain=reselected_gains["interference_gain"],
            step_size=reselected_gains["step_size"],
            hard_cap=hard_cap,
        )
        scale_by_ablation[ablation] = fixed_scale
        shift = summarize_scale_shift(
            terms=baseline_terms,
            scale=fixed_scale,
            baseline_scale=baseline_fixed_scale,
            hard_cap=hard_cap,
        )
        rows.append(
            {
                "ablation": ablation,
                "description": description,
                "fixed_candidate_id": selected_id,
                "fixed_mechanistic_objective": float(fixed["mechanistic_objective"]),
                "fixed_objective_delta": float(fixed["mechanistic_objective"]) - float(baseline_fixed["mechanistic_objective"]),
                "fixed_ablation_objective": float(fixed_ablation["mechanistic_objective"]),
                "fixed_nonbase_mass_retention": float(fixed["nonbase_mass_retention"]),
                "fixed_nonbase_mass_retention_delta": float(fixed["nonbase_mass_retention"])
                - float(baseline_fixed["nonbase_mass_retention"]),
                "fixed_risk_weighted_predicted_delta": float(fixed["risk_weighted_predicted_delta"]),
                "fixed_risk_weighted_predicted_delta_delta": float(fixed["risk_weighted_predicted_delta"])
                - float(baseline_fixed["risk_weighted_predicted_delta"]),
                "fixed_benefit_weighted_scale": float(fixed["benefit_weighted_scale"]),
                "fixed_mean_scale": float(fixed["mean_scale"]),
                "fixed_min_scale": float(fixed["min_scale"]),
                "fixed_max_predicted_relative_delta": float(fixed["max_predicted_relative_delta"]),
                "fixed_hard_cap_violation_count": int(fixed["hard_cap_violation_count"]),
                "reselected_candidate_id": reselected_id,
                "reselected_mechanistic_objective": float(reselected["mechanistic_objective"]),
                "reselected_objective_delta": float(reselected["mechanistic_objective"])
                - float(baseline_reselected["mechanistic_objective"]),
                "reselected_ablation_objective": float(reselected_ablation["mechanistic_objective"]),
                "reselected_nonbase_mass_retention": float(reselected["nonbase_mass_retention"]),
                "reselected_risk_weighted_predicted_delta": float(reselected["risk_weighted_predicted_delta"]),
                **shift,
            }
        )
    ablations = pd.DataFrame(rows)
    ablations["is_baseline"] = ablations["ablation"].astype(str) == "baseline"
    ablations = ablations.sort_values(["is_baseline", "fixed_objective_delta"], ascending=[False, False])
    correlations = build_feature_correlations(baseline_terms, baseline_fixed_scale)
    affected = build_affected_groups(
        baseline_terms,
        baseline_fixed_scale,
        scale_by_ablation,
        per_ablation=max(1, int(args.top_groups_per_ablation)),
    )

    nonbaseline = ablations[ablations["ablation"] != "baseline"].copy()
    strongest_objective = (
        nonbaseline.sort_values("fixed_objective_delta", ascending=False).iloc[0].to_dict()
        if not nonbaseline.empty
        else {}
    )
    strongest_scale = (
        nonbaseline.sort_values("route_mass_weighted_abs_scale_shift", ascending=False).iloc[0].to_dict()
        if not nonbaseline.empty
        else {}
    )

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablations_path = output_dir / "feature_family_ablation.csv"
    correlations_path = output_dir / "feature_correlations.csv"
    affected_path = output_dir / "top_affected_groups.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    summary = {
        "schema_version": 1,
        "status": "mechanistic_sensitivity_ready",
        "baseline_candidate_id": selected_id,
        "baseline_reselected_candidate_id": str(baseline_reselected["candidate_id"]),
        "effective_hard_cap": hard_cap,
        "min_retention": min_retention,
        "ablation_count": int(len(ablations)),
        "feature_correlation_count": int(len(correlations)),
        "affected_group_rows": int(len(affected)),
        "baseline_fixed_objective": float(baseline_fixed["mechanistic_objective"]),
        "baseline_reselected_objective": float(baseline_reselected["mechanistic_objective"]),
        "baseline_nonbase_mass_retention": float(baseline_fixed["nonbase_mass_retention"]),
        "baseline_risk_weighted_predicted_delta": float(baseline_fixed["risk_weighted_predicted_delta"]),
        "strongest_fixed_objective_regression": {
            "ablation": strongest_objective.get("ablation"),
            "fixed_objective_delta": fnum(strongest_objective.get("fixed_objective_delta")),
            "fixed_nonbase_mass_retention_delta": fnum(
                strongest_objective.get("fixed_nonbase_mass_retention_delta")
            ),
            "reselected_candidate_id": strongest_objective.get("reselected_candidate_id"),
        },
        "strongest_scale_sensitivity": {
            "ablation": strongest_scale.get("ablation"),
            "route_mass_weighted_abs_scale_shift": fnum(
                strongest_scale.get("route_mass_weighted_abs_scale_shift")
            ),
            "groups_changed_gt_0_01": int(strongest_scale.get("groups_changed_gt_0_01", 0))
            if strongest_scale
            else 0,
        },
        "top_shrink_feature": correlations.iloc[0].to_dict() if not correlations.empty else {},
        "outputs": {
            "feature_family_ablation": rel(ablations_path),
            "feature_correlations": rel(correlations_path),
            "top_affected_groups": rel(affected_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    ablations.to_csv(ablations_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    affected.to_csv(affected_path, index=False)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, ablations, correlations), encoding="utf-8")
    print(f"Wrote Qwen3 MoE mechanistic sensitivity to {output_dir.resolve()}")
    print(
        "Status: "
        f"{summary['status']}; strongest={summary['strongest_fixed_objective_regression']['ablation']}; "
        f"scale={summary['strongest_scale_sensitivity']['ablation']}"
    )


if __name__ == "__main__":
    main()
