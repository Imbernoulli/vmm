#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    total = float(weights.sum())
    if total <= EPS:
        return float(values.mean()) if len(values) else 0.0
    return float((values * weights).sum() / total)


def spearman(x: pd.Series, y: pd.Series) -> float:
    x_rank = pd.to_numeric(x, errors="coerce").fillna(0.0).rank(method="average")
    y_rank = pd.to_numeric(y, errors="coerce").fillna(0.0).rank(method="average")
    value = x_rank.corr(y_rank)
    if value is None or math.isnan(float(value)):
        return 0.0
    return float(value)


def decile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    ranks = values.rank(method="first", pct=True)
    return (ranks.mul(10).apply(math.ceil).clip(1, 10)).astype(int)


def add_mechanistic_terms(df: pd.DataFrame, selected: dict[str, Any], hard_cap: float) -> pd.DataFrame:
    out = df.copy()
    benefit_gain = float(selected.get("benefit_gain", 1.0))
    curvature_gain = float(selected.get("curvature_gain", 1.0))
    interference_gain = float(selected.get("interference_gain", 1.0))
    out["route_mass"] = numeric(out, "total_topk_fraction")
    out["prior_scale"] = numeric(out, "prior_scale", 1.0).clip(0.0, 1.0)
    out["selected_scale"] = numeric(out, "mechanistic_selected_scale", 1.0).clip(0.0, 1.0)
    out["scale_delta"] = out["selected_scale"] - out["prior_scale"]
    out["delta_norm"] = numeric(out, "audit_max_relative_delta_norm")
    out["selected_expected_delta_norm"] = out["delta_norm"] * out["selected_scale"]
    out["prior_expected_delta_norm"] = out["delta_norm"] * out["prior_scale"]
    out["benefit_term"] = benefit_gain * numeric(out, "marginal_benefit_proxy", 0.0)
    out["curvature_delta_term"] = (
        curvature_gain * numeric(out, "local_curvature_proxy", 0.0) * out["delta_norm"].pow(2)
    )
    out["interference_term"] = interference_gain * numeric(out, "marginal_interference_proxy", 0.0)
    out["solver_cost_term"] = out["curvature_delta_term"] + 2.0 * out["interference_term"]
    out["solver_gradient_at_prior"] = out["benefit_term"] - out["solver_cost_term"] * out["prior_scale"]
    out["objective_gradient_at_prior"] = out["solver_gradient_at_prior"]
    out["solver_fixed_point_scale"] = (
        out["benefit_term"] / out["solver_cost_term"].clip(lower=EPS)
    ).clip(0.0, 2.0)
    out["objective_unconstrained_scale"] = (
        out["benefit_term"] / out["solver_cost_term"].clip(lower=EPS)
    ).clip(0.0, 2.0)
    out["hard_cap_scale"] = (hard_cap / out["delta_norm"].clip(lower=EPS)).clip(0.0, 1.0)
    out["hard_cap_bound"] = (
        (out["selected_expected_delta_norm"] >= hard_cap - 1e-6)
        | (out["selected_scale"] >= out["hard_cap_scale"] - 1e-6)
    )
    out["prior_objective_proxy"] = (
        0.5 * out["curvature_delta_term"] * out["prior_scale"].pow(2)
        + out["interference_term"] * out["prior_scale"].pow(2)
        - out["benefit_term"] * out["prior_scale"]
    )
    out["selected_objective_proxy"] = (
        0.5 * out["curvature_delta_term"] * out["selected_scale"].pow(2)
        + out["interference_term"] * out["selected_scale"].pow(2)
        - out["benefit_term"] * out["selected_scale"]
    )
    out["objective_proxy_gain_vs_prior"] = out["prior_objective_proxy"] - out["selected_objective_proxy"]
    out["gradient_sign_agrees_with_scale"] = (
        ((out["solver_gradient_at_prior"] < -1e-9) & (out["selected_scale"] <= out["prior_scale"] + 1e-9))
        | ((out["solver_gradient_at_prior"] > 1e-9) & (out["selected_scale"] >= out["prior_scale"] - 1e-9))
        | (out["solver_gradient_at_prior"].abs() <= 1e-9)
    )
    out["mechanistic_binding"] = "damped_preserve"
    out.loc[out["hard_cap_bound"], "mechanistic_binding"] = "hard_cap_bound"
    out.loc[
        (~out["hard_cap_bound"])
        & (out["solver_gradient_at_prior"] < 0.0)
        & (out["selected_scale"] < out["prior_scale"] - 1e-6),
        "mechanistic_binding",
    ] = "cost_gradient_shrink"
    out.loc[
        (~out["hard_cap_bound"])
        & (out["solver_gradient_at_prior"] > 0.0)
        & (out["selected_scale"] > out["prior_scale"] + 1e-6),
        "mechanistic_binding",
    ] = "benefit_gradient_restore"
    subspace = numeric(out, "feature_subspace_conflict")
    router = numeric(out, "feature_router_instability")
    out.loc[
        (out["mechanistic_binding"] == "damped_preserve")
        & (subspace >= 0.72)
        & (out["selected_scale"] < out["prior_scale"] - 1e-6),
        "mechanistic_binding",
    ] = "subspace_conflict_shrink"
    out.loc[
        (out["mechanistic_binding"] == "damped_preserve")
        & (router >= 0.50)
        & (out["selected_scale"] < out["prior_scale"] - 1e-6),
        "mechanistic_binding",
    ] = "router_boundary_shrink"
    return out


def summarize_bindings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for binding, group in df.groupby("mechanistic_binding", sort=False):
        weights = group["route_mass"]
        rows.append(
            {
                "mechanistic_binding": binding,
                "group_count": int(len(group)),
                "route_mass_sum": float(weights.sum()),
                "mean_prior_scale": float(group["prior_scale"].mean()),
                "mean_selected_scale": float(group["selected_scale"].mean()),
                "mean_scale_delta": float(group["scale_delta"].mean()),
                "mean_benefit_term": weighted_mean(group["benefit_term"], weights),
                "mean_curvature_delta_term": weighted_mean(group["curvature_delta_term"], weights),
                "mean_interference_term": weighted_mean(group["interference_term"], weights),
                "mean_solver_gradient_at_prior": weighted_mean(group["solver_gradient_at_prior"], weights),
                "mean_selected_expected_delta_norm": weighted_mean(
                    group["selected_expected_delta_norm"], weights
                ),
                "hard_cap_bound_fraction": float(group["hard_cap_bound"].mean()),
                "objective_gain_vs_prior": weighted_mean(group["objective_proxy_gain_vs_prior"], weights),
            }
        )
    return pd.DataFrame(rows).sort_values(["route_mass_sum", "group_count"], ascending=[False, False])


def summarize_feature_deciles(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = numeric(df, feature)
        bins = decile(values)
        for bucket, group in df.assign(_feature_value=values, _decile=bins).groupby("_decile"):
            weights = group["route_mass"]
            rows.append(
                {
                    "feature": feature,
                    "decile": int(bucket),
                    "group_count": int(len(group)),
                    "route_mass_sum": float(weights.sum()),
                    "mean_feature_value": weighted_mean(group["_feature_value"], weights),
                    "mean_selected_scale": weighted_mean(group["selected_scale"], weights),
                    "mean_scale_delta": weighted_mean(group["scale_delta"], weights),
                    "mean_selected_expected_delta_norm": weighted_mean(
                        group["selected_expected_delta_norm"], weights
                    ),
                    "hard_cap_bound_fraction": float(group["hard_cap_bound"].mean()),
                    "mean_benefit_term": weighted_mean(group["benefit_term"], weights),
                    "mean_curvature_delta_term": weighted_mean(group["curvature_delta_term"], weights),
                    "mean_interference_term": weighted_mean(group["interference_term"], weights),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "decile"])


def summarize_feature_correlations(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "spearman_feature_vs_selected_scale": spearman(numeric(df, feature), df["selected_scale"]),
                "spearman_feature_vs_scale_delta": spearman(numeric(df, feature), df["scale_delta"]),
                "spearman_feature_vs_expected_delta": spearman(
                    numeric(df, feature), df["selected_expected_delta_norm"]
                ),
                "route_mass_weighted_feature": weighted_mean(numeric(df, feature), df["route_mass"]),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_feature_vs_selected_scale")


def build_hard_cases(df: pd.DataFrame) -> pd.DataFrame:
    cases = []
    specs = [
        (
            "high_benefit_but_shrunk",
            (numeric(df, "benefit_score") >= 0.62) & (df["selected_scale"] < df["prior_scale"] - 0.05),
            ["route_mass", "benefit_term"],
        ),
        (
            "high_interference_preserved",
            (numeric(df, "interference_score") >= 0.62) & (df["selected_scale"] >= 0.95),
            ["route_mass", "interference_term"],
        ),
        (
            "high_subspace_cap_bound",
            (numeric(df, "feature_subspace_conflict") >= 0.72) & df["hard_cap_bound"],
            ["route_mass", "feature_subspace_conflict"],
        ),
        (
            "benefit_gradient_restored",
            df["mechanistic_binding"].eq("benefit_gradient_restore"),
            ["route_mass", "solver_gradient_at_prior"],
        ),
        (
            "cost_gradient_shrunk",
            df["mechanistic_binding"].eq("cost_gradient_shrink"),
            ["route_mass", "solver_gradient_at_prior"],
        ),
    ]
    keep = [
        "case_type",
        "layer_id",
        "expert_id",
        "total_topk_fraction",
        "dominant_source",
        "dominant_category",
        "prior_scale",
        "selected_scale",
        "scale_delta",
        "delta_norm",
        "selected_expected_delta_norm",
        "benefit_score",
        "curvature_score",
        "interference_score",
        "benefit_term",
        "curvature_delta_term",
        "interference_term",
        "solver_gradient_at_prior",
        "objective_unconstrained_scale",
        "hard_cap_scale",
        "mechanistic_binding",
        "mechanistic_reason",
        "trust_risk_flags",
        "tensor_pattern",
    ]
    for case_type, mask, sort_cols in specs:
        subset = df[mask].copy()
        if subset.empty:
            continue
        subset["case_type"] = case_type
        subset = subset.sort_values(sort_cols, ascending=False).head(40)
        cases.append(subset[[column for column in keep if column in subset.columns]])
    return pd.concat(cases, ignore_index=True) if cases else pd.DataFrame(columns=keep)


def build_summary(
    df: pd.DataFrame,
    selected: dict[str, Any],
    candidate_summary: dict[str, Any],
    binding_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    route_mass = df["route_mass"]
    positive_gain = df["objective_proxy_gain_vs_prior"] > 0.0
    top_negative = correlations.sort_values("spearman_feature_vs_selected_scale").head(3)
    top_positive = correlations.sort_values("spearman_feature_vs_selected_scale", ascending=False).head(3)
    return {
        "status": "mechanistic_evidence_audit_ready",
        "selected_candidate_id": selected.get("candidate_id"),
        "group_count": int(len(df)),
        "route_mass_sum": float(route_mass.sum()),
        "hard_cap": float(
            candidate_summary.get(
                "effective_hard_cap",
                candidate_summary.get("hard_cap", selected.get("max_predicted_relative_delta", 0.65)),
            )
        ),
        "nominal_hard_cap": clean_value(candidate_summary.get("nominal_hard_cap")),
        "materialization_safety_margin": clean_value(candidate_summary.get("materialization_safety_margin")),
        "effective_hard_cap": clean_value(
            candidate_summary.get("effective_hard_cap", candidate_summary.get("hard_cap"))
        ),
        "hard_cap_bound_group_count": int(df["hard_cap_bound"].sum()),
        "hard_cap_bound_route_mass": float(route_mass[df["hard_cap_bound"]].sum()),
        "gradient_sign_agreement_rate": float(df["gradient_sign_agrees_with_scale"].mean()),
        "objective_proxy_improved_group_fraction": float(positive_gain.mean()),
        "route_mass_weighted_objective_gain_vs_prior": weighted_mean(
            df["objective_proxy_gain_vs_prior"], route_mass
        ),
        "route_mass_weighted_selected_scale": weighted_mean(df["selected_scale"], route_mass),
        "route_mass_weighted_scale_delta": weighted_mean(df["scale_delta"], route_mass),
        "route_mass_weighted_benefit_term": weighted_mean(df["benefit_term"], route_mass),
        "route_mass_weighted_curvature_delta_term": weighted_mean(df["curvature_delta_term"], route_mass),
        "route_mass_weighted_interference_term": weighted_mean(df["interference_term"], route_mass),
        "dominant_binding": binding_summary.iloc[0]["mechanistic_binding"] if not binding_summary.empty else None,
        "dominant_binding_group_count": int(binding_summary.iloc[0]["group_count"])
        if not binding_summary.empty
        else 0,
        "dominant_binding_route_mass": float(binding_summary.iloc[0]["route_mass_sum"])
        if not binding_summary.empty
        else 0.0,
        "most_scale_suppressing_features": [clean_value(item) for item in top_negative["feature"].tolist()],
        "most_scale_preserving_features": [clean_value(item) for item in top_positive["feature"].tolist()],
        "outputs": {
            "report": rel(output_dir / "report.md"),
            "summary": rel(output_dir / "summary.json"),
            "group_evidence": rel(output_dir / "group_mechanistic_evidence.csv"),
            "binding_summary": rel(output_dir / "binding_summary.csv"),
            "feature_deciles": rel(output_dir / "feature_decile_response.csv"),
            "feature_correlations": rel(output_dir / "feature_correlations.csv"),
            "hard_cases": rel(output_dir / "hard_cases.csv"),
        },
    }


def build_report(
    summary: dict[str, Any],
    binding_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    hard_cases: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen3 MoE Mechanistic Evidence Audit",
        "",
        "这个审计把 mechanistic unified candidate 的每个 routed expert group 拆成 `B/H/I`、梯度、hard-cap 绑定和 scale 响应。它不是再报一个算法名，而是检查当前统一 scale law 是否真的由内部参数驱动。",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Selected candidate: `{summary['selected_candidate_id']}`",
        f"- Expert groups: `{summary['group_count']}`",
        f"- Nominal/effective hard cap: `{fmt(summary['nominal_hard_cap'])}` / `{fmt(summary['effective_hard_cap'])}`",
        f"- Materialization safety margin: `{fmt(summary['materialization_safety_margin'])}`",
        f"- Hard-cap bound groups: `{summary['hard_cap_bound_group_count']}`",
        f"- Gradient/sign agreement: `{fmt(summary['gradient_sign_agreement_rate'])}`",
        f"- Objective proxy improved groups: `{fmt(summary['objective_proxy_improved_group_fraction'])}`",
        f"- Route-mass weighted objective gain vs prior: `{fmt(summary['route_mass_weighted_objective_gain_vs_prior'])}`",
        f"- Route-mass weighted selected scale: `{fmt(summary['route_mass_weighted_selected_scale'])}`",
        f"- Dominant binding: `{summary['dominant_binding']}`",
        "",
        "## Local Scale Law",
        "",
        "当前同结构 average 被写成每个 expert group 的局部二次近似：",
        "",
        "`J_g(s) = 1/2 H_g ||delta_g||^2 s^2 + I_g s^2 - B_g s`",
        "",
        "`B_g` 是保留非 base source 的收益代理量；`H_g ||delta_g||^2` 是沿当前权重方向移动的局部曲率成本；`I_g` 是 source conflict、router/top-k 边界、load 与 subspace conflict 的干扰成本。若没有 damping、retention 和 hard cap，驻点是 `s*=B/(H||delta||^2+2I)`；当前候选使用更保守的一步更新，并把 `s * ||delta|| <= hard_cap` 当成硬约束。",
        "",
        "## Binding Summary",
        "",
        "| binding | groups | route mass | mean scale | scale delta | B | Hdelta2 | I | grad@prior | cap frac | obj gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in binding_summary.head(10).iterrows():
        lines.append(
            "| "
            f"`{row['mechanistic_binding']}` | "
            f"{int(row['group_count'])} | "
            f"{fmt(row['route_mass_sum'])} | "
            f"{fmt(row['mean_selected_scale'])} | "
            f"{fmt(row['mean_scale_delta'])} | "
            f"{fmt(row['mean_benefit_term'])} | "
            f"{fmt(row['mean_curvature_delta_term'])} | "
            f"{fmt(row['mean_interference_term'])} | "
            f"{fmt(row['mean_solver_gradient_at_prior'])} | "
            f"{fmt(row['hard_cap_bound_fraction'])} | "
            f"{fmt(row['objective_gain_vs_prior'])} |"
        )
    lines.extend(
        [
            "",
            "## Feature Response",
            "",
            "| feature | corr(scale) | corr(delta scale) | route-weighted feature |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in correlations.iterrows():
        lines.append(
            "| "
            f"`{row['feature']}` | "
            f"{fmt(row['spearman_feature_vs_selected_scale'])} | "
            f"{fmt(row['spearman_feature_vs_scale_delta'])} | "
            f"{fmt(row['route_mass_weighted_feature'])} |"
        )
    lines.extend(
        [
            "",
            "## Hard Cases",
            "",
            f"- Rows exported: `{len(hard_cases)}`",
            f"- File: `{summary['outputs']['hard_cases']}`",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(repo_path(args.group_rules))
    selected = read_json(args.selected_candidate)
    candidate_summary = read_json(args.candidate_summary)
    hard_cap = float(
        args.hard_cap
        if args.hard_cap is not None
        else candidate_summary.get("effective_hard_cap", candidate_summary.get("hard_cap", 0.65))
    )
    evidence = add_mechanistic_terms(groups, selected, hard_cap)
    features = [
        "benefit_score",
        "curvature_score",
        "interference_score",
        "route_mass_pressure",
        "coder_route_share",
        "delta_pressure",
        "feature_subspace_conflict",
        "feature_subspace_route_conflict",
        "feature_expert_route_geometry",
        "feature_expert_internal_geometry",
        "feature_router_instability",
        "load_pressure",
    ]
    binding_summary = summarize_bindings(evidence)
    deciles = summarize_feature_deciles(evidence, features)
    correlations = summarize_feature_correlations(evidence, features)
    hard_cases = build_hard_cases(evidence)
    summary = build_summary(evidence, selected, candidate_summary, binding_summary, correlations, output_dir)
    evidence.to_csv(output_dir / "group_mechanistic_evidence.csv", index=False)
    binding_summary.to_csv(output_dir / "binding_summary.csv", index=False)
    deciles.to_csv(output_dir / "feature_decile_response.csv", index=False)
    correlations.to_csv(output_dir / "feature_correlations.csv", index=False)
    hard_cases.to_csv(output_dir / "hard_cases.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(summary, binding_summary, correlations, hard_cases), encoding="utf-8")
    print(
        "Status: "
        f"{summary['status']}; "
        f"gradient_agreement={summary['gradient_sign_agreement_rate']:.4f}; "
        f"hard_cap_bound={summary['hard_cap_bound_group_count']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit mechanistic evidence behind the Qwen3 MoE scale law.")
    parser.add_argument(
        "--group-rules",
        default="results/qwen3_moe_mechanistic_unified_candidate/mechanistic_group_rules.csv",
    )
    parser.add_argument(
        "--selected-candidate",
        default="results/qwen3_moe_mechanistic_unified_candidate/selected_candidate.json",
    )
    parser.add_argument(
        "--candidate-summary",
        default="results/qwen3_moe_mechanistic_unified_candidate/summary.json",
    )
    parser.add_argument("--hard-cap", type=float, default=None)
    parser.add_argument("--output-dir", default="results/qwen3_moe_mechanistic_evidence_audit")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
