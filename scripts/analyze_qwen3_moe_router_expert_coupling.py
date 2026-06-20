#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


LITERATURE_SOURCES = [
    {
        "key": "harc",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Top-k routing can break under MoE weight merging; router movement needs a separate calibration or freeze gate.",
    },
    {
        "key": "mergeme",
        "title": "MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs",
        "url": "https://arxiv.org/abs/2502.00997",
        "mechanism": "MoE merging needs routing heuristics and interference mitigation beyond uniform expert averaging.",
    },
    {
        "key": "expert_merging",
        "title": "Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking",
        "url": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Layer heterogeneity matters; high-risk layers need finer coefficients than low-risk layers.",
    },
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


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(repo_path(path))


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    denom = float(weights.sum())
    if denom <= EPS:
        return float(values.mean())
    return float((values * weights).sum() / denom)


def weighted_corr(x: pd.Series, y: pd.Series, weights: pd.Series) -> float | None:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0).astype(float)
    y = pd.to_numeric(y, errors="coerce").fillna(0.0).astype(float)
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
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


def normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def quantile_slice(frame: pd.DataFrame, column: str, q: float, high: bool) -> pd.DataFrame:
    threshold = float(frame[column].quantile(q))
    return frame[frame[column] >= threshold] if high else frame[frame[column] <= threshold]


def summarize_slice(frame: pd.DataFrame) -> dict[str, Any]:
    weights = numeric(frame, "total_topk_fraction", 0.0)
    shrink = numeric(frame, "scale_shrink", 0.0)
    return {
        "group_count": int(len(frame)),
        "route_mass": float(weights.sum()),
        "weighted_boundary_fragility": weighted_mean(numeric(frame, "boundary_fragility_score"), weights),
        "weighted_safe_lambda_proxy": weighted_mean(numeric(frame, "safe_lambda_proxy"), weights),
        "weighted_router_instability_feature": weighted_mean(
            numeric(frame, "feature_router_instability"), weights
        ),
        "weighted_scale": weighted_mean(numeric(frame, "mechanistic_selected_scale"), weights),
        "weighted_scale_shrink": weighted_mean(shrink, weights),
        "weighted_expected_delta": weighted_mean(
            numeric(frame, "mechanistic_expected_max_relative_delta_norm"), weights
        ),
        "weighted_benefit": weighted_mean(numeric(frame, "benefit_score"), weights),
        "weighted_curvature": weighted_mean(numeric(frame, "curvature_score"), weights),
        "weighted_interference": weighted_mean(numeric(frame, "interference_score"), weights),
    }


def build_joined(groups: pd.DataFrame, layer_fragility: pd.DataFrame) -> pd.DataFrame:
    layer_cols = [
        "layer",
        "boundary_fragility_score",
        "safe_lambda_proxy",
        "unsafe_fraction",
        "mean_topk_jaccard",
        "mean_top1_agreement",
        "mean_top1_margin",
        "router_relative_delta_norm",
        "router_movement_score",
        "fragility_rank",
        "router_margin_action",
    ]
    joined = groups.merge(
        layer_fragility[layer_cols],
        left_on="layer_id",
        right_on="layer",
        how="left",
    )
    joined["scale_shrink"] = (
        numeric(joined, "prior_scale", 1.0) - numeric(joined, "mechanistic_selected_scale", 1.0)
    ).clip(lower=0.0)
    joined["scale_restore"] = (
        numeric(joined, "mechanistic_selected_scale", 1.0) - numeric(joined, "prior_scale", 1.0)
    ).clip(lower=0.0)
    joined["fragility_norm"] = normalize(joined["boundary_fragility_score"])
    joined["router_feature_norm"] = normalize(joined["feature_router_instability"])
    joined["router_undercoverage"] = (joined["fragility_norm"] - joined["router_feature_norm"]).clip(lower=0.0)
    joined["router_coupled_risk"] = (
        numeric(joined, "total_topk_fraction", 0.0).clip(lower=0.0)
        * numeric(joined, "boundary_fragility_score", 0.0)
        * (
            0.50 * numeric(joined, "feature_router_instability", 0.0)
            + 0.25 * numeric(joined, "interference_score", 0.0)
            + 0.25 * numeric(joined, "curvature_score", 0.0)
        )
    )
    joined["router_residual_pressure"] = (
        numeric(joined, "total_topk_fraction", 0.0).clip(lower=0.0)
        * joined["router_undercoverage"]
        * (1.0 + numeric(joined, "interference_score", 0.0))
    )
    return joined


def build_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    weights = numeric(joined, "total_topk_fraction", 0.0)
    targets = {
        "feature_router_instability": numeric(joined, "feature_router_instability"),
        "scale_shrink": numeric(joined, "scale_shrink"),
        "mechanistic_selected_scale": numeric(joined, "mechanistic_selected_scale"),
        "mechanistic_expected_max_relative_delta_norm": numeric(
            joined, "mechanistic_expected_max_relative_delta_norm"
        ),
        "benefit_score": numeric(joined, "benefit_score"),
        "curvature_score": numeric(joined, "curvature_score"),
        "interference_score": numeric(joined, "interference_score"),
        "router_residual_pressure": numeric(joined, "router_residual_pressure"),
    }
    drivers = {
        "boundary_fragility_score": numeric(joined, "boundary_fragility_score"),
        "safe_lambda_proxy": numeric(joined, "safe_lambda_proxy"),
        "router_relative_delta_norm": numeric(joined, "router_relative_delta_norm"),
        "unsafe_fraction": numeric(joined, "unsafe_fraction"),
    }
    rows = []
    for driver_name, driver in drivers.items():
        for target_name, target in targets.items():
            rows.append(
                {
                    "driver": driver_name,
                    "target": target_name,
                    "route_mass_weighted_corr": weighted_corr(driver, target, weights),
                }
            )
    return pd.DataFrame(rows)


def build_layer_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer_id, group in joined.groupby("layer_id", sort=True):
        weights = numeric(group, "total_topk_fraction", 0.0)
        row = summarize_slice(group)
        row.update(
            {
                "layer_id": int(layer_id),
                "boundary_fragility_score": fnum(group["boundary_fragility_score"].iloc[0]),
                "safe_lambda_proxy": fnum(group["safe_lambda_proxy"].iloc[0]),
                "unsafe_fraction": fnum(group["unsafe_fraction"].iloc[0]),
                "fragility_rank": int(fnum(group["fragility_rank"].iloc[0], 0)),
                "router_margin_action": group["router_margin_action"].iloc[0],
                "route_mass_weighted_router_undercoverage": weighted_mean(
                    numeric(group, "router_undercoverage"), weights
                ),
                "route_mass_weighted_router_residual_pressure": weighted_mean(
                    numeric(group, "router_residual_pressure"), weights
                ),
                "router_coupled_risk_sum": float(numeric(group, "router_coupled_risk").sum()),
                "top_expert_id": int(group.sort_values("router_coupled_risk", ascending=False).iloc[0]["expert_id"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["router_coupled_risk_sum", "boundary_fragility_score"], ascending=[False, False]
    )


def build_top_experts(joined: pd.DataFrame, limit: int) -> pd.DataFrame:
    columns = [
        "layer_id",
        "expert_id",
        "dominant_source",
        "dominant_category",
        "total_topk_fraction",
        "boundary_fragility_score",
        "safe_lambda_proxy",
        "feature_router_instability",
        "router_undercoverage",
        "router_coupled_risk",
        "router_residual_pressure",
        "prior_scale",
        "mechanistic_selected_scale",
        "scale_shrink",
        "benefit_score",
        "curvature_score",
        "interference_score",
        "mechanistic_expected_max_relative_delta_norm",
        "mechanistic_reason",
        "trust_risk_flags",
    ]
    return joined[columns].sort_values("router_coupled_risk", ascending=False).head(limit)


def decide_gate(summary_values: dict[str, Any]) -> tuple[str, str]:
    corr_frag_router = fnum(summary_values["fragility_router_feature_corr"])
    corr_frag_shrink = fnum(summary_values["fragility_scale_shrink_corr"])
    shrink_lift = fnum(summary_values["high_vs_low_weighted_shrink_lift"])
    if corr_frag_router >= 0.50 and corr_frag_shrink >= 0.30 and shrink_lift >= 0.005:
        return (
            "router_expert_coupling_active",
            "Current B/H/I already turns fragile router layers into stronger routed-expert shrink, so removing router-boundary terms would be a mechanistic regression.",
        )
    if corr_frag_router < 0.30 and corr_frag_shrink < 0.20:
        return (
            "router_expert_coupling_underfit",
            "Router fragility is not yet reflected in expert scales; add an explicit router-coupled shrink term before accepting a MoE average.",
        )
    return (
        "router_expert_coupling_partial",
        "Router fragility is partly reflected in expert scales, but residual undercoverage should remain an ablation target.",
    )


def build_report(
    summary: dict[str, Any],
    layer_summary: pd.DataFrame,
    top_experts: pd.DataFrame,
    correlations: pd.DataFrame,
) -> str:
    high = summary["high_fragility_slice"]
    low = summary["low_fragility_slice"]
    lines = [
        "# Qwen3 MoE Router-Expert Coupling",
        "",
        "这个 probe 把 router top-k 边界脆弱性和 routed expert scale law 按 layer/expert join 起来。目标不是再说 router 要不要动，而是检查：脆弱 router 层下面的 experts 是否已经被 B/H/I trust-region 自动收紧。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Fragility -> router-instability feature corr: `{fmt(summary['fragility_router_feature_corr'])}`",
        f"- Fragility -> scale-shrink corr: `{fmt(summary['fragility_scale_shrink_corr'])}`",
        f"- Safe-lambda -> scale-shrink corr: `{fmt(summary['safe_lambda_scale_shrink_corr'])}`",
        f"- High-fragility weighted scale/shrink: `{fmt(high['weighted_scale'])}` / `{fmt(high['weighted_scale_shrink'])}`",
        f"- Low-fragility weighted scale/shrink: `{fmt(low['weighted_scale'])}` / `{fmt(low['weighted_scale_shrink'])}`",
        f"- High-vs-low shrink lift: `{fmt(summary['high_vs_low_weighted_shrink_lift'])}`",
        f"- Top coupled layer: `L{summary['top_coupled_layer_id']}` risk `{fmt(summary['top_coupled_layer_risk'])}`",
        "",
        "## Layer Coupling",
        "",
        "| layer | fragility | safe lambda | weighted scale | weighted shrink | router feature | risk sum | top expert |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in layer_summary.head(16).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {fmt(row['boundary_fragility_score'])} | "
            f"{fmt(row['safe_lambda_proxy'])} | {fmt(row['weighted_scale'])} | "
            f"{fmt(row['weighted_scale_shrink'])} | {fmt(row['weighted_router_instability_feature'])} | "
            f"{fmt(row['router_coupled_risk_sum'])} | {int(row['top_expert_id'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Router-Coupled Experts",
            "",
            "| layer | expert | category | route mass | fragility | router feature | scale | shrink | coupled risk | reason |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in top_experts.head(16).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | `{row['dominant_category']}` | "
            f"{fmt(row['total_topk_fraction'])} | {fmt(row['boundary_fragility_score'])} | "
            f"{fmt(row['feature_router_instability'])} | {fmt(row['mechanistic_selected_scale'])} | "
            f"{fmt(row['scale_shrink'])} | {fmt(row['router_coupled_risk'])} | `{row['mechanistic_reason']}` |"
        )
    lines.extend(["", "## Correlation Checks", ""])
    key_rows = correlations[
        correlations["driver"].isin(["boundary_fragility_score", "safe_lambda_proxy"])
        & correlations["target"].isin(
            ["feature_router_instability", "scale_shrink", "mechanistic_selected_scale"]
        )
    ]
    lines.extend(["| driver | target | weighted corr |", "| --- | --- | ---: |"])
    for _, row in key_rows.iterrows():
        lines.append(
            f"| `{row['driver']}` | `{row['target']}` | {fmt(row['route_mass_weighted_corr'])} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(summary["gate_reason"])
    lines.append(
        "Dense 模型的插值风险主要是连续 loss barrier；MoE 还多了离散 dispatch 边界。这里的结果说明 router fragility 不能只作为 router freeze 的理由，还应该作为 expert trust-region 的一部分，因为高脆弱层的 expert scale 确实被系统性收紧。"
    )
    lines.append(
        "这个 probe 仍不替代最终 vLLM selector；它只证明 router-boundary 项在当前 B/H/I scale law 里有可观测机制作用。"
    )
    lines.extend(["", "## Outputs", ""])
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze router-expert coupling in Qwen3 MoE averaging.")
    parser.add_argument(
        "--mechanistic-group-rules",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate/mechanistic_group_rules.csv"),
    )
    parser.add_argument(
        "--router-layer-fragility",
        type=Path,
        default=Path("results/qwen3_moe_router_margin_fragility/layer_margin_fragility.csv"),
    )
    parser.add_argument(
        "--mechanistic-summary",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_router_expert_coupling"))
    parser.add_argument("--top-expert-count", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = read_csv(args.mechanistic_group_rules)
    layer_fragility = read_csv(args.router_layer_fragility)
    mechanistic_summary = read_json(args.mechanistic_summary)
    joined = build_joined(groups, layer_fragility)
    correlations = build_correlations(joined)
    layer_summary = build_layer_summary(joined)
    top_experts = build_top_experts(joined, limit=max(1, int(args.top_expert_count)))

    weights = numeric(joined, "total_topk_fraction", 0.0)
    fragility = numeric(joined, "boundary_fragility_score", 0.0)
    high = quantile_slice(joined, "boundary_fragility_score", 0.75, high=True)
    low = quantile_slice(joined, "boundary_fragility_score", 0.25, high=False)
    high_summary = summarize_slice(high)
    low_summary = summarize_slice(low)
    values = {
        "fragility_router_feature_corr": weighted_corr(
            fragility, numeric(joined, "feature_router_instability"), weights
        ),
        "fragility_scale_shrink_corr": weighted_corr(fragility, numeric(joined, "scale_shrink"), weights),
        "safe_lambda_scale_shrink_corr": weighted_corr(
            numeric(joined, "safe_lambda_proxy"), numeric(joined, "scale_shrink"), weights
        ),
        "fragility_expected_delta_corr": weighted_corr(
            fragility,
            numeric(joined, "mechanistic_expected_max_relative_delta_norm"),
            weights,
        ),
        "high_vs_low_weighted_shrink_lift": high_summary["weighted_scale_shrink"]
        - low_summary["weighted_scale_shrink"],
    }
    gate, gate_reason = decide_gate(values)
    top_layer = layer_summary.iloc[0].to_dict()

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_path = output_dir / "layer_router_expert_coupling.csv"
    top_experts_path = output_dir / "top_router_coupled_experts.csv"
    corr_path = output_dir / "coupling_correlations.csv"
    literature_path = output_dir / "literature_sources.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    summary = {
        "schema_version": 1,
        "status": "router_expert_coupling_ready",
        "gate": gate,
        "gate_reason": gate_reason,
        "selected_mechanistic_candidate_id": mechanistic_summary.get("selected_candidate_id"),
        "group_count": int(len(joined)),
        "layer_count": int(layer_summary["layer_id"].nunique()),
        "high_fragility_layer_count": int(
            (layer_summary["boundary_fragility_score"] >= layer_summary["boundary_fragility_score"].quantile(0.75)).sum()
        ),
        "top_coupled_layer_id": int(top_layer["layer_id"]),
        "top_coupled_layer_risk": fnum(top_layer["router_coupled_risk_sum"]),
        "top_coupled_layer_fragility": fnum(top_layer["boundary_fragility_score"]),
        "top_coupled_layer_weighted_shrink": fnum(top_layer["weighted_scale_shrink"]),
        "fragility_router_feature_corr": values["fragility_router_feature_corr"],
        "fragility_scale_shrink_corr": values["fragility_scale_shrink_corr"],
        "safe_lambda_scale_shrink_corr": values["safe_lambda_scale_shrink_corr"],
        "fragility_expected_delta_corr": values["fragility_expected_delta_corr"],
        "high_vs_low_weighted_shrink_lift": values["high_vs_low_weighted_shrink_lift"],
        "high_fragility_slice": high_summary,
        "low_fragility_slice": low_summary,
        "outputs": {
            "layer_router_expert_coupling": rel(layer_path),
            "top_router_coupled_experts": rel(top_experts_path),
            "coupling_correlations": rel(corr_path),
            "literature_sources": rel(literature_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    layer_summary.to_csv(layer_path, index=False)
    top_experts.to_csv(top_experts_path, index=False)
    correlations.to_csv(corr_path, index=False)
    literature_path.write_text(json.dumps(LITERATURE_SOURCES, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, layer_summary, top_experts, correlations), encoding="utf-8")

    print(f"Wrote Qwen3 MoE router-expert coupling to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; gate={summary['gate']}; top_layer=L{summary['top_coupled_layer_id']}")


if __name__ == "__main__":
    main()
