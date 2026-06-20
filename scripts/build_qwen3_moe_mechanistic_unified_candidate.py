#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = "instruct"
NONBASE_SOURCE = "coder"
EPS = 1e-12


LITERATURE_PRIORS = [
    {
        "key": "loss_landscape",
        "source": "https://arxiv.org/abs/1712.09913",
        "mechanism": "Weight-space pictures are only useful when the plotted directions are tied to the model perturbation being tested.",
    },
    {
        "key": "mode_connectivity",
        "source": "https://arxiv.org/abs/1802.10026",
        "mechanism": "Low-barrier connectivity is the condition under which a direct average is plausible.",
    },
    {
        "key": "model_soups",
        "source": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Same-basin finetunes can be averaged, but endpoint fallback and validation remain part of the method.",
    },
    {
        "key": "git_rebasin",
        "source": "https://arxiv.org/abs/2209.04836",
        "mechanism": "Permutation/gauge symmetry means hidden units or experts must be aligned before same-name averaging is trusted.",
    },
    {
        "key": "ties",
        "source": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Parameter conflict should be measured and suppressed instead of blindly averaging every coordinate.",
    },
    {
        "key": "expert_merging",
        "source": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Layer/chunk heterogeneity and unlabeled hidden/logit alignment provide better merge coefficients than one global weight.",
    },
    {
        "key": "nash_expert_merging",
        "source": "https://arxiv.org/abs/2510.16138",
        "mechanism": "Expert contributions are cooperative/competitive, so weights should depend on marginal utility rather than a fixed average.",
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


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def boolish(series: pd.Series, default: bool = False) -> pd.Series:
    if series.empty:
        return pd.Series(default, index=series.index, dtype=bool)
    if series.dtype == bool:
        return series.fillna(default)
    return series.fillna(default).map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def robust01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.quantile(0.05))
    hi = float(values.quantile(0.95))
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def parse_flags(raw: Any) -> set[str]:
    raw = clean_value(raw)
    if raw is None:
        return set()
    return {part.strip() for part in str(raw).split("|") if part.strip()}


def category_prior(category: Any) -> float:
    category = str(clean_value(category) or "").strip().lower()
    if category in {"code", "programming"}:
        return 1.00
    if category in {"math", "reasoning"}:
        return 0.48
    if category in {"general"}:
        return 0.34
    if category in {"safety", "alignment"}:
        return 0.12
    return 0.25


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_group_rules(path: Path, fallback_path: Path) -> pd.DataFrame:
    path = repo_path(path)
    fallback_path = repo_path(fallback_path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.read_csv(fallback_path)


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "total_topk_fraction": 0.0,
        "original_weight_instruct": 0.0,
        "original_weight_coder": 0.0,
        "selected_weight_instruct": None,
        "selected_weight_coder": None,
        "selected_scale": 1.0,
        "feedback_selected_scale": None,
        "feedback_scale_delta": 0.0,
        "audit_max_relative_delta_norm": 0.0,
        "selected_expected_max_relative_delta_norm": None,
        "feedback_expected_max_relative_delta_norm": None,
        "mechanism_risk_score": 0.0,
        "feature_delta_pressure": None,
        "feature_router_instability": 0.0,
        "feature_load_pressure": 0.0,
        "feature_source_conflict": 0.0,
        "feature_low_route_mass": None,
        "feature_expert_route_geometry": None,
        "feature_expert_internal_geometry": None,
        "feature_layer_geometry": None,
        "feature_subspace_conflict": None,
        "feature_subspace_route_conflict": None,
        "subspace_conflict_score": None,
        "subspace_route_weighted_conflict_score": None,
        "route_geometry_risk_score": 0.0,
        "internal_geometry_risk_score": 0.0,
        "layer_route_geometry_risk_score": 0.0,
        "layer_prior_coder_coefficient": 1.0,
        "geometry_combined_relative_delta": 0.0,
        "geometry_combined_cosine": 1.0,
        "route_mass_instruct": 0.0,
        "route_mass_coder": 0.0,
        "max_topk_over_uniform": 0.0,
        "dominant_source": "",
        "dominant_category": "",
        "trust_risk_flags": "",
        "tensor_pattern": "",
        "feedback_action": "",
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default
    for column in [
        "total_topk_fraction",
        "original_weight_instruct",
        "original_weight_coder",
        "selected_scale",
        "audit_max_relative_delta_norm",
        "mechanism_risk_score",
        "feature_router_instability",
        "feature_load_pressure",
        "feature_source_conflict",
        "route_geometry_risk_score",
        "internal_geometry_risk_score",
        "layer_route_geometry_risk_score",
        "layer_prior_coder_coefficient",
        "geometry_combined_relative_delta",
        "geometry_combined_cosine",
        "route_mass_instruct",
        "route_mass_coder",
        "max_topk_over_uniform",
        "feedback_scale_delta",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(float(defaults.get(column, 0.0)))

    out["selected_weight_instruct"] = numeric(out, "selected_weight_instruct", 0.0)
    missing_selected_instruct = out["selected_weight_instruct"].abs() <= EPS
    out.loc[missing_selected_instruct, "selected_weight_instruct"] = out.loc[
        missing_selected_instruct, "original_weight_instruct"
    ]
    out["selected_scale"] = numeric(out, "selected_scale", 1.0).clip(0.0, 1.0)
    out["selected_weight_coder"] = numeric(out, "selected_weight_coder", 0.0)
    missing_selected_coder = out["selected_weight_coder"].abs() <= EPS
    out.loc[missing_selected_coder, "selected_weight_coder"] = (
        out.loc[missing_selected_coder, "original_weight_coder"]
        * out.loc[missing_selected_coder, "selected_scale"]
    )
    feedback_scale = pd.to_numeric(out["feedback_selected_scale"], errors="coerce")
    out["prior_scale"] = feedback_scale.fillna(out["selected_scale"]).clip(0.0, 1.0)
    out["prior_weight_coder"] = out["original_weight_coder"] * out["prior_scale"]
    out["prior_weight_instruct"] = out["selected_weight_instruct"]
    expected = pd.to_numeric(out["feedback_expected_max_relative_delta_norm"], errors="coerce")
    expected = expected.fillna(pd.to_numeric(out["selected_expected_max_relative_delta_norm"], errors="coerce"))
    out["prior_expected_max_relative_delta_norm"] = expected.fillna(
        out["audit_max_relative_delta_norm"] * out["prior_scale"]
    )
    for column in ["dominant_source", "dominant_category", "trust_risk_flags", "tensor_pattern", "feedback_action"]:
        out[column] = out[column].fillna("").astype(str)
    return out


def add_mechanistic_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = add_missing_columns(df)
    route_mass = out["total_topk_fraction"].clip(lower=0.0)
    out["route_mass_pressure"] = robust01(route_mass)
    out["delta_pressure"] = robust01(out["audit_max_relative_delta_norm"])
    out["load_pressure"] = robust01(out["max_topk_over_uniform"])
    out["low_route_evidence_pressure"] = 1.0 - out["route_mass_pressure"]
    route_total = (out["route_mass_instruct"] + out["route_mass_coder"]).clip(lower=EPS)
    out["coder_route_share"] = (out["route_mass_coder"] / route_total).clip(0.0, 1.0)
    out.loc[route_total <= EPS, "coder_route_share"] = (out["dominant_source"].str.lower() == NONBASE_SOURCE).astype(float)
    out["category_coder_prior"] = out["dominant_category"].map(category_prior)
    flags = out["trust_risk_flags"].apply(parse_flags)
    out["flag_high_load"] = flags.apply(lambda item: float("high_load_expert" in item or "high_load_weight_limit" in item))
    out["flag_fragile_router"] = flags.apply(lambda item: float("fragile_router_layer" in item))
    out["flag_low_route_evidence"] = flags.apply(lambda item: float("low_route_evidence" in item))
    out["flag_category_mismatch"] = flags.apply(lambda item: float("category_source_mismatch" in item))
    out["flag_shared_mixed"] = flags.apply(lambda item: float("shared_mixed_expert" in item))
    out["dominant_coder_flag"] = (out["dominant_source"].str.lower() == NONBASE_SOURCE).astype(float)
    out["feature_delta_pressure"] = pd.to_numeric(out["feature_delta_pressure"], errors="coerce").fillna(
        out["delta_pressure"]
    )
    out["feature_low_route_mass"] = pd.to_numeric(out["feature_low_route_mass"], errors="coerce").fillna(
        out["low_route_evidence_pressure"]
    )
    out["feature_expert_route_geometry"] = pd.to_numeric(
        out["feature_expert_route_geometry"], errors="coerce"
    ).fillna(out["route_geometry_risk_score"])
    out["feature_expert_internal_geometry"] = pd.to_numeric(
        out["feature_expert_internal_geometry"], errors="coerce"
    ).fillna(out["internal_geometry_risk_score"])
    out["feature_layer_geometry"] = pd.to_numeric(out["feature_layer_geometry"], errors="coerce").fillna(
        out["layer_route_geometry_risk_score"]
    )
    out["feature_subspace_conflict"] = pd.to_numeric(out["feature_subspace_conflict"], errors="coerce").fillna(
        pd.to_numeric(out["subspace_conflict_score"], errors="coerce").fillna(0.0)
    )
    out["feature_subspace_route_conflict"] = pd.to_numeric(
        out["feature_subspace_route_conflict"], errors="coerce"
    ).fillna(pd.to_numeric(out["subspace_route_weighted_conflict_score"], errors="coerce").fillna(0.0))

    out["benefit_score"] = (
        0.30 * out["route_mass_pressure"]
        + 0.24 * out["coder_route_share"]
        + 0.20 * out["dominant_coder_flag"]
        + 0.16 * out["category_coder_prior"]
        + 0.10 * robust01(out["original_weight_coder"])
    ).clip(0.0, 1.0)
    out["curvature_score"] = (
        0.20 * out["feature_delta_pressure"].clip(0.0, 1.0)
        + 0.16 * out["feature_expert_route_geometry"].clip(0.0, 1.0)
        + 0.13 * out["feature_expert_internal_geometry"].clip(0.0, 1.0)
        + 0.12 * out["feature_layer_geometry"].clip(0.0, 1.0)
        + 0.15 * out["feature_subspace_conflict"].clip(0.0, 1.0)
        + 0.10 * out["feature_subspace_route_conflict"].clip(0.0, 1.0)
        + 0.08 * out["feature_router_instability"].clip(0.0, 1.0)
        + 0.06 * out["load_pressure"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["interference_score"] = (
        0.26 * out["feature_source_conflict"].clip(0.0, 1.0)
        + 0.18 * out["feature_subspace_route_conflict"].clip(0.0, 1.0)
        + 0.14 * out["feature_subspace_conflict"].clip(0.0, 1.0)
        + 0.12 * out["feature_router_instability"].clip(0.0, 1.0)
        + 0.10 * out["flag_fragile_router"]
        + 0.08 * out["flag_high_load"]
        + 0.06 * out["flag_low_route_evidence"]
        + 0.04 * out["flag_category_mismatch"]
        + 0.02 * out["flag_shared_mixed"]
    ).clip(0.0, 1.0)
    out["local_curvature_proxy"] = (0.20 + 1.80 * out["curvature_score"]).clip(0.20, 2.00)
    out["marginal_interference_proxy"] = (0.05 + 1.35 * out["interference_score"]).clip(0.05, 1.40)
    out["marginal_benefit_proxy"] = (
        0.10
        + out["benefit_score"] * (0.55 + 0.45 * out["route_mass_pressure"])
        + 0.15 * out["dominant_coder_flag"]
    ).clip(0.05, 1.30)
    out["mechanistic_reason"] = "balanced"
    out.loc[
        (out["benefit_score"] >= 0.62) & (out["curvature_score"] <= 0.55),
        "mechanistic_reason",
    ] = "preserve_high_benefit_low_curvature"
    out.loc[
        (out["benefit_score"] >= 0.62)
        & (out["curvature_score"] <= 0.55)
        & (out["audit_max_relative_delta_norm"] > 0.65),
        "mechanistic_reason",
    ] = "preserve_benefit_but_delta_cap_limited"
    out.loc[
        (out["interference_score"] >= 0.62) & (out["benefit_score"] < 0.50),
        "mechanistic_reason",
    ] = "shrink_high_interference_low_benefit"
    out.loc[
        (out["curvature_score"] >= 0.70) & (out["feature_subspace_conflict"] >= 0.70),
        "mechanistic_reason",
    ] = "shrink_curved_subspace_conflict"
    out.loc[
        (out["flag_fragile_router"] > 0.5) & (out["route_mass_pressure"] >= 0.60),
        "mechanistic_reason",
    ] = "route_important_fragile_boundary"
    return out


def solve_scale(
    df: pd.DataFrame,
    *,
    benefit_gain: float,
    curvature_gain: float,
    interference_gain: float,
    step_size: float,
    hard_cap: float,
    min_scale: float,
) -> pd.Series:
    delta = df["audit_max_relative_delta_norm"].clip(lower=0.0)
    prior = df["prior_scale"].clip(0.0, 1.0)
    benefit = benefit_gain * df["marginal_benefit_proxy"].clip(lower=EPS)
    curvature_cost = curvature_gain * df["local_curvature_proxy"] * (delta**2)
    interference_cost = interference_gain * df["marginal_interference_proxy"]
    gradient = benefit - (curvature_cost + interference_cost) * prior
    denominator = (0.50 + curvature_cost + interference_cost).clip(lower=EPS)
    update = step_size * (gradient / denominator).clip(-1.0, 1.0)
    restore_gate = (0.35 + 0.65 * df["benefit_score"]).clip(0.0, 1.0)
    shrink_gate = (0.45 + 0.55 * df["interference_score"]).clip(0.0, 1.0)
    update = update.clip(lower=0.0) * restore_gate + update.clip(upper=0.0) * shrink_gate
    scale = (prior + update).clip(min_scale, 1.0)
    feedback_action = df["feedback_action"].fillna("").astype(str).str.lower()
    feedback_shrink = feedback_action.str.contains("shrink|regression")
    scale.loc[feedback_shrink] = pd.concat([scale.loc[feedback_shrink], prior.loc[feedback_shrink]], axis=1).min(axis=1)
    cap_scale = pd.Series(1.0, index=df.index)
    needs_cap = delta * scale > hard_cap
    cap_scale.loc[needs_cap] = hard_cap / delta.loc[needs_cap].clip(lower=EPS)
    return pd.concat([scale, cap_scale.clip(0.0, 1.0)], axis=1).min(axis=1).clip(min_scale, 1.0)


def metrics_from_scale(
    df: pd.DataFrame,
    scale: pd.Series,
    *,
    candidate_id: str,
    benefit_gain: float,
    curvature_gain: float,
    interference_gain: float,
    step_size: float,
    hard_cap: float,
) -> dict[str, Any]:
    route_mass = df["total_topk_fraction"].clip(lower=0.0)
    delta = df["audit_max_relative_delta_norm"].clip(lower=0.0)
    predicted_delta = delta * scale
    original_nonbase_mass = float((route_mass * df["original_weight_coder"].clip(lower=0.0)).sum())
    selected_nonbase_mass = float((route_mass * df["original_weight_coder"].clip(lower=0.0) * scale).sum())
    retention = selected_nonbase_mass / max(EPS, original_nonbase_mass)
    risk_weight = route_mass * (0.50 * df["curvature_score"] + 0.50 * df["interference_score"])
    benefit_weight = route_mass * df["benefit_score"]
    route_weight = route_mass.clip(lower=0.0)
    loss_terms = route_mass * (
        0.5 * df["local_curvature_proxy"] * (predicted_delta**2)
        + df["marginal_interference_proxy"] * (scale**2)
        - df["marginal_benefit_proxy"] * scale
    )
    weighted_predicted_delta = float((route_weight * predicted_delta).sum() / max(EPS, float(route_weight.sum())))
    risk_weighted_delta = float((risk_weight * predicted_delta).sum() / max(EPS, float(risk_weight.sum())))
    benefit_weighted_scale = float((benefit_weight * scale).sum() / max(EPS, float(benefit_weight.sum())))
    high_benefit_low_risk = (df["benefit_score"] >= 0.62) & (df["curvature_score"] <= 0.55)
    high_interference_low_benefit = (df["interference_score"] >= 0.62) & (df["benefit_score"] < 0.50)
    high_subspace = df["feature_subspace_conflict"] >= 0.72
    hard_violations = predicted_delta > hard_cap + 1e-9
    objective = (
        2000.0 * int(hard_violations.sum())
        + 60.0 * max(0.0, float(predicted_delta.max()) - hard_cap)
        + 2.5 * max(0.0, 0.965 - retention)
        + 1.40 * risk_weighted_delta
        + 0.80 * weighted_predicted_delta
        - 0.35 * benefit_weighted_scale
        + float(loss_terms.mean())
    )
    return {
        "candidate_id": candidate_id,
        "benefit_gain": benefit_gain,
        "curvature_gain": curvature_gain,
        "interference_gain": interference_gain,
        "step_size": step_size,
        "group_count": int(len(df)),
        "scaled_group_count": int((scale < df["prior_scale"] - 1e-9).sum()),
        "restored_group_count": int((scale > df["prior_scale"] + 1e-9).sum()),
        "max_predicted_relative_delta": float(predicted_delta.max()),
        "p99_predicted_relative_delta": float(predicted_delta.quantile(0.99)),
        "hard_cap_violation_count": int(hard_violations.sum()),
        "nonbase_mass_retention": retention,
        "route_mass_weighted_predicted_delta": weighted_predicted_delta,
        "risk_weighted_predicted_delta": risk_weighted_delta,
        "benefit_weighted_scale": benefit_weighted_scale,
        "mean_mechanistic_loss_proxy": float(loss_terms.mean()),
        "mean_scale": float(scale.mean()),
        "min_scale": float(scale.min()),
        "high_benefit_low_risk_group_count": int(high_benefit_low_risk.sum()),
        "high_benefit_low_risk_mean_scale": float(scale.loc[high_benefit_low_risk].mean())
        if bool(high_benefit_low_risk.any())
        else 1.0,
        "high_interference_low_benefit_group_count": int(high_interference_low_benefit.sum()),
        "high_interference_low_benefit_mean_scale": float(scale.loc[high_interference_low_benefit].mean())
        if bool(high_interference_low_benefit.any())
        else 1.0,
        "high_subspace_group_count": int(high_subspace.sum()),
        "high_subspace_mean_scale": float(scale.loc[high_subspace].mean()) if bool(high_subspace.any()) else 1.0,
        "mechanistic_objective": objective,
        "passes_hard_cap": bool(not hard_violations.any()),
    }


def candidate_grid(
    df: pd.DataFrame,
    *,
    hard_cap: float,
    min_scale: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows = []
    scales: dict[str, pd.Series] = {}
    for step_size in (0.04, 0.08, 0.12, 0.16):
        for benefit_gain in (0.90, 1.15, 1.40, 1.65):
            for curvature_gain in (0.75, 1.00, 1.25):
                for interference_gain in (0.75, 1.00, 1.25):
                    candidate_id = (
                        f"s{step_size:.2f}_b{benefit_gain:.2f}_h{curvature_gain:.2f}_i{interference_gain:.2f}"
                    )
                    scale = solve_scale(
                        df,
                        benefit_gain=benefit_gain,
                        curvature_gain=curvature_gain,
                        interference_gain=interference_gain,
                        step_size=step_size,
                        hard_cap=hard_cap,
                        min_scale=min_scale,
                    )
                    rows.append(
                        metrics_from_scale(
                            df,
                            scale,
                            candidate_id=candidate_id,
                            benefit_gain=benefit_gain,
                            curvature_gain=curvature_gain,
                            interference_gain=interference_gain,
                            step_size=step_size,
                            hard_cap=hard_cap,
                        )
                    )
                    scales[candidate_id] = scale
    search = pd.DataFrame(rows).sort_values(
        [
            "passes_hard_cap",
            "mechanistic_objective",
            "risk_weighted_predicted_delta",
            "nonbase_mass_retention",
        ],
        ascending=[False, True, True, False],
    )
    return search, scales


def select_candidate(search: pd.DataFrame, *, min_retention: float) -> pd.Series:
    feasible = search[search["passes_hard_cap"]].copy()
    if feasible.empty:
        return search.sort_values(["hard_cap_violation_count", "mechanistic_objective"]).iloc[0]
    retained = feasible[feasible["nonbase_mass_retention"] >= min_retention].copy()
    if retained.empty:
        return feasible.sort_values(["nonbase_mass_retention", "mechanistic_objective"], ascending=[False, True]).iloc[0]
    return retained.sort_values(
        [
            "mechanistic_objective",
            "risk_weighted_predicted_delta",
            "benefit_weighted_scale",
            "nonbase_mass_retention",
        ],
        ascending=[True, True, False, False],
    ).iloc[0]


def build_group_rules(df: pd.DataFrame, scale: pd.Series, selected: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["mechanistic_candidate_id"] = str(selected["candidate_id"])
    out["mechanistic_selected_scale"] = scale
    out["mechanistic_scale_delta_vs_prior"] = out["mechanistic_selected_scale"] - out["prior_scale"]
    out["mechanistic_weight_coder"] = out["original_weight_coder"] * out["mechanistic_selected_scale"]
    out["mechanistic_weight_instruct"] = out["prior_weight_instruct"]
    out["mechanistic_expected_max_relative_delta_norm"] = (
        out["audit_max_relative_delta_norm"] * out["mechanistic_selected_scale"]
    )
    out["mechanistic_action"] = "keep_prior"
    out.loc[out["mechanistic_scale_delta_vs_prior"] < -1e-6, "mechanistic_action"] = "shrink_from_prior"
    out.loc[out["mechanistic_scale_delta_vs_prior"] > 1e-6, "mechanistic_action"] = "restore_from_prior"
    out.loc[
        (out["mechanistic_action"] == "keep_prior") & (out["mechanistic_reason"] != "balanced"),
        "mechanistic_action",
    ] = out["mechanistic_reason"]
    columns = [
        "layer_id",
        "expert_id",
        "total_topk_fraction",
        "dominant_source",
        "dominant_category",
        "original_weight_instruct",
        "original_weight_coder",
        "prior_scale",
        "prior_weight_instruct",
        "prior_weight_coder",
        "mechanistic_selected_scale",
        "mechanistic_scale_delta_vs_prior",
        "mechanistic_weight_instruct",
        "mechanistic_weight_coder",
        "audit_max_relative_delta_norm",
        "mechanistic_expected_max_relative_delta_norm",
        "benefit_score",
        "curvature_score",
        "interference_score",
        "local_curvature_proxy",
        "marginal_interference_proxy",
        "marginal_benefit_proxy",
        "route_mass_pressure",
        "coder_route_share",
        "category_coder_prior",
        "feature_subspace_conflict",
        "feature_subspace_route_conflict",
        "feature_expert_route_geometry",
        "feature_expert_internal_geometry",
        "feature_router_instability",
        "load_pressure",
        "delta_pressure",
        "mechanistic_reason",
        "mechanistic_action",
        "trust_risk_flags",
        "feedback_action",
        "tensor_pattern",
    ]
    present = [column for column in columns if column in out.columns]
    return out[present].sort_values(["layer_id", "expert_id"])


def extract_writer_context(command_path: Path) -> tuple[str, dict[str, str]]:
    command_path = repo_path(command_path)
    command = command_path.read_text(encoding="utf-8").strip()
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


def write_rules_and_commands(group_rules: pd.DataFrame, output_dir: Path, writer_context_command: Path) -> dict[str, str]:
    base_path, sources = extract_writer_context(writer_context_command)
    rules_path = output_dir / "tensor_rules.txt"
    checkpoint_output_dir = "results/checkpoints/qwen3_moe_mechanistic_unified_candidate"
    lines = [
        "# Mechanistic unified Qwen3 MoE expert rules.",
        "# Router and shared attention remain frozen; per-expert nonbase scale is solved from benefit/curvature/interference.",
    ]
    for _, row in group_rules.sort_values(["layer_id", "expert_id"]).iterrows():
        pattern = str(row["tensor_pattern"])
        if not pattern:
            continue
        lines.append(
            f"{pattern}::{BASE_SOURCE}={float(row['mechanistic_weight_instruct']):.6g},"
            f"{NONBASE_SOURCE}={float(row['mechanistic_weight_coder']):.6g}"
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


def write_literature(path: Path) -> None:
    path.write_text(json.dumps(LITERATURE_PRIORS, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_report(summary: dict[str, Any], search: pd.DataFrame, group_rules: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Mechanistic Unified Candidate",
        "",
        "这个候选不再把 Average 写成“选某个算法名”。它把同结构 MoE 平均写成每个 routed expert group 的近似二次目标：",
        "",
        "`J_g(s) = 1/2 H_g ||delta_g||^2 s^2 + I_g s^2 - B_g s`",
        "",
        "其中 `B_g` 来自 route mass、Coder route share、source specialization 和当前非 base 权重；`H_g` 来自 delta pressure、expert geometry、layer geometry、router fragility 和 subspace conflict；`I_g` 来自 source conflict、fragile top-k boundary、high-load/shared experts 和 category mismatch。求出的 `s` 是同结构规则里的非 base scale，并继续受 hard-cap 与 vLLM feedback gate 约束。",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Expert groups: `{summary['expert_group_count']}`",
        f"- Selected candidate: `{summary['selected_candidate_id']}`",
        f"- Search points: `{summary['candidate_count']}`",
        f"- Nonbase route-mass retention: `{fmt(summary['selected_nonbase_mass_retention'])}`",
        f"- Max predicted routed relative delta: `{fmt(summary['selected_max_predicted_relative_delta'])}`",
        f"- Hard-cap violations: `{summary['selected_hard_cap_violation_count']}`",
        f"- Risk-weighted predicted delta: `{fmt(summary['selected_risk_weighted_predicted_delta'])}`",
        f"- Benefit-weighted scale: `{fmt(summary['selected_benefit_weighted_scale'])}`",
        f"- Mean mechanistic loss proxy: `{fmt(summary['selected_mean_mechanistic_loss_proxy'])}`",
        f"- Mean scale: `{fmt(summary['selected_mean_scale'])}`",
        f"- High-benefit low-risk mean scale: `{fmt(summary['selected_high_benefit_low_risk_mean_scale'])}`",
        f"- High-interference low-benefit mean scale: `{fmt(summary['selected_high_interference_low_benefit_mean_scale'])}`",
        f"- High-subspace-conflict mean scale: `{fmt(summary['selected_high_subspace_mean_scale'])}`",
        "",
        "## Mechanism Interpretation",
        "",
        "平均是否有效取决于同一个结构里的两个条件：第一，两个 checkpoint 的对应功能单元是否在同一个低损失 basin 或已对齐；第二，该功能单元的边际收益是否大于沿这条 weight-space 方向移动造成的曲率和干扰成本。Dense 模型里这通常表现为 mode connectivity、符号冲突和局部 curvature；MoE 里还多了 top-k 路由边界、expert identity、expert load、channel/chunk 子空间冲突。这个脚本把这些内部参数统一成 `B/H/I`，因此输出的是“为什么这组 expert 该动多少”，不是“某场景套某算法”。",
        "",
        "`scale` 不是收益本身；高收益 group 如果原始 `delta` 已经很大，也会被 hard cap 限制。报告里的 reason 因此区分了 `preserve_high_benefit_low_curvature` 和 `preserve_benefit_but_delta_cap_limited`。",
        "",
        "## Candidate Search",
        "",
        "| candidate | pass cap | retention | max delta | risk delta | benefit scale | loss proxy | mean scale | objective |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in search.head(14).iterrows():
        lines.append(
            f"| `{row['candidate_id']}` | `{bool(row['passes_hard_cap'])}` | "
            f"{fmt(float(row['nonbase_mass_retention']))} | "
            f"{fmt(float(row['max_predicted_relative_delta']))} | "
            f"{fmt(float(row['risk_weighted_predicted_delta']))} | "
            f"{fmt(float(row['benefit_weighted_scale']))} | "
            f"{fmt(float(row['mean_mechanistic_loss_proxy']))} | "
            f"{fmt(float(row['mean_scale']))} | "
            f"{fmt(float(row['mechanistic_objective']))} |"
        )
    lines.extend(
        [
            "",
            "## Top Shrink Reasons",
            "",
            "| reason | groups | route mass | mean benefit | mean curvature | mean interference | mean scale |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    reason_rows = (
        group_rules.groupby("mechanistic_reason", as_index=False)
        .agg(
            groups=("expert_id", "count"),
            route_mass=("total_topk_fraction", "sum"),
            mean_benefit=("benefit_score", "mean"),
            mean_curvature=("curvature_score", "mean"),
            mean_interference=("interference_score", "mean"),
            mean_scale=("mechanistic_selected_scale", "mean"),
        )
        .sort_values(["route_mass", "groups"], ascending=[False, False])
    )
    for _, row in reason_rows.iterrows():
        lines.append(
            f"| `{row['mechanistic_reason']}` | {int(row['groups'])} | {fmt(float(row['route_mass']))} | "
            f"{fmt(float(row['mean_benefit']))} | {fmt(float(row['mean_curvature']))} | "
            f"{fmt(float(row['mean_interference']))} | {fmt(float(row['mean_scale']))} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def smoke_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer_id": 0,
                "expert_id": 0,
                "total_topk_fraction": 0.85,
                "dominant_source": "coder",
                "dominant_category": "code",
                "original_weight_instruct": 0.20,
                "original_weight_coder": 0.75,
                "selected_scale": 0.92,
                "audit_max_relative_delta_norm": 0.42,
                "route_mass_instruct": 0.10,
                "route_mass_coder": 0.75,
                "feature_source_conflict": 0.20,
                "feature_router_instability": 0.10,
                "feature_subspace_conflict": 0.15,
                "feature_subspace_route_conflict": 0.10,
                "route_geometry_risk_score": 0.10,
                "internal_geometry_risk_score": 0.10,
                "trust_risk_flags": "",
                "tensor_pattern": "L0E0",
            },
            {
                "layer_id": 0,
                "expert_id": 1,
                "total_topk_fraction": 0.80,
                "dominant_source": "instruct",
                "dominant_category": "safety",
                "original_weight_instruct": 0.75,
                "original_weight_coder": 0.20,
                "selected_scale": 1.00,
                "audit_max_relative_delta_norm": 0.50,
                "route_mass_instruct": 0.72,
                "route_mass_coder": 0.08,
                "feature_source_conflict": 0.75,
                "feature_router_instability": 0.72,
                "feature_subspace_conflict": 0.82,
                "feature_subspace_route_conflict": 0.76,
                "route_geometry_risk_score": 0.80,
                "internal_geometry_risk_score": 0.65,
                "trust_risk_flags": "fragile_router_layer|high_load_expert|shared_mixed_expert",
                "tensor_pattern": "L0E1",
            },
            {
                "layer_id": 1,
                "expert_id": 0,
                "total_topk_fraction": 0.55,
                "dominant_source": "coder",
                "dominant_category": "code",
                "original_weight_instruct": 0.25,
                "original_weight_coder": 0.70,
                "selected_scale": 1.00,
                "audit_max_relative_delta_norm": 1.20,
                "route_mass_instruct": 0.15,
                "route_mass_coder": 0.40,
                "feature_source_conflict": 0.40,
                "feature_router_instability": 0.40,
                "feature_subspace_conflict": 0.45,
                "feature_subspace_route_conflict": 0.35,
                "route_geometry_risk_score": 0.35,
                "internal_geometry_risk_score": 0.35,
                "trust_risk_flags": "",
                "tensor_pattern": "L1E0",
            },
            {
                "layer_id": 1,
                "expert_id": 1,
                "total_topk_fraction": 0.40,
                "dominant_source": "instruct",
                "dominant_category": "general",
                "original_weight_instruct": 0.60,
                "original_weight_coder": 0.35,
                "selected_scale": 0.85,
                "feedback_selected_scale": 0.55,
                "audit_max_relative_delta_norm": 0.60,
                "route_mass_instruct": 0.32,
                "route_mass_coder": 0.08,
                "feature_source_conflict": 0.68,
                "feature_router_instability": 0.55,
                "feature_subspace_conflict": 0.70,
                "feature_subspace_route_conflict": 0.62,
                "route_geometry_risk_score": 0.55,
                "internal_geometry_risk_score": 0.55,
                "trust_risk_flags": "fragile_router_layer",
                "feedback_action": "shrink_nonbase_for_source_regression",
                "tensor_pattern": "L1E1",
            },
        ]
    )


def run_smoke(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = add_mechanistic_terms(smoke_frame())
    search, scales = candidate_grid(df, hard_cap=args.hard_cap, min_scale=args.min_scale)
    selected = select_candidate(search, min_retention=0.50)
    scale = scales[str(selected["candidate_id"])]
    row_by_expert = {
        int(row["expert_id"]): float(scale.loc[idx])
        for idx, row in df[df["layer_id"] == 0].iterrows()
    }
    hard_cap_scale = float(scale.loc[(df["layer_id"] == 1) & (df["expert_id"] == 0)].iloc[0])
    feedback_scale = float(scale.loc[(df["layer_id"] == 1) & (df["expert_id"] == 1)].iloc[0])
    matrix = pd.DataFrame(
        [
            {
                "case": "benefit_vs_interference",
                "assertion": "high_benefit_low_risk_kept_above_high_interference",
                "expected": "expert0 scale > expert1 scale",
                "actual": f"{row_by_expert[0]:.6f} > {row_by_expert[1]:.6f}",
                "passed": row_by_expert[0] > row_by_expert[1],
            },
            {
                "case": "hard_cap",
                "assertion": "delta_cap_enforced",
                "expected": f"scale <= {args.hard_cap / 1.20:.6f}",
                "actual": f"{hard_cap_scale:.6f}",
                "passed": hard_cap_scale <= args.hard_cap / 1.20 + 1e-9,
            },
            {
                "case": "feedback_prior",
                "assertion": "feedback_shrink_respected",
                "expected": "<=0.55",
                "actual": f"{feedback_scale:.6f}",
                "passed": feedback_scale <= 0.55 + 1e-9,
            },
            {
                "case": "candidate_search",
                "assertion": "selected_candidate_passes_hard_cap",
                "expected": "True",
                "actual": str(bool(selected["passes_hard_cap"])),
                "passed": bool(selected["passes_hard_cap"]),
            },
        ]
    )
    summary = {
        "status": "passed" if bool(matrix["passed"].all()) else "failed",
        "case_count": int(len(matrix)),
        "passed_case_count": int(matrix["passed"].sum()),
        "failed_case_count": int((~matrix["passed"]).sum()),
        "selected_candidate_id": str(selected["candidate_id"]),
        "outputs": {
            "matrix": rel(output_dir / "mechanistic_unified_smoke_matrix.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    matrix.to_csv(output_dir / "mechanistic_unified_smoke_matrix.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Qwen3 MoE Mechanistic Unified Candidate Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases passed: `{summary['passed_case_count']}/{summary['case_count']}`",
        "",
        "| case | assertion | expected | actual | passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['assertion']}` | `{row['expected']}` | `{row['actual']}` | `{row['passed']}` |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Status: {summary['status']}; cases={summary['passed_case_count']}/{summary['case_count']}")
    if summary["status"] != "passed":
        raise SystemExit(1)


def run_real(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_group_rules(args.group_rules, args.fallback_group_rules)
    df = add_mechanistic_terms(df)
    search, scales = candidate_grid(df, hard_cap=args.hard_cap, min_scale=args.min_scale)
    selected = select_candidate(search, min_retention=args.min_retention)
    scale = scales[str(selected["candidate_id"])]
    group_rules = build_group_rules(df, scale, selected)

    search_path = output_dir / "candidate_search.csv"
    group_rules_path = output_dir / "mechanistic_group_rules.csv"
    selected_path = output_dir / "selected_candidate.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    literature_path = output_dir / "literature_sources.json"
    artifacts = write_rules_and_commands(group_rules, output_dir, args.writer_context_command)
    write_literature(literature_path)
    search.to_csv(search_path, index=False)
    group_rules.to_csv(group_rules_path, index=False)
    selected_path.write_text(
        json.dumps(json_safe(selected.to_dict()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    feedback_summary = read_json(repo_path(args.feedback_summary))
    feedback_status = feedback_summary.get("status") or "not_available"
    summary = {
        "status": "mechanistic_unified_candidate_ready",
        "expert_group_count": int(len(group_rules)),
        "candidate_count": int(len(search)),
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_nonbase_mass_retention": float(selected["nonbase_mass_retention"]),
        "selected_max_predicted_relative_delta": float(selected["max_predicted_relative_delta"]),
        "selected_hard_cap_violation_count": int(selected["hard_cap_violation_count"]),
        "selected_risk_weighted_predicted_delta": float(selected["risk_weighted_predicted_delta"]),
        "selected_benefit_weighted_scale": float(selected["benefit_weighted_scale"]),
        "selected_mean_mechanistic_loss_proxy": float(selected["mean_mechanistic_loss_proxy"]),
        "selected_mean_scale": float(selected["mean_scale"]),
        "selected_min_scale": float(selected["min_scale"]),
        "selected_high_benefit_low_risk_mean_scale": float(selected["high_benefit_low_risk_mean_scale"]),
        "selected_high_interference_low_benefit_mean_scale": float(
            selected["high_interference_low_benefit_mean_scale"]
        ),
        "selected_high_subspace_mean_scale": float(selected["high_subspace_mean_scale"]),
        "hard_cap": args.hard_cap,
        "min_retention": args.min_retention,
        "feedback_status": feedback_status,
        "feedback_materialization_gate": feedback_summary.get("materialization_gate", "not_available"),
        "outputs": {
            "report": rel(report_path),
            "summary": rel(summary_path),
            "candidate_search": rel(search_path),
            "mechanistic_group_rules": rel(group_rules_path),
            "selected_candidate": rel(selected_path),
            "tensor_rules": artifacts["tensor_rules"],
            "writer_command": artifacts["writer_command"],
            "dry_run_command": artifacts["dry_run_command"],
            "literature_sources": rel(literature_path),
            "checkpoint_output_dir": artifacts["checkpoint_output_dir"],
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, search, group_rules), encoding="utf-8")
    print(
        "Status: "
        f"{summary['status']}; selected={summary['selected_candidate_id']}; "
        f"retention={summary['selected_nonbase_mass_retention']:.4f}; "
        f"max_delta={summary['selected_max_predicted_relative_delta']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mechanistic unified Qwen3 MoE average candidate.")
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
        "--feedback-summary",
        type=Path,
        default=Path("results/qwen3_moe_feedback_optimizer/summary.json"),
    )
    parser.add_argument(
        "--writer-context-command",
        type=Path,
        default=Path("results/qwen3_moe_feedback_optimizer/writer_command.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_mechanistic_unified_candidate"))
    parser.add_argument("--hard-cap", type=float, default=0.65)
    parser.add_argument("--min-retention", type=float, default=0.965)
    parser.add_argument("--min-scale", type=float, default=0.0)
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_matrix:
        run_smoke(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
