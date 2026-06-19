#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
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
        "key": "mode_connectivity",
        "source": "https://arxiv.org/abs/1802.10026",
        "mechanism": "Use low-loss connectivity as a gate: a straight average is only trusted when the path probe is safe.",
    },
    {
        "key": "model_soups",
        "source": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Averaging is plausible inside one basin; endpoint fallback remains necessary outside it.",
    },
    {
        "key": "fisher_merging",
        "source": "https://arxiv.org/abs/2111.09832",
        "mechanism": "Local curvature motivates trust regions, but nonlocal route barriers still need held-out/vLLM gates.",
    },
    {
        "key": "ties",
        "source": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Coordinate conflict is a real mechanism, but for MoE it must be subordinated to expert identity and routing.",
    },
    {
        "key": "git_rebasin",
        "source": "https://arxiv.org/abs/2209.04836",
        "mechanism": "Permutation symmetry means expert identity/alignment is a precondition for same-name averaging.",
    },
    {
        "key": "moe_routing_breakdown",
        "source": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Router perturbations can break top-k dispatch, so this candidate freezes router and treats calibration separately.",
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
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return clean_value(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_flags(raw: Any) -> set[str]:
    raw = clean_value(raw)
    if raw is None:
        return set()
    return {part.strip() for part in str(raw).split("|") if part.strip()}


def robust01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.quantile(0.05))
    hi = float(values.quantile(0.95))
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flags = out["trust_risk_flags"].apply(parse_flags)
    out["feature_high_load_flag"] = flags.apply(
        lambda item: float("high_load_expert" in item or "high_load_weight_limit" in item)
    )
    out["feature_shared_mixed_flag"] = flags.apply(lambda item: float("shared_mixed_expert" in item))
    out["feature_fragile_router_flag"] = flags.apply(lambda item: float("fragile_router_layer" in item))
    out["feature_low_route_evidence_flag"] = flags.apply(lambda item: float("low_route_evidence" in item))
    out["feature_category_mismatch_flag"] = flags.apply(lambda item: float("category_source_mismatch" in item))
    out["feature_delta_pressure"] = robust01(out["audit_max_relative_delta_norm"])
    out["feature_route_mass"] = robust01(out["total_topk_fraction"])
    out["feature_low_route_mass"] = 1.0 - out["feature_route_mass"]
    out["feature_load_pressure"] = robust01(out["max_topk_over_uniform"])
    out["feature_router_instability"] = (
        robust01(1.0 - pd.to_numeric(out["min_topk_jaccard"], errors="coerce").fillna(1.0))
        + robust01(1.0 - pd.to_numeric(out["min_top1_agreement"], errors="coerce").fillna(1.0))
        + robust01(out["max_router_risk_score"])
    ) / 3.0
    source_sum = (out["weight_instruct"].abs() + out["weight_coder"].abs()).clip(lower=EPS)
    out["feature_source_conflict"] = (1.0 - (out["weight_instruct"] - out["weight_coder"]).abs() / source_sum).clip(
        0.0, 1.0
    )
    out["mechanism_risk_score"] = (
        0.30 * out["feature_delta_pressure"]
        + 0.20 * out["feature_router_instability"]
        + 0.15 * out["feature_load_pressure"]
        + 0.10 * out["feature_source_conflict"]
        + 0.10 * out["feature_low_route_mass"]
        + 0.05 * out["feature_high_load_flag"]
        + 0.05 * out["feature_low_route_evidence_flag"]
        + 0.03 * out["feature_category_mismatch_flag"]
        + 0.02 * out["feature_shared_mixed_flag"]
    ).clip(0.0, 1.0)
    return out


def scale_from_target_cap(df: pd.DataFrame, target_cap: pd.Series) -> pd.Series:
    audit_max = df["audit_max_relative_delta_norm"].clip(lower=0.0)
    scale = pd.Series(1.0, index=df.index)
    needs_scale = audit_max > target_cap
    scale.loc[needs_scale] = target_cap.loc[needs_scale] / audit_max.loc[needs_scale].clip(lower=EPS)
    return scale.clip(lower=0.0, upper=1.0)


def candidate_target_caps(
    df: pd.DataFrame,
    *,
    hard_cap: float,
    min_cap: float,
) -> list[tuple[str, str, pd.Series]]:
    risk = df["mechanism_risk_score"].clip(0.0, 1.0)
    low_evidence = df["feature_low_route_mass"].clip(0.0, 1.0)
    router_instability = df["feature_router_instability"].clip(0.0, 1.0)
    load_pressure = df["feature_load_pressure"].clip(0.0, 1.0)
    rows: list[tuple[str, str, pd.Series]] = [
        (
            f"uniform_{hard_cap:.2f}",
            "threshold_efficient_cap",
            pd.Series(hard_cap, index=df.index),
        )
    ]
    for strength in (0.25, 0.50, 0.75, 1.00):
        cap = hard_cap - strength * (hard_cap - min_cap) * risk
        rows.append((f"smooth_risk_s{strength:.2f}", "continuous_mechanism_risk", cap.clip(lower=min_cap)))
    for strength in (0.25, 0.50, 0.75):
        mixed = (0.55 * risk + 0.25 * router_instability + 0.20 * low_evidence).clip(0.0, 1.0)
        cap = hard_cap - strength * (hard_cap - min_cap) * mixed
        rows.append((f"router_evidence_risk_s{strength:.2f}", "router_and_evidence_weighted_risk", cap.clip(lower=min_cap)))
    for strength in (0.25, 0.50):
        mixed = (0.60 * risk + 0.40 * load_pressure).clip(0.0, 1.0)
        cap = hard_cap - strength * (hard_cap - min_cap) * mixed
        rows.append((f"load_aware_risk_s{strength:.2f}", "load_weighted_risk", cap.clip(lower=min_cap)))
    return rows


def metrics_from_scale(
    df: pd.DataFrame,
    scale: pd.Series,
    target_cap: pd.Series,
    *,
    candidate_id: str,
    candidate_family: str,
    hard_cap: float,
) -> dict[str, Any]:
    audit_max = df["audit_max_relative_delta_norm"].clip(lower=0.0)
    predicted_max = audit_max * scale
    route_mass = df["total_topk_fraction"].clip(lower=0.0)
    original_nonbase = df["original_effective_nonbase_weight"].clip(lower=0.0)
    audit_delta_norm = df["audit_delta_norm"].clip(lower=0.0)
    original_mass = float((route_mass * original_nonbase).sum())
    preserved_mass = float((route_mass * original_nonbase * scale).sum())
    original_norm = float(math.sqrt(float((audit_delta_norm**2).sum())))
    predicted_norm = float(math.sqrt(float(((audit_delta_norm * scale) ** 2).sum())))
    risk = df["mechanism_risk_score"].clip(0.0, 1.0)
    risk_weight = (route_mass * risk).clip(lower=0.0)
    risk_weight_sum = float(risk_weight.sum())
    risk_weighted_delta = float((risk_weight * predicted_max).sum() / max(EPS, risk_weight_sum))
    hard_violation = (predicted_max > hard_cap + 1e-9)
    retention = preserved_mass / max(EPS, original_mass)
    norm_ratio = predicted_norm / max(EPS, original_norm)
    objective = (
        1500.0 * int(hard_violation.sum())
        + 40.0 * max(0.0, float(predicted_max.max()) - hard_cap)
        + 12.0 * (1.0 - retention)
        + 1.0 * norm_ratio
        + 0.25 * risk_weighted_delta
    )
    return {
        "candidate_id": candidate_id,
        "candidate_family": candidate_family,
        "scaled_group_count": int((scale < 0.999999).sum()),
        "mean_target_cap": float(target_cap.mean()),
        "min_target_cap": float(target_cap.min()),
        "max_target_cap": float(target_cap.max()),
        "max_predicted_relative_delta": float(predicted_max.max()),
        "p99_predicted_relative_delta": float(predicted_max.quantile(0.99)),
        "routed_gt_hard_cap_groups": int(hard_violation.sum()),
        "routed_gt_075_groups": int((predicted_max > 0.75 + 1e-9).sum()),
        "routed_gt_065_groups": int((predicted_max > 0.65 + 1e-9).sum()),
        "routed_gt_050_groups": int((predicted_max > 0.50 + 1e-9).sum()),
        "route_mass_weighted_original_nonbase": original_mass,
        "route_mass_weighted_preserved_nonbase": preserved_mass,
        "nonbase_mass_retention": retention,
        "delta_norm_proxy": predicted_norm,
        "delta_norm_proxy_ratio_vs_uncapped": norm_ratio,
        "risk_weighted_predicted_relative_delta": risk_weighted_delta,
        "mean_delta_scale": float(scale.mean()),
        "min_delta_scale": float(scale.min()),
        "unified_objective": objective,
        "passes_hard_cap": bool(not hard_violation.any()),
    }


def search_candidates(df: pd.DataFrame, *, hard_cap: float, min_cap: float) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows = []
    scales: dict[str, pd.Series] = {}
    for candidate_id, family, target_cap in candidate_target_caps(df, hard_cap=hard_cap, min_cap=min_cap):
        scale = scale_from_target_cap(df, target_cap)
        rows.append(
            metrics_from_scale(
                df,
                scale,
                target_cap,
                candidate_id=candidate_id,
                candidate_family=family,
                hard_cap=hard_cap,
            )
        )
        scales[candidate_id] = scale
    search = pd.DataFrame(rows).sort_values(
        [
            "passes_hard_cap",
            "nonbase_mass_retention",
            "unified_objective",
            "delta_norm_proxy_ratio_vs_uncapped",
        ],
        ascending=[False, False, True, True],
    )
    return search, scales


def select_candidate(search: pd.DataFrame) -> pd.Series:
    feasible = search[search["passes_hard_cap"]].copy()
    if feasible.empty:
        return search.sort_values(["routed_gt_hard_cap_groups", "unified_objective"]).iloc[0]
    best_retention = float(feasible["nonbase_mass_retention"].max())
    threshold_efficient = feasible[feasible["nonbase_mass_retention"] >= best_retention - 1e-12]
    return threshold_efficient.sort_values(["unified_objective", "delta_norm_proxy_ratio_vs_uncapped"]).iloc[0]


def extract_writer_context(command_path: Path) -> tuple[str, dict[str, str]]:
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


def build_rules_and_commands(
    df: pd.DataFrame,
    scale: pd.Series,
    output_dir: Path,
    writer_context_command: Path,
) -> dict[str, Any]:
    base_path, sources = extract_writer_context(writer_context_command)
    rules_path = output_dir / "tensor_rules.txt"
    checkpoint_output_dir = "results/checkpoints/qwen3_moe_unified_mechanism_candidate"
    with rules_path.open("w", encoding="utf-8") as handle:
        handle.write("# Unified mechanism-aware Qwen3 MoE expert rules.\n")
        handle.write("# Router and shared attention are frozen by omission / --freeze-router.\n")
        for idx, row in df.sort_values(["layer_id", "expert_id"]).iterrows():
            coder_weight = float(row["original_weight_coder"]) * float(scale.loc[idx])
            instruct_weight = float(row["original_weight_instruct"])
            handle.write(
                f"{row['tensor_pattern']}::{BASE_SOURCE}={instruct_weight:.6g},{NONBASE_SOURCE}={coder_weight:.6g}\n"
            )
    materialize_command = (
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
    writer_command_path.write_text(materialize_command + "\n", encoding="utf-8")
    dry_run_command_path.write_text(materialize_command + " --dry-run\n", encoding="utf-8")
    return {
        "tensor_rules": rel(rules_path),
        "writer_command": rel(writer_command_path),
        "dry_run_command": rel(dry_run_command_path),
        "checkpoint_output_dir": checkpoint_output_dir,
    }


def parse_tensor_rules(path: Path) -> dict[str, dict[str, float]]:
    rules: dict[str, dict[str, float]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pattern, raw_weights = line.split("::", 1)
        weights: dict[str, float] = {}
        for part in raw_weights.split(","):
            name, value = part.split("=", 1)
            weights[name] = float(value)
        rules[pattern] = weights
    return rules


def compare_tensor_rules(left: Path, right: Path) -> dict[str, Any]:
    left_rules = parse_tensor_rules(left)
    if not right.exists():
        return {
            "validated_reference_rules": rel(right),
            "validated_reference_exists": False,
            "matches_validated_reference_rules": False,
            "reference_rule_count": 0,
            "candidate_rule_count": len(left_rules),
            "max_reference_weight_abs_diff": None,
        }
    right_rules = parse_tensor_rules(right)
    missing_patterns = sorted(set(right_rules) - set(left_rules))
    extra_patterns = sorted(set(left_rules) - set(right_rules))
    max_diff = 0.0
    for pattern in set(left_rules) & set(right_rules):
        sources = set(left_rules[pattern]) | set(right_rules[pattern])
        for source in sources:
            max_diff = max(max_diff, abs(left_rules[pattern].get(source, 0.0) - right_rules[pattern].get(source, 0.0)))
    return {
        "validated_reference_rules": rel(right),
        "validated_reference_exists": True,
        "matches_validated_reference_rules": not missing_patterns and not extra_patterns and max_diff <= 1e-12,
        "reference_rule_count": len(right_rules),
        "candidate_rule_count": len(left_rules),
        "missing_reference_patterns": len(missing_patterns),
        "extra_candidate_patterns": len(extra_patterns),
        "max_reference_weight_abs_diff": max_diff,
    }


def mechanism_feature_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "mean_mechanism_risk_score": float(df["mechanism_risk_score"].mean()),
        "p95_mechanism_risk_score": float(df["mechanism_risk_score"].quantile(0.95)),
        "high_risk_group_count": int((df["mechanism_risk_score"] >= 0.75).sum()),
        "flag_high_load_group_count": int(df["feature_high_load_flag"].sum()),
        "flag_low_route_evidence_group_count": int(df["feature_low_route_evidence_flag"].sum()),
        "flag_fragile_router_group_count": int(df["feature_fragile_router_flag"].sum()),
        "flag_category_mismatch_group_count": int(df["feature_category_mismatch_flag"].sum()),
        "flag_shared_mixed_group_count": int(df["feature_shared_mixed_flag"].sum()),
    }


def build_group_rules(df: pd.DataFrame, scale: pd.Series, selected: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["selected_candidate_id"] = str(selected["candidate_id"])
    out["selected_scale"] = scale
    out["selected_weight_coder"] = out["original_weight_coder"] * out["selected_scale"]
    out["selected_weight_instruct"] = out["original_weight_instruct"]
    out["selected_expected_max_relative_delta_norm"] = out["audit_max_relative_delta_norm"] * out["selected_scale"]
    columns = [
        "layer_id",
        "expert_id",
        "total_topk_fraction",
        "dominant_source",
        "original_weight_instruct",
        "original_weight_coder",
        "selected_weight_instruct",
        "selected_weight_coder",
        "selected_scale",
        "audit_max_relative_delta_norm",
        "selected_expected_max_relative_delta_norm",
        "mechanism_risk_score",
        "feature_delta_pressure",
        "feature_router_instability",
        "feature_load_pressure",
        "feature_source_conflict",
        "feature_low_route_mass",
        "trust_risk_flags",
        "tensor_pattern",
        "tensor_rule",
    ]
    return out[columns].sort_values(["layer_id", "expert_id"])


def build_report(summary: dict[str, Any], search: pd.DataFrame, selected: pd.Series) -> str:
    lines = [
        "# Qwen3 MoE Unified Mechanism Candidate",
        "",
        "这个实验把“Average”写成同结构约束优化，而不是命名算法选择：先冻结高风险 router，保持 expert identity，再用真实 route mass、router fragility、load、source conflict 和 safetensors delta probe 生成 per-expert 缩放。",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Expert groups: `{summary['expert_group_count']}`",
        f"- Searched candidates: `{summary['candidate_count']}`",
        f"- Selected candidate: `{summary['selected_candidate_id']}`",
        f"- Selection family: `{summary['selected_candidate_family']}`",
        f"- Nonbase route-mass retention: `{fmt(summary['selected_nonbase_mass_retention'])}`",
        f"- Max predicted routed relative delta: `{fmt(summary['selected_max_predicted_relative_delta'])}`",
        f"- Groups over hard cap `{summary['hard_cap']}`: `{summary['selected_routed_gt_hard_cap_groups']}`",
        f"- Matches validated no-gt-0.65 rules: `{summary['matches_validated_reference_rules']}`",
        "",
        "## Why This Is The Current Unified Rule",
        "",
        "理论上，uniform average、task-vector merge、TIES、Fisher/RegMean 都可以看成在同一个参数空间里求一个同结构解；真正的区别是约束和局部几何假设。对当前 Qwen3 MoE，最强的内部证据不是“某个算法名更好”，而是 router/top-k dispatch 对扰动敏感、expert identity 必须先固定、routed expert 的 high-delta tail 必须被限制。",
        "",
        "因此本脚本求的是：在不改结构、不改 router、不增加 expert 的条件下，最大化 route-mass-weighted Coder contribution，同时让 routed expert 的预测 relative-delta tail 不超过 hard cap。更复杂的 risk-dependent cap 也参与搜索；如果它只降低 retention 而不进一步降低 hard-tail violation，就被自动拒绝。",
        "",
        "## Candidate Search",
        "",
        "| candidate | family | pass cap | retention | norm ratio | max rel-delta | risk-weighted rel-delta | objective |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in search.head(12).iterrows():
        lines.append(
            f"| `{row['candidate_id']}` | `{row['candidate_family']}` | `{bool(row['passes_hard_cap'])}` | "
            f"{fmt(float(row['nonbase_mass_retention']))} | "
            f"{fmt(float(row['delta_norm_proxy_ratio_vs_uncapped']))} | "
            f"{fmt(float(row['max_predicted_relative_delta']))} | "
            f"{fmt(float(row['risk_weighted_predicted_relative_delta']))} | "
            f"{fmt(float(row['unified_objective']))} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Constraints",
            "",
            "- Expert identity: `same-name expert average only after identity gate`; current Qwen3 gate allows identity expert rules.",
            "- Router: `freeze_router`; router calibration remains a separately gated ablation.",
            "- Shared attention: frozen in this candidate because delta frontier says attention utility needs downstream eval, not norm-only evidence.",
            "- Endpoint fallback: downstream selector must still reject this candidate if source endpoints dominate it.",
            "",
            "## Literature Priors",
            "",
        ]
    )
    for item in LITERATURE_PRIORS:
        lines.append(f"- `{item['key']}`: {item['source']}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified mechanism-aware Qwen3 MoE same-shape candidate.")
    parser.add_argument(
        "--expert-rules",
        type=Path,
        default=Path("results/qwen3_moe_trust_region_candidate/trust_region_source_weights_by_expert.csv"),
    )
    parser.add_argument(
        "--writer-context-command",
        type=Path,
        default=Path("results/qwen3_moe_expert_only_trust_region_candidate/writer_command.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_unified_mechanism_candidate"))
    parser.add_argument("--hard-cap", type=float, default=0.65)
    parser.add_argument("--min-cap", type=float, default=0.55)
    parser.add_argument(
        "--validated-reference-rules",
        type=Path,
        default=Path("results/qwen3_moe_trust_region_cap_search/searched_no_gt065_max_retention_tensor_rules.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(repo_path(args.expert_rules))
    df = add_mechanism_features(df)
    search, scales = search_candidates(df, hard_cap=args.hard_cap, min_cap=args.min_cap)
    selected = select_candidate(search)
    selected_id = str(selected["candidate_id"])
    selected_scale = scales[selected_id]

    search_path = output_dir / "candidate_search.csv"
    group_rules_path = output_dir / "unified_group_rules.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    selected_path = output_dir / "selected_candidate.json"

    group_rules = build_group_rules(df, selected_scale, selected)
    artifacts = build_rules_and_commands(df, selected_scale, output_dir, repo_path(args.writer_context_command))
    reference_check = compare_tensor_rules(output_dir / "tensor_rules.txt", repo_path(args.validated_reference_rules))
    search.to_csv(search_path, index=False)
    group_rules.to_csv(group_rules_path, index=False)

    selected_record = json_safe(selected.to_dict())
    selected_path.write_text(json.dumps(selected_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "status": "unified_mechanism_candidate_ready",
        "expert_group_count": int(len(df)),
        "candidate_count": int(len(search)),
        "hard_cap": float(args.hard_cap),
        "min_cap": float(args.min_cap),
        "selected_candidate_id": selected_id,
        "selected_candidate_family": str(selected["candidate_family"]),
        "selected_nonbase_mass_retention": float(selected["nonbase_mass_retention"]),
        "selected_delta_norm_proxy_ratio_vs_uncapped": float(selected["delta_norm_proxy_ratio_vs_uncapped"]),
        "selected_max_predicted_relative_delta": float(selected["max_predicted_relative_delta"]),
        "selected_routed_gt_hard_cap_groups": int(selected["routed_gt_hard_cap_groups"]),
        "selected_routed_gt_065_groups": int(selected["routed_gt_065_groups"]),
        "selected_routed_gt_075_groups": int(selected["routed_gt_075_groups"]),
        "selected_scaled_group_count": int(selected["scaled_group_count"]),
        "selected_mean_delta_scale": float(selected["mean_delta_scale"]),
        **reference_check,
        "router_policy": "freeze_router",
        "shared_attention_policy": "freeze_shared_attention_pending_downstream_eval",
        "selection_rule": (
            "Select the feasible candidate that satisfies the hard routed-expert cap and maximizes "
            "route-mass-weighted nonbase retention; use the unified objective only as a tie-breaker."
        ),
        "mechanism_features": mechanism_feature_summary(df),
        "literature_priors": LITERATURE_PRIORS,
        "outputs": {
            "report": rel(report_path),
            "summary": rel(summary_path),
            "selected_candidate": rel(selected_path),
            "candidate_search": rel(search_path),
            "unified_group_rules": rel(group_rules_path),
            **artifacts,
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, search, selected), encoding="utf-8")
    print(f"Wrote Qwen3 MoE unified mechanism candidate to {output_dir.resolve()}")
    print(
        f"Selected {selected_id}: retention={float(selected['nonbase_mass_retention']):.6f}, "
        f"max_rel_delta={float(selected['max_predicted_relative_delta']):.6f}, "
        f"hard_cap_violations={int(selected['routed_gt_hard_cap_groups'])}"
    )


if __name__ == "__main__":
    main()
