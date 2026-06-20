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
        "key": "expert_merging_chunking",
        "url": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Importance-guided layer/chunk coefficients address inter-layer heterogeneity in model merging.",
    },
    {
        "key": "harc_routing_breakdown",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Router perturbation can break MoE top-k dispatch; this candidate keeps router frozen.",
    },
    {
        "key": "router_kd_calibration",
        "url": "https://arxiv.org/abs/2603.02217",
        "mechanism": "Router-expert mismatch should be handled by a separate calibrated router delta, not direct router averaging.",
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


def norm01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def replace_cli_arg(command: str, option: str, value: str) -> str:
    tokens = shlex.split(command)
    updated = False
    for idx, token in enumerate(tokens):
        if token == option and idx + 1 < len(tokens):
            tokens[idx + 1] = value
            updated = True
            break
        prefix = option + "="
        if token.startswith(prefix):
            tokens[idx] = f"{option}={value}"
            updated = True
            break
    if not updated:
        tokens.extend([option, value])
    return " ".join(shlex.quote(token) for token in tokens)


def load_inputs(group_rules_path: Path, chunking_path: Path) -> pd.DataFrame:
    groups = pd.read_csv(repo_path(group_rules_path))
    chunks = pd.read_csv(repo_path(chunking_path))
    chunks = chunks.rename(columns={"layer": "layer_id"})
    keep = [
        "layer_id",
        "layer_importance_score",
        "chunk_policy",
        "recommended_coefficient_slots",
        "calibrate_fraction",
        "router_relative_delta_norm",
        "min_topk_jaccard",
    ]
    merged = groups.merge(chunks[keep], on="layer_id", how="left")
    merged["layer_importance_score"] = pd.to_numeric(
        merged["layer_importance_score"], errors="coerce"
    ).fillna(0.0)
    merged["layer_importance_norm"] = norm01(merged["layer_importance_score"])
    merged["risk_norm"] = norm01(merged["mechanism_risk_score"])
    merged["route_mass"] = pd.to_numeric(merged["total_topk_fraction"], errors="coerce").fillna(0.0)
    merged["base_coder_weight"] = pd.to_numeric(
        merged["selected_weight_coder"], errors="coerce"
    ).fillna(0.0)
    merged["base_instruct_weight"] = pd.to_numeric(
        merged["selected_weight_instruct"], errors="coerce"
    ).fillna(0.0)
    merged["base_expected_delta"] = pd.to_numeric(
        merged["selected_expected_max_relative_delta_norm"], errors="coerce"
    ).fillna(0.0)
    return merged


def schedule_coefficients(df: pd.DataFrame, schedule_id: str) -> pd.Series:
    if schedule_id == "baseline_unified":
        return pd.Series(1.0, index=df.index)
    if schedule_id.startswith("continuous_importance_s"):
        strength = float(schedule_id.rsplit("s", 1)[1])
        return (1.0 - strength * df["layer_importance_norm"]).clip(0.0, 1.0)
    if schedule_id.startswith("continuous_risk_layer_s"):
        strength = float(schedule_id.rsplit("s", 1)[1])
        mixed = (0.55 * df["layer_importance_norm"] + 0.45 * df["risk_norm"]).clip(0.0, 1.0)
        return (1.0 - strength * mixed).clip(0.0, 1.0)
    policy_values = {
        "policy_098_099_100": (0.98, 0.99, 1.00),
        "policy_095_098_100": (0.95, 0.98, 1.00),
        "policy_092_096_100": (0.92, 0.96, 1.00),
        "policy_090_095_100": (0.90, 0.95, 1.00),
        "policy_085_092_100": (0.85, 0.92, 1.00),
        "policy_080_090_098": (0.80, 0.90, 0.98),
    }
    if schedule_id not in policy_values:
        raise ValueError(f"Unknown schedule: {schedule_id}")
    per_layer, two_layer, coarse = policy_values[schedule_id]
    coeff = pd.Series(coarse, index=df.index)
    coeff.loc[df["chunk_policy"] == "two_layer_chunk_coefficients"] = two_layer
    coeff.loc[df["chunk_policy"] == "per_layer_coefficients"] = per_layer
    return coeff.clip(0.0, 1.0)


def candidate_schedules() -> list[str]:
    return [
        "baseline_unified",
        "policy_098_099_100",
        "policy_095_098_100",
        "policy_092_096_100",
        "policy_090_095_100",
        "policy_085_092_100",
        "policy_080_090_098",
        "continuous_importance_s0.05",
        "continuous_importance_s0.10",
        "continuous_importance_s0.15",
        "continuous_importance_s0.20",
        "continuous_risk_layer_s0.05",
        "continuous_risk_layer_s0.10",
        "continuous_risk_layer_s0.15",
        "continuous_risk_layer_s0.20",
    ]


def metrics_from_coefficients(df: pd.DataFrame, coeff: pd.Series, schedule_id: str) -> dict[str, Any]:
    base_mass = float((df["route_mass"] * df["base_coder_weight"]).sum())
    mass = float((df["route_mass"] * df["base_coder_weight"] * coeff).sum())
    base_delta = df["base_expected_delta"].clip(lower=0.0)
    predicted_delta = base_delta * coeff
    risk_weight = (
        df["route_mass"].clip(lower=0.0)
        * (0.50 * df["risk_norm"].clip(0.0, 1.0) + 0.50 * df["layer_importance_norm"].clip(0.0, 1.0))
    ).clip(lower=0.0)
    risk_weight_sum = float(risk_weight.sum())
    base_risk_delta = float((risk_weight * base_delta).sum() / max(EPS, risk_weight_sum))
    risk_delta = float((risk_weight * predicted_delta).sum() / max(EPS, risk_weight_sum))
    high_mask = df["chunk_policy"] == "per_layer_coefficients"
    high_base_mass = float((df.loc[high_mask, "route_mass"] * df.loc[high_mask, "base_coder_weight"]).sum())
    high_mass = float(
        (df.loc[high_mask, "route_mass"] * df.loc[high_mask, "base_coder_weight"] * coeff.loc[high_mask]).sum()
    )
    delta_norm = float(math.sqrt(float((predicted_delta**2).sum())))
    base_delta_norm = float(math.sqrt(float((base_delta**2).sum())))
    retention = mass / max(EPS, base_mass)
    risk_reduction = 1.0 - (risk_delta / max(EPS, base_risk_delta))
    return {
        "schedule_id": schedule_id,
        "min_layer_coefficient": float(coeff.min()),
        "mean_layer_coefficient": float(coeff.mean()),
        "route_mass_weighted_coder_retention": retention,
        "fine_layer_coder_retention": high_mass / max(EPS, high_base_mass),
        "risk_weighted_delta": risk_delta,
        "risk_weighted_delta_reduction": risk_reduction,
        "delta_norm_proxy": delta_norm,
        "delta_norm_proxy_ratio": delta_norm / max(EPS, base_delta_norm),
        "max_predicted_relative_delta": float(predicted_delta.max()),
        "p99_predicted_relative_delta": float(predicted_delta.quantile(0.99)),
        "groups_gt_065": int((predicted_delta > 0.65 + 1e-9).sum()),
        "groups_gt_050": int((predicted_delta > 0.50 + 1e-9).sum()),
        "changed_group_count": int((coeff < 0.999999).sum()),
        "objective": risk_reduction - 0.35 * (1.0 - retention),
    }


def search_schedules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows = []
    coeffs: dict[str, pd.Series] = {}
    for schedule_id in candidate_schedules():
        coeff = schedule_coefficients(df, schedule_id)
        coeffs[schedule_id] = coeff
        rows.append(metrics_from_coefficients(df, coeff, schedule_id))
    search = pd.DataFrame(rows).sort_values(
        ["objective", "route_mass_weighted_coder_retention", "risk_weighted_delta_reduction"],
        ascending=[False, False, False],
    )
    return search, coeffs


def add_selection_constraints(search: pd.DataFrame, *, min_retention: float, hard_cap: float) -> pd.DataFrame:
    search = search.copy()
    search["passes_retention_constraint"] = (
        search["route_mass_weighted_coder_retention"] >= min_retention
    )
    search["passes_hard_cap_constraint"] = (
        search["max_predicted_relative_delta"] <= hard_cap + 1e-9
    )
    search["feasible_for_selection"] = (
        search["passes_retention_constraint"] & search["passes_hard_cap_constraint"]
    )
    return search


def select_schedule(search: pd.DataFrame, *, min_retention: float, hard_cap: float) -> pd.Series:
    feasible = search[
        (search["route_mass_weighted_coder_retention"] >= min_retention)
        & (search["max_predicted_relative_delta"] <= hard_cap + 1e-9)
    ].copy()
    if feasible.empty:
        return search.sort_values(["route_mass_weighted_coder_retention", "objective"], ascending=[False, False]).iloc[0]
    return feasible.sort_values(["objective", "risk_weighted_delta_reduction"], ascending=[False, False]).iloc[0]


def build_layer_coefficients(df: pd.DataFrame, coeff: pd.Series, selected_id: str) -> pd.DataFrame:
    out = df[[
        "layer_id",
        "layer_importance_score",
        "layer_importance_norm",
        "chunk_policy",
        "recommended_coefficient_slots",
        "calibrate_fraction",
        "router_relative_delta_norm",
        "min_topk_jaccard",
    ]].drop_duplicates("layer_id").copy()
    layer_coeff = pd.DataFrame({"layer_id": df["layer_id"], "group_coeff": coeff}).groupby("layer_id", as_index=False).mean()
    out = out.merge(layer_coeff, on="layer_id", how="left")
    out["schedule_id"] = selected_id
    out = out.rename(columns={"group_coeff": "selected_layer_coder_coefficient"})
    return out.sort_values(["chunk_policy", "layer_importance_score"], ascending=[False, False])


def build_selected_group_rules(df: pd.DataFrame, coeff: pd.Series, selected_id: str) -> pd.DataFrame:
    out = df.copy()
    out["schedule_id"] = selected_id
    out["layer_chunk_coder_coefficient"] = coeff
    out["layer_chunk_weight_coder"] = out["base_coder_weight"] * coeff
    out["layer_chunk_weight_instruct"] = out["base_instruct_weight"]
    out["layer_chunk_expected_relative_delta"] = out["base_expected_delta"] * coeff
    columns = [
        "layer_id",
        "expert_id",
        "total_topk_fraction",
        "dominant_source",
        "base_instruct_weight",
        "base_coder_weight",
        "layer_chunk_weight_instruct",
        "layer_chunk_weight_coder",
        "layer_chunk_coder_coefficient",
        "base_expected_delta",
        "layer_chunk_expected_relative_delta",
        "mechanism_risk_score",
        "layer_importance_score",
        "chunk_policy",
        "trust_risk_flags",
        "tensor_pattern",
    ]
    return out[columns].sort_values(["layer_id", "expert_id"])


def write_tensor_rules(group_rules: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "tensor_rules.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Qwen3 MoE layer/chunk sensitivity candidate rules.\n")
        handle.write("# Router and shared attention remain frozen; only routed expert Coder coefficients are shrunk.\n")
        for _, row in group_rules.iterrows():
            handle.write(
                f"{row['tensor_pattern']}::{BASE_SOURCE}={float(row['layer_chunk_weight_instruct']):.6g},"
                f"{NONBASE_SOURCE}={float(row['layer_chunk_weight_coder']):.6g}\n"
            )
    return path


def read_dry_run_summary(checkpoint_output_dir: str) -> dict[str, Any]:
    manifest_path = repo_path(checkpoint_output_dir) / "merge_manifest.json"
    summary: dict[str, Any] = {
        "dry_run_validated": False,
        "dry_run_manifest": rel(manifest_path),
    }
    if not manifest_path.exists():
        return summary
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        summary["dry_run_error"] = str(exc)
        return summary

    rule_counts = manifest.get("rule_counts", {})
    if not isinstance(rule_counts, dict):
        rule_counts = {}
    tensor_rule_hit_count = sum(
        int(value) for key, value in rule_counts.items() if str(key).startswith("tensor_rule:")
    )
    tensor_rule_count = sum(1 for key in rule_counts if str(key).startswith("tensor_rule:"))
    floating_tensors = int(manifest.get("floating_tensors", 0) or 0)
    frozen_tensors = int(manifest.get("frozen_tensors", 0) or 0)
    freeze_router = bool(manifest.get("freeze_router", False))
    dry_run = bool(manifest.get("dry_run", False))
    summary.update(
        {
            "dry_run_validated": dry_run and freeze_router and floating_tensors > 0 and tensor_rule_hit_count > 0,
            "dry_run": dry_run,
            "dry_run_floating_tensors": floating_tensors,
            "dry_run_frozen_tensors": frozen_tensors,
            "dry_run_freeze_router": freeze_router,
            "dry_run_default_tensor_count": int(rule_counts.get("default", 0) or 0),
            "dry_run_freeze_router_hits": int(rule_counts.get("freeze_router", 0) or 0),
            "dry_run_tensor_rule_count": tensor_rule_count,
            "dry_run_tensor_rule_hit_count": tensor_rule_hit_count,
            "dry_run_method_counts": manifest.get("method_counts", {}),
            "dry_run_default_weights": manifest.get("default_weights", {}),
        }
    )
    return summary


def build_commands(writer_command_path: Path, rules_path: Path, checkpoint_output_dir: str, output_dir: Path) -> dict[str, str]:
    command = repo_path(writer_command_path).read_text(encoding="utf-8").strip()
    command = replace_cli_arg(command, "--tensor-rule-file", rel(rules_path))
    command = replace_cli_arg(command, "--output-dir", checkpoint_output_dir)
    writer_path = output_dir / "writer_command.txt"
    dry_run_path = output_dir / "dry_run_command.txt"
    writer_path.write_text(command + "\n", encoding="utf-8")
    dry_run_path.write_text(command + " --dry-run\n", encoding="utf-8")
    return {
        "writer_command": rel(writer_path),
        "dry_run_command": rel(dry_run_path),
        "checkpoint_output_dir": checkpoint_output_dir,
    }


def fmt(value: Any) -> str:
    value = clean_value(value)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_report(summary: dict[str, Any], search: pd.DataFrame, layers: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Layer/Chunk Coefficient Candidate",
        "",
        "这个候选把 mechanism leverage map 里的 layer/chunk 敏感性变成同结构 tensor rules：router 和 shared attention 继续冻结，只对高敏感 routed experts 的 Coder contribution 做小幅 shrink。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Selected schedule: `{summary['selected_schedule_id']}`",
        f"- Route-mass Coder retention: `{fmt(summary['selected_route_mass_weighted_coder_retention'])}`",
        f"- Risk-weighted delta reduction: `{fmt(summary['selected_risk_weighted_delta_reduction'])}`",
        f"- Fine-layer Coder retention: `{fmt(summary['selected_fine_layer_coder_retention'])}`",
        f"- Max predicted relative delta: `{fmt(summary['selected_max_predicted_relative_delta'])}`",
        (
            f"- Selection constraints: retention >= `{fmt(summary['selection_min_retention'])}`, "
            f"max relative delta <= `{fmt(summary['selection_hard_cap'])}`"
        ),
        (
            f"- Writer dry-run: `{summary['dry_run_validated']}` "
            f"({summary.get('dry_run_floating_tensors', 0)} floating tensors, "
            f"{summary.get('dry_run_frozen_tensors', 0)} frozen tensors, "
            f"{summary.get('dry_run_tensor_rule_hit_count', 0)} tensor-rule hits)"
        ),
        "",
        "## Candidate Search",
        "",
        "| schedule | feasible | retention | fine retention | risk reduction | delta norm ratio | max rel-delta | changed groups | objective |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in search.iterrows():
        lines.append(
            f"| `{row['schedule_id']}` | `{bool(row.get('feasible_for_selection', False))}` | "
            f"{fmt(row['route_mass_weighted_coder_retention'])} | "
            f"{fmt(row['fine_layer_coder_retention'])} | {fmt(row['risk_weighted_delta_reduction'])} | "
            f"{fmt(row['delta_norm_proxy_ratio'])} | {fmt(row['max_predicted_relative_delta'])} | "
            f"{int(row['changed_group_count'])} | {fmt(row['objective'])} |"
        )
    lines.extend(
        [
            "",
            "## Layer Coefficients",
            "",
            "| layer | policy | coeff | importance | router rel | min Jaccard |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in layers.head(20).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | `{row['chunk_policy']}` | "
            f"{fmt(row['selected_layer_coder_coefficient'])} | {fmt(row['layer_importance_score'])} | "
            f"{fmt(row['router_relative_delta_norm'])} | {fmt(row['min_topk_jaccard'])} |"
        )
    lines.extend(
        [
            "",
            "## Why This Is A Candidate, Not A Conclusion",
            "",
            "内部 proxy 只能说明这个 schedule 在 retention 和 delta hard-cap 约束内降低了高敏感层的风险加权 Coder delta；它不能证明下游任务更好。Candidate search 里 objective 更高但 feasible 为 false 的 schedule 是因为 route-mass Coder retention 太低，不能直接选。这个 checkpoint 必须和 source、tail-trimmed、searched no-gt-0.65 在同一套 budgeted vLLM eval 下比较。",
            "",
            "## Literature Priors",
            "",
        ]
    )
    for item in LITERATURE_PRIORS:
        lines.append(f"- `{item['key']}`: {item['url']}")
    lines.extend(["", "## Outputs", ""])
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    df: pd.DataFrame,
    search: pd.DataFrame,
    coeffs: dict[str, pd.Series],
    selected: pd.Series,
    *,
    writer_command_path: Path,
    min_retention: float,
    hard_cap: float,
) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_id = str(selected["schedule_id"])
    coeff = coeffs[selected_id]
    layers = build_layer_coefficients(df, coeff, selected_id)
    selected_groups = build_selected_group_rules(df, coeff, selected_id)
    tensor_rules_path = write_tensor_rules(selected_groups, output_dir)
    checkpoint_output_dir = "results/checkpoints/qwen3_moe_layer_chunk_candidate"
    commands = build_commands(
        writer_command_path,
        tensor_rules_path,
        checkpoint_output_dir,
        output_dir,
    )
    dry_run_summary = read_dry_run_summary(checkpoint_output_dir)

    search_path = output_dir / "schedule_search.csv"
    layers_path = output_dir / "layer_coefficients.csv"
    groups_path = output_dir / "selected_group_rules.csv"
    selected_path = output_dir / "selected_schedule.json"
    literature_path = output_dir / "literature_sources.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    search.to_csv(search_path, index=False)
    layers.to_csv(layers_path, index=False)
    selected_groups.to_csv(groups_path, index=False)
    selected_path.write_text(json.dumps(json_safe(selected.to_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    literature_path.write_text(json.dumps(json_safe(LITERATURE_PRIORS), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "status": "layer_chunk_candidate_ready",
        "expert_group_count": int(len(df)),
        "schedule_count": int(len(search)),
        "selected_schedule_id": selected_id,
        "selected_route_mass_weighted_coder_retention": float(selected["route_mass_weighted_coder_retention"]),
        "selected_fine_layer_coder_retention": float(selected["fine_layer_coder_retention"]),
        "selected_risk_weighted_delta_reduction": float(selected["risk_weighted_delta_reduction"]),
        "selected_delta_norm_proxy_ratio": float(selected["delta_norm_proxy_ratio"]),
        "selected_max_predicted_relative_delta": float(selected["max_predicted_relative_delta"]),
        "selected_changed_group_count": int(selected["changed_group_count"]),
        "selection_min_retention": float(min_retention),
        "selection_hard_cap": float(hard_cap),
        "same_shape_policy": "same tensor names/shapes; freeze router and shared attention; shrink Coder coefficient in routed expert tensor rules",
        "selection_rule": "maximize risk-weighted delta reduction under route-mass Coder retention and hard-cap constraints",
        **dry_run_summary,
        "outputs": {
            "report": rel(report_path),
            "summary": rel(summary_path),
            "selected_schedule": rel(selected_path),
            "schedule_search": rel(search_path),
            "layer_coefficients": rel(layers_path),
            "selected_group_rules": rel(groups_path),
            "tensor_rules": rel(tensor_rules_path),
            "literature_sources": rel(literature_path),
            **commands,
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, search, layers), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen3 MoE layer/chunk coefficient candidate.")
    parser.add_argument(
        "--group-rules",
        type=Path,
        default=Path("results/qwen3_moe_unified_mechanism_candidate/unified_group_rules.csv"),
    )
    parser.add_argument(
        "--chunking-plan",
        type=Path,
        default=Path("results/qwen3_moe_mechanism_levers/layer_chunking_plan.csv"),
    )
    parser.add_argument(
        "--writer-command",
        type=Path,
        default=Path("results/qwen3_moe_unified_mechanism_candidate/writer_command.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_layer_chunk_candidate"))
    parser.add_argument("--min-retention", type=float, default=0.975)
    parser.add_argument("--hard-cap", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_inputs(args.group_rules, args.chunking_plan)
    search, coeffs = search_schedules(df)
    search = add_selection_constraints(search, min_retention=args.min_retention, hard_cap=args.hard_cap)
    selected = select_schedule(search, min_retention=args.min_retention, hard_cap=args.hard_cap)
    summary = write_outputs(
        args.output_dir,
        df,
        search,
        coeffs,
        selected,
        writer_command_path=args.writer_command,
        min_retention=args.min_retention,
        hard_cap=args.hard_cap,
    )
    print(f"Wrote Qwen3 MoE layer/chunk candidate to {repo_path(args.output_dir).resolve()}")
    print(
        f"Selected {summary['selected_schedule_id']}: retention="
        f"{summary['selected_route_mass_weighted_coder_retention']:.6f}, "
        f"risk_delta_reduction={summary['selected_risk_weighted_delta_reduction']:.6f}"
    )


if __name__ == "__main__":
    main()
