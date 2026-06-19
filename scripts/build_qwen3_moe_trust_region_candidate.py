#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERT_TENSOR_RE = re.compile(
    r"layers\.(?P<layer>\d+).*experts\.(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight"
)
LAYER_RE = re.compile(r"layers\.(?P<layer>\d+)")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def same_path(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in record.items():
        if isinstance(value, float) and math.isnan(value):
            out[key] = None
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def summarize_dry_run_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "writer_dry_run_validated": False,
            "writer_dry_run_manifest": rel(path),
            "writer_dry_run_expert_tensor_rule_hits": 0,
            "writer_dry_run_shared_attention_hits": 0,
            "writer_dry_run_freeze_router_hits": 0,
            "writer_dry_run_floating_tensors": 0,
            "writer_dry_run_frozen_tensors": 0,
            "writer_dry_run_shards": 0,
        }
    manifest = read_json(path)
    rule_counts = manifest.get("rule_counts") or {}
    expert_hits = sum(
        int(value) for key, value in rule_counts.items() if str(key).startswith("tensor_rule:.*layers\\.")
    )
    return {
        "writer_dry_run_validated": bool(manifest.get("dry_run")),
        "writer_dry_run_manifest": rel(path),
        "writer_dry_run_expert_tensor_rule_hits": int(expert_hits),
        "writer_dry_run_shared_attention_hits": int(rule_counts.get("tensor_rule:.*self_attn.*", 0)),
        "writer_dry_run_freeze_router_hits": int(rule_counts.get("freeze_router", 0)),
        "writer_dry_run_floating_tensors": int(manifest.get("floating_tensors", 0)),
        "writer_dry_run_frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        "writer_dry_run_shards": int(len(manifest.get("shards") or [])),
    }


def summarize_materialized_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "writer_checkpoint_materialized": False,
            "writer_materialized_manifest": rel(path),
            "writer_materialized_floating_tensors": 0,
            "writer_materialized_frozen_tensors": 0,
            "writer_materialized_shards": 0,
        }
    manifest = read_json(path)
    return {
        "writer_checkpoint_materialized": not bool(manifest.get("dry_run", True)),
        "writer_materialized_manifest": rel(path),
        "writer_materialized_floating_tensors": int(manifest.get("floating_tensors", 0)),
        "writer_materialized_frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        "writer_materialized_shards": int(len(manifest.get("shards") or [])),
    }


def source_names_from_columns(source_weights: pd.DataFrame) -> list[str]:
    return sorted(col.removeprefix("weight_") for col in source_weights.columns if col.startswith("weight_"))


def parse_layer(router: str) -> int | None:
    match = LAYER_RE.search(str(router))
    return None if match is None else int(match.group("layer"))


def parse_expert_tensor(name: str) -> tuple[int, int, str] | None:
    match = EXPERT_TENSOR_RE.search(name)
    if match is None:
        return None
    return int(match.group("layer")), int(match.group("expert")), match.group("projection")


def build_expert_audit(delta_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    routed = delta_audit[delta_audit["group"] == "routed_expert_ffn"].copy()
    for _, row in routed.iterrows():
        parsed = parse_expert_tensor(str(row["tensor"]))
        if parsed is None:
            continue
        layer, expert, projection = parsed
        rows.append(
            {
                "layer_id": layer,
                "expert_id": expert,
                "projection": projection,
                "tensor": row["tensor"],
                "changed": bool(row["changed"]),
                "audit_relative_delta_norm": float(row["relative_delta_norm"]),
                "audit_delta_norm": float(row["delta_norm"]),
                "audit_base_norm": float(row["base_norm"]),
                "audit_max_abs_delta": float(row["max_abs_delta"]),
                "audit_numel": int(row["numel"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "layer_id",
                "expert_id",
                "audit_tensor_count",
                "audit_changed_tensors",
                "audit_max_relative_delta_norm",
                "audit_mean_relative_delta_norm",
                "audit_max_abs_delta",
                "audit_delta_norm",
            ]
        )
    grouped = []
    for (layer, expert), group in frame.groupby(["layer_id", "expert_id"], sort=True):
        grouped.append(
            {
                "layer_id": int(layer),
                "expert_id": int(expert),
                "audit_tensor_count": int(len(group)),
                "audit_changed_tensors": int(group["changed"].sum()),
                "audit_max_relative_delta_norm": float(group["audit_relative_delta_norm"].max()),
                "audit_mean_relative_delta_norm": float(group["audit_relative_delta_norm"].mean()),
                "audit_max_abs_delta": float(group["audit_max_abs_delta"].max()),
                "audit_delta_norm": float(math.sqrt(float((group["audit_delta_norm"] ** 2).sum()))),
            }
        )
    return pd.DataFrame(grouped)


def aggregate_expert_load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["layer_id", "expert_id"])
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["layer_id", "expert_id"])
    frame["layer_id"] = frame["router"].map(parse_layer)
    frame = frame.dropna(subset=["layer_id"]).copy()
    frame["layer_id"] = frame["layer_id"].astype(int)
    frame["is_high_load_action"] = frame["recommended_action"].astype(str).eq("protect_or_source_weight_high_load_expert")
    grouped = []
    for (layer, expert), group in frame.groupby(["layer_id", "expert_id"], sort=True):
        grouped.append(
            {
                "layer_id": int(layer),
                "expert_id": int(expert),
                "load_probe_rows": int(len(group)),
                "load_probe_category_count": int(group["category"].astype(str).nunique()),
                "max_topk_over_uniform": float(group["topk_over_uniform"].max()),
                "mean_topk_over_uniform": float(group["topk_over_uniform"].mean()),
                "max_topk_fraction": float(group["topk_fraction"].max()),
                "mean_topk_fraction": float(group["topk_fraction"].mean()),
                "high_load_action_rows": int(group["is_high_load_action"].sum()),
            }
        )
    return pd.DataFrame(grouped)


def aggregate_category_specialization(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["layer_id", "expert_id"])
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["layer_id", "expert_id"])
    frame["layer_id"] = frame["router"].map(parse_layer)
    frame = frame.dropna(subset=["layer_id"]).copy()
    frame["layer_id"] = frame["layer_id"].astype(int)
    frame["is_specialized_action"] = frame["recommended_action"].astype(str).eq("category_specialized_route_weight")
    grouped = []
    for (layer, expert), group in frame.groupby(["layer_id", "expert_id"], sort=True):
        strongest = group.sort_values("dominant_category_share", ascending=False).iloc[0]
        grouped.append(
            {
                "layer_id": int(layer),
                "expert_id": int(expert),
                "specialization_probe_rows": int(len(group)),
                "dominant_category": str(strongest["dominant_category"]),
                "max_dominant_category_share": float(group["dominant_category_share"].max()),
                "mean_dominant_category_share": float(group["dominant_category_share"].mean()),
                "min_categories_observed": int(group["categories_observed"].min()),
                "specialized_action_rows": int(group["is_specialized_action"].sum()),
            }
        )
    return pd.DataFrame(grouped)


def aggregate_router_risk(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["layer_id"])
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["layer_id"])
    frame["layer_id"] = frame["router"].map(parse_layer)
    frame = frame.dropna(subset=["layer_id"]).copy()
    frame["layer_id"] = frame["layer_id"].astype(int)
    frame["is_calibrate_action"] = frame["recommended_action"].astype(str).eq("calibrate_router_before_average")
    grouped = []
    for layer, group in frame.groupby("layer_id", sort=True):
        grouped.append(
            {
                "layer_id": int(layer),
                "router_probe_rows": int(len(group)),
                "router_calibrate_rows": int(group["is_calibrate_action"].sum()),
                "mean_router_risk_score": float(group["risk_score"].mean()),
                "max_router_risk_score": float(group["risk_score"].max()),
                "min_topk_jaccard": float(group["topk_jaccard"].min()),
                "mean_topk_jaccard": float(group["topk_jaccard"].mean()),
                "min_top1_agreement": float(group["top1_agreement"].min()),
                "mean_top1_agreement": float(group["top1_agreement"].mean()),
            }
        )
    return pd.DataFrame(grouped)


def route_to_source(category: str) -> str:
    normalized = str(category).lower().replace("-", "_")
    if "code" in normalized or "program" in normalized or "software" in normalized:
        return "coder"
    return "instruct"


def add_risk_and_scale(row: pd.Series, args: argparse.Namespace, nonbase_sources: list[str]) -> dict[str, Any]:
    flags = []
    target_cap = float(args.base_target_relative_delta)
    max_relative = float(row.get("audit_max_relative_delta_norm", 0.0) or 0.0)
    total_topk = float(row.get("total_topk_fraction", 0.0) or 0.0)
    dominant_source = str(row.get("dominant_source", ""))
    dominant_category = str(row.get("dominant_category", ""))
    expected_source = route_to_source(dominant_category) if dominant_category else ""

    if max_relative > args.base_target_relative_delta:
        flags.append("delta_above_base_cap")
    if float(row.get("max_topk_over_uniform", 0.0) or 0.0) >= args.high_load_over_uniform or int(
        row.get("high_load_action_rows", 0) or 0
    ) > 0:
        flags.append("high_load_expert")
        target_cap = min(target_cap, args.high_load_target_relative_delta)
    if (
        int(row.get("min_categories_observed", 0) or 0) >= args.shared_categories_min
        and float(row.get("max_dominant_category_share", 0.0) or 0.0) <= args.shared_category_share_max
        and total_topk >= args.shared_min_route_fraction
    ):
        flags.append("shared_mixed_expert")
        target_cap = min(target_cap, args.shared_target_relative_delta)
    if (
        float(row.get("mean_topk_jaccard", 1.0) or 1.0) <= args.fragile_mean_topk_jaccard
        or float(row.get("min_topk_jaccard", 1.0) or 1.0) <= args.fragile_min_topk_jaccard
    ):
        flags.append("fragile_router_layer")
        target_cap = min(target_cap, args.fragile_router_target_relative_delta)
    if 0.0 < total_topk < args.low_route_fraction:
        flags.append("low_route_evidence")
        target_cap = min(target_cap, args.low_route_target_relative_delta)
    if (
        expected_source
        and expected_source in {"instruct", "coder"}
        and dominant_source
        and expected_source != dominant_source
        and float(row.get("max_dominant_category_share", 0.0) or 0.0) >= args.category_mismatch_share
    ):
        flags.append("category_source_mismatch")
        target_cap = min(target_cap, args.category_mismatch_target_relative_delta)

    scale = 1.0
    if max_relative > target_cap and max_relative > 0.0:
        scale = min(scale, target_cap / max_relative)

    original_effective = sum(abs(float(row.get(f"weight_{source}", 0.0) or 0.0)) for source in nonbase_sources)
    if "high_load_expert" in flags and original_effective > args.high_load_max_nonbase_weight > 0.0:
        scale = min(scale, args.high_load_max_nonbase_weight / original_effective)
        flags.append("high_load_weight_limit")
    if "shared_mixed_expert" in flags and original_effective > args.shared_max_nonbase_weight > 0.0:
        scale = min(scale, args.shared_max_nonbase_weight / original_effective)
        flags.append("shared_weight_limit")

    scale = max(0.0, min(1.0, scale))
    if scale <= args.freeze_scale_threshold:
        scale = 0.0
        flags.append("freeze_extreme_trust_risk")

    expected_max_relative = max_relative * scale if original_effective > 0.0 else 0.0
    if scale == 0.0 and original_effective > 0.0:
        action = "freeze_nonbase_delta_extreme_trust_risk"
    elif scale < 1.0 and original_effective > 0.0:
        action = "scale_nonbase_delta_to_moe_trust_region"
    elif flags:
        action = "keep_with_moe_risk_monitor"
    else:
        action = "keep_route_weight_rule"

    return {
        "trust_target_relative_delta": target_cap,
        "trust_delta_scale": scale,
        "trust_risk_flags": "|".join(flags) if flags else "",
        "trust_gate_action": action,
        "expected_max_relative_delta_norm": expected_max_relative,
    }


def estimate_tensor_deltas(delta_audit: pd.DataFrame, expert_rules: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    scale_by_expert = {
        (int(row["layer_id"]), int(row["expert_id"])): float(row["trust_delta_scale"])
        for _, row in expert_rules.iterrows()
    }
    rows = []
    for _, row in delta_audit.iterrows():
        group = str(row["group"])
        scale = 1.0
        layer = None if pd.isna(row.get("layer")) else int(row["layer"])
        expert = None
        projection = ""
        if group == "routed_expert_ffn":
            parsed = parse_expert_tensor(str(row["tensor"]))
            if parsed is not None:
                layer, expert, projection = parsed
                scale = scale_by_expert.get((layer, expert), 1.0)
        delta_norm = float(row["delta_norm"]) * scale
        relative_delta_norm = float(row["relative_delta_norm"]) * scale
        rows.append(
            {
                "tensor": row["tensor"],
                "group": group,
                "layer": layer,
                "expert_id": expert,
                "projection": projection,
                "numel": int(row["numel"]),
                "base_norm": float(row["base_norm"]),
                "route_delta_norm": float(row["delta_norm"]),
                "route_relative_delta_norm": float(row["relative_delta_norm"]),
                "trust_delta_scale": scale,
                "estimated_delta_norm": delta_norm,
                "estimated_relative_delta_norm": relative_delta_norm,
                "estimated_changed": bool(delta_norm > 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    total_delta_norm2 = float((frame["estimated_delta_norm"].fillna(0.0) ** 2).sum())
    total_base_norm2 = float((frame["base_norm"].fillna(0.0) ** 2).sum())
    routed = frame[frame["group"] == "routed_expert_ffn"]
    summary = {
        "estimated_total_delta_norm": math.sqrt(total_delta_norm2),
        "estimated_total_base_norm": math.sqrt(total_base_norm2),
        "estimated_relative_delta_norm": math.sqrt(total_delta_norm2) / max(1e-12, math.sqrt(total_base_norm2)),
        "estimated_changed_tensors": int(frame["estimated_changed"].sum()),
        "estimated_routed_tensor_max_relative_delta": float(routed["estimated_relative_delta_norm"].max())
        if not routed.empty
        else 0.0,
        "estimated_routed_tensors_relative_delta_gt_1": int((routed["estimated_relative_delta_norm"] > 1.0).sum())
        if not routed.empty
        else 0,
        "estimated_routed_tensors_relative_delta_gt_075": int((routed["estimated_relative_delta_norm"] > 0.75).sum())
        if not routed.empty
        else 0,
        "estimated_routed_tensors_relative_delta_gt_065": int((routed["estimated_relative_delta_norm"] > 0.65).sum())
        if not routed.empty
        else 0,
    }
    return frame, summary


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    route_dir = repo_path(args.route_candidate_dir)
    audit_dir = repo_path(args.delta_audit_dir)
    readiness_dir = repo_path(args.routing_readiness_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_summary = read_json(route_dir / "summary.json")
    source_weights = pd.read_csv(route_dir / "source_weights_by_expert.csv")
    delta_audit = pd.read_csv(audit_dir / "tensor_delta_audit.csv")
    expert_audit = build_expert_audit(delta_audit)
    expert_load = aggregate_expert_load(readiness_dir / "expert_load_risks.csv")
    specialization = aggregate_category_specialization(readiness_dir / "category_specialization.csv")
    router_risk = aggregate_router_risk(readiness_dir / "router_readiness.csv")

    source_names = source_names_from_columns(source_weights)
    base_path = str(route_summary.get("base", ""))
    source_paths = {str(k): str(v) for k, v in (route_summary.get("source_paths") or {}).items()}
    base_sources = sorted(source for source, path in source_paths.items() if same_path(path, base_path))
    nonbase_sources = [source for source in source_names if source not in set(base_sources)]
    if not nonbase_sources:
        nonbase_sources = [source for source in source_names if source != (base_sources[0] if base_sources else "")]

    merged = source_weights.merge(expert_audit, on=["layer_id", "expert_id"], how="left")
    merged = merged.merge(expert_load, on=["layer_id", "expert_id"], how="left")
    merged = merged.merge(specialization, on=["layer_id", "expert_id"], how="left")
    merged = merged.merge(router_risk, on="layer_id", how="left")

    numeric_defaults = {
        "audit_tensor_count": 0,
        "audit_changed_tensors": 0,
        "audit_max_relative_delta_norm": 0.0,
        "audit_mean_relative_delta_norm": 0.0,
        "audit_max_abs_delta": 0.0,
        "audit_delta_norm": 0.0,
        "load_probe_rows": 0,
        "load_probe_category_count": 0,
        "max_topk_over_uniform": 0.0,
        "mean_topk_over_uniform": 0.0,
        "max_topk_fraction": 0.0,
        "mean_topk_fraction": 0.0,
        "high_load_action_rows": 0,
        "specialization_probe_rows": 0,
        "max_dominant_category_share": 0.0,
        "mean_dominant_category_share": 0.0,
        "min_categories_observed": 0,
        "specialized_action_rows": 0,
        "router_probe_rows": 0,
        "router_calibrate_rows": 0,
        "mean_router_risk_score": 0.0,
        "max_router_risk_score": 0.0,
        "min_topk_jaccard": 1.0,
        "mean_topk_jaccard": 1.0,
        "min_top1_agreement": 1.0,
        "mean_top1_agreement": 1.0,
    }
    for col, default in numeric_defaults.items():
        if col not in merged:
            merged[col] = default
        merged[col] = merged[col].fillna(default)
    merged["dominant_category"] = merged.get("dominant_category", "").fillna("")

    calibrated_rows = []
    for _, row in merged.iterrows():
        risk = add_risk_and_scale(row, args, nonbase_sources)
        out = row.to_dict()
        out.update(risk)
        original_effective = 0.0
        new_effective = 0.0
        for source in source_names:
            original = float(row.get(f"weight_{source}", 0.0) or 0.0)
            if source in nonbase_sources:
                updated = original * float(risk["trust_delta_scale"])
                original_effective += abs(original)
                new_effective += abs(updated)
            else:
                updated = original
            out[f"original_weight_{source}"] = original
            out[f"weight_{source}"] = updated
        out["original_effective_nonbase_weight"] = original_effective
        out["effective_nonbase_weight"] = new_effective
        weight_chunks = [f"{source}={out[f'weight_{source}']:.6g}" for source in source_names]
        out["tensor_rule"] = f"{row['tensor_pattern']}::" + ",".join(weight_chunks)
        calibrated_rows.append(out)

    calibrated = pd.DataFrame(calibrated_rows)
    calibrated_csv = output_dir / "trust_region_source_weights_by_expert.csv"
    calibrated.to_csv(calibrated_csv, index=False)

    estimated_tensor_delta, estimated_summary = estimate_tensor_deltas(delta_audit, calibrated)
    estimated_tensor_csv = output_dir / "estimated_tensor_delta.csv"
    estimated_tensor_delta.to_csv(estimated_tensor_csv, index=False)

    tensor_rules_path = output_dir / "tensor_rules.txt"
    attention_weights = route_summary.get("shared_attention_weights") or {}
    attention_rule = ",".join(f"{source}={float(attention_weights.get(source, 0.0)):.6g}" for source in source_names)
    with tensor_rules_path.open("w", encoding="utf-8") as handle:
        handle.write("# Shared attention rule. It remains a small shared step; MoE trust region gates routed experts only.\n")
        handle.write(f".*self_attn.*::{attention_rule}\n")
        handle.write("# MoE trust-region expert rules from route load, specialization, router fragility, and delta audit probes.\n")
        for _, row in calibrated.sort_values(["layer_id", "expert_id"]).iterrows():
            handle.write(str(row["tensor_rule"]) + "\n")

    checkpoint_output_dir = str(args.checkpoint_output_dir)
    dry_run_output_dir = str(args.dry_run_output_dir)
    source_args = " ".join(f"--source {source}={source_paths[source]}" for source in source_names)
    source_weight_args = " ".join(f"--source-weight {source}=0.0" for source in source_names)
    materialize_command = (
        "python scripts/write_same_shape_average_checkpoint.py "
        f"--base {base_path} "
        f"{source_args} "
        f"{source_weight_args} "
        "--freeze-router "
        f"--tensor-rule-file {rel(tensor_rules_path)} "
        f"--output-dir {checkpoint_output_dir}"
    )
    dry_run_command = materialize_command.replace(f"--output-dir {checkpoint_output_dir}", f"--output-dir {dry_run_output_dir}") + " --dry-run"

    writer_command_path = output_dir / "writer_command.txt"
    dry_run_command_path = output_dir / "dry_run_command.txt"
    writer_command_path.write_text(materialize_command + "\n", encoding="utf-8")
    dry_run_command_path.write_text(dry_run_command + "\n", encoding="utf-8")

    action_counts = {str(key): int(value) for key, value in calibrated["trust_gate_action"].value_counts().sort_index().items()}
    flag_counter: Counter[str] = Counter()
    for raw in calibrated["trust_risk_flags"].fillna("").astype(str):
        for flag in raw.split("|"):
            if flag:
                flag_counter[flag] += 1
    scaled = calibrated[calibrated["trust_delta_scale"] < 1.0]
    scaled_beyond_delta = scaled[~scaled["trust_risk_flags"].fillna("").astype(str).str.contains("delta_above_base_cap")]

    dry_run_manifest = repo_path(dry_run_output_dir) / "merge_manifest.json"
    materialized_manifest = repo_path(checkpoint_output_dir) / "merge_manifest.json"
    summary = {
        "schema_version": 1,
        "status": "trust_region_rules_ready",
        "base": base_path,
        "source_paths": source_paths,
        "source_names": source_names,
        "base_sources": base_sources,
        "nonbase_sources": nonbase_sources,
        "expert_rule_count": int(len(calibrated)),
        "scaled_expert_rule_count": int(len(scaled)),
        "scaled_beyond_delta_cap_count": int(len(scaled_beyond_delta)),
        "mean_original_effective_nonbase_weight": float(calibrated["original_effective_nonbase_weight"].mean()),
        "mean_effective_nonbase_weight": float(calibrated["effective_nonbase_weight"].mean()),
        "max_original_effective_nonbase_weight": float(calibrated["original_effective_nonbase_weight"].max()),
        "max_effective_nonbase_weight": float(calibrated["effective_nonbase_weight"].max()),
        "base_target_relative_delta": args.base_target_relative_delta,
        "min_trust_target_relative_delta": float(calibrated["trust_target_relative_delta"].min()),
        "max_route_audit_relative_delta_before_scale": float(calibrated["audit_max_relative_delta_norm"].max()),
        "max_expected_relative_delta_after_scale": float(calibrated["expected_max_relative_delta_norm"].max()),
        "min_trust_delta_scale": float(calibrated["trust_delta_scale"].min()),
        "action_counts": action_counts,
        "risk_flag_counts": {str(key): int(value) for key, value in sorted(flag_counter.items())},
        **summarize_dry_run_manifest(dry_run_manifest),
        **summarize_materialized_manifest(materialized_manifest),
        **estimated_summary,
        "outputs": {
            "trust_region_source_weights": rel(calibrated_csv),
            "estimated_tensor_delta": rel(estimated_tensor_csv),
            "tensor_rules": rel(tensor_rules_path),
            "writer_command": rel(writer_command_path),
            "dry_run_command": rel(dry_run_command_path),
            "dry_run_manifest": rel(dry_run_manifest),
            "materialized_manifest": rel(materialized_manifest),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, calibrated), encoding="utf-8")
    return summary


def build_report(summary: dict[str, Any], calibrated: pd.DataFrame) -> str:
    strongest = calibrated.sort_values("trust_delta_scale").head(20)
    risk = calibrated[calibrated["trust_gate_action"] != "keep_route_weight_rule"].sort_values(
        ["trust_delta_scale", "audit_max_relative_delta_norm"], ascending=[True, False]
    ).head(30)
    lines = [
        "# Qwen3 MoE Trust-Region Candidate",
        "",
        "这个候选把 route-frequency expert weights、routing/load readiness 和 materialized delta audit 合在一起。",
        "核心约束是：router 仍冻结，目标结构不变；只按 MoE 内部风险缩小非 base source delta。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Expert rules: `{summary['expert_rule_count']}`",
        f"- Scaled expert rules: `{summary['scaled_expert_rule_count']}`",
        f"- Scaled beyond simple delta cap: `{summary['scaled_beyond_delta_cap_count']}`",
        (
            f"- Mean effective nonbase weight: `{fmt(summary['mean_original_effective_nonbase_weight'])}`"
            f" -> `{fmt(summary['mean_effective_nonbase_weight'])}`"
        ),
        (
            f"- Max routed expert relative delta estimate: `{fmt(summary['max_route_audit_relative_delta_before_scale'])}`"
            f" -> `{fmt(summary['max_expected_relative_delta_after_scale'])}`"
        ),
        f"- Estimated total relative delta norm: `{fmt(summary['estimated_relative_delta_norm'])}`",
        f"- Estimated routed tensors >1.0 / >0.75 / >0.65: `{summary['estimated_routed_tensors_relative_delta_gt_1']}` / `{summary['estimated_routed_tensors_relative_delta_gt_075']}` / `{summary['estimated_routed_tensors_relative_delta_gt_065']}`",
        (
            f"- Writer dry-run: `{summary['writer_dry_run_validated']}`; "
            f"expert/attention/router hits `{summary['writer_dry_run_expert_tensor_rule_hits']}` / "
            f"`{summary['writer_dry_run_shared_attention_hits']}` / "
            f"`{summary['writer_dry_run_freeze_router_hits']}`"
        ),
        (
            f"- Materialized checkpoint: `{summary['writer_checkpoint_materialized']}`; "
            f"shards `{summary['writer_materialized_shards']}`"
        ),
        "",
        "## Risk Flags",
        "",
        "| flag | count |",
        "| --- | ---: |",
    ]
    for flag, count in summary["risk_flag_counts"].items():
        lines.append(f"| `{flag}` | {count} |")
    lines.extend(
        [
            "",
            "## Action Counts",
            "",
            "| action | count |",
            "| --- | ---: |",
        ]
    )
    for action, count in summary["action_counts"].items():
        lines.append(f"| `{action}` | {count} |")
    lines.extend(
        [
            "",
            "## Strongest Trust-Region Gates",
            "",
            "| layer | expert | original nonbase | new nonbase | target cap | route max rel | expected max rel | scale | flags |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in strongest.iterrows():
        lines.append(
            "| "
            f"{int(row['layer_id'])} | {int(row['expert_id'])} | "
            f"{fmt(float(row['original_effective_nonbase_weight']))} | "
            f"{fmt(float(row['effective_nonbase_weight']))} | "
            f"{fmt(float(row['trust_target_relative_delta']))} | "
            f"{fmt(float(row['audit_max_relative_delta_norm']))} | "
            f"{fmt(float(row['expected_max_relative_delta_norm']))} | "
            f"{fmt(float(row['trust_delta_scale']))} | "
            f"`{row['trust_risk_flags']}` |"
        )
    lines.extend(
        [
            "",
            "## Highest-Risk Kept Or Scaled Experts",
            "",
            "| layer | expert | action | total route | max load/uniform | mean top-k Jaccard | categories | category share | route max rel | scale | flags |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in risk.iterrows():
        lines.append(
            "| "
            f"{int(row['layer_id'])} | {int(row['expert_id'])} | "
            f"`{row['trust_gate_action']}` | "
            f"{fmt(float(row['total_topk_fraction']))} | "
            f"{fmt(float(row['max_topk_over_uniform']))} | "
            f"{fmt(float(row['mean_topk_jaccard']))} | "
            f"{int(row['min_categories_observed'])} | "
            f"{fmt(float(row['max_dominant_category_share']))} | "
            f"{fmt(float(row['audit_max_relative_delta_norm']))} | "
            f"{fmt(float(row['trust_delta_scale']))} | "
            f"`{row['trust_risk_flags']}` |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['trust_region_source_weights']}`",
            f"- `{summary['outputs']['estimated_tensor_delta']}`",
            f"- `{summary['outputs']['tensor_rules']}`",
            f"- `{summary['outputs']['writer_command']}`",
            f"- `{summary['outputs']['dry_run_command']}`",
            f"- `{summary['outputs']['dry_run_manifest']}`",
            f"- `{summary['outputs']['materialized_manifest']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-candidate-dir", default="results/qwen3_moe_unified_route_guarded_candidate")
    parser.add_argument("--delta-audit-dir", default="results/qwen3_moe_materialized_delta_audit")
    parser.add_argument("--routing-readiness-dir", default="results/moe_routing_readiness/qwen3_30b_instruct_vs_coder")
    parser.add_argument("--output-dir", default="results/qwen3_moe_trust_region_candidate")
    parser.add_argument("--checkpoint-output-dir", default="results/checkpoints/qwen3_moe_trust_region_candidate")
    parser.add_argument("--dry-run-output-dir", default="results/qwen3_moe_trust_region_candidate/dry_run")
    parser.add_argument("--base-target-relative-delta", type=float, default=0.75)
    parser.add_argument("--high-load-target-relative-delta", type=float, default=0.65)
    parser.add_argument("--shared-target-relative-delta", type=float, default=0.68)
    parser.add_argument("--fragile-router-target-relative-delta", type=float, default=0.70)
    parser.add_argument("--low-route-target-relative-delta", type=float, default=0.60)
    parser.add_argument("--category-mismatch-target-relative-delta", type=float, default=0.62)
    parser.add_argument("--high-load-over-uniform", type=float, default=8.0)
    parser.add_argument("--high-load-max-nonbase-weight", type=float, default=0.65)
    parser.add_argument("--shared-max-nonbase-weight", type=float, default=0.70)
    parser.add_argument("--shared-categories-min", type=int, default=6)
    parser.add_argument("--shared-category-share-max", type=float, default=0.35)
    parser.add_argument("--shared-min-route-fraction", type=float, default=0.02)
    parser.add_argument("--fragile-mean-topk-jaccard", type=float, default=0.45)
    parser.add_argument("--fragile-min-topk-jaccard", type=float, default=0.35)
    parser.add_argument("--low-route-fraction", type=float, default=0.02)
    parser.add_argument("--category-mismatch-share", type=float, default=0.70)
    parser.add_argument("--freeze-scale-threshold", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    summary = build_candidate(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
