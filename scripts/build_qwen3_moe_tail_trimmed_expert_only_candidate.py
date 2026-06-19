#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RULE_RE = re.compile(r"layers\\\.(?P<layer>\d+)\\\..*experts\\\.(?P<expert>\d+)\\\.")
TENSOR_RE = re.compile(r"layers\.(?P<layer>\d+).*experts\.(?P<expert>\d+)\.")


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


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def parse_weights(raw: str) -> dict[str, float]:
    weights = {}
    for item in raw.split(","):
        key, value = item.split("=", 1)
        weights[key] = float(value)
    return weights


def format_weights(weights: dict[str, float]) -> str:
    return ",".join(f"{key}={weights[key]:.8g}" for key in sorted(weights))


def parse_rule_layer_expert(pattern: str) -> tuple[int, int] | None:
    match = RULE_RE.search(pattern)
    if match is None:
        return None
    return int(match.group("layer")), int(match.group("expert"))


def parse_tensor_layer_expert(tensor: str) -> tuple[int, int] | None:
    match = TENSOR_RE.search(tensor)
    if match is None:
        return None
    return int(match.group("layer")), int(match.group("expert"))


def summarize_manifest(path: Path, prefix: str) -> dict[str, Any]:
    if not path.exists():
        return {
            f"{prefix}_validated": False,
            f"{prefix}_manifest": rel(path),
            f"{prefix}_expert_tensor_rule_hits": 0,
            f"{prefix}_shared_attention_hits": 0,
            f"{prefix}_freeze_router_hits": 0,
            f"{prefix}_floating_tensors": 0,
            f"{prefix}_frozen_tensors": 0,
            f"{prefix}_shards": 0,
        }
    manifest = read_json(path)
    rule_counts = manifest.get("rule_counts") or {}
    expert_hits = sum(
        int(value) for key, value in rule_counts.items() if str(key).startswith("tensor_rule:.*layers\\.")
    )
    return {
        f"{prefix}_validated": bool(manifest.get("dry_run", False)) if prefix.endswith("dry_run") else True,
        f"{prefix}_manifest": rel(path),
        f"{prefix}_expert_tensor_rule_hits": int(expert_hits),
        f"{prefix}_shared_attention_hits": int(rule_counts.get("tensor_rule:.*self_attn.*", 0)),
        f"{prefix}_freeze_router_hits": int(rule_counts.get("freeze_router", 0)),
        f"{prefix}_floating_tensors": int(manifest.get("floating_tensors", 0)),
        f"{prefix}_frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        f"{prefix}_shards": int(len(manifest.get("shards") or [])),
    }


def build_group_scales(delta_audit: pd.DataFrame, target_cap: float) -> pd.DataFrame:
    routed = delta_audit[delta_audit["group"] == "routed_expert_ffn"].copy()
    records = []
    for _, row in routed.iterrows():
        parsed = parse_tensor_layer_expert(str(row["tensor"]))
        if parsed is None:
            continue
        layer, expert = parsed
        records.append(
            {
                "layer": layer,
                "expert": expert,
                "tensor": row["tensor"],
                "relative_delta_norm": float(row["relative_delta_norm"]),
                "delta_norm": float(row["delta_norm"]),
                "base_norm": float(row["base_norm"]),
                "changed": bool(row["changed"]),
            }
        )
    tensors = pd.DataFrame(records)
    rows = []
    for (layer, expert), group in tensors.groupby(["layer", "expert"], sort=True):
        max_rel = float(group["relative_delta_norm"].max())
        scale = 1.0 if max_rel <= target_cap else target_cap / max_rel
        rows.append(
            {
                "layer": int(layer),
                "expert": int(expert),
                "tensor_count": int(len(group)),
                "changed_tensors": int(group["changed"].sum()),
                "max_relative_delta_norm": max_rel,
                "mean_relative_delta_norm": float(group["relative_delta_norm"].mean()),
                "target_cap": target_cap,
                "delta_scale": scale,
                "scaled": scale < 0.999999,
                "predicted_max_relative_delta_norm": max_rel * scale,
            }
        )
    return pd.DataFrame(rows)


def rewrite_tensor_rules(
    input_path: Path,
    output_path: Path,
    group_scales: pd.DataFrame,
    nonbase_source: str,
) -> pd.DataFrame:
    scale_lookup = {
        (int(row.layer), int(row.expert)): float(row.delta_scale)
        for _, row in group_scales.iterrows()
    }
    audit_lookup = {
        (int(row.layer), int(row.expert)): row
        for _, row in group_scales.iterrows()
    }
    rule_rows = []
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        dst.write("# Tail-trimmed expert-only rules. Shared attention and router remain frozen.\n")
        for raw in src:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "::" not in line:
                dst.write(line + "\n")
                continue
            pattern, weight_text = line.split("::", 1)
            weights = parse_weights(weight_text)
            parsed = parse_rule_layer_expert(pattern)
            scale = 1.0 if parsed is None else scale_lookup.get(parsed, 1.0)
            old_nonbase = float(weights.get(nonbase_source, 0.0))
            weights[nonbase_source] = old_nonbase * scale
            dst.write(f"{pattern}::{format_weights(weights)}\n")
            audit = audit_lookup.get(parsed) if parsed is not None else None
            rule_rows.append(
                {
                    "pattern": pattern,
                    "layer": None if parsed is None else parsed[0],
                    "expert": None if parsed is None else parsed[1],
                    "old_nonbase_weight": old_nonbase,
                    "new_nonbase_weight": weights[nonbase_source],
                    "delta_scale": scale,
                    "scaled": scale < 0.999999,
                    "max_relative_delta_norm": None if audit is None else float(audit.max_relative_delta_norm),
                    "predicted_max_relative_delta_norm": None
                    if audit is None
                    else float(audit.predicted_max_relative_delta_norm),
                }
            )
    return pd.DataFrame(rule_rows)


def estimate_delta(
    delta_audit: pd.DataFrame,
    group_scales: pd.DataFrame,
    total_base_norm: float,
) -> dict[str, Any]:
    scale_lookup = {
        (int(row.layer), int(row.expert)): float(row.delta_scale)
        for _, row in group_scales.iterrows()
    }
    delta_sq = 0.0
    routed_rows = []
    for _, row in delta_audit.iterrows():
        group = str(row["group"])
        delta_norm = float(row["delta_norm"])
        scale = 1.0
        if group == "attention":
            scale = 0.0
        elif group == "routed_expert_ffn":
            parsed = parse_tensor_layer_expert(str(row["tensor"]))
            if parsed is not None:
                scale = scale_lookup.get(parsed, 1.0)
        scaled_delta = delta_norm * scale
        delta_sq += scaled_delta * scaled_delta
        if group == "routed_expert_ffn":
            rel = float(row["relative_delta_norm"]) * scale
            routed_rows.append(rel)
    routed = pd.Series(routed_rows)
    total_delta = math.sqrt(delta_sq)
    return {
        "estimated_total_delta_norm": total_delta,
        "estimated_total_relative_delta_norm": total_delta / max(1e-12, total_base_norm),
        "estimated_routed_max_relative_delta": float(routed.max()) if not routed.empty else 0.0,
        "estimated_routed_p99_relative_delta": float(routed.quantile(0.99)) if not routed.empty else 0.0,
        "estimated_routed_tensors_gt_1": int((routed > 1.0).sum()) if not routed.empty else 0,
        "estimated_routed_tensors_gt_075": int((routed > 0.75).sum()) if not routed.empty else 0,
        "estimated_routed_tensors_gt_065": int((routed > 0.65).sum()) if not routed.empty else 0,
        "estimated_routed_tensors_gt_05": int((routed > 0.5).sum()) if not routed.empty else 0,
    }


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = repo_path(args.source_candidate_dir)
    audit_dir = repo_path(args.source_delta_audit_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summary = read_json(source_dir / "summary.json")
    audit_summary = read_json(audit_dir / "summary.json")
    delta_audit = pd.read_csv(audit_dir / "tensor_delta_audit.csv")
    source_names = list(source_summary.get("source_names") or [])
    source_paths = {str(k): str(v) for k, v in (source_summary.get("source_paths") or {}).items()}
    base_path = str(source_summary.get("base"))
    if args.nonbase_source not in source_names:
        raise ValueError(f"{args.nonbase_source} not in source names: {source_names}")

    group_scales = build_group_scales(delta_audit, args.target_cap)
    tensor_rules_path = output_dir / "tensor_rules.txt"
    rule_rows = rewrite_tensor_rules(
        source_dir / "tensor_rules.txt",
        tensor_rules_path,
        group_scales,
        args.nonbase_source,
    )
    scale_path = output_dir / "expert_tail_scales.csv"
    rules_path = output_dir / "tail_trimmed_rules.csv"
    group_scales.to_csv(scale_path, index=False)
    rule_rows.to_csv(rules_path, index=False)

    estimate = estimate_delta(delta_audit, group_scales, float(audit_summary["total_base_norm"]))
    checkpoint_output_dir = str(args.checkpoint_output_dir)
    dry_run_output_dir = str(args.dry_run_output_dir)
    source_args = " ".join(f"--source {source}={source_paths[source]}" for source in source_names)
    source_weight_args = " ".join(f"--source-weight {source}=0.0" for source in source_names)
    writer_command = (
        "python scripts/write_same_shape_average_checkpoint.py "
        f"--base {base_path} "
        f"{source_args} "
        f"{source_weight_args} "
        "--freeze-router "
        f"--tensor-rule-file {rel(tensor_rules_path)} "
        f"--output-dir {checkpoint_output_dir}"
    )
    dry_run_command = (
        writer_command.replace(f"--output-dir {checkpoint_output_dir}", f"--output-dir {dry_run_output_dir}")
        + " --dry-run"
    )
    writer_command_path = output_dir / "writer_command.txt"
    dry_run_command_path = output_dir / "dry_run_command.txt"
    writer_command_path.write_text(writer_command + "\n", encoding="utf-8")
    dry_run_command_path.write_text(dry_run_command + "\n", encoding="utf-8")
    dry_run_manifest = repo_path(dry_run_output_dir) / "merge_manifest.json"
    materialized_manifest = repo_path(checkpoint_output_dir) / "merge_manifest.json"

    scaled_rules = rule_rows[rule_rows["scaled"].astype(bool)]
    scaled_groups = group_scales[group_scales["scaled"].astype(bool)]
    summary = {
        "schema_version": 1,
        "status": "tail_trimmed_rules_ready",
        "source_candidate_dir": rel(source_dir),
        "source_delta_audit_dir": rel(audit_dir),
        "base": base_path,
        "source_names": source_names,
        "source_paths": source_paths,
        "nonbase_source": args.nonbase_source,
        "target_cap": args.target_cap,
        "expert_group_count": int(len(group_scales)),
        "scaled_expert_group_count": int(len(scaled_groups)),
        "rule_count": int(len(rule_rows)),
        "scaled_rule_count": int(len(scaled_rules)),
        "mean_delta_scale_scaled_groups": None
        if scaled_groups.empty
        else float(scaled_groups["delta_scale"].mean()),
        "min_delta_scale": float(group_scales["delta_scale"].min()) if not group_scales.empty else 1.0,
        "source_total_relative_delta_norm": float(audit_summary["relative_delta_norm"]),
        **estimate,
        **summarize_manifest(dry_run_manifest, "writer_dry_run"),
        **summarize_manifest(materialized_manifest, "writer_materialized"),
        "outputs": {
            "expert_tail_scales": rel(scale_path),
            "tail_trimmed_rules": rel(rules_path),
            "tensor_rules": rel(tensor_rules_path),
            "writer_command": rel(writer_command_path),
            "dry_run_command": rel(dry_run_command_path),
            "dry_run_manifest": rel(dry_run_manifest),
            "materialized_manifest": rel(materialized_manifest),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(summary, group_scales), encoding="utf-8")
    return summary


def build_report(summary: dict[str, Any], group_scales: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Tail-Trimmed Expert-Only Candidate",
        "",
        "This candidate starts from the materialized expert-only trust-region candidate and trims only the remaining high routed-expert delta tail.",
        "Shared attention remains frozen, router remains frozen, and the output checkpoint keeps the same model structure.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Target routed tensor cap: `{fmt(summary['target_cap'])}`",
        f"- Expert groups: `{summary['expert_group_count']}`",
        f"- Scaled expert groups: `{summary['scaled_expert_group_count']}`",
        f"- Scaled rules: `{summary['scaled_rule_count']}`",
        f"- Source total relative delta norm: `{fmt(summary['source_total_relative_delta_norm'])}`",
        f"- Estimated total relative delta norm: `{fmt(summary['estimated_total_relative_delta_norm'])}`",
        f"- Estimated routed max relative delta: `{fmt(summary['estimated_routed_max_relative_delta'])}`",
        f"- Estimated routed tensors >0.65: `{summary['estimated_routed_tensors_gt_065']}`",
        f"- Writer dry-run: `{summary['writer_dry_run_validated']}`",
        f"- Materialized: `{summary['writer_materialized_validated']}`",
        "",
        "## Top Scaled Expert Groups",
        "",
        "| layer | expert | tensors | max rel | scale | predicted max rel |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    top = group_scales[group_scales["scaled"].astype(bool)].sort_values(
        ["max_relative_delta_norm", "layer", "expert"],
        ascending=[False, True, True],
    )
    for _, row in top.head(25).iterrows():
        lines.append(
            "| "
            f"{int(row['layer'])} | {int(row['expert'])} | {int(row['tensor_count'])} | "
            f"{fmt(float(row['max_relative_delta_norm']))} | "
            f"{fmt(float(row['delta_scale']))} | "
            f"{fmt(float(row['predicted_max_relative_delta_norm']))} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
        ]
    )
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidate-dir", default="results/qwen3_moe_expert_only_trust_region_candidate")
    parser.add_argument("--source-delta-audit-dir", default="results/qwen3_moe_expert_only_delta_audit")
    parser.add_argument("--output-dir", default="results/qwen3_moe_tail_trimmed_expert_only_candidate")
    parser.add_argument("--checkpoint-output-dir", default="results/checkpoints/qwen3_moe_tail_trimmed_expert_only_candidate")
    parser.add_argument("--dry-run-output-dir", default="results/qwen3_moe_tail_trimmed_expert_only_candidate/dry_run")
    parser.add_argument("--target-cap", type=float, default=0.65)
    parser.add_argument("--nonbase-source", default="coder")
    return parser.parse_args()


def main() -> None:
    summary = build_candidate(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
