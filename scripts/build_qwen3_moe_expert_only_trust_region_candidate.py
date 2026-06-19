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
ATTENTION_KIND_RE = re.compile(r"self_attn\.([^\.]+)\.weight")


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


def summarize_manifest(path: Path) -> dict[str, Any]:
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


def attention_kind(tensor: str) -> str:
    match = ATTENTION_KIND_RE.search(tensor)
    return match.group(1) if match else "unknown"


def summarize_attention(delta_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    attention = delta_audit[delta_audit["group"] == "attention"].copy()
    attention["attention_kind"] = attention["tensor"].map(attention_kind)
    by_kind = []
    for kind, group in attention.groupby("attention_kind", sort=True):
        by_kind.append(
            {
                "attention_kind": kind,
                "tensor_count": int(len(group)),
                "delta_norm": float(math.sqrt(float((group["delta_norm"] ** 2).sum()))),
                "base_norm": float(math.sqrt(float((group["base_norm"] ** 2).sum()))),
                "relative_delta_norm": float(
                    math.sqrt(float((group["delta_norm"] ** 2).sum()))
                    / max(1e-12, math.sqrt(float((group["base_norm"] ** 2).sum())))
                ),
                "mean_tensor_relative_delta_norm": float(group["relative_delta_norm"].mean()),
                "p95_tensor_relative_delta_norm": float(group["relative_delta_norm"].quantile(0.95)),
                "max_tensor_relative_delta_norm": float(group["relative_delta_norm"].max()),
                "max_abs_delta": float(group["max_abs_delta"].max()),
            }
        )
    by_layer = []
    for layer, group in attention.groupby("layer", sort=True):
        by_layer.append(
            {
                "layer": int(layer),
                "tensor_count": int(len(group)),
                "delta_norm": float(math.sqrt(float((group["delta_norm"] ** 2).sum()))),
                "base_norm": float(math.sqrt(float((group["base_norm"] ** 2).sum()))),
                "relative_delta_norm": float(
                    math.sqrt(float((group["delta_norm"] ** 2).sum()))
                    / max(1e-12, math.sqrt(float((group["base_norm"] ** 2).sum())))
                ),
                "max_tensor_relative_delta_norm": float(group["relative_delta_norm"].max()),
            }
        )
    return pd.DataFrame(by_kind), pd.DataFrame(by_layer).sort_values("relative_delta_norm", ascending=False)


def build_tensor_rules(source_rules: Path, output_path: Path, source_names: list[str]) -> int:
    attention_rule = ",".join(f"{source}=0" for source in source_names)
    expert_rule_count = 0
    with source_rules.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        dst.write("# Shared attention ablation. This freezes attention to isolate routed expert trust-region effects.\n")
        dst.write(f".*self_attn.*::{attention_rule}\n")
        dst.write("# Expert rules copied from Qwen3 MoE trust-region candidate.\n")
        for raw in src:
            line = raw.strip()
            if not line or line.startswith("#") or "self_attn" in line:
                continue
            dst.write(line + "\n")
            expert_rule_count += 1
    return expert_rule_count


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    trust_dir = repo_path(args.trust_region_dir)
    audit_dir = repo_path(args.trust_region_delta_audit_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trust_summary = read_json(trust_dir / "summary.json")
    audit_summary = read_json(audit_dir / "summary.json")
    group_summary = pd.read_csv(audit_dir / "group_delta_summary.csv")
    delta_audit = pd.read_csv(audit_dir / "tensor_delta_audit.csv")
    by_kind, by_layer = summarize_attention(delta_audit)

    source_names = list(trust_summary.get("source_names") or [])
    source_paths = {str(k): str(v) for k, v in (trust_summary.get("source_paths") or {}).items()}
    base_path = str(trust_summary.get("base"))
    if not source_names or not source_paths or not base_path:
        raise ValueError(f"{trust_dir / 'summary.json'} does not contain source_names/source_paths/base")

    tensor_rules_path = output_dir / "tensor_rules.txt"
    expert_rule_count = build_tensor_rules(trust_dir / "tensor_rules.txt", tensor_rules_path, source_names)
    by_kind_path = output_dir / "attention_kind_summary.csv"
    by_layer_path = output_dir / "attention_layer_summary.csv"
    by_kind.to_csv(by_kind_path, index=False)
    by_layer.to_csv(by_layer_path, index=False)

    attention_group = group_summary[group_summary["group"] == "attention"].iloc[0]
    total_delta_norm = float(audit_summary["total_delta_norm"])
    total_base_norm = float(audit_summary["total_base_norm"])
    attention_delta_norm = float(attention_group["delta_norm"])
    no_attention_delta_norm = math.sqrt(max(0.0, total_delta_norm**2 - attention_delta_norm**2))
    no_attention_relative_delta_norm = no_attention_delta_norm / max(1e-12, total_base_norm)
    current_relative_delta_norm = float(audit_summary["relative_delta_norm"])
    attention_delta_energy_fraction = attention_delta_norm**2 / max(1e-12, total_delta_norm**2)

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

    dry_run_manifest = repo_path(dry_run_output_dir) / "merge_manifest.json"
    materialized_manifest = repo_path(checkpoint_output_dir) / "merge_manifest.json"
    summary = {
        "schema_version": 1,
        "status": "attention_ablation_rules_ready",
        "base": base_path,
        "source_names": source_names,
        "source_paths": source_paths,
        "expert_rule_count": int(expert_rule_count),
        "attention_rule": "freeze_shared_attention",
        "current_trust_region_relative_delta_norm": current_relative_delta_norm,
        "estimated_expert_only_relative_delta_norm": no_attention_relative_delta_norm,
        "estimated_relative_delta_norm_reduction": current_relative_delta_norm - no_attention_relative_delta_norm,
        "attention_delta_norm": attention_delta_norm,
        "attention_relative_delta_norm": float(attention_group["relative_delta_norm"]),
        "attention_delta_energy_fraction": attention_delta_energy_fraction,
        "attention_changed_tensors_removed": int(attention_group["changed_tensors"]),
        **summarize_manifest(dry_run_manifest),
        **summarize_materialized_manifest(materialized_manifest),
        "outputs": {
            "attention_kind_summary": rel(by_kind_path),
            "attention_layer_summary": rel(by_layer_path),
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
    (output_dir / "report.md").write_text(build_report(summary, by_kind, by_layer), encoding="utf-8")
    return summary


def build_report(summary: dict[str, Any], by_kind: pd.DataFrame, by_layer: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Expert-Only Trust-Region Candidate",
        "",
        "这个候选是一个 attention ablation：保留 trust-region routed expert rules，但冻结 shared attention。",
        "目的不是声称一定更好，而是在后续 vLLM eval 中隔离 Coder attention delta 是否真的有收益。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Expert rules: `{summary['expert_rule_count']}`",
        f"- Attention rule: `{summary['attention_rule']}`",
        f"- Trust-region relative delta norm: `{fmt(summary['current_trust_region_relative_delta_norm'])}`",
        f"- Estimated expert-only relative delta norm: `{fmt(summary['estimated_expert_only_relative_delta_norm'])}`",
        f"- Estimated reduction: `{fmt(summary['estimated_relative_delta_norm_reduction'])}`",
        f"- Attention relative delta norm: `{fmt(summary['attention_relative_delta_norm'])}`",
        f"- Attention delta energy fraction: `{fmt(summary['attention_delta_energy_fraction'])}`",
        f"- Writer dry-run: `{summary['writer_dry_run_validated']}`",
        f"- Checkpoint materialized: `{summary['writer_checkpoint_materialized']}`",
        f"- Materialized shards: `{summary['writer_materialized_shards']}`",
        "",
        "## Attention Projection Summary",
        "",
        "| kind | tensors | relative delta norm | mean tensor rel | p95 tensor rel | max tensor rel | max abs delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in by_kind.iterrows():
        lines.append(
            "| "
            f"`{row['attention_kind']}` | {int(row['tensor_count'])} | "
            f"{fmt(float(row['relative_delta_norm']))} | "
            f"{fmt(float(row['mean_tensor_relative_delta_norm']))} | "
            f"{fmt(float(row['p95_tensor_relative_delta_norm']))} | "
            f"{fmt(float(row['max_tensor_relative_delta_norm']))} | "
            f"{fmt(float(row['max_abs_delta']))} |"
        )
    lines.extend(
        [
            "",
            "## Highest Attention Layers",
            "",
            "| layer | tensors | relative delta norm | max tensor rel |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in by_layer.head(20).iterrows():
        lines.append(
            "| "
            f"{int(row['layer'])} | {int(row['tensor_count'])} | "
            f"{fmt(float(row['relative_delta_norm']))} | "
            f"{fmt(float(row['max_tensor_relative_delta_norm']))} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['attention_kind_summary']}`",
            f"- `{summary['outputs']['attention_layer_summary']}`",
            f"- `{summary['outputs']['tensor_rules']}`",
            f"- `{summary['outputs']['writer_command']}`",
            f"- `{summary['outputs']['dry_run_command']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-region-dir", default="results/qwen3_moe_trust_region_candidate")
    parser.add_argument("--trust-region-delta-audit-dir", default="results/qwen3_moe_trust_region_delta_audit")
    parser.add_argument("--output-dir", default="results/qwen3_moe_expert_only_trust_region_candidate")
    parser.add_argument("--checkpoint-output-dir", default="results/checkpoints/qwen3_moe_expert_only_trust_region_candidate")
    parser.add_argument("--dry-run-output-dir", default="results/qwen3_moe_expert_only_trust_region_candidate/dry_run")
    return parser.parse_args()


def main() -> None:
    summary = build_candidate(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
