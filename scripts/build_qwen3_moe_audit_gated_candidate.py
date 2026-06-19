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
EXPERT_TENSOR_RE = re.compile(r"layers\.(?P<layer>\d+).*experts\.(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight")


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
                "relative_delta_norm": float(row["relative_delta_norm"]),
                "delta_norm": float(row["delta_norm"]),
                "max_abs_delta": float(row["max_abs_delta"]),
                "numel": int(row["numel"]),
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
                "audit_max_relative_delta_norm": float(group["relative_delta_norm"].max()),
                "audit_mean_relative_delta_norm": float(group["relative_delta_norm"].mean()),
                "audit_max_abs_delta": float(group["max_abs_delta"].max()),
                "audit_delta_norm": float(math.sqrt(float((group["delta_norm"] ** 2).sum()))),
            }
        )
    return pd.DataFrame(grouped)


def source_names_from_columns(source_weights: pd.DataFrame) -> list[str]:
    return sorted(col.removeprefix("weight_") for col in source_weights.columns if col.startswith("weight_"))


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    route_dir = repo_path(args.route_candidate_dir)
    audit_dir = repo_path(args.delta_audit_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_summary = read_json(route_dir / "summary.json")
    source_weights = pd.read_csv(route_dir / "source_weights_by_expert.csv")
    delta_audit = pd.read_csv(audit_dir / "tensor_delta_audit.csv")
    expert_audit = build_expert_audit(delta_audit)

    source_names = source_names_from_columns(source_weights)
    base_path = str(route_summary.get("base", ""))
    source_paths = {str(k): str(v) for k, v in (route_summary.get("source_paths") or {}).items()}
    base_sources = sorted(source for source, path in source_paths.items() if same_path(path, base_path))
    nonbase_sources = [source for source in source_names if source not in set(base_sources)]
    if not nonbase_sources:
        nonbase_sources = [source for source in source_names if source != (base_sources[0] if base_sources else "")]

    merged = source_weights.merge(expert_audit, on=["layer_id", "expert_id"], how="left")
    for col in [
        "audit_tensor_count",
        "audit_changed_tensors",
        "audit_max_relative_delta_norm",
        "audit_mean_relative_delta_norm",
        "audit_max_abs_delta",
        "audit_delta_norm",
    ]:
        merged[col] = merged[col].fillna(0.0)

    calibrated_rows = []
    scaled_count = 0
    frozen_by_cap_count = 0
    for _, row in merged.iterrows():
        max_relative = float(row["audit_max_relative_delta_norm"])
        if max_relative > args.target_max_relative_delta and max_relative > 0:
            scale = args.target_max_relative_delta / max_relative
        else:
            scale = 1.0
        scale = max(0.0, min(1.0, scale))
        if scale < 1.0:
            scaled_count += 1
        if scale <= args.freeze_scale_threshold:
            frozen_by_cap_count += 1
            scale = 0.0

        out = row.to_dict()
        original_effective = 0.0
        new_effective = 0.0
        for source in source_names:
            original = float(row.get(f"weight_{source}", 0.0))
            if source in nonbase_sources:
                updated = original * scale
                original_effective += abs(original)
                new_effective += abs(updated)
            else:
                updated = original
            out[f"original_weight_{source}"] = original
            out[f"weight_{source}"] = updated
        out["audit_delta_scale"] = scale
        out["original_effective_nonbase_weight"] = original_effective
        out["effective_nonbase_weight"] = new_effective
        if scale == 0.0 and original_effective > 0.0:
            out["audit_gate_action"] = "freeze_nonbase_delta_extreme_audit_risk"
        elif scale < 1.0 and original_effective > 0.0:
            out["audit_gate_action"] = "scale_nonbase_delta_to_relative_norm_cap"
        else:
            out["audit_gate_action"] = "keep_route_weight_rule"
        weight_chunks = [f"{source}={out[f'weight_{source}']:.6g}" for source in source_names]
        out["tensor_rule"] = f"{row['tensor_pattern']}::" + ",".join(weight_chunks)
        calibrated_rows.append(out)

    calibrated = pd.DataFrame(calibrated_rows)
    calibrated_csv = output_dir / "audit_gated_source_weights_by_expert.csv"
    calibrated.to_csv(calibrated_csv, index=False)

    tensor_rules_path = output_dir / "tensor_rules.txt"
    attention_weights = route_summary.get("shared_attention_weights") or {}
    attention_rule = ",".join(f"{source}={float(attention_weights.get(source, 0.0)):.6g}" for source in source_names)
    with tensor_rules_path.open("w", encoding="utf-8") as handle:
        handle.write("# Shared attention rule. Kept unchanged because materialized audit shows group-level relative delta below cap.\n")
        handle.write(f".*self_attn.*::{attention_rule}\n")
        handle.write("# Audit-gated route-frequency expert rules.\n")
        for _, row in calibrated.sort_values(["layer_id", "expert_id"]).iterrows():
            handle.write(str(row["tensor_rule"]) + "\n")

    checkpoint_output_dir = str(args.checkpoint_output_dir)
    source_args = " ".join(f"--source {source}={source_paths[source]}" for source in source_names)
    source_weight_args = " ".join(f"--source-weight {source}=0.0" for source in source_names)
    writer_command = (
        "python scripts/write_same_shape_average_checkpoint.py "
        f"--base {base_path} "
        f"{source_args} "
        f"{source_weight_args} "
        "--freeze-router "
        f"--tensor-rule-file {rel(tensor_rules_path)} "
        f"--output-dir {checkpoint_output_dir} "
        "--dry-run"
    )
    writer_command_path = output_dir / "writer_command.txt"
    writer_command_path.write_text(writer_command + "\n", encoding="utf-8")

    report_path = output_dir / "report.md"
    summary = {
        "schema_version": 1,
        "status": "audit_gated_rules_ready",
        "base": base_path,
        "source_paths": source_paths,
        "source_names": source_names,
        "base_sources": base_sources,
        "nonbase_sources": nonbase_sources,
        "target_max_relative_delta": args.target_max_relative_delta,
        "freeze_scale_threshold": args.freeze_scale_threshold,
        "expert_rule_count": int(len(calibrated)),
        "scaled_expert_rule_count": int(scaled_count),
        "frozen_by_audit_cap_count": int(frozen_by_cap_count),
        "mean_original_effective_nonbase_weight": float(calibrated["original_effective_nonbase_weight"].mean()),
        "mean_effective_nonbase_weight": float(calibrated["effective_nonbase_weight"].mean()),
        "max_original_effective_nonbase_weight": float(calibrated["original_effective_nonbase_weight"].max()),
        "max_effective_nonbase_weight": float(calibrated["effective_nonbase_weight"].max()),
        "max_audit_relative_delta_before_cap": float(calibrated["audit_max_relative_delta_norm"].max()),
        "min_audit_delta_scale": float(calibrated["audit_delta_scale"].min()),
        "action_counts": {
            str(key): int(value) for key, value in calibrated["audit_gate_action"].value_counts().sort_index().items()
        },
        "outputs": {
            "audit_gated_source_weights": rel(calibrated_csv),
            "tensor_rules": rel(tensor_rules_path),
            "writer_command": rel(writer_command_path),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(report_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, calibrated), encoding="utf-8")
    return summary


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def build_report(summary: dict[str, Any], calibrated: pd.DataFrame) -> str:
    risky = calibrated.sort_values("audit_max_relative_delta_norm", ascending=False).head(20)
    scaled = calibrated[calibrated["audit_gate_action"] != "keep_route_weight_rule"].sort_values(
        "audit_delta_scale"
    ).head(20)
    lines = [
        "# Qwen3 MoE Audit-Gated Candidate",
        "",
        "这个候选把 route-frequency expert weights 再经过 materialized checkpoint delta audit 约束：如果某个 expert 的实际 FFN tensor relative delta norm 超过阈值，就按比例缩小非 base source 的 delta weight。目标是保留 route-aware 机制，同时避免少数 expert projection 被移动到超过 base norm 的幅度。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Target max relative delta: `{fmt(summary['target_max_relative_delta'])}`",
        f"- Expert rules: `{summary['expert_rule_count']}`",
        f"- Scaled expert rules: `{summary['scaled_expert_rule_count']}`",
        f"- Frozen by audit cap: `{summary['frozen_by_audit_cap_count']}`",
        f"- Mean effective nonbase weight: `{fmt(summary['mean_original_effective_nonbase_weight'])}` -> `{fmt(summary['mean_effective_nonbase_weight'])}`",
        f"- Max effective nonbase weight: `{fmt(summary['max_original_effective_nonbase_weight'])}` -> `{fmt(summary['max_effective_nonbase_weight'])}`",
        f"- Max audited relative delta before cap: `{fmt(summary['max_audit_relative_delta_before_cap'])}`",
        f"- Min audit delta scale: `{fmt(summary['min_audit_delta_scale'])}`",
        "",
        "## Action Counts",
        "",
        "| action | count |",
        "| --- | ---: |",
    ]
    for action, count in summary["action_counts"].items():
        lines.append(f"| `{action}` | {count} |")
    lines.extend(
        [
            "",
            "## Largest Audited Expert Deltas",
            "",
            "| layer | expert | original nonbase | new nonbase | max rel delta | scale | action |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in risky.iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | "
            f"{fmt(float(row['original_effective_nonbase_weight']))} | {fmt(float(row['effective_nonbase_weight']))} | "
            f"{fmt(float(row['audit_max_relative_delta_norm']))} | {fmt(float(row['audit_delta_scale']))} | "
            f"`{row['audit_gate_action']}` |"
        )
    lines.extend(
        [
            "",
            "## Strongest Audit Gates",
            "",
            "| layer | expert | original nonbase | new nonbase | max rel delta | scale | action |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in scaled.iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | "
            f"{fmt(float(row['original_effective_nonbase_weight']))} | {fmt(float(row['effective_nonbase_weight']))} | "
            f"{fmt(float(row['audit_max_relative_delta_norm']))} | {fmt(float(row['audit_delta_scale']))} | "
            f"`{row['audit_gate_action']}` |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['audit_gated_source_weights']}`",
            f"- `{summary['outputs']['tensor_rules']}`",
            f"- `{summary['outputs']['writer_command']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an audit-gated Qwen3 MoE route-aware same-shape candidate.")
    parser.add_argument("--route-candidate-dir", default="results/qwen3_moe_unified_route_guarded_candidate")
    parser.add_argument("--delta-audit-dir", default="results/qwen3_moe_materialized_delta_audit")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_audit_gated_candidate"))
    parser.add_argument("--checkpoint-output-dir", default="results/checkpoints/qwen3_moe_audit_gated_candidate")
    parser.add_argument("--target-max-relative-delta", type=float, default=0.75)
    parser.add_argument("--freeze-scale-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_candidate(args)
    print(f"Wrote Qwen3 MoE audit-gated candidate to {repo_path(args.output_dir).resolve()}")
    print(f"Scaled expert rules: {summary['scaled_expert_rule_count']}")


if __name__ == "__main__":
    main()
