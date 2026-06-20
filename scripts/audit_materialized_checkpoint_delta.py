#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from write_same_shape_average_checkpoint import discover_safetensors, is_router_tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
EXPERT_RE = re.compile(r"(?:^|\.)(?:experts|routed_experts)(?:\.|$)")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def classify_tensor(name: str) -> str:
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding_or_head"
    if is_router_tensor(name):
        return "router"
    if "self_attn" in name:
        return "attention"
    if "shared_expert" in name:
        return "shared_expert"
    if EXPERT_RE.search(name):
        return "routed_expert_ffn"
    if ".mlp." in name:
        return "dense_or_shared_mlp"
    if "norm" in name:
        return "norm"
    return "other"


def layer_id(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return None if match is None else int(match.group(1))


def shape_numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


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


def tensor_stats(base_tensor: torch.Tensor, candidate_tensor: torch.Tensor) -> dict[str, float | bool]:
    if not torch.is_floating_point(base_tensor):
        changed = not torch.equal(base_tensor, candidate_tensor)
        return {
            "base_norm2": 0.0,
            "delta_norm2": float("nan") if changed else 0.0,
            "max_abs_delta": float("nan") if changed else 0.0,
            "changed": changed,
        }
    base_fp32 = base_tensor.to(torch.float32)
    candidate_fp32 = candidate_tensor.to(torch.float32)
    delta = candidate_fp32 - base_fp32
    base_norm2 = float(torch.sum(base_fp32 * base_fp32).item())
    delta_norm2 = float(torch.sum(delta * delta).item())
    max_abs_delta = float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0
    return {
        "base_norm2": base_norm2,
        "delta_norm2": delta_norm2,
        "max_abs_delta": max_abs_delta,
        "changed": bool(delta_norm2 > 0.0 or max_abs_delta > 0.0),
    }


def summarize_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    summaries = []
    for value, group in frame.groupby(key, dropna=False):
        delta_norm2 = float(group["delta_norm2"].fillna(0.0).sum())
        base_norm2 = float(group["base_norm2"].fillna(0.0).sum())
        numel = int(group["numel"].sum())
        changed_numel = int(group.loc[group["changed"], "numel"].sum())
        summaries.append(
            {
                key: "" if pd.isna(value) else value,
                "tensor_count": int(len(group)),
                "changed_tensors": int(group["changed"].sum()),
                "numel": numel,
                "changed_numel": changed_numel,
                "changed_numel_fraction": changed_numel / max(1, numel),
                "delta_norm": math.sqrt(delta_norm2),
                "base_norm": math.sqrt(base_norm2),
                "relative_delta_norm": math.sqrt(delta_norm2) / max(1e-12, math.sqrt(base_norm2)),
                "max_abs_delta": float(group["max_abs_delta"].fillna(0.0).max()),
            }
        )
    return summaries


def audit_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = discover_safetensors(args.base)
    candidate = discover_safetensors(args.candidate)
    base_names = set(base.tensor_info)
    candidate_names = set(candidate.tensor_info)
    common_names = sorted(base_names & candidate_names)

    rows: list[dict[str, Any]] = []
    shape_mismatches = []
    dtype_mismatches = []
    shard_pairs: dict[tuple[Path, Path], list[str]] = {}
    for name in common_names:
        base_info = base.tensor_info[name]
        candidate_info = candidate.tensor_info[name]
        if base_info.shape != candidate_info.shape:
            shape_mismatches.append(name)
            continue
        if base_info.dtype != candidate_info.dtype:
            dtype_mismatches.append(name)
        shard_pairs.setdefault((base_info.shard, candidate_info.shard), []).append(name)

    for (base_shard, candidate_shard), names in tqdm(sorted(shard_pairs.items()), desc="audit shard pairs"):
        with torch.no_grad():
            from safetensors import safe_open

            with safe_open(str(base_shard), framework="pt", device="cpu") as base_handle, safe_open(
                str(candidate_shard), framework="pt", device="cpu"
            ) as candidate_handle:
                for name in names:
                    base_tensor = base_handle.get_tensor(name)
                    candidate_tensor = candidate_handle.get_tensor(name)
                    stats = tensor_stats(base_tensor, candidate_tensor)
                    numel = shape_numel(base.tensor_info[name].shape)
                    delta_norm = (
                        math.sqrt(float(stats["delta_norm2"]))
                        if not math.isnan(float(stats["delta_norm2"]))
                        else float("nan")
                    )
                    base_norm = math.sqrt(float(stats["base_norm2"]))
                    rows.append(
                        {
                            "tensor": name,
                            "group": classify_tensor(name),
                            "layer": layer_id(name),
                            "shape": "x".join(str(dim) for dim in base.tensor_info[name].shape),
                            "dtype": str(base.tensor_info[name].dtype).replace("torch.", ""),
                            "numel": numel,
                            "changed": bool(stats["changed"]),
                            "base_norm2": float(stats["base_norm2"]),
                            "delta_norm2": float(stats["delta_norm2"]),
                            "base_norm": base_norm,
                            "delta_norm": delta_norm,
                            "relative_delta_norm": delta_norm / max(1e-12, base_norm) if not math.isnan(delta_norm) else float("nan"),
                            "max_abs_delta": stats["max_abs_delta"],
                            "base_shard": base_shard.name,
                            "candidate_shard": candidate_shard.name,
                        }
                    )

    tensor_frame = pd.DataFrame(rows).sort_values(["changed", "delta_norm"], ascending=[False, False])
    group_summary = pd.DataFrame(summarize_groups(rows, "group")).sort_values("delta_norm", ascending=False)
    layer_summary = pd.DataFrame(summarize_groups(rows, "layer"))
    layer_summary["_sort_layer"] = pd.to_numeric(layer_summary["layer"], errors="coerce").fillna(-1)
    layer_summary = layer_summary.sort_values("_sort_layer").drop(columns=["_sort_layer"])

    tensor_csv = output_dir / "tensor_delta_audit.csv"
    group_csv = output_dir / "group_delta_summary.csv"
    layer_csv = output_dir / "layer_delta_summary.csv"
    tensor_frame.to_csv(tensor_csv, index=False)
    group_summary.to_csv(group_csv, index=False)
    layer_summary.to_csv(layer_csv, index=False)

    router = tensor_frame[tensor_frame["group"] == "router"]
    attention = tensor_frame[tensor_frame["group"] == "attention"]
    routed = tensor_frame[tensor_frame["group"] == "routed_expert_ffn"]
    routed_relative_delta = pd.to_numeric(routed["relative_delta_norm"], errors="coerce").fillna(0.0)
    changed = tensor_frame[tensor_frame["changed"]]
    total_delta_norm2 = float((tensor_frame["delta_norm"].fillna(0.0) ** 2).sum())
    total_base_norm2 = float((tensor_frame["base_norm"].fillna(0.0) ** 2).sum())
    summary = {
        "schema_version": 1,
        "status": "passed"
        if not shape_mismatches and not dtype_mismatches and int(router["changed"].sum()) == 0 and len(changed) > 0
        else "needs_review",
        "base": str(args.base),
        "candidate": str(args.candidate),
        "manifest": str(args.manifest) if args.manifest else "",
        "tensor_count": int(len(tensor_frame)),
        "missing_from_candidate": int(len(base_names - candidate_names)),
        "extra_in_candidate": int(len(candidate_names - base_names)),
        "shape_mismatch_count": int(len(shape_mismatches)),
        "dtype_mismatch_count": int(len(dtype_mismatches)),
        "changed_tensors": int(len(changed)),
        "changed_numel": int(changed["numel"].sum()) if not changed.empty else 0,
        "total_numel": int(tensor_frame["numel"].sum()) if not tensor_frame.empty else 0,
        "total_delta_norm": math.sqrt(total_delta_norm2),
        "total_base_norm": math.sqrt(total_base_norm2),
        "relative_delta_norm": math.sqrt(total_delta_norm2) / max(1e-12, math.sqrt(total_base_norm2)),
        "total_relative_delta_norm": math.sqrt(total_delta_norm2) / max(1e-12, math.sqrt(total_base_norm2)),
        "max_abs_delta": float(tensor_frame["max_abs_delta"].fillna(0.0).max()) if not tensor_frame.empty else 0.0,
        "attention_tensors": int(len(attention)),
        "attention_changed_tensors": int(attention["changed"].sum()) if not attention.empty else 0,
        "router_tensors": int(len(router)),
        "router_changed_tensors": int(router["changed"].sum()) if not router.empty else 0,
        "routed_expert_ffn_tensors": int(len(routed)),
        "routed_expert_ffn_changed_tensors": int(routed["changed"].sum()) if not routed.empty else 0,
        "routed_expert_ffn_relative_delta_norm": float(
            group_summary.loc[
                group_summary["group"].eq("routed_expert_ffn"),
                "relative_delta_norm",
            ].iloc[0]
        )
        if bool(group_summary["group"].eq("routed_expert_ffn").any())
        else 0.0,
        "max_routed_expert_ffn_relative_delta_norm": float(routed_relative_delta.max())
        if not routed_relative_delta.empty
        else 0.0,
        "routed_expert_ffn_tensors_gt_0_65": int((routed_relative_delta > 0.65).sum()),
        "routed_expert_ffn_tensors_gt_0_6505": int((routed_relative_delta > 0.6505).sum()),
        "routed_tensors_gt_0_65": int((routed_relative_delta > 0.65).sum()),
        "routed_tensors_gt_0_6505": int((routed_relative_delta > 0.6505).sum()),
        "top_changed_tensors": [
            {
                "tensor": str(row["tensor"]),
                "group": str(row["group"]),
                "layer": None if pd.isna(row["layer"]) else int(row["layer"]),
                "numel": int(row["numel"]),
                "delta_norm": float(row["delta_norm"]),
                "relative_delta_norm": float(row["relative_delta_norm"]),
                "max_abs_delta": float(row["max_abs_delta"]),
            }
            for _, row in changed.sort_values("delta_norm", ascending=False).head(args.top_k).iterrows()
        ],
        "group_summary": [clean_record(row) for row in group_summary.to_dict(orient="records")],
        "outputs": {
            "tensor_delta_audit": rel(tensor_csv),
            "group_delta_summary": rel(group_csv),
            "layer_delta_summary": rel(layer_csv),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, group_summary, layer_summary, changed, args.top_k), encoding="utf-8")
    return summary


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    summary: dict[str, Any],
    group_summary: pd.DataFrame,
    layer_summary: pd.DataFrame,
    changed: pd.DataFrame,
    top_k: int,
) -> str:
    lines = [
        "# Materialized Checkpoint Delta Audit",
        "",
        "这个审计直接读取已物化的 safetensors checkpoint，而不是只相信 writer manifest。它检查输出 tensor 是否与 base 同构，并量化 candidate 相对 base 的实际参数改动。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Tensor count: `{summary['tensor_count']}`",
        f"- Changed tensors: `{summary['changed_tensors']}`",
        f"- Changed numel fraction: `{fmt(summary['changed_numel'] / max(1, summary['total_numel']))}`",
        f"- Total relative delta norm: `{fmt(summary['relative_delta_norm'])}`",
        f"- Max abs delta: `{fmt(summary['max_abs_delta'])}`",
        f"- Routed expert FFN relative delta norm: `{fmt(summary['routed_expert_ffn_relative_delta_norm'])}`",
        f"- Max routed expert FFN tensor relative delta: `{fmt(summary['max_routed_expert_ffn_relative_delta_norm'])}`",
        f"- Routed expert FFN tensors >0.65 / >0.6505: `{summary['routed_expert_ffn_tensors_gt_0_65']}` / `{summary['routed_expert_ffn_tensors_gt_0_6505']}`",
        f"- Attention changed tensors: `{summary['attention_changed_tensors']}/{summary['attention_tensors']}`",
        f"- Router changed tensors: `{summary['router_changed_tensors']}/{summary['router_tensors']}`",
        "",
        "## Group Summary",
        "",
        "| group | tensors | changed | changed numel frac | delta norm | relative delta norm | max abs delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in group_summary.iterrows():
        lines.append(
            f"| `{row['group']}` | {int(row['tensor_count'])} | {int(row['changed_tensors'])} | "
            f"{fmt(float(row['changed_numel_fraction']))} | {fmt(float(row['delta_norm']))} | "
            f"{fmt(float(row['relative_delta_norm']))} | {fmt(float(row['max_abs_delta']))} |"
        )
    lines.extend(
        [
            "",
            "## Layer Summary",
            "",
            "| layer | tensors | changed | changed numel frac | delta norm | relative delta norm |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in layer_summary.iterrows():
        layer = row["layer"]
        layer_text = "" if pd.isna(layer) or layer == "" else str(int(layer))
        lines.append(
            f"| {layer_text} | {int(row['tensor_count'])} | {int(row['changed_tensors'])} | "
            f"{fmt(float(row['changed_numel_fraction']))} | {fmt(float(row['delta_norm']))} | "
            f"{fmt(float(row['relative_delta_norm']))} |"
        )
    lines.extend(
        [
            "",
            f"## Top {top_k} Changed Tensors",
            "",
            "| tensor | group | layer | numel | delta norm | relative delta norm | max abs delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in changed.sort_values("delta_norm", ascending=False).head(top_k).iterrows():
        layer = row["layer"]
        layer_text = "" if pd.isna(layer) else str(int(layer))
        lines.append(
            f"| `{row['tensor']}` | `{row['group']}` | {layer_text} | {int(row['numel'])} | "
            f"{fmt(float(row['delta_norm']))} | {fmt(float(row['relative_delta_norm']))} | "
            f"{fmt(float(row['max_abs_delta']))} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['tensor_delta_audit']}`",
            f"- `{summary['outputs']['group_delta_summary']}`",
            f"- `{summary['outputs']['layer_delta_summary']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit actual tensor deltas in a materialized same-shape checkpoint.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("results/materialized_checkpoint_delta_audit"))
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_checkpoint(args)
    print(f"Wrote materialized checkpoint delta audit to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}")


if __name__ == "__main__":
    main()
