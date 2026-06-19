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
EXPERT_TENSOR_RE = re.compile(
    r"layers\.(?P<layer>\d+).*experts\.(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight"
)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_expert_tensor(name: str) -> tuple[int, int, str] | None:
    match = EXPERT_TENSOR_RE.search(name)
    if match is None:
        return None
    return int(match.group("layer")), int(match.group("expert")), match.group("projection")


def fmt(value: Any, digits: int = 6) -> str:
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


def attach_expert_ids(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        parsed = parse_expert_tensor(str(row["tensor"]))
        if parsed is None:
            rows.append({"parsed_layer_id": None, "parsed_expert_id": None, "projection": ""})
        else:
            layer, expert, projection = parsed
            rows.append({"parsed_layer_id": layer, "parsed_expert_id": expert, "projection": projection})
    parsed = pd.DataFrame(rows)
    return pd.concat([frame.reset_index(drop=True), parsed], axis=1)


def summarize_group(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        record = {col: ("" if pd.isna(value) else value) for col, value in zip(group_cols, values)}
        record.update(
            {
                "tensor_count": int(len(group)),
                "actual_changed_tensors": int(group["actual_changed"].sum()),
                "mean_abs_relative_error": float(group["relative_delta_abs_error"].mean()),
                "p99_abs_relative_error": float(group["relative_delta_abs_error"].quantile(0.99)),
                "max_abs_relative_error": float(group["relative_delta_abs_error"].max()),
                "mean_abs_delta_norm_error": float(group["delta_norm_abs_error"].mean()),
                "max_abs_delta_norm_error": float(group["delta_norm_abs_error"].max()),
                "max_actual_relative_delta_norm": float(group["actual_relative_delta_norm"].max()),
                "max_predicted_relative_delta_norm": float(group["predicted_relative_delta_norm"].max()),
                "actual_relative_delta_gt_075": int((group["actual_relative_delta_norm"] > 0.75).sum()),
                "predicted_relative_delta_gt_075": int((group["predicted_relative_delta_norm"] > 0.75).sum()),
                "rounding_slop_gt_075": int(
                    ((group["actual_relative_delta_norm"] > 0.75) & (group["actual_relative_delta_norm"] <= 0.751)).sum()
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def explode_risk_flags(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expert_cols = [
        "parsed_layer_id",
        "parsed_expert_id",
        "tensor",
        "actual_changed",
        "actual_delta_norm",
        "actual_relative_delta_norm",
        "predicted_delta_norm",
        "predicted_relative_delta_norm",
        "delta_norm_abs_error",
        "relative_delta_abs_error",
        "trust_gate_action",
        "trust_risk_flags",
        "above_075_rounding_slop",
    ]
    for _, row in frame[frame["group"] == "routed_expert_ffn"].iterrows():
        raw = str(row.get("trust_risk_flags", "") or "")
        flags = [flag for flag in raw.split("|") if flag]
        if not flags:
            flags = ["no_risk_flag"]
        for flag in flags:
            out = {col: row.get(col) for col in expert_cols}
            out["risk_flag"] = flag
            rows.append(out)
    return pd.DataFrame(rows)


def build_validation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted = pd.read_csv(repo_path(args.predicted_tensor_delta))
    actual = pd.read_csv(repo_path(args.actual_tensor_delta))
    expert_rules = pd.read_csv(repo_path(args.expert_rules))

    predicted = predicted.rename(
        columns={
            "estimated_delta_norm": "predicted_delta_norm",
            "estimated_relative_delta_norm": "predicted_relative_delta_norm",
            "estimated_changed": "predicted_changed",
        }
    )
    actual = actual.rename(
        columns={
            "delta_norm": "actual_delta_norm",
            "relative_delta_norm": "actual_relative_delta_norm",
            "changed": "actual_changed",
            "max_abs_delta": "actual_max_abs_delta",
        }
    )
    frame = actual.merge(
        predicted[
            [
                "tensor",
                "route_delta_norm",
                "route_relative_delta_norm",
                "trust_delta_scale",
                "predicted_delta_norm",
                "predicted_relative_delta_norm",
                "predicted_changed",
            ]
        ],
        on="tensor",
        how="left",
    )
    frame = attach_expert_ids(frame)
    frame = frame.merge(
        expert_rules[
            [
                "layer_id",
                "expert_id",
                "trust_gate_action",
                "trust_risk_flags",
                "trust_target_relative_delta",
                "trust_delta_scale",
                "original_effective_nonbase_weight",
                "effective_nonbase_weight",
            ]
        ],
        left_on=["parsed_layer_id", "parsed_expert_id"],
        right_on=["layer_id", "expert_id"],
        how="left",
        suffixes=("", "_expert"),
    )
    for col in ["predicted_delta_norm", "predicted_relative_delta_norm", "route_delta_norm", "route_relative_delta_norm"]:
        frame[col] = frame[col].fillna(0.0)
    frame["trust_gate_action"] = frame["trust_gate_action"].fillna("non_expert_or_unruled")
    frame["trust_risk_flags"] = frame["trust_risk_flags"].fillna("")
    frame["relative_delta_error"] = frame["actual_relative_delta_norm"] - frame["predicted_relative_delta_norm"]
    frame["relative_delta_abs_error"] = frame["relative_delta_error"].abs()
    frame["delta_norm_error"] = frame["actual_delta_norm"] - frame["predicted_delta_norm"]
    frame["delta_norm_abs_error"] = frame["delta_norm_error"].abs()
    frame["above_075_rounding_slop"] = (frame["actual_relative_delta_norm"] > 0.75) & (
        frame["actual_relative_delta_norm"] <= args.rounding_slop_threshold
    )
    frame["prediction_passes_tolerance"] = frame["relative_delta_abs_error"] <= args.relative_tolerance

    tensor_csv = output_dir / "tensor_prediction_error.csv"
    group_csv = output_dir / "group_prediction_error_summary.csv"
    action_csv = output_dir / "action_prediction_error_summary.csv"
    risk_csv = output_dir / "risk_flag_prediction_error_summary.csv"
    residual_csv = output_dir / "threshold_residuals.csv"

    group_summary = summarize_group(frame, ["group"])
    action_summary = summarize_group(frame[frame["group"] == "routed_expert_ffn"], ["trust_gate_action"])
    risk_rows = explode_risk_flags(frame)
    risk_summary = summarize_group(risk_rows, ["risk_flag"]) if not risk_rows.empty else pd.DataFrame()
    residuals = frame[(frame["group"] == "routed_expert_ffn") & (frame["actual_relative_delta_norm"] > 0.75)].copy()
    residuals = residuals.sort_values("actual_relative_delta_norm", ascending=False)

    frame.to_csv(tensor_csv, index=False)
    group_summary.to_csv(group_csv, index=False)
    action_summary.to_csv(action_csv, index=False)
    risk_summary.to_csv(risk_csv, index=False)
    residuals.to_csv(residual_csv, index=False)

    routed = frame[frame["group"] == "routed_expert_ffn"]
    summary = {
        "schema_version": 1,
        "status": "passed"
        if int((frame["prediction_passes_tolerance"] | frame["above_075_rounding_slop"]).sum()) == len(frame)
        else "needs_review",
        "tensor_count": int(len(frame)),
        "prediction_relative_tolerance": args.relative_tolerance,
        "rounding_slop_threshold": args.rounding_slop_threshold,
        "max_abs_relative_error": float(frame["relative_delta_abs_error"].max()),
        "p99_abs_relative_error": float(frame["relative_delta_abs_error"].quantile(0.99)),
        "mean_abs_relative_error": float(frame["relative_delta_abs_error"].mean()),
        "tensors_above_relative_tolerance": int((frame["relative_delta_abs_error"] > args.relative_tolerance).sum()),
        "routed_tensor_count": int(len(routed)),
        "routed_max_abs_relative_error": float(routed["relative_delta_abs_error"].max()) if not routed.empty else 0.0,
        "routed_p99_abs_relative_error": float(routed["relative_delta_abs_error"].quantile(0.99)) if not routed.empty else 0.0,
        "routed_actual_relative_delta_gt_075": int((routed["actual_relative_delta_norm"] > 0.75).sum()),
        "routed_predicted_relative_delta_gt_075": int((routed["predicted_relative_delta_norm"] > 0.75).sum()),
        "routed_above_075_rounding_slop": int(routed["above_075_rounding_slop"].sum()) if not routed.empty else 0,
        "routed_max_actual_relative_delta_norm": float(routed["actual_relative_delta_norm"].max()) if not routed.empty else 0.0,
        "routed_max_predicted_relative_delta_norm": float(routed["predicted_relative_delta_norm"].max()) if not routed.empty else 0.0,
        "scaled_tensor_count": int((routed["trust_delta_scale_expert"].fillna(1.0) < 1.0).sum()) if not routed.empty else 0,
        "outputs": {
            "tensor_prediction_error": rel(tensor_csv),
            "group_prediction_error_summary": rel(group_csv),
            "action_prediction_error_summary": rel(action_csv),
            "risk_flag_prediction_error_summary": rel(risk_csv),
            "threshold_residuals": rel(residual_csv),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        build_report(summary, group_summary, action_summary, risk_summary, residuals),
        encoding="utf-8",
    )
    return summary


def build_report(
    summary: dict[str, Any],
    group_summary: pd.DataFrame,
    action_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    residuals: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen3 MoE Trust-Region Delta Validation",
        "",
        "这个验证把 trust-region 规则的逐 tensor delta 预测和真实物化 safetensors 的 delta audit 对齐。",
        "目标是确认 expert-level nonbase delta 缩放确实按预期进入参数文件，而不是只看 group-level 汇总。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Tensor count: `{summary['tensor_count']}`",
        f"- Max abs relative-delta prediction error: `{fmt(summary['max_abs_relative_error'])}`",
        f"- P99 abs relative-delta prediction error: `{fmt(summary['p99_abs_relative_error'])}`",
        f"- Tensors above tolerance: `{summary['tensors_above_relative_tolerance']}`",
        f"- Routed actual/predicted tensors >0.75: `{summary['routed_actual_relative_delta_gt_075']}` / `{summary['routed_predicted_relative_delta_gt_075']}`",
        f"- Routed >0.75 within rounding slop: `{summary['routed_above_075_rounding_slop']}`",
        f"- Routed max actual/predicted relative delta: `{fmt(summary['routed_max_actual_relative_delta_norm'])}` / `{fmt(summary['routed_max_predicted_relative_delta_norm'])}`",
        "",
        "## Group Prediction Error",
        "",
        "| group | tensors | max abs rel error | p99 abs rel error | actual >0.75 | predicted >0.75 | rounding slop >0.75 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in group_summary.iterrows():
        lines.append(
            "| "
            f"`{row['group']}` | {int(row['tensor_count'])} | "
            f"{fmt(float(row['max_abs_relative_error']))} | "
            f"{fmt(float(row['p99_abs_relative_error']))} | "
            f"{int(row['actual_relative_delta_gt_075'])} | "
            f"{int(row['predicted_relative_delta_gt_075'])} | "
            f"{int(row['rounding_slop_gt_075'])} |"
        )
    lines.extend(
        [
            "",
            "## Action Prediction Error",
            "",
            "| action | tensors | max abs rel error | actual >0.75 | rounding slop >0.75 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in action_summary.iterrows():
        lines.append(
            "| "
            f"`{row['trust_gate_action']}` | {int(row['tensor_count'])} | "
            f"{fmt(float(row['max_abs_relative_error']))} | "
            f"{int(row['actual_relative_delta_gt_075'])} | "
            f"{int(row['rounding_slop_gt_075'])} |"
        )
    lines.extend(
        [
            "",
            "## Risk Flag Prediction Error",
            "",
            "| risk flag | tensors | max abs rel error | actual >0.75 | rounding slop >0.75 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if not risk_summary.empty:
        for _, row in risk_summary.sort_values("actual_relative_delta_gt_075", ascending=False).iterrows():
            lines.append(
                "| "
                f"`{row['risk_flag']}` | {int(row['tensor_count'])} | "
                f"{fmt(float(row['max_abs_relative_error']))} | "
                f"{int(row['actual_relative_delta_gt_075'])} | "
                f"{int(row['rounding_slop_gt_075'])} |"
            )
    lines.extend(
        [
            "",
            "## Routed Threshold Residuals",
            "",
            "| tensor | action | actual rel | predicted rel | abs error | flags |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in residuals.head(30).iterrows():
        lines.append(
            "| "
            f"`{row['tensor']}` | `{row['trust_gate_action']}` | "
            f"{fmt(float(row['actual_relative_delta_norm']))} | "
            f"{fmt(float(row['predicted_relative_delta_norm']))} | "
            f"{fmt(float(row['relative_delta_abs_error']))} | "
            f"`{row['trust_risk_flags']}` |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['tensor_prediction_error']}`",
            f"- `{summary['outputs']['group_prediction_error_summary']}`",
            f"- `{summary['outputs']['action_prediction_error_summary']}`",
            f"- `{summary['outputs']['risk_flag_prediction_error_summary']}`",
            f"- `{summary['outputs']['threshold_residuals']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predicted-tensor-delta",
        default="results/qwen3_moe_trust_region_candidate/estimated_tensor_delta.csv",
    )
    parser.add_argument(
        "--actual-tensor-delta",
        default="results/qwen3_moe_trust_region_delta_audit/tensor_delta_audit.csv",
    )
    parser.add_argument(
        "--expert-rules",
        default="results/qwen3_moe_trust_region_candidate/trust_region_source_weights_by_expert.csv",
    )
    parser.add_argument("--output-dir", default="results/qwen3_moe_trust_region_delta_validation")
    parser.add_argument("--relative-tolerance", type=float, default=0.002)
    parser.add_argument("--rounding-slop-threshold", type=float, default=0.751)
    return parser.parse_args()


def main() -> None:
    summary = build_validation(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
