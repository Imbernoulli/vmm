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


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def layer_id_from_router(router: str, layer_regex: re.Pattern[str]) -> str:
    match = layer_regex.search(router)
    return match.group(1) if match else ""


def normalize_method_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "method" not in df.columns:
        if "model" in df.columns:
            df["method"] = df["model"].astype(str)
        else:
            df["method"] = "probe_model"
    return df


def load_expert_loads(router_dirs: list[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    inputs = []
    for router_dir in router_dirs:
        path = router_dir / "expert_load.csv"
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            inputs.append({"router_dir": rel(router_dir), "rows": 0, "status": "missing_or_empty"})
            continue
        required = {"router", "expert_id", "topk_fraction"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        df = normalize_method_column(df)
        df["router_dir"] = rel(router_dir)
        frames.append(df)
        inputs.append({"router_dir": rel(router_dir), "rows": int(len(df)), "status": "loaded"})
    if not frames:
        return pd.DataFrame(), inputs
    return pd.concat(frames, ignore_index=True), inputs


def build_slice_loads(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slice_cols = ["method", "router"] + [col for col in ("category", "prompt_idx") if col in df.columns]
    for slice_key, group in df.groupby(slice_cols, sort=True):
        slice_values = dict(zip(slice_cols, slice_key if isinstance(slice_key, tuple) else (slice_key,)))
        if "topk_count" in group.columns:
            counts = group.groupby("expert_id", sort=True)["topk_count"].sum()
            total_assignments = float(counts.sum())
            for expert_id, count in counts.items():
                rows.append(
                    {
                        **slice_values,
                        "expert_id": int(expert_id),
                        "topk_assignments": float(count),
                        "total_topk_assignments": total_assignments,
                        "observed_topk_fraction": float(count) / max(1.0, total_assignments),
                    }
                )
        else:
            masses = group.groupby("expert_id", sort=True)["topk_fraction"].sum()
            total_mass = float(masses.sum())
            for expert_id, mass in masses.items():
                rows.append(
                    {
                        **slice_values,
                        "expert_id": int(expert_id),
                        "topk_assignments": float(mass),
                        "total_topk_assignments": total_mass,
                        "observed_topk_fraction": float(mass) / max(1e-12, total_mass),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_router_loads(df: pd.DataFrame, *, load_stat: str, load_quantile: float) -> pd.DataFrame:
    slice_loads = build_slice_loads(df)
    if slice_loads.empty:
        return slice_loads
    rows = []
    for (method, router, expert_id), group in slice_loads.groupby(["method", "router", "expert_id"], sort=True):
        values = group["observed_topk_fraction"].astype(float)
        if load_stat == "mean":
            observed = float(values.mean())
        elif load_stat == "quantile":
            observed = float(values.quantile(load_quantile))
        elif load_stat == "worst":
            observed = float(values.max())
        else:
            raise ValueError(f"Unsupported load_stat: {load_stat}")
        rows.append(
            {
                "method": method,
                "router": router,
                "expert_id": int(expert_id),
                "topk_assignments": float(group["topk_assignments"].sum()),
                "total_topk_assignments": float(group["total_topk_assignments"].sum()),
                "observed_topk_fraction": observed,
                "load_stat": load_stat,
                "load_quantile": load_quantile if load_stat == "quantile" else None,
            }
        )
    return pd.DataFrame(rows)


def build_bias_rows(
    loads: pd.DataFrame,
    *,
    capacity_factor: float,
    target_topk_fraction: float | None,
    bias_step: float,
    max_abs_delta: float,
    min_abs_delta: float,
    min_fraction: float,
    router_bias_template: str,
    layer_regex: re.Pattern[str],
    num_experts: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if loads.empty:
        return rows
    for (method, router), group in loads.groupby(["method", "router"], sort=True):
        inferred_experts = int(group["expert_id"].max()) + 1
        n_experts = int(num_experts or inferred_experts)
        observed_by_expert = {int(row["expert_id"]): maybe_float(row["observed_topk_fraction"]) for _, row in group.iterrows()}
        total_assignments = maybe_float(group["total_topk_assignments"].max())
        capacity_fraction = float(target_topk_fraction) if target_topk_fraction is not None else float(capacity_factor) / max(1, n_experts)
        uniform_fraction = 1.0 / max(1, n_experts)
        raw_deltas = []
        for expert_id in range(n_experts):
            observed = max(min_fraction, observed_by_expert.get(expert_id, 0.0))
            load_ratio = observed / max(min_fraction, capacity_fraction)
            raw_deltas.append(-float(bias_step) * math.log(load_ratio))
        mean_delta = sum(raw_deltas) / max(1, len(raw_deltas))
        layer_id = layer_id_from_router(str(router), layer_regex)
        for expert_id, raw_delta in enumerate(raw_deltas):
            observed = observed_by_expert.get(expert_id, 0.0)
            centered_delta = raw_delta - mean_delta
            clipped_delta = max(-max_abs_delta, min(max_abs_delta, centered_delta))
            if abs(clipped_delta) < min_abs_delta:
                clipped_delta = 0.0
            capacity_ratio = observed / max(min_fraction, capacity_fraction)
            if capacity_ratio > 1.0 and clipped_delta < 0:
                reason = "reduce_overloaded_expert_logit"
            elif observed < uniform_fraction and clipped_delta > 0:
                reason = "lift_underused_expert_logit"
            elif clipped_delta < 0:
                reason = "center_reduce_expert_logit"
            elif clipped_delta > 0:
                reason = "center_lift_expert_logit"
            else:
                reason = "no_effect_after_centering"
            tensor_name = router_bias_template.format(
                router=router,
                layer_id=layer_id,
                expert_id=expert_id,
                method=method,
            )
            rows.append(
                {
                    "method": method,
                    "router": router,
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "tensor": tensor_name,
                    "index": expert_id,
                    "delta": clipped_delta,
                    "raw_delta": raw_delta,
                    "centered_delta": centered_delta,
                    "observed_topk_fraction": observed,
                    "uniform_topk_fraction": uniform_fraction,
                    "capacity_topk_fraction": capacity_fraction,
                    "capacity_ratio": capacity_ratio,
                    "topk_assignments_total": total_assignments,
                    "load_stat": str(group["load_stat"].iloc[0]) if "load_stat" in group else "",
                    "load_quantile": maybe_float(group["load_quantile"].iloc[0], default=0.0) if "load_quantile" in group else 0.0,
                    "same_shape_action": "router_bias_additive_capacity_correction",
                    "reason": reason,
                }
            )
    return rows


def build_report(summary: dict[str, Any], plan_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# MoE Router Bias Plan",
        "",
        "这个 recipe 把 routing probe 的 `expert_load.csv` 转成 writer 可用的 router-bias additive delta。它不改变 expert 数、router shape 或模型结构；只是给已有 bias tensor 的每个 expert logit 加一个离线计算出的标量。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Methods: `{', '.join(summary['methods']) if summary['methods'] else 'none'}`",
        f"- Router dirs: `{', '.join(summary['router_dirs']) if summary['router_dirs'] else 'none'}`",
        f"- Routers: `{summary['router_count']}`",
        f"- Delta rows: `{summary['delta_rows']}`",
        f"- Nonzero deltas: `{summary['nonzero_delta_rows']}`",
        f"- Capacity factor: `{summary['capacity_factor']}`",
        f"- Load statistic: `{summary['load_stat']}`",
        f"- Bias step / clip: `{summary['bias_step']}` / `{summary['max_abs_delta']}`",
        "",
        "## 规则",
        "",
        "```text",
        "observed_topk_fraction[e] = worst/mean/quantile over prompt-category slices",
        "capacity_fraction = capacity_factor / num_experts",
        "raw_delta[e] = -bias_step * log(observed_topk_fraction[e] / capacity_fraction)",
        "delta[e] = clip(raw_delta[e] - mean_e(raw_delta[e]), -max_abs_delta, max_abs_delta)",
        "```",
        "",
        "过载 expert 的 logit 会被压低，低载 expert 的 logit 会被抬高；同一 router 内做中心化，避免引入无意义的整体 logit 平移。这个 CSV 是离线候选修正，仍然要用 held-out 下游任务和 capacity-aware 指标验收。",
        "",
    ]
    if summary.get("writer_csv_ready"):
        lines.extend(
            [
                "## Writer 用法",
                "",
                "```bash",
                "python scripts/write_same_shape_average_checkpoint.py --base MOE_BASE_OR_ANCHOR_PATH "
                "--source general=GENERAL_MODEL_PATH --source code=CODE_MODEL_PATH "
                "--source-weight general=0.0 --source-weight code=0.0 --freeze-router "
                f"--tensor-add-csv {summary['outputs']['router_bias_deltas']} "
                "--output-dir results/checkpoints/moe_bias_calibrated_candidate --dry-run",
                "```",
                "",
                "如果真实 checkpoint 没有对应 bias tensor，writer 会在校验阶段报错；这表示该模型需要改用 router weight 小步校准或保持 router freeze，而不是强行改结构。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Writer 用法",
                "",
                "当前没有写出 writer CSV，通常是因为输入里有多个 method。请用 `--method METHOD_NAME` 选定一个候选后重新生成。",
                "",
            ]
        )
    if plan_rows:
        df = pd.DataFrame(plan_rows).sort_values("capacity_ratio", ascending=False)
        lines.extend(
            [
                "## 过载 Expert 预览",
                "",
                "| method | router | expert | observed top-k | capacity | ratio | delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in df.head(20).iterrows():
            lines.append(
                f"| `{row['method']}` | `{row['router']}` | {int(row['expert_id'])} | "
                f"{float(row['observed_topk_fraction']):.4f} | {float(row['capacity_topk_fraction']):.4f} | "
                f"{float(row['capacity_ratio']):.3f} | {float(row['delta']):.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            f"- `{summary['outputs']['router_bias_plan']}`",
            f"- `{summary['outputs']['router_bias_deltas']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build additive router-bias correction CSVs from MoE expert_load.csv probes.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_router_bias_plan"))
    parser.add_argument("--router-dir", action="append", default=[], help="Directory containing expert_load.csv.")
    parser.add_argument("--method", action="append", default=[], help="Method/model name to keep when expert_load.csv contains multiple candidates.")
    parser.add_argument("--num-experts", type=int, default=None, help="Override inferred expert count per router.")
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--target-topk-fraction", type=float, default=None)
    parser.add_argument("--load-stat", choices=["worst", "mean", "quantile"], default="worst")
    parser.add_argument("--load-quantile", type=float, default=0.95)
    parser.add_argument("--bias-step", type=float, default=0.25)
    parser.add_argument("--max-abs-delta", type=float, default=0.5)
    parser.add_argument("--min-abs-delta", type=float, default=1e-6)
    parser.add_argument("--min-fraction", type=float, default=1e-6)
    parser.add_argument("--layer-regex", default=r"layers\.(\d+)")
    parser.add_argument("--router-bias-template", default="{router}.bias")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    router_dirs = [repo_path(path) for path in args.router_dir]
    expert_loads, input_summaries = load_expert_loads(router_dirs)
    if not expert_loads.empty and args.method:
        expert_loads = expert_loads[expert_loads["method"].isin(args.method)].copy()
    methods = sorted(expert_loads["method"].unique().tolist()) if not expert_loads.empty else []
    loads = (
        aggregate_router_loads(expert_loads, load_stat=args.load_stat, load_quantile=args.load_quantile)
        if not expert_loads.empty
        else pd.DataFrame()
    )
    plan_rows = build_bias_rows(
        loads,
        capacity_factor=args.capacity_factor,
        target_topk_fraction=args.target_topk_fraction,
        bias_step=args.bias_step,
        max_abs_delta=args.max_abs_delta,
        min_abs_delta=args.min_abs_delta,
        min_fraction=args.min_fraction,
        router_bias_template=args.router_bias_template,
        layer_regex=re.compile(args.layer_regex),
        num_experts=args.num_experts,
    )
    plan_df = pd.DataFrame(plan_rows)
    plan_path = output_dir / "router_bias_plan.csv"
    delta_path = output_dir / "router_bias_deltas.csv"
    plan_df.to_csv(plan_path, index=False)

    writer_csv_ready = len(methods) == 1 and bool(plan_rows)
    if writer_csv_ready:
        delta_df = plan_df[plan_df["delta"].abs() >= args.min_abs_delta][["tensor", "index", "delta", "reason"]].copy()
    else:
        delta_df = pd.DataFrame(columns=["tensor", "index", "delta", "reason"])
    delta_df.to_csv(delta_path, index=False)

    summary = {
        "schema_version": 1,
        "status": "router_bias_delta_ready" if writer_csv_ready else "select_single_method_before_writer_csv",
        "router_dirs": [rel(path) for path in router_dirs],
        "input_summaries": input_summaries,
        "methods": methods,
        "method_filter": args.method,
        "router_count": int(plan_df["router"].nunique()) if not plan_df.empty else 0,
        "delta_rows": int(len(plan_df)),
        "nonzero_delta_rows": int(len(delta_df)),
        "writer_csv_ready": writer_csv_ready,
        "capacity_factor": args.capacity_factor,
        "target_topk_fraction": args.target_topk_fraction,
        "load_stat": args.load_stat,
        "load_quantile": args.load_quantile,
        "bias_step": args.bias_step,
        "max_abs_delta": args.max_abs_delta,
        "router_bias_template": args.router_bias_template,
        "same_shape_constraint": "The generated CSV only adds scalar deltas to existing router-bias tensors; it does not add experts, layers, or router outputs.",
        "outputs": {
            "router_bias_plan": rel(plan_path),
            "router_bias_deltas": rel(delta_path),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, plan_rows), encoding="utf-8")
    print(f"Wrote MoE router bias plan to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
