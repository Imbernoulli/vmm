#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_endpoint(alpha: float | None, beta: float | None, eps: float = 1e-8) -> bool:
    if alpha is None or beta is None:
        return False
    candidates = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    return any(abs(alpha - a) <= eps and abs(beta - b) <= eps for a, b in candidates)


def is_interior(alpha: float | None, beta: float | None) -> bool:
    if alpha is None or beta is None:
        return False
    return not is_endpoint(alpha, beta) and alpha >= 0.0 and beta >= 0.0


def qwen_writer_command(
    *,
    qwen_summary: dict[str, Any],
    alpha: float,
    beta: float,
    output_dir: str,
    dry_run: bool,
) -> str:
    base = qwen_summary["base"]
    experts = qwen_summary["experts"]
    parts = [
        "python scripts/write_same_shape_average_checkpoint.py",
        f"--base {base}",
        f"--source instruct={experts['instruct']}",
        f"--source coder={experts['coder']}",
        f"--source-weight instruct={alpha:g}",
        f"--source-weight coder={beta:g}",
        f"--output-dir {output_dir}",
    ]
    if dry_run:
        parts.append("--dry-run")
    return " ".join(parts)


def build_dense_rows(decision_table: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, item in decision_table.iterrows():
        experiment = str(item["experiment"])
        if experiment == "qwen_instruct_coder":
            continue
        alpha = float(item["best_grid_alpha"]) if finite(item.get("best_grid_alpha")) else None
        beta = float(item["best_grid_beta"]) if finite(item.get("best_grid_beta")) else None
        verdict = str(item["verdict"])
        if is_endpoint(alpha, beta):
            status = "skip_endpoint_only"
            reason = "best grid 是端点，不证明存在有价值的 average。"
        elif verdict == "avoid_uniform_average" and is_interior(alpha, beta):
            status = "candidate_interior_weighted_average"
            reason = "uniform average 被否定，但 validation grid 找到了 interior 同构平均系数。"
        elif verdict in {"coefficient_search", "uniform_average_ok", "average_with_conflict_probe"}:
            status = "candidate_weighted_average"
            reason = "probe 证据支持 coefficient-selected 同构平均。"
        else:
            status = "needs_more_probe"
            reason = "还没有安全的 materialization recipe。"
        rows.append(
            {
                "experiment": experiment,
                "candidate": "best_grid_same_shape_average",
                "status": status,
                "alpha": alpha,
                "beta": beta,
                "writer_command": "",
                "reason": reason,
                "source": "average_decision_report",
            }
        )
    return rows


def build_qwen_rows(
    decision_table: pd.DataFrame,
    qwen_summary: dict[str, Any] | None,
    output_prefix: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    qwen_rows = decision_table[decision_table["experiment"] == "qwen_instruct_coder"]
    if qwen_rows.empty:
        return []
    item = qwen_rows.iloc[0]
    alpha = float(item["best_grid_alpha"]) if finite(item.get("best_grid_alpha")) else None
    beta = float(item["best_grid_beta"]) if finite(item.get("best_grid_beta")) else None
    rows: list[dict[str, Any]] = []

    if qwen_summary is None:
        rows.append(
            {
                "experiment": "qwen_instruct_coder",
                "candidate": "best_grid_same_shape_average",
                "status": "missing_model_paths",
                "alpha": alpha,
                "beta": beta,
                "writer_command": "",
                "reason": "没有提供包含 base/expert 路径的 Qwen summary。",
                "source": "average_decision_report",
            }
        )
        return rows

    if is_endpoint(alpha, beta):
        rows.append(
            {
                "experiment": "qwen_instruct_coder",
                "candidate": "best_grid_endpoint",
                "status": "skip_endpoint_only",
                "alpha": alpha,
                "beta": beta,
                "writer_command": "",
                "reason": "best grid 是端点；materialize 只会复制一个 source delta，不是有价值的 average。",
                "source": "qwen_multi_expert_merge",
            }
        )
    elif is_interior(alpha, beta):
        rows.append(
            {
                "experiment": "qwen_instruct_coder",
                "candidate": "best_grid_same_shape_average",
                "status": "materialize_dry_run_first",
                "alpha": alpha,
                "beta": beta,
                "writer_command": qwen_writer_command(
                    qwen_summary=qwen_summary,
                    alpha=alpha,
                    beta=beta,
                    output_dir=f"{output_prefix}/qwen_instruct_coder_best_grid",
                    dry_run=dry_run,
                ),
                "reason": "best grid 是 interior 系数；先跑 writer dry-run，再 materialize 并评测 held-out slices。",
                "source": "qwen_multi_expert_merge",
            }
        )

    rows.append(
        {
            "experiment": "qwen_instruct_coder",
            "candidate": "uniform_average_baseline",
            "status": "skip_rejected_by_probe" if str(item["verdict"]) == "avoid_uniform_average" else "optional_baseline",
            "alpha": 0.5,
            "beta": 0.5,
            "writer_command": ""
            if str(item["verdict"]) == "avoid_uniform_average"
            else qwen_writer_command(
                qwen_summary=qwen_summary,
                alpha=0.5,
                beta=0.5,
                output_dir=f"{output_prefix}/qwen_instruct_coder_uniform",
                dry_run=dry_run,
            ),
            "reason": "当前 probe 显示 linear average 落在高 worst-NLL ridge 上；只保留作负 baseline。"
            if str(item["verdict"]) == "avoid_uniform_average"
            else "uniform average 可作为 baseline materialize。",
            "source": "qwen_multi_expert_merge",
        }
    )
    return rows


def build_moe_template_rows(output_prefix: str, dry_run: bool) -> list[dict[str, Any]]:
    command = (
        "python scripts/write_same_shape_average_checkpoint.py "
        "--base MOE_ANCHOR_PATH "
        "--source general=MOE_GENERAL_PATH "
        "--source code=MOE_CODE_PATH "
        "--source-weight general=0.0 "
        "--source-weight code=0.0 "
        "--freeze-router "
        "--tensor-rule '.*self_attn.*::general=0.5,code=0.5' "
        "--tensor-rule '.*experts.*::general=ROUTE_WEIGHT_GENERAL,code=ROUTE_WEIGHT_CODE' "
        f"--output-dir {output_prefix}/moe_route_aware_candidate"
    )
    if dry_run:
        command += " --dry-run"
    return [
        {
            "experiment": "qwen_moe_template",
            "candidate": "router_frozen_route_aware_average",
            "status": "template_waiting_for_routing_probe",
            "alpha": None,
            "beta": None,
            "writer_command": command,
            "reason": "需要真实 MoE routing/expert-load probe 填入 expert-wise route weights；保持 router frozen 和输出结构不变。",
            "source": "moe_average_plan",
        }
    ]


def build_report(rows: list[dict[str, Any]]) -> str:
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    lines = [
        "# Average Candidate Recipes",
        "",
        "这个报告把 Average decision report 转成可执行或可跳过的 candidate recipes。它的核心原则是保守 materialization：endpoint-only 不算有效 average，已被 probe 否定的 uniform average 不生成 writer 命令，MoE template 必须等 routing/expert-load probe 填入权重。",
        "",
        f"Status counts: `{json.dumps(counts, ensure_ascii=False)}`",
        "",
        "| experiment | candidate | status | alpha | beta | reason |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        alpha = "n/a" if row["alpha"] is None else f"{float(row['alpha']):.4g}"
        beta = "n/a" if row["beta"] is None else f"{float(row['beta']):.4g}"
        lines.append(
            f"| {row['experiment']} | {row['candidate']} | `{row['status']}` | {alpha} | {beta} | {row['reason']} |"
        )
    lines.extend(["", "## Writer Commands", ""])
    commands = [row for row in rows if row.get("writer_command")]
    if not commands:
        lines.append("当前没有可以直接 materialize 的 checkpoint writer 命令。")
    for row in commands:
        lines.extend(
            [
                f"### {row['experiment']} / {row['candidate']}",
                "",
                "```bash",
                row["writer_command"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `skip_endpoint_only`：best grid 是 base 或某个 endpoint，说明当前候选模型集合里还没有找到有价值的 average。",
            "- `skip_rejected_by_probe`：已有 probe 直接显示该平均点退化，只保留作负 baseline。",
            "- `materialize_dry_run_first`：先用 writer `--dry-run` 做同构检查，再写真实 checkpoint 并跑 held-out eval。",
            "- `template_waiting_for_routing_probe`：MoE 需要 routing/expert-load probe 填入 expert-wise 权重后才能 materialize。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative same-shape Average candidate recipes from decision reports.")
    parser.add_argument("--decision-table", default="results/average_decision_report/decision_table.csv")
    parser.add_argument("--qwen-summary", default="results/qwen_multi_expert_merge/summary.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results/average_candidate_recipes"))
    parser.add_argument("--checkpoint-output-prefix", default="results/checkpoints")
    parser.add_argument("--materialize", action="store_true", help="Emit writer commands without --dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_table = pd.read_csv(repo_path(args.decision_table))
    qwen_summary_path = repo_path(args.qwen_summary)
    qwen_summary = read_json(qwen_summary_path) if qwen_summary_path.exists() else None
    rows = []
    rows.extend(build_dense_rows(decision_table))
    rows.extend(
        build_qwen_rows(
            decision_table,
            qwen_summary=qwen_summary,
            output_prefix=args.checkpoint_output_prefix,
            dry_run=not args.materialize,
        )
    )
    rows.extend(build_moe_template_rows(output_prefix=args.checkpoint_output_prefix, dry_run=not args.materialize))
    pd.DataFrame(rows).to_csv(output_dir / "candidate_recipes.csv", index=False)
    summary = {
        "rows": rows,
        "status_counts": pd.Series([row["status"] for row in rows]).value_counts().to_dict(),
        "dry_run_commands": not args.materialize,
        "same_shape_constraint": "Recipes only target same-shape checkpoints; MoE templates keep router/expert counts unchanged.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(rows), encoding="utf-8")
    print(f"Wrote Average candidate recipes to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
