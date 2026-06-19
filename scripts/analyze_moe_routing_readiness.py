#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_COLUMNS = [
    "router_dir",
    "category",
    "prompt_idx",
    "router",
    "num_experts",
    "max_top1_fraction",
    "effective_top1_experts",
    "effective_top1_fraction",
    "top1_margin_mean",
    "top1_agreement",
    "topk_jaccard",
    "risk_score",
    "risk_flags",
    "recommended_action",
    "reason",
]
EXPERT_COLUMNS = [
    "router_dir",
    "category",
    "prompt_idx",
    "router",
    "expert_id",
    "num_experts",
    "top1_fraction",
    "topk_fraction",
    "topk_over_uniform",
    "top1_over_uniform",
    "risk_flags",
    "recommended_action",
    "reason",
]
SPECIALIZATION_COLUMNS = [
    "router_dir",
    "router",
    "expert_id",
    "total_topk_fraction",
    "dominant_category",
    "dominant_category_share",
    "categories_observed",
    "recommended_action",
    "reason",
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


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def maybe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def discover_moe_topology(summary_path: Path, model_name: str | None) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for model in payload.get("models", []):
        config = model.get("config", {})
        if model_name is not None and model.get("name") != model_name:
            continue
        if config.get("is_moe_config"):
            return {
                "name": model.get("name"),
                "model_type": config.get("model_type"),
                "num_hidden_layers": config.get("num_hidden_layers"),
                "num_experts": config.get("num_experts"),
                "num_experts_per_tok": config.get("num_experts_per_tok"),
                "active_expert_fraction_per_token": config.get("active_expert_fraction_per_token"),
                "weights_available": model.get("headers", {}).get("weights_available"),
            }
    return None


def merge_overlap(router_summary: pd.DataFrame, route_overlap: pd.DataFrame | None) -> pd.DataFrame:
    merged = router_summary.copy()
    if route_overlap is None or route_overlap.empty:
        return merged
    join_cols = [col for col in ("category", "prompt_idx", "router") if col in merged.columns and col in route_overlap.columns]
    if not join_cols:
        return merged
    keep_cols = join_cols + [col for col in ("top1_agreement", "topk_jaccard") if col in route_overlap.columns]
    return merged.merge(route_overlap[keep_cols], on=join_cols, how="left")


def router_row_action(
    item: dict[str, Any],
    *,
    collapse_threshold: float,
    min_effective_fraction: float,
    min_topk_jaccard: float,
    min_top1_agreement: float,
    low_margin_threshold: float,
) -> tuple[int, list[str], str, str]:
    flags: list[str] = []
    num_experts = maybe_float(item.get("num_experts"))
    max_top1 = maybe_float(item.get("max_top1_fraction"))
    effective = maybe_float(item.get("effective_top1_experts"))
    margin = maybe_float(item.get("top1_margin_mean"))
    topk_jaccard = maybe_float(item.get("topk_jaccard"))
    top1_agreement = maybe_float(item.get("top1_agreement"))
    if max_top1 is not None and max_top1 >= collapse_threshold:
        flags.append("top1_load_concentration")
    if num_experts and effective is not None and effective / max(num_experts, 1.0) < min_effective_fraction:
        flags.append("low_effective_expert_fraction")
    if topk_jaccard is not None and topk_jaccard < min_topk_jaccard:
        flags.append("low_topk_route_overlap")
    if top1_agreement is not None and top1_agreement < min_top1_agreement:
        flags.append("low_top1_route_agreement")
    if margin is not None and margin < low_margin_threshold:
        flags.append("fragile_topk_boundary")

    score = len(flags)
    if "low_topk_route_overlap" in flags or "low_top1_route_agreement" in flags:
        action = "calibrate_router_before_average"
        reason = "source routers disagree on token-to-expert assignment; freeze the anchor router or apply router calibration before opening router deltas."
    elif "top1_load_concentration" in flags or "low_effective_expert_fraction" in flags:
        action = "freeze_router_and_check_load_balance"
        reason = "expert load is concentrated; direct router averaging risks collapse."
    elif "fragile_topk_boundary" in flags:
        action = "small_lambda_router_with_overlap_guard"
        reason = "router margins are small; use small router lambda and verify route overlap after materialization."
    else:
        action = "router_probe_passed_for_small_lambda"
        reason = "no routing-collapse flag is triggered under the current thresholds."
    return score, flags, action, reason


def build_router_readiness(
    router_dirs: list[Path],
    *,
    collapse_threshold: float,
    min_effective_fraction: float,
    min_topk_jaccard: float,
    min_top1_agreement: float,
    low_margin_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for router_dir in router_dirs:
        router_summary = read_csv_if_exists(router_dir / "router_summary.csv")
        if router_summary is None or router_summary.empty:
            continue
        route_overlap = read_csv_if_exists(router_dir / "route_overlap.csv")
        merged = merge_overlap(router_summary, route_overlap)
        for _, item in merged.iterrows():
            item_dict = {str(key): clean_value(value) for key, value in item.items()}
            num_experts = maybe_float(item_dict.get("num_experts"))
            effective = maybe_float(item_dict.get("effective_top1_experts"))
            score, flags, action, reason = router_row_action(
                item_dict,
                collapse_threshold=collapse_threshold,
                min_effective_fraction=min_effective_fraction,
                min_topk_jaccard=min_topk_jaccard,
                min_top1_agreement=min_top1_agreement,
                low_margin_threshold=low_margin_threshold,
            )
            rows.append(
                {
                    "router_dir": rel(router_dir),
                    "category": item_dict.get("category"),
                    "prompt_idx": item_dict.get("prompt_idx"),
                    "router": item_dict.get("router"),
                    "num_experts": num_experts,
                    "max_top1_fraction": maybe_float(item_dict.get("max_top1_fraction")),
                    "effective_top1_experts": effective,
                    "effective_top1_fraction": (effective / num_experts) if effective is not None and num_experts else None,
                    "top1_margin_mean": maybe_float(item_dict.get("top1_margin_mean")),
                    "top1_agreement": maybe_float(item_dict.get("top1_agreement")),
                    "topk_jaccard": maybe_float(item_dict.get("topk_jaccard")),
                    "risk_score": score,
                    "risk_flags": "|".join(flags),
                    "recommended_action": action,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=ROUTER_COLUMNS)


def build_expert_load_risks(
    router_dirs: list[Path],
    *,
    overuse_ratio_threshold: float,
    underuse_fraction_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for router_dir in router_dirs:
        expert_load = read_csv_if_exists(router_dir / "expert_load.csv")
        if expert_load is None or expert_load.empty:
            continue
        group_cols = [col for col in ("category", "prompt_idx", "router") if col in expert_load.columns]
        expert_load = expert_load.copy()
        if "num_experts" not in expert_load.columns:
            if group_cols:
                expert_load["num_experts"] = expert_load.groupby(group_cols)["expert_id"].transform("nunique")
            else:
                expert_load["num_experts"] = expert_load["expert_id"].nunique()
        for _, item in expert_load.iterrows():
            item_dict = {str(key): clean_value(value) for key, value in item.items()}
            num_experts = maybe_float(item_dict.get("num_experts")) or 0.0
            expected = 1.0 / num_experts if num_experts > 0 else None
            topk = maybe_float(item_dict.get("topk_fraction"), 0.0) or 0.0
            top1 = maybe_float(item_dict.get("top1_fraction"), 0.0) or 0.0
            topk_over = topk / expected if expected else None
            top1_over = top1 / expected if expected else None
            flags: list[str] = []
            if topk_over is not None and topk_over >= overuse_ratio_threshold:
                flags.append("overused_expert")
            if topk <= underuse_fraction_threshold and top1 <= underuse_fraction_threshold:
                flags.append("underused_expert")
            if "overused_expert" in flags:
                action = "protect_or_source_weight_high_load_expert"
                reason = "expert carries disproportionate route mass; average only with route/category-aware weights and verify NLL sensitivity."
            elif "underused_expert" in flags:
                action = "anchor_heavy_until_rare_task_probe"
                reason = "expert has little observed load; keep near anchor unless rare-task prompts prove it matters."
            else:
                action = "low_lambda_or_route_frequency_average"
                reason = "expert load is within the current risk thresholds."
            rows.append(
                {
                    "router_dir": rel(router_dir),
                    "category": item_dict.get("category"),
                    "prompt_idx": item_dict.get("prompt_idx"),
                    "router": item_dict.get("router"),
                    "expert_id": item_dict.get("expert_id"),
                    "num_experts": num_experts,
                    "top1_fraction": top1,
                    "topk_fraction": topk,
                    "topk_over_uniform": topk_over,
                    "top1_over_uniform": top1_over,
                    "risk_flags": "|".join(flags),
                    "recommended_action": action,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=EXPERT_COLUMNS)


def build_category_specialization(router_dirs: list[Path], specialization_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for router_dir in router_dirs:
        expert_load = read_csv_if_exists(router_dir / "expert_load.csv")
        if expert_load is None or expert_load.empty:
            continue
        required = {"category", "router", "expert_id", "topk_fraction"}
        if not required.issubset(set(expert_load.columns)):
            continue
        masses: dict[tuple[str, Any], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for _, item in expert_load.iterrows():
            router = str(item["router"])
            expert_id = item["expert_id"]
            category = str(item["category"])
            masses[(router, expert_id)][category] += maybe_float(item.get("topk_fraction"), 0.0) or 0.0
        for (router, expert_id), category_mass in sorted(masses.items(), key=lambda value: (value[0][0], value[0][1])):
            total = sum(category_mass.values())
            if total <= 0:
                continue
            dominant_category = max(category_mass, key=lambda key: category_mass[key])
            share = category_mass[dominant_category] / total
            if share >= specialization_threshold:
                action = "category_specialized_route_weight"
                reason = "one prompt category dominates this expert; route-weighted merging should preserve the corresponding source delta."
            else:
                action = "shared_or_mixed_expert"
                reason = "route mass is spread across categories; use mixed source weights or keep closer to anchor."
            rows.append(
                {
                    "router_dir": rel(router_dir),
                    "router": router,
                    "expert_id": expert_id,
                    "total_topk_fraction": total,
                    "dominant_category": dominant_category,
                    "dominant_category_share": share,
                    "categories_observed": len(category_mass),
                    "recommended_action": action,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=SPECIALIZATION_COLUMNS)


def status_from_tables(router_readiness: pd.DataFrame, router_dirs: list[Path]) -> str:
    if not router_dirs:
        return "waiting_for_routing_probe"
    if router_readiness.empty:
        return "missing_router_summary"
    if (router_readiness["risk_score"] >= 2).any():
        return "high_risk_calibrate_router_before_merge"
    if (router_readiness["risk_score"] == 1).any():
        return "medium_risk_small_lambda_with_guards"
    return "ready_for_route_guarded_materialization"


def table_action_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "recommended_action" not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df["recommended_action"].value_counts().to_dict().items()}


def fmt_optional(value: Any, digits: int = 4) -> str:
    numeric = maybe_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}g}"


def build_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    router_readiness: pd.DataFrame,
    expert_risks: pd.DataFrame,
    specialization: pd.DataFrame,
) -> str:
    lines = [
        "# MoE Routing Readiness",
        "",
        "这个报告把 MoE routing probe 的原始 CSV 转成合并前的风险诊断。它回答的不是“该不该做 MoE”，而是更具体的四个问题：router 是否会 collapse，两个 source 的路由是否漂移，top-k 边界是否脆弱，以及哪些 experts 需要 route/category-aware 权重。",
        "",
        f"- Readiness status: `{summary['readiness_status']}`",
        f"- Router dirs: `{', '.join(summary['router_dirs']) if summary['router_dirs'] else 'none'}`",
        f"- Router rows: `{summary['router_rows']}`；expert rows: `{summary['expert_rows']}`；specialization rows: `{summary['specialization_rows']}`",
        "",
    ]
    topology = summary.get("topology")
    if topology:
        lines.extend(
            [
                "## 拓扑线索",
                "",
                f"- MoE model: `{topology.get('name')}` / `{topology.get('model_type')}`",
                f"- Experts: `{topology.get('num_experts')}`；active per token: `{topology.get('num_experts_per_tok')}`；active fraction: `{topology.get('active_expert_fraction_per_token')}`",
                f"- Local weights available: `{topology.get('weights_available')}`",
                "",
            ]
        )

    lines.extend(["## Router Readiness", ""])
    if router_readiness.empty:
        lines.append("当前没有真实 `router_summary.csv`。先运行 MoE routing probe，再重新生成本报告。")
    else:
        lines.append(f"Router action counts: `{json.dumps(summary['router_action_counts'], ensure_ascii=False)}`")
        lines.append("")
        lines.extend(
            [
                "| router | category | max top1 | effective fraction | top-k Jaccard | risk flags | action |",
                "| --- | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for _, row in router_readiness.sort_values(["risk_score"], ascending=False).head(20).iterrows():
            lines.append(
                f"| {row['router']} | {row['category']} | {fmt_optional(row['max_top1_fraction'])} | "
                f"{fmt_optional(row['effective_top1_fraction'])} | {fmt_optional(row['topk_jaccard'])} | "
                f"{row['risk_flags'] or 'none'} | `{row['recommended_action']}` |"
            )

    lines.extend(["", "## Expert Load Risks", ""])
    if expert_risks.empty:
        lines.append("当前没有真实 `expert_load.csv`。route-weight recipes 也会保持 `waiting_for_routing_probe`。")
    else:
        lines.append(f"Expert action counts: `{json.dumps(summary['expert_action_counts'], ensure_ascii=False)}`")
        lines.append("")
        lines.extend(
            [
                "| router | category | expert | top-k over uniform | flags | action |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        ranked = expert_risks.assign(abs_overuse=expert_risks["topk_over_uniform"].fillna(0.0)).sort_values(
            "abs_overuse", ascending=False
        )
        for _, row in ranked.head(20).iterrows():
            lines.append(
                f"| {row['router']} | {row['category']} | {row['expert_id']} | "
                f"{fmt_optional(row['topk_over_uniform'])} | {row['risk_flags'] or 'none'} | `{row['recommended_action']}` |"
            )

    lines.extend(["", "## Category Specialization", ""])
    if specialization.empty:
        lines.append("当前没有 category-level expert specialization 证据。")
    else:
        lines.append(f"Specialization action counts: `{json.dumps(summary['specialization_action_counts'], ensure_ascii=False)}`")
        lines.append("")
        lines.extend(
            [
                "| router | expert | dominant category | share | action |",
                "| --- | ---: | --- | ---: | --- |",
            ]
        )
        for _, row in specialization.sort_values(["dominant_category_share"], ascending=False).head(20).iterrows():
            lines.append(
                f"| {row['router']} | {row['expert_id']} | {row['dominant_category']} | "
                f"{fmt_optional(row['dominant_category_share'])} | `{row['recommended_action']}` |"
            )

    lines.extend(
        [
            "",
            "## 规则依据",
            "",
            "- [HARC / routing-breakdown](https://arxiv.org/abs/2606.03391) 分析说明 MoE router 的 softmax/top-k 对参数扰动敏感，因此 `low_topk_route_overlap`、`low_top1_route_agreement` 和 load concentration 都应阻止直接 router average。",
            "- [Sub-MoE](https://arxiv.org/abs/2506.23266) / [MergeMoE](https://arxiv.org/abs/2510.14436) 强调 expert specialization 和 expert output alignment，因此高负载或强 category-specialized experts 应先做 route/source-aware 权重，而不是同权平均。",
            "- [Expert Merging](https://arxiv.org/abs/2509.25712) 强调 layer/chunk-wise coefficients；本报告输出的风险表应和 `build_moe_route_weight_recipes.py` 的 tensor rules 以及后续 layer-wise coefficient search 联动。",
            "",
            "## 下一步命令",
            "",
            "```bash",
            "python scripts/probe_moe_routing.py --model Qwen/Qwen3-30B-A3B --compare-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --use-chat-template --output-dir results/moe_routing_probe/qwen3_30b_general_vs_code",
            "PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code",
            "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code --category-source general=general --category-source code=code --category-source math=general --category-source safety=general",
            "```",
            "",
            "## Files",
            "",
            f"- `{rel(output_dir / 'router_readiness.csv')}`",
            f"- `{rel(output_dir / 'expert_load_risks.csv')}`",
            f"- `{rel(output_dir / 'category_specialization.csv')}`",
            f"- `{rel(output_dir / 'summary.json')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MoE routing probe readiness before same-shape model averaging.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_routing_readiness"))
    parser.add_argument("--router-dir", action="append", default=[], help="Output directory from scripts/probe_moe_routing.py.")
    parser.add_argument("--collapse-threshold", type=float, default=0.50)
    parser.add_argument("--min-effective-fraction", type=float, default=0.02)
    parser.add_argument("--min-topk-jaccard", type=float, default=0.50)
    parser.add_argument("--min-top1-agreement", type=float, default=0.50)
    parser.add_argument("--low-margin-threshold", type=float, default=0.05)
    parser.add_argument("--overuse-ratio-threshold", type=float, default=4.0)
    parser.add_argument("--underuse-fraction-threshold", type=float, default=0.0)
    parser.add_argument("--specialization-threshold", type=float, default=0.70)
    parser.add_argument("--topology-summary", default="results/checkpoint_topology_inspect/summary.json")
    parser.add_argument("--topology-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    router_dirs = [repo_path(path) for path in args.router_dir]

    router_readiness = build_router_readiness(
        router_dirs,
        collapse_threshold=args.collapse_threshold,
        min_effective_fraction=args.min_effective_fraction,
        min_topk_jaccard=args.min_topk_jaccard,
        min_top1_agreement=args.min_top1_agreement,
        low_margin_threshold=args.low_margin_threshold,
    )
    expert_risks = build_expert_load_risks(
        router_dirs,
        overuse_ratio_threshold=args.overuse_ratio_threshold,
        underuse_fraction_threshold=args.underuse_fraction_threshold,
    )
    specialization = build_category_specialization(router_dirs, args.specialization_threshold)
    readiness_status = status_from_tables(router_readiness, router_dirs)
    router_readiness.to_csv(output_dir / "router_readiness.csv", index=False)
    expert_risks.to_csv(output_dir / "expert_load_risks.csv", index=False)
    specialization.to_csv(output_dir / "category_specialization.csv", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness_status": readiness_status,
        "router_dirs": [rel(path) for path in router_dirs],
        "router_rows": int(len(router_readiness)),
        "expert_rows": int(len(expert_risks)),
        "specialization_rows": int(len(specialization)),
        "router_action_counts": table_action_counts(router_readiness),
        "expert_action_counts": table_action_counts(expert_risks),
        "specialization_action_counts": table_action_counts(specialization),
        "thresholds": {
            "collapse_threshold": args.collapse_threshold,
            "min_effective_fraction": args.min_effective_fraction,
            "min_topk_jaccard": args.min_topk_jaccard,
            "min_top1_agreement": args.min_top1_agreement,
            "low_margin_threshold": args.low_margin_threshold,
            "overuse_ratio_threshold": args.overuse_ratio_threshold,
            "underuse_fraction_threshold": args.underuse_fraction_threshold,
            "specialization_threshold": args.specialization_threshold,
        },
        "topology": discover_moe_topology(repo_path(args.topology_summary), args.topology_model),
        "same_shape_constraint": "Readiness actions preserve router shape, expert count, hidden size, tokenizer, and model class.",
        "outputs": {
            "router_readiness": rel(output_dir / "router_readiness.csv"),
            "expert_load_risks": rel(output_dir / "expert_load_risks.csv"),
            "category_specialization": rel(output_dir / "category_specialization.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        build_report(
            output_dir=output_dir,
            summary=summary,
            router_readiness=router_readiness,
            expert_risks=expert_risks,
            specialization=specialization,
        ),
        encoding="utf-8",
    )
    print(f"Wrote MoE routing readiness to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
