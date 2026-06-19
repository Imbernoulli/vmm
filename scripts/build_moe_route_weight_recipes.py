#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WEIGHT_COLUMNS = [
    "layer_id",
    "router",
    "expert_id",
    "total_topk_fraction",
    "dominant_source",
    "dominant_weight",
    "same_shape_action",
    "tensor_pattern",
    "tensor_rule",
    "reason",
]
DEFAULT_PROBE_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_PROBE_COMPARE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_PROBE_PROMPTS = "prompts/qwen_moe_route_probe_prompts.jsonl"
DEFAULT_PROBE_OUTPUT_DIR = "results/moe_routing_probe/qwen3_30b_general_vs_code"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_pair(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"Expected non-empty KEY=VALUE, got: {raw}")
    return key, value


def parse_weight(raw: str) -> tuple[str, float]:
    key, value = parse_pair(raw)
    return key, float(value)


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


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_moe_topology(summary_path: Path, model_name: str | None) -> dict[str, Any] | None:
    if not summary_path.exists() or not summary_path.is_file():
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


def source_for_category(category: str, category_sources: dict[str, str], source_names: list[str], fallback: str) -> str:
    if category in category_sources:
        return category_sources[category]
    normalized = category.lower().replace("-", "_")
    if "code" in source_names and (
        normalized == "code"
        or normalized.endswith("_code")
        or "coding" in normalized
        or "programming" in normalized
        or "software" in normalized
    ):
        return "code"
    if category in source_names:
        return category
    return fallback


def category_mapping_kind(category: str, category_sources: dict[str, str], source_names: list[str], fallback_source: str) -> str:
    if category in category_sources:
        return "explicit"
    source = source_for_category(category, category_sources, source_names, fallback_source)
    if category in source_names and source == category:
        return "source_name_match"
    if source in source_names and source != fallback_source:
        return "category_heuristic"
    return "fallback"


def build_category_source_plan(
    *,
    prompt_path: Path,
    source_names: list[str],
    category_sources: dict[str, str],
    fallback_source: str,
) -> list[dict[str, Any]]:
    prompts = read_jsonl_if_exists(prompt_path)
    counts: dict[str, int] = defaultdict(int)
    for row in prompts:
        counts[str(row.get("category", "default"))] += 1
    rows = []
    for category, count in sorted(counts.items()):
        source = source_for_category(category, category_sources, source_names, fallback_source)
        rows.append(
            {
                "category": category,
                "prompt_count": count,
                "mapped_source": source,
                "mapping_kind": category_mapping_kind(category, category_sources, source_names, fallback_source),
            }
        )
    return rows


def build_probe_command(
    *,
    model: str,
    compare_model: str | None,
    prompt_path: Path,
    output_dir: Path,
    device_map: str,
    dtype: str,
    max_length: int,
    use_chat_template: bool,
) -> str:
    parts = [
        "python scripts/probe_moe_routing.py",
        f"--model {model}",
    ]
    if compare_model:
        parts.append(f"--compare-model {compare_model}")
    parts.extend(
        [
            f"--prompts {rel(prompt_path)}",
            f"--device-map {device_map}",
            f"--dtype {dtype}",
            f"--max-length {max_length}",
        ]
    )
    if use_chat_template:
        parts.append("--use-chat-template")
    parts.extend([f"--output-dir {rel(output_dir)}"])
    return " ".join(parts)


def build_routing_probe_plan(
    *,
    model: str,
    compare_model: str | None,
    prompt_path: Path,
    output_dir: Path,
    source_names: list[str],
    category_source_rows: list[dict[str, Any]],
    device_map: str,
    dtype: str,
    max_length: int,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    category_args = " ".join(
        f"--category-source {row['category']}={row['mapped_source']}" for row in category_source_rows
    )
    command = build_probe_command(
        model=model,
        compare_model=compare_model,
        prompt_path=prompt_path,
        output_dir=output_dir,
        device_map=device_map,
        dtype=dtype,
        max_length=max_length,
        use_chat_template=use_chat_template,
    )
    return [
        {
            "probe_name": output_dir.name,
            "model": model,
            "compare_model": compare_model,
            "source_names": ",".join(source_names),
            "prompt_pack": rel(prompt_path),
            "prompt_count": int(sum(row["prompt_count"] for row in category_source_rows)),
            "category_count": int(len(category_source_rows)),
            "output_dir": rel(output_dir),
            "expected_expert_load": rel(output_dir / "expert_load.csv"),
            "expected_route_overlap": rel(output_dir / "route_overlap.csv") if compare_model else "",
            "device_map": device_map,
            "dtype": dtype,
            "max_length": max_length,
            "use_chat_template": use_chat_template,
            "command": command,
            "next_recipe_command": (
                "PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py "
                f"--router-dir {rel(output_dir)} "
                + " ".join(f"--source {source}" for source in source_names)
                + (f" {category_args}" if category_args else "")
            ),
        }
    ]


def layer_id_from_router(router: str, layer_regex: re.Pattern[str]) -> str:
    match = layer_regex.search(router)
    return match.group(1) if match else ""


def load_route_masses(
    *,
    router_dirs: list[Path],
    source_names: list[str],
    category_sources: dict[str, str],
    fallback_source: str,
    layer_regex: re.Pattern[str],
) -> tuple[dict[tuple[str, str, int], dict[str, float]], list[dict[str, Any]]]:
    route_masses: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: {name: 0.0 for name in source_names})
    inputs: list[dict[str, Any]] = []
    for router_dir in router_dirs:
        expert_load = read_csv_if_exists(router_dir / "expert_load.csv")
        if expert_load is None or expert_load.empty:
            inputs.append({"router_dir": rel(router_dir), "expert_load_rows": 0, "status": "missing_or_empty"})
            continue
        required = {"category", "router", "expert_id", "topk_fraction"}
        missing = sorted(required - set(expert_load.columns))
        if missing:
            raise ValueError(f"{router_dir / 'expert_load.csv'} is missing columns: {missing}")
        used_rows = 0
        for _, item in expert_load.iterrows():
            category = str(item["category"])
            source = source_for_category(category, category_sources, source_names, fallback_source)
            if source not in source_names:
                raise ValueError(f"Category {category!r} maps to unknown source {source!r}")
            router = str(item["router"])
            expert_id = int(item["expert_id"])
            layer_id = layer_id_from_router(router, layer_regex)
            topk_fraction = maybe_float(item.get("topk_fraction"))
            if topk_fraction <= 0:
                continue
            route_masses[(layer_id, router, expert_id)][source] += topk_fraction
            used_rows += 1
        inputs.append({"router_dir": rel(router_dir), "expert_load_rows": int(len(expert_load)), "used_rows": used_rows, "status": "loaded"})
    return route_masses, inputs


def normalize_source_weights(
    masses: dict[str, float],
    *,
    source_names: list[str],
    anchor_floor: float,
    min_source_weight: float,
) -> dict[str, float]:
    total = sum(max(0.0, masses.get(name, 0.0)) for name in source_names)
    if total <= 0:
        return {name: 0.0 for name in source_names}
    raw = {name: max(0.0, masses.get(name, 0.0)) / total for name in source_names}
    kept = {name: value for name, value in raw.items() if value >= min_source_weight}
    if kept:
        kept_total = sum(kept.values())
        raw = {name: kept.get(name, 0.0) / kept_total for name in source_names}
    scale = max(0.0, min(1.0, 1.0 - anchor_floor))
    return {name: raw[name] * scale for name in source_names}


def format_weights(weights: dict[str, float], source_names: list[str]) -> str:
    return ",".join(f"{name}={weights[name]:.6g}" for name in source_names)


def build_tensor_pattern(layer_id: str, expert_id: int, template: str, fallback_template: str) -> str:
    if layer_id:
        return template.format(layer_id=layer_id, expert_id=expert_id)
    return fallback_template.format(expert_id=expert_id)


def build_source_weight_rows(
    route_masses: dict[tuple[str, str, int], dict[str, float]],
    *,
    source_names: list[str],
    anchor_floor: float,
    min_source_weight: float,
    low_route_threshold: float,
    expert_tensor_pattern_template: str,
    fallback_expert_tensor_pattern_template: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (layer_id, router, expert_id), masses in sorted(route_masses.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        total_topk = sum(max(0.0, masses.get(name, 0.0)) for name in source_names)
        pattern = build_tensor_pattern(
            layer_id,
            expert_id,
            expert_tensor_pattern_template,
            fallback_expert_tensor_pattern_template,
        )
        if total_topk < low_route_threshold:
            weights = {name: 0.0 for name in source_names}
            action = "anchor_heavy_or_freeze"
            reason = "route mass below threshold; keep this expert close to base/anchor until sensitivity probe proves it matters."
        else:
            weights = normalize_source_weights(
                masses,
                source_names=source_names,
                anchor_floor=anchor_floor,
                min_source_weight=min_source_weight,
            )
            max_source = max(source_names, key=lambda name: weights[name])
            max_weight = weights[max_source]
            if max_weight >= (1.0 - anchor_floor) * 0.80:
                action = "dominant_source_expert_delta"
                reason = "one prompt/source family dominates this expert's route mass; use a source-skewed delta."
            else:
                action = "mixed_source_expert_delta"
                reason = "multiple prompt/source families use this expert; keep a mixed route-frequency average."
        dominant_source = max(source_names, key=lambda name: weights[name]) if source_names else ""
        tensor_rule = f"{pattern}::{format_weights(weights, source_names)}"
        row = {
            "layer_id": layer_id,
            "router": router,
            "expert_id": expert_id,
            "total_topk_fraction": total_topk,
            "dominant_source": dominant_source,
            "dominant_weight": weights.get(dominant_source, 0.0),
            "same_shape_action": action,
            "tensor_pattern": pattern,
            "tensor_rule": tensor_rule,
            "reason": reason,
        }
        for source in source_names:
            row[f"weight_{source}"] = weights[source]
            row[f"route_mass_{source}"] = masses.get(source, 0.0)
        rows.append(row)
    return rows


def load_explicit_expert_weight_rows(
    *,
    paths: list[Path],
    source_names: list[str],
    category_filter: str | None,
    expert_tensor_pattern_template: str,
    fallback_expert_tensor_pattern_template: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        weights_df = read_csv_if_exists(path)
        if weights_df is None or weights_df.empty:
            inputs.append({"expert_weight_csv": rel(path), "rows": 0, "status": "missing_or_empty"})
            continue
        original_rows = int(len(weights_df))
        if category_filter is not None:
            if "category" not in weights_df.columns:
                raise ValueError(f"{path} uses --expert-weight-category but has no category column")
            weights_df = weights_df[weights_df["category"].astype(str) == category_filter].copy()
            if weights_df.empty:
                inputs.append(
                    {
                        "expert_weight_csv": rel(path),
                        "rows": original_rows,
                        "filtered_rows": 0,
                        "category_filter": category_filter,
                        "status": "filtered_empty",
                    }
                )
                continue
        required = {"expert_id"} | {f"weight_{source}" for source in source_names}
        missing = sorted(required - set(weights_df.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        used_rows = 0
        for _, item in weights_df.iterrows():
            expert_id = int(item["expert_id"])
            layer_id = "" if "layer_id" not in item or pd.isna(item["layer_id"]) else str(item["layer_id"])
            router = "" if "router" not in item or pd.isna(item["router"]) else str(item["router"])
            weights = {source: maybe_float(item.get(f"weight_{source}")) for source in source_names}
            total_weight = sum(max(0.0, weights[source]) for source in source_names)
            route_masses = {source: maybe_float(item.get(f"route_mass_{source}")) for source in source_names}
            total_route = sum(max(0.0, route_masses[source]) for source in source_names)
            pattern = build_tensor_pattern(
                layer_id,
                expert_id,
                expert_tensor_pattern_template,
                fallback_expert_tensor_pattern_template,
            )
            dominant_source = max(source_names, key=lambda source: weights[source]) if source_names else ""
            row = {
                "layer_id": layer_id,
                "router": router,
                "expert_id": expert_id,
                "total_topk_fraction": total_route,
                "dominant_source": dominant_source,
                "dominant_weight": weights.get(dominant_source, 0.0),
                "same_shape_action": str(item.get("same_shape_action", "explicit_expert_delta_weights")),
                "tensor_pattern": pattern,
                "tensor_rule": f"{pattern}::{format_weights(weights, source_names)}",
                "reason": (
                    "explicit per-expert source weights supplied by calibration/search"
                    if total_weight > 0
                    else "explicit row freezes this expert to the anchor/base"
                ),
            }
            for source in source_names:
                row[f"weight_{source}"] = weights[source]
                row[f"route_mass_{source}"] = route_masses[source]
            rows.append(row)
            used_rows += 1
        inputs.append(
            {
                "expert_weight_csv": rel(path),
                "rows": original_rows,
                "filtered_rows": int(len(weights_df)),
                "used_rows": used_rows,
                "category_filter": category_filter,
                "status": "loaded",
            }
        )
    return rows, inputs


def shared_weights(source_names: list[str], raw_weights: list[str]) -> dict[str, float]:
    if raw_weights:
        weights = {name: 0.0 for name in source_names}
        for raw in raw_weights:
            name, value = parse_weight(raw)
            if name not in weights:
                raise ValueError(f"Shared weight references unknown source {name!r}")
            weights[name] = value
        return weights
    weight = 1.0 / max(1, len(source_names))
    return {name: weight for name in source_names}


def build_tensor_rules(
    *,
    rows: list[dict[str, Any]],
    source_names: list[str],
    shared_attention_weights: dict[str, float],
    shared_mlp_weights: dict[str, float] | None,
    expert_rule_label: str,
) -> list[str]:
    rules = [
        "# Shared attention rule. Keep before expert rules because writer uses first match.",
        f".*self_attn.*::{format_weights(shared_attention_weights, source_names)}",
    ]
    if rows:
        rules.append(f"# {expert_rule_label}")
        rules.extend(str(row["tensor_rule"]) for row in rows)
    else:
        rules.append("# Expert rules are omitted until routing probe expert_load.csv is available.")
    if shared_mlp_weights is not None:
        rules.extend(
            [
                "# Shared dense MLP fallback. It is placed after expert rules because writer uses first match.",
                f".*(^|\\.)mlp\\..*(gate_proj|up_proj|down_proj).*::{format_weights(shared_mlp_weights, source_names)}",
            ]
        )
    return rules


def build_writer_command(
    *,
    source_names: list[str],
    tensor_rule_file: Path,
    output_checkpoint_dir: str,
    freeze_router: bool,
) -> str:
    parts = [
        "python scripts/write_same_shape_average_checkpoint.py",
        "--base MOE_BASE_OR_ANCHOR_PATH",
    ]
    for source in source_names:
        parts.append(f"--source {source}={source.upper()}_MODEL_PATH")
    for source in source_names:
        parts.append(f"--source-weight {source}=0.0")
    if freeze_router:
        parts.append("--freeze-router")
    parts.extend(
        [
            f"--tensor-rule-file {rel(tensor_rule_file)}",
            f"--output-dir {output_checkpoint_dir}",
            "--dry-run",
        ]
    )
    return " ".join(parts)


def build_report(
    *,
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    tensor_rule_file: Path,
    writer_command: str,
) -> str:
    status = summary["recipe_status"]
    recipe_kind = summary.get("recipe_kind", "route_frequency")
    lines = [
        "# MoE Route-Weight Recipes",
        "",
        "这个报告把 MoE routing/expert-load probe 或显式 expert 权重转成同构 checkpoint writer 可以读取的 tensor-rule 权重。目标不是增加 experts 或做 ensemble，而是在原 expert 数、原 router shape 下，给每个 expert 设置更合理的 source delta 系数。",
        "",
        f"- Recipe status: `{status}`",
        f"- Recipe kind: `{recipe_kind}`",
        f"- Sources: `{', '.join(summary['source_names'])}`",
        f"- Router dirs: `{', '.join(summary['router_dirs']) if summary['router_dirs'] else 'none'}`",
        f"- Expert weight CSVs: `{', '.join(summary.get('expert_weight_csvs', [])) if summary.get('expert_weight_csvs') else 'none'}`",
        f"- Expert weight category filter: `{summary.get('expert_weight_category') or 'none'}`",
        f"- Expert tensor rules: `{summary['expert_rule_count']}`",
        f"- Tensor rule file: `{rel(tensor_rule_file)}`",
        "",
        "## 权重规则",
        "",
        "如果传入 `--expert-weight-csv`，脚本直接使用 `weight_<source>` 列；否则先按 prompt category 把 route mass 分给对应 source，然后做归一化：",
        "",
        "```text",
        "route_mass[source, layer, expert] = sum topk_fraction(category -> source)",
        "writer_weight[source, layer, expert] = (1 - anchor_floor) * normalize(route_mass)",
        "```",
        "",
        "剩下的 `anchor_floor` 留给 base/anchor checkpoint，低使用率 expert 默认保持 anchor-heavy/frozen。这样做是为了避免 MoE average 直接把低证据、低路由频率的 experts 拉偏。",
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
    lines.extend(["## 当前专家权重摘要", ""])
    if source_rows:
        df = pd.DataFrame(source_rows)
        action_counts = df["same_shape_action"].value_counts().to_dict()
        lines.append(f"Action counts: `{json.dumps(action_counts, ensure_ascii=False)}`")
        lines.append("")
        preview_columns = ["layer_id", "expert_id", "total_topk_fraction", "dominant_source", "dominant_weight", "same_shape_action"]
        lines.extend(["| layer | expert | total top-k fraction | dominant source | dominant weight | action |", "| --- | ---: | ---: | --- | ---: | --- |"])
        for _, row in df.head(20).iterrows():
            lines.append(
                f"| {row['layer_id']} | {int(row['expert_id'])} | {float(row['total_topk_fraction']):.4g} | "
                f"{row['dominant_source']} | {float(row['dominant_weight']):.4g} | `{row['same_shape_action']}` |"
            )
    else:
        lines.append("当前没有真实 `expert_load.csv` 或 explicit expert-weight CSV，因此只生成 shared-module 规则和 writer 模板。下一步需要先跑 MoE routing probe 或传入搜索权重。")
    category_source_plan = summary.get("category_source_plan", [])
    if category_source_plan:
        lines.extend(
            [
                "",
                "## Prompt Category Source Map",
                "",
                "| category | prompts | mapped source | mapping |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for row in category_source_plan:
            lines.append(
                f"| {row['category']} | {int(row['prompt_count'])} | {row['mapped_source']} | {row['mapping_kind']} |"
            )
    routing_probe_plan = summary.get("routing_probe_plan", [])
    if routing_probe_plan:
        lines.extend(
            [
                "",
                "## Routing Probe Plan",
                "",
                "| probe | model | compare model | prompts | output |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for row in routing_probe_plan:
            lines.append(
                f"| {row['probe_name']} | `{row['model']}` | `{row.get('compare_model') or ''}` | "
                f"{int(row['prompt_count'])} | `{row['output_dir']}` |"
            )
    lines.extend(
        [
            "",
            "## Writer Dry-Run Command",
            "",
            "```bash",
            writer_command,
            "```",
            "",
            "## 需要先跑的 Routing Probe",
            "",
            "```bash",
            routing_probe_plan[0]["command"] if routing_probe_plan else "",
            "```",
            "",
            "然后重新生成 route weights：",
            "",
            "```bash",
            routing_probe_plan[0]["next_recipe_command"] if routing_probe_plan else "",
            "```",
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['source_weights']}`",
            f"- `{summary['outputs']['tensor_rules']}`",
            f"- `{summary['outputs']['writer_command']}`",
            f"- `{summary['outputs']['routing_probe_plan']}`",
            f"- `{summary['outputs']['category_source_plan']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build route-frequency tensor-rule recipes for same-shape MoE averaging.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_route_weight_recipes"))
    parser.add_argument("--router-dir", action="append", default=[], help="Output directory from scripts/probe_moe_routing.py.")
    parser.add_argument("--expert-weight-csv", action="append", default=[], help="CSV with expert_id and weight_<source> columns from a calibration/search step.")
    parser.add_argument("--expert-weight-category", default=None, help="When expert-weight CSV has a category column, keep only rows with this category.")
    parser.add_argument("--source", action="append", default=[], help="Source name used by the checkpoint writer, e.g. general or code.")
    parser.add_argument("--category-source", action="append", default=[], help="Map prompt category to source, e.g. code=code.")
    parser.add_argument("--fallback-source", default=None, help="Source used for unmapped prompt categories. Defaults to first --source.")
    parser.add_argument("--anchor-floor", type=float, default=0.15, help="Unmerged base/anchor reserve for expert deltas.")
    parser.add_argument("--min-source-weight", type=float, default=0.05, help="Trim tiny normalized source weights before rescaling.")
    parser.add_argument("--low-route-threshold", type=float, default=0.005)
    parser.add_argument("--shared-source-weight", action="append", default=[], help="Shared attention source weight NAME=VALUE.")
    parser.add_argument("--shared-mlp-source-weight", action="append", default=[], help="Optional shared dense MLP source weight NAME=VALUE.")
    parser.add_argument("--no-freeze-router", dest="freeze_router", action="store_false")
    parser.add_argument("--layer-regex", default=r"layers\.(\d+)")
    parser.add_argument("--expert-tensor-pattern-template", default=r".*layers\.{layer_id}\..*experts\.{expert_id}\..*")
    parser.add_argument("--fallback-expert-tensor-pattern-template", default=r".*experts\.{expert_id}\..*")
    parser.add_argument("--topology-summary", default="results/checkpoint_topology_inspect/summary.json")
    parser.add_argument("--topology-model", default=None)
    parser.add_argument("--checkpoint-output-dir", default="results/checkpoints/moe_route_aware_candidate")
    parser.add_argument("--probe-model", default=DEFAULT_PROBE_MODEL)
    parser.add_argument("--probe-compare-model", default=DEFAULT_PROBE_COMPARE_MODEL)
    parser.add_argument("--probe-prompts", default=DEFAULT_PROBE_PROMPTS)
    parser.add_argument("--probe-output-dir", default=DEFAULT_PROBE_OUTPUT_DIR)
    parser.add_argument("--probe-device-map", default="auto")
    parser.add_argument("--probe-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--probe-max-length", type=int, default=768)
    parser.add_argument("--no-probe-chat-template", dest="probe_chat_template", action="store_false")
    parser.set_defaults(freeze_router=True)
    parser.set_defaults(probe_chat_template=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_names = args.source or ["general", "code"]
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"Source names must be unique: {source_names}")
    category_sources = dict(parse_pair(raw) for raw in args.category_source)
    fallback_source = args.fallback_source or source_names[0]
    if fallback_source not in source_names:
        raise ValueError(f"Fallback source {fallback_source!r} is not in source list {source_names}")
    for category, source in category_sources.items():
        if source not in source_names:
            raise ValueError(f"Category {category!r} maps to unknown source {source!r}")

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    router_dirs = [repo_path(path) for path in args.router_dir]
    probe_prompt_path = repo_path(args.probe_prompts)
    probe_output_dir = repo_path(args.probe_output_dir)
    category_source_rows = build_category_source_plan(
        prompt_path=probe_prompt_path,
        source_names=source_names,
        category_sources=category_sources,
        fallback_source=fallback_source,
    )
    routing_probe_plan = build_routing_probe_plan(
        model=args.probe_model,
        compare_model=args.probe_compare_model,
        prompt_path=probe_prompt_path,
        output_dir=probe_output_dir,
        source_names=source_names,
        category_source_rows=category_source_rows,
        device_map=args.probe_device_map,
        dtype=args.probe_dtype,
        max_length=args.probe_max_length,
        use_chat_template=args.probe_chat_template,
    )
    pd.DataFrame(category_source_rows).to_csv(output_dir / "category_source_plan.csv", index=False)
    pd.DataFrame(routing_probe_plan).to_csv(output_dir / "routing_probe_plan.csv", index=False)
    (output_dir / "routing_probe_plan.json").write_text(
        json.dumps(routing_probe_plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    layer_regex = re.compile(args.layer_regex)
    expert_weight_csvs = [repo_path(path) for path in args.expert_weight_csv]
    if expert_weight_csvs:
        source_rows, expert_weight_inputs = load_explicit_expert_weight_rows(
            paths=expert_weight_csvs,
            source_names=source_names,
            category_filter=args.expert_weight_category,
            expert_tensor_pattern_template=args.expert_tensor_pattern_template,
            fallback_expert_tensor_pattern_template=args.fallback_expert_tensor_pattern_template,
        )
        input_summaries = expert_weight_inputs
        recipe_kind = "explicit_expert_weights"
    else:
        route_masses, input_summaries = load_route_masses(
            router_dirs=router_dirs,
            source_names=source_names,
            category_sources=category_sources,
            fallback_source=fallback_source,
            layer_regex=layer_regex,
        )
        source_rows = build_source_weight_rows(
            route_masses,
            source_names=source_names,
            anchor_floor=args.anchor_floor,
            min_source_weight=args.min_source_weight,
            low_route_threshold=args.low_route_threshold,
            expert_tensor_pattern_template=args.expert_tensor_pattern_template,
            fallback_expert_tensor_pattern_template=args.fallback_expert_tensor_pattern_template,
        )
        recipe_kind = "route_frequency"
    shared_attention = shared_weights(source_names, args.shared_source_weight)
    shared_mlp = shared_weights(source_names, args.shared_mlp_source_weight) if args.shared_mlp_source_weight else None
    tensor_rules = build_tensor_rules(
        rows=source_rows,
        source_names=source_names,
        shared_attention_weights=shared_attention,
        shared_mlp_weights=shared_mlp,
        expert_rule_label="Explicit expert-weight rules." if expert_weight_csvs else "Route-frequency expert rules.",
    )

    tensor_rule_file = output_dir / "tensor_rules.txt"
    tensor_rule_file.write_text("\n".join(tensor_rules) + "\n", encoding="utf-8")
    writer_command = build_writer_command(
        source_names=source_names,
        tensor_rule_file=tensor_rule_file,
        output_checkpoint_dir=args.checkpoint_output_dir,
        freeze_router=args.freeze_router,
    )
    (output_dir / "writer_command.txt").write_text(writer_command + "\n", encoding="utf-8")
    weight_columns = SOURCE_WEIGHT_COLUMNS + [f"weight_{name}" for name in source_names] + [f"route_mass_{name}" for name in source_names]
    pd.DataFrame(source_rows, columns=weight_columns).to_csv(output_dir / "source_weights_by_expert.csv", index=False)

    topology = discover_moe_topology(repo_path(args.topology_summary), args.topology_model)
    summary = {
        "recipe_status": (
            "explicit_expert_weight_rules_ready"
            if source_rows and expert_weight_csvs
            else "route_weight_rules_ready"
            if source_rows
            else "waiting_for_routing_probe"
        ),
        "recipe_kind": recipe_kind,
        "source_names": source_names,
        "category_sources": category_sources,
        "fallback_source": fallback_source,
        "router_dirs": [rel(path) for path in router_dirs],
        "expert_weight_csvs": [rel(path) for path in expert_weight_csvs],
        "expert_weight_category": args.expert_weight_category,
        "input_summaries": input_summaries,
        "routing_probe_plan_rows": len(routing_probe_plan),
        "routing_probe_plan": routing_probe_plan,
        "category_source_plan_rows": len(category_source_rows),
        "category_source_plan": category_source_rows,
        "prompt_pack": rel(probe_prompt_path),
        "anchor_floor": args.anchor_floor,
        "min_source_weight": args.min_source_weight,
        "low_route_threshold": args.low_route_threshold,
        "freeze_router": args.freeze_router,
        "shared_attention_weights": shared_attention,
        "shared_mlp_weights": shared_mlp,
        "expert_rule_count": len(source_rows),
        "tensor_rule_count": sum(1 for line in tensor_rules if line and not line.startswith("#")),
        "topology": topology,
        "same_shape_constraint": "Tensor rules only change weights inside the existing model structure; expert count, router shape, tokenizer, and layer count stay fixed.",
        "outputs": {
            "source_weights": rel(output_dir / "source_weights_by_expert.csv"),
            "tensor_rules": rel(tensor_rule_file),
            "writer_command": rel(output_dir / "writer_command.txt"),
            "routing_probe_plan": rel(output_dir / "routing_probe_plan.csv"),
            "routing_probe_plan_json": rel(output_dir / "routing_probe_plan.json"),
            "category_source_plan": rel(output_dir / "category_source_plan.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        build_report(
            summary=summary,
            source_rows=source_rows,
            tensor_rule_file=tensor_rule_file,
            writer_command=writer_command,
        ),
        encoding="utf-8",
    )
    print(f"Wrote MoE route-weight recipes to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
