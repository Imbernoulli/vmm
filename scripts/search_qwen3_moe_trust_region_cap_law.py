#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NONBASE_SOURCE = "coder"
BASE_SOURCE = "instruct"
EPS = 1e-12


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return clean_value(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_flags(raw: Any) -> set[str]:
    if clean_value(raw) is None:
        return set()
    return {part.strip() for part in str(raw).split("|") if part.strip()}


def add_boolean_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["trust_risk_flags"].apply(parse_flags)
    out["flag_high_load"] = parsed.apply(lambda flags: "high_load_expert" in flags or "high_load_weight_limit" in flags)
    out["flag_shared_mixed"] = parsed.apply(lambda flags: "shared_mixed_expert" in flags)
    out["flag_fragile_router"] = parsed.apply(lambda flags: "fragile_router_layer" in flags)
    out["flag_low_route_evidence"] = parsed.apply(lambda flags: "low_route_evidence" in flags)
    out["flag_category_mismatch"] = parsed.apply(lambda flags: "category_source_mismatch" in flags)
    out["flag_delta_above_base"] = parsed.apply(lambda flags: "delta_above_base_cap" in flags)
    return out


def cap_law_grid() -> list[dict[str, float]]:
    rows = []
    for base_cap, min_cap, high_load, shared, fragile, low_evidence, mismatch in itertools.product(
        [0.75, 0.70, 0.65],
        [0.60, 0.65],
        [0.00, 0.05, 0.10],
        [0.00, 0.03],
        [0.00, 0.03, 0.05],
        [0.00, 0.05],
        [0.00, 0.05],
    ):
        if min_cap > base_cap:
            continue
        rows.append(
            {
                "base_cap": base_cap,
                "min_cap": min_cap,
                "high_load_penalty": high_load,
                "shared_mixed_penalty": shared,
                "fragile_router_penalty": fragile,
                "low_route_evidence_penalty": low_evidence,
                "category_mismatch_penalty": mismatch,
            }
        )
    return rows


def target_cap_for_law(df: pd.DataFrame, law: dict[str, float]) -> pd.Series:
    penalty = (
        df["flag_high_load"].astype(float) * law["high_load_penalty"]
        + df["flag_shared_mixed"].astype(float) * law["shared_mixed_penalty"]
        + df["flag_fragile_router"].astype(float) * law["fragile_router_penalty"]
        + df["flag_low_route_evidence"].astype(float) * law["low_route_evidence_penalty"]
        + df["flag_category_mismatch"].astype(float) * law["category_mismatch_penalty"]
    )
    return (law["base_cap"] - penalty).clip(lower=law["min_cap"])


def metrics_from_scale(
    df: pd.DataFrame,
    scale: pd.Series,
    *,
    law_id: str,
    law: dict[str, float] | None = None,
    kind: str,
) -> dict[str, Any]:
    audit_max = df["audit_max_relative_delta_norm"].clip(lower=0.0)
    predicted_max = audit_max * scale
    route_mass = df["total_topk_fraction"].clip(lower=0.0)
    original_nonbase = df["original_effective_nonbase_weight"].clip(lower=0.0)
    audit_delta_norm = df["audit_delta_norm"].clip(lower=0.0)
    original_mass = float((route_mass * original_nonbase).sum())
    preserved_mass = float((route_mass * original_nonbase * scale).sum())
    original_delta_norm = float(math.sqrt(float((audit_delta_norm**2).sum())))
    predicted_delta_norm = float(math.sqrt(float(((audit_delta_norm * scale) ** 2).sum())))
    out: dict[str, Any] = {
        "law_id": law_id,
        "kind": kind,
        "base_cap": None if law is None else law["base_cap"],
        "min_cap": None if law is None else law["min_cap"],
        "high_load_penalty": None if law is None else law["high_load_penalty"],
        "shared_mixed_penalty": None if law is None else law["shared_mixed_penalty"],
        "fragile_router_penalty": None if law is None else law["fragile_router_penalty"],
        "low_route_evidence_penalty": None if law is None else law["low_route_evidence_penalty"],
        "category_mismatch_penalty": None if law is None else law["category_mismatch_penalty"],
        "scaled_group_count": int((scale < 0.999999).sum()),
        "max_predicted_relative_delta": float(predicted_max.max()),
        "p99_predicted_relative_delta": float(predicted_max.quantile(0.99)),
        "p95_predicted_relative_delta": float(predicted_max.quantile(0.95)),
        "routed_gt_075_groups": int((predicted_max > 0.75 + 1e-9).sum()),
        "routed_gt_065_groups": int((predicted_max > 0.65 + 1e-9).sum()),
        "routed_gt_050_groups": int((predicted_max > 0.50 + 1e-9).sum()),
        "route_mass_weighted_original_nonbase": original_mass,
        "route_mass_weighted_preserved_nonbase": preserved_mass,
        "nonbase_mass_retention": preserved_mass / max(EPS, original_mass),
        "delta_norm_proxy": predicted_delta_norm,
        "delta_norm_proxy_ratio_vs_uncapped": predicted_delta_norm / max(EPS, original_delta_norm),
        "mean_delta_scale": float(scale.mean()),
        "min_delta_scale": float(scale.min()),
    }
    out["internal_risk_score"] = (
        out["routed_gt_075_groups"] * 1000.0
        + out["routed_gt_065_groups"] * 5.0
        + out["routed_gt_050_groups"] * 0.1
        + max(0.0, out["max_predicted_relative_delta"] - 0.65) * 100.0
        + (1.0 - out["nonbase_mass_retention"]) * 20.0
        + out["delta_norm_proxy_ratio_vs_uncapped"]
    )
    return out


def simulate_law(df: pd.DataFrame, law: dict[str, float], idx: int) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    target_cap = target_cap_for_law(df, law)
    audit_max = df["audit_max_relative_delta_norm"].clip(lower=0.0)
    scale = pd.Series(1.0, index=df.index)
    needs_scale = audit_max > target_cap
    scale.loc[needs_scale] = target_cap.loc[needs_scale] / audit_max.loc[needs_scale].clip(lower=EPS)
    law_id = (
        f"grid_{idx:05d}_b{law['base_cap']:.2f}_m{law['min_cap']:.2f}_"
        f"h{law['high_load_penalty']:.2f}_s{law['shared_mixed_penalty']:.2f}_"
        f"r{law['fragile_router_penalty']:.2f}_l{law['low_route_evidence_penalty']:.2f}_"
        f"c{law['category_mismatch_penalty']:.2f}"
    )
    return metrics_from_scale(df, scale, law_id=law_id, law=law, kind="searched_cap_law"), target_cap, scale


def current_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    uncapped = pd.Series(1.0, index=df.index)
    current_trust = df["trust_delta_scale"].clip(lower=0.0, upper=1.0)
    uniform_065 = pd.Series(1.0, index=df.index)
    audit_max = df["audit_max_relative_delta_norm"].clip(lower=0.0)
    uniform_065.loc[audit_max > 0.65] = 0.65 / audit_max.loc[audit_max > 0.65].clip(lower=EPS)
    return [
        metrics_from_scale(df, uncapped, law_id="route_guarded_uncapped_expert_rules", law=None, kind="reference"),
        metrics_from_scale(df, current_trust, law_id="current_trust_region_cap_law", law=None, kind="reference"),
        metrics_from_scale(df, uniform_065, law_id="uniform_065_tail_cap", law=None, kind="reference"),
    ]


def pareto_frontier(search: pd.DataFrame) -> pd.DataFrame:
    representatives = []
    for _, group in search.groupby(["routed_gt_075_groups", "routed_gt_065_groups"], sort=False):
        representatives.append(
            group.sort_values(
                ["nonbase_mass_retention", "delta_norm_proxy_ratio_vs_uncapped"],
                ascending=[False, True],
            ).head(4)
        )
        representatives.append(
            group.sort_values(
                ["delta_norm_proxy_ratio_vs_uncapped", "nonbase_mass_retention"],
                ascending=[True, False],
            ).head(4)
        )
        representatives.append(
            group.sort_values(
                ["internal_risk_score", "nonbase_mass_retention"],
                ascending=[True, False],
            ).head(4)
        )
    reduced = pd.concat(representatives, ignore_index=False).drop_duplicates("law_id").reset_index(drop=True)
    objectives = reduced[
        [
            "routed_gt_075_groups",
            "routed_gt_065_groups",
            "delta_norm_proxy_ratio_vs_uncapped",
            "nonbase_mass_retention",
        ]
    ].to_numpy(float)
    n = len(reduced)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        left = objectives[i]
        for j in range(n):
            if i == j:
                continue
            right = objectives[j]
            no_worse = (
                right[0] <= left[0] + 1e-12
                and right[1] <= left[1] + 1e-12
                and right[2] <= left[2] + 1e-12
                and right[3] >= left[3] - 1e-12
            )
            strictly_better = (
                right[0] < left[0] - 1e-12
                or right[1] < left[1] - 1e-12
                or right[2] < left[2] - 1e-12
                or right[3] > left[3] + 1e-12
            )
            if no_worse and strictly_better:
                dominated[i] = True
                break
    frontier = reduced.loc[[not flag for flag in dominated]].copy()
    return frontier.sort_values(
        ["routed_gt_075_groups", "routed_gt_065_groups", "nonbase_mass_retention"],
        ascending=[True, True, False],
    )


def select_laws(search: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ("route_guarded_uncapped_expert_rules", "current_trust_region_cap_law", "uniform_065_tail_cap"):
        row = references[references["law_id"] == name].iloc[0].to_dict()
        row["selection_role"] = "reference"
        rows.append(row)

    no_gt075 = search[search["routed_gt_075_groups"] == 0].copy()
    if not no_gt075.empty:
        row = no_gt075.sort_values(
            ["nonbase_mass_retention", "routed_gt_065_groups", "delta_norm_proxy_ratio_vs_uncapped"],
            ascending=[False, True, True],
        ).iloc[0].to_dict()
        row["selection_role"] = "searched_no_gt075_max_retention"
        rows.append(row)

    no_gt065 = search[search["routed_gt_065_groups"] == 0].copy()
    if not no_gt065.empty:
        row = no_gt065.sort_values(
            ["nonbase_mass_retention", "delta_norm_proxy_ratio_vs_uncapped"],
            ascending=[False, True],
        ).iloc[0].to_dict()
        row["selection_role"] = "searched_no_gt065_max_retention"
        rows.append(row)

    balanced = search.sort_values(
        ["internal_risk_score", "nonbase_mass_retention"],
        ascending=[True, False],
    ).iloc[0].to_dict()
    balanced["selection_role"] = "searched_min_internal_risk_score"
    rows.append(balanced)
    return pd.DataFrame(rows)


def risk_flag_ablation(df: pd.DataFrame) -> pd.DataFrame:
    base = {
        "base_cap": 0.75,
        "min_cap": 0.55,
        "high_load_penalty": 0.0,
        "shared_mixed_penalty": 0.0,
        "fragile_router_penalty": 0.0,
        "low_route_evidence_penalty": 0.0,
        "category_mismatch_penalty": 0.0,
    }
    _, _, base_scale = simulate_law(df, base, 0)
    base_metrics = metrics_from_scale(df, base_scale, law_id="base_075", law=base, kind="ablation_base")
    rows = []
    penalties = {
        "high_load": "high_load_penalty",
        "shared_mixed": "shared_mixed_penalty",
        "fragile_router": "fragile_router_penalty",
        "low_route_evidence": "low_route_evidence_penalty",
        "category_mismatch": "category_mismatch_penalty",
    }
    for label, column in penalties.items():
        law = dict(base)
        law[column] = 0.05
        metrics, _, scale = simulate_law(df, law, 0)
        flag_col = f"flag_{label}"
        rows.append(
            {
                "flag": label,
                "flagged_groups": int(df[flag_col].sum()),
                "additional_scaled_groups": int((scale < base_scale - 1e-12).sum()),
                "routed_gt_075_reduction": int(base_metrics["routed_gt_075_groups"] - metrics["routed_gt_075_groups"]),
                "routed_gt_065_reduction": int(base_metrics["routed_gt_065_groups"] - metrics["routed_gt_065_groups"]),
                "delta_norm_proxy_reduction": float(base_metrics["delta_norm_proxy"] - metrics["delta_norm_proxy"]),
                "nonbase_mass_retention_loss": float(
                    base_metrics["nonbase_mass_retention"] - metrics["nonbase_mass_retention"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["routed_gt_075_reduction", "routed_gt_065_reduction", "nonbase_mass_retention_loss"],
        ascending=[False, False, True],
    )


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def extract_writer_context(command_path: Path) -> tuple[str, dict[str, str]]:
    command = command_path.read_text(encoding="utf-8").strip()
    parts = shlex.split(command)
    base = ""
    sources: dict[str, str] = {}
    idx = 0
    while idx < len(parts):
        if parts[idx] == "--base" and idx + 1 < len(parts):
            base = parts[idx + 1]
            idx += 2
            continue
        if parts[idx] == "--source" and idx + 1 < len(parts):
            raw = parts[idx + 1]
            if "=" in raw:
                name, path = raw.split("=", 1)
                sources[name] = path
            idx += 2
            continue
        idx += 1
    if not base or BASE_SOURCE not in sources or NONBASE_SOURCE not in sources:
        raise ValueError(f"Could not recover base/source paths from {command_path}")
    return base, sources


def build_selected_rules(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    search_by_id: dict[str, tuple[pd.Series, pd.Series]],
    output_dir: Path,
    writer_context_command: Path,
) -> list[dict[str, Any]]:
    base_path, sources = extract_writer_context(writer_context_command)
    rows = []
    for _, selected_row in selected.iterrows():
        role = str(selected_row["selection_role"])
        if not role.startswith("searched_"):
            continue
        law_id = str(selected_row["law_id"])
        if law_id not in search_by_id:
            continue
        _, scale = search_by_id[law_id]
        safe_role = sanitize_id(role)
        rules_path = output_dir / f"{safe_role}_tensor_rules.txt"
        checkpoint_output_dir = f"results/checkpoints/qwen3_moe_{safe_role}_candidate"
        with rules_path.open("w", encoding="utf-8") as handle:
            handle.write("# Searched Qwen3 MoE expert cap-law rules. Shared attention is frozen by omission.\n")
            handle.write("# Router is frozen by the writer command.\n")
            for idx, row in df.sort_values(["layer_id", "expert_id"]).iterrows():
                coder_weight = float(row["original_weight_coder"]) * float(scale.loc[idx])
                instruct_weight = float(row["original_weight_instruct"])
                handle.write(
                    f"{row['tensor_pattern']}::{BASE_SOURCE}={instruct_weight:.6g},{NONBASE_SOURCE}={coder_weight:.6g}\n"
                )
        materialize_command = (
            "python scripts/write_same_shape_average_checkpoint.py "
            f"--base {shlex.quote(base_path)} "
            f"--source {BASE_SOURCE}={shlex.quote(sources[BASE_SOURCE])} "
            f"--source {NONBASE_SOURCE}={shlex.quote(sources[NONBASE_SOURCE])} "
            f"--source-weight {BASE_SOURCE}=0.0 --source-weight {NONBASE_SOURCE}=0.0 "
            f"--freeze-router --tensor-rule-file {rel(rules_path)} "
            f"--output-dir {checkpoint_output_dir}"
        )
        dry_run_command = f"{materialize_command} --dry-run"
        (output_dir / f"{safe_role}_writer_command.txt").write_text(materialize_command + "\n", encoding="utf-8")
        (output_dir / f"{safe_role}_dry_run_command.txt").write_text(dry_run_command + "\n", encoding="utf-8")
        manifest_path = repo_path(checkpoint_output_dir) / "merge_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rule_counts = manifest.get("rule_counts", {})
        expert_hits = sum(
            int(value) for key, value in rule_counts.items() if str(key).startswith("tensor_rule:.*layers\\.")
        )
        rows.append(
            {
                "selection_role": role,
                "law_id": law_id,
                "tensor_rules": rel(rules_path),
                "writer_command": rel(output_dir / f"{safe_role}_writer_command.txt"),
                "dry_run_command": rel(output_dir / f"{safe_role}_dry_run_command.txt"),
                "checkpoint_output_dir": checkpoint_output_dir,
                "dry_run_manifest": rel(manifest_path) if manifest_path.exists() else None,
                "dry_run_validated": bool(manifest.get("dry_run", False)),
                "dry_run_floating_tensors": int(manifest.get("floating_tensors", 0)),
                "dry_run_frozen_tensors": int(manifest.get("frozen_tensors", 0)),
                "dry_run_expert_rule_hits": int(expert_hits),
                "dry_run_freeze_router_hits": int(rule_counts.get("freeze_router", 0)),
            }
        )
    return rows


def build_report(
    summary: dict[str, Any],
    selected: pd.DataFrame,
    frontier: pd.DataFrame,
    ablation: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen3 MoE Trust-Region Cap-Law 搜索",
        "",
        "这个实验不是静态比较算法名，而是对 Qwen3 MoE 合并规则本身做一次内部参数优化：在真实 route mass、risk flag 和 safetensors delta probe 上搜索可解释的 expert cap law。它不替代 vLLM 下游评测；它的作用是把“为什么要这么合并”转成可检查、可物化的规则。",
        "",
        "## 结果",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 真实 expert groups：`{summary['expert_group_count']}`",
        f"- 搜索 cap laws：`{summary['searched_law_count']}`",
        f"- Pareto frontier laws：`{summary['pareto_frontier_count']}`",
        f"- 选中的 no `>0.75` law：`{summary['selected_no_gt075_law']}`",
        f"- 选中的 no `>0.65` law：`{summary['selected_no_gt065_law']}`",
        f"- 当前 trust-region retention：`{fmt(summary['current_trust_retention'])}`，仍有 `{summary['current_trust_routed_gt_065_groups']}` 个 group 高于 `0.65`",
        f"- 简单 uniform `0.65` cap retention：`{fmt(summary['uniform_065_retention'])}`，高于 `0.65` 的 group 为 `{summary['uniform_065_routed_gt_065_groups']}`",
        "",
        "## 为什么要做这个 Probe",
        "",
        "对 routed expert group `g`，当前 same-shape 合并规则可以近似写成：",
        "",
        "```text",
        "theta_out[g] = theta_base[g] + s_g * w_g * (theta_coder[g] - theta_base[g])",
        "s_g = min(1, cap_g / relative_delta_g)",
        "cap_g = base_cap - penalties(route_load, mixed_source, router_fragility, low_evidence, category_mismatch)",
        "```",
        "",
        "`w_g` 是 route/source 规则给 Coder delta 的原始权重，`s_g` 是 delta trust region 给它加的缩放。搜索目标不是下游分数，而是一个安全/效用代理：尽量压低 routed experts 里的高 relative-delta tail，同时尽量保留 route-mass-weighted 的 Coder contribution。",
        "",
        "主要发现是：当前手写 risk penalties 自身不是 delta-threshold efficient。简单 uniform `0.65` cap 可以去掉剩余高 tail，而且 route-weighted nonbase mass retention 还略高于当前 trust-region 规则。这并不证明它下游一定更好；它证明下一轮 vLLM gate 应该把“简单 tail cap”和“更复杂的风险标记 law”放到同一组 source/candidate eval 里判定。",
        "",
        "## 选中的规则",
        "",
        "| role | law | >0.75 groups | >0.65 groups | retention | norm ratio | max rel-delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| `{row['selection_role']}` | `{row['law_id']}` | "
            f"{int(row['routed_gt_075_groups'])} | {int(row['routed_gt_065_groups'])} | "
            f"{fmt(float(row['nonbase_mass_retention']))} | "
            f"{fmt(float(row['delta_norm_proxy_ratio_vs_uncapped']))} | "
            f"{fmt(float(row['max_predicted_relative_delta']))} |"
        )
    lines.extend(
        [
            "",
            "## Risk-Flag Ablation",
            "",
            "这里检查每个风险标记单独加 `0.05` penalty 时，是否真的减少高 delta tail。结果显示它们主要只是降低 norm/retention，不能单独减少 `>0.75` 或 `>0.65` group，因此不能只凭这些 flag 就说复杂 law 更优。",
            "",
            "| flag | groups | extra scaled | >0.75 reduction | >0.65 reduction | retention loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in ablation.iterrows():
        lines.append(
            f"| `{row['flag']}` | {int(row['flagged_groups'])} | {int(row['additional_scaled_groups'])} | "
            f"{int(row['routed_gt_075_reduction'])} | {int(row['routed_gt_065_reduction'])} | "
            f"{fmt(float(row['nonbase_mass_retention_loss']))} |"
        )
    lines.extend(
        [
            "",
            "## Pareto Frontier 样例",
            "",
            "| law | >0.75 | >0.65 | retention | norm ratio | internal risk |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in frontier.head(12).iterrows():
        lines.append(
            f"| `{row['law_id']}` | {int(row['routed_gt_075_groups'])} | "
            f"{int(row['routed_gt_065_groups'])} | {fmt(float(row['nonbase_mass_retention']))} | "
            f"{fmt(float(row['delta_norm_proxy_ratio_vs_uncapped']))} | "
            f"{fmt(float(row['internal_risk_score']))} |"
        )
    lines.extend(
        [
            "",
            "## 文献连接",
            "",
            "- HARC (2026) 指出 MoE routing breakdown 会来自 softmax/top-k router 扰动；这里先冻结 router，把 expert delta 单独优化： https://arxiv.org/abs/2606.03391",
            "- Expert Merging (2025) 用 unlabeled calibration behavior 学 layer/chunk-wise coefficients；这里对应到 layer/expert 粒度的参数审计和 cap-law 搜索： https://arxiv.org/abs/2509.25712",
            "- 近期 LLM model-merging 系统研究报告很多通用 merge 算法在 LLM 上会失败，所以这里保留 endpoint fallback 和 vLLM gate： https://arxiv.org/abs/2511.21437",
            "- Sub-MoE 用 expert-output similarity/subspace 做 expert 合并/压缩，这支持把 expert identity/subspace probe 和 router probe 分开处理： https://arxiv.org/abs/2506.23266",
            "",
            "## 输出",
            "",
        ]
    )
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search interpretable Qwen3 MoE expert trust-region cap laws.")
    parser.add_argument(
        "--expert-rules",
        type=Path,
        default=Path("results/qwen3_moe_trust_region_candidate/trust_region_source_weights_by_expert.csv"),
    )
    parser.add_argument(
        "--writer-context-command",
        type=Path,
        default=Path("results/qwen3_moe_expert_only_trust_region_candidate/writer_command.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_trust_region_cap_search"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(repo_path(args.expert_rules))
    df = add_boolean_flags(df)

    search_rows = []
    scale_lookup: dict[str, tuple[pd.Series, pd.Series]] = {}
    for idx, law in enumerate(cap_law_grid()):
        metrics, target_cap, scale = simulate_law(df, law, idx)
        search_rows.append(metrics)
        scale_lookup[str(metrics["law_id"])] = (target_cap, scale)
    search = pd.DataFrame(search_rows).sort_values(
        ["routed_gt_075_groups", "routed_gt_065_groups", "nonbase_mass_retention"],
        ascending=[True, True, False],
    )
    references = pd.DataFrame(current_metrics(df))
    frontier = pareto_frontier(search)
    selected = select_laws(search, references)
    ablation = risk_flag_ablation(df)
    selected_artifacts = build_selected_rules(df, selected, scale_lookup, output_dir, repo_path(args.writer_context_command))

    search_path = output_dir / "cap_law_search.csv"
    references_path = output_dir / "reference_laws.csv"
    frontier_path = output_dir / "pareto_frontier.csv"
    selected_path = output_dir / "selected_cap_laws.csv"
    ablation_path = output_dir / "risk_flag_ablation.csv"
    artifacts_path = output_dir / "selected_rule_artifacts.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    search.to_csv(search_path, index=False)
    references.to_csv(references_path, index=False)
    frontier.to_csv(frontier_path, index=False)
    selected.to_csv(selected_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    artifacts_path.write_text(json.dumps(json_safe(selected_artifacts), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    no_gt075 = selected[selected["selection_role"] == "searched_no_gt075_max_retention"]
    no_gt065 = selected[selected["selection_role"] == "searched_no_gt065_max_retention"]
    current = references[references["law_id"] == "current_trust_region_cap_law"].iloc[0]
    uniform_065 = references[references["law_id"] == "uniform_065_tail_cap"].iloc[0]
    summary = {
        "schema_version": 1,
        "status": "cap_law_search_ready",
        "expert_group_count": int(len(df)),
        "searched_law_count": int(len(search)),
        "pareto_frontier_count": int(len(frontier)),
        "selected_no_gt075_law": None if no_gt075.empty else str(no_gt075.iloc[0]["law_id"]),
        "selected_no_gt065_law": None if no_gt065.empty else str(no_gt065.iloc[0]["law_id"]),
        "current_trust_retention": float(current["nonbase_mass_retention"]),
        "current_trust_routed_gt_075_groups": int(current["routed_gt_075_groups"]),
        "current_trust_routed_gt_065_groups": int(current["routed_gt_065_groups"]),
        "uniform_065_retention": float(uniform_065["nonbase_mass_retention"]),
        "uniform_065_routed_gt_075_groups": int(uniform_065["routed_gt_075_groups"]),
        "uniform_065_routed_gt_065_groups": int(uniform_065["routed_gt_065_groups"]),
        "uniform_065_retention_delta_vs_current_trust": float(
            uniform_065["nonbase_mass_retention"] - current["nonbase_mass_retention"]
        ),
        "current_extra_risk_penalties_delta_threshold_efficient": bool(
            current["nonbase_mass_retention"] >= uniform_065["nonbase_mass_retention"]
            or current["routed_gt_065_groups"] < uniform_065["routed_gt_065_groups"]
        ),
        "selected_rule_artifact_count": len(selected_artifacts),
        "interpretation": (
            "Use this search as an internal risk/utility frontier. It can suggest the next cap law, "
            "but vLLM source/candidate evaluation remains the decision gate."
        ),
        "outputs": {
            "report": rel(report_path),
            "summary": rel(summary_path),
            "cap_law_search": rel(search_path),
            "reference_laws": rel(references_path),
            "pareto_frontier": rel(frontier_path),
            "selected_cap_laws": rel(selected_path),
            "risk_flag_ablation": rel(ablation_path),
            "selected_rule_artifacts": rel(artifacts_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, selected, frontier, ablation), encoding="utf-8")
    print(f"Wrote Qwen3 MoE trust-region cap-law search to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
