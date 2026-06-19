#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_NAME


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GroupRule:
    group: str
    regex: str
    instruct_weight: float
    coder_weight: float
    reason: str


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def shell_quote(value: str | Path) -> str:
    raw = str(value)
    if not raw:
        return "''"
    if all(ch.isalnum() or ch in "/._-:=," for ch in raw):
        return raw
    return "'" + raw.replace("'", "'\"'\"'") + "'"


def resolve_safetensors_file(path: str | Path) -> str:
    model_path = Path(path)
    if model_path.is_file() and model_path.name.endswith(".safetensors"):
        return str(model_path)
    single = model_path / SAFE_WEIGHTS_NAME
    if single.exists():
        return str(single)
    snapshot_singles = sorted((model_path / "snapshots").glob(f"*/{SAFE_WEIGHTS_NAME}"))
    if snapshot_singles:
        return str(snapshot_singles[0])
    raise FileNotFoundError(f"No single-file safetensors checkpoint found under {model_path}")


def tensor_group(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "embedding_anchor"
    if name == "model.norm.weight" or "layernorm" in name:
        return "norm_anchor"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "other"


def projection_kind(name: str) -> str:
    if ".self_attn.q_proj" in name:
        return "attn_q"
    if ".self_attn.k_proj" in name:
        return "attn_k"
    if ".self_attn.v_proj" in name:
        return "attn_v"
    if ".self_attn.o_proj" in name:
        return "attn_o"
    if ".mlp.gate_proj" in name:
        return "mlp_gate"
    if ".mlp.up_proj" in name:
        return "mlp_up"
    if ".mlp.down_proj" in name:
        return "mlp_down"
    return tensor_group(name)


def layer_index(name: str) -> int | None:
    match = re.search(r"model\.layers\.(\d+)\.", name)
    if not match:
        return None
    return int(match.group(1))


def safe_ratio(num: float, den: float) -> float | None:
    if den <= 0.0:
        return None
    return num / den


def compute_tensor_stats(base_path: str, instruct_path: str, coder_path: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with (
        safe_open(base_path, framework="pt", device="cpu") as base,
        safe_open(instruct_path, framework="pt", device="cpu") as instruct,
        safe_open(coder_path, framework="pt", device="cpu") as coder,
    ):
        base_names = sorted(base.keys())
        missing = (set(base_names) - set(instruct.keys())) | (set(base_names) - set(coder.keys()))
        if missing:
            raise ValueError(f"Missing source tensors: {sorted(missing)[:5]}")
        for name in base_names:
            base_tensor = base.get_tensor(name)
            if not torch.is_floating_point(base_tensor):
                continue
            base_fp = base_tensor.to(torch.float32)
            instruct_delta = instruct.get_tensor(name).to(torch.float32) - base_fp
            coder_delta = coder.get_tensor(name).to(torch.float32) - base_fp
            dot = float((instruct_delta * coder_delta).sum().item())
            instruct_sq = float((instruct_delta * instruct_delta).sum().item())
            coder_sq = float((coder_delta * coder_delta).sum().item())
            denom = math.sqrt(instruct_sq) * math.sqrt(coder_sq)
            cosine = safe_ratio(dot, denom)
            nonzero = (instruct_delta != 0) | (coder_delta != 0)
            nonzero_count = int(nonzero.sum().item())
            if nonzero_count:
                sign_conflict = float(((instruct_delta * coder_delta) < 0).sum().item() / nonzero_count)
            else:
                sign_conflict = None
            rows.append(
                {
                    "tensor": name,
                    "group": tensor_group(name),
                    "projection": projection_kind(name),
                    "layer": layer_index(name),
                    "numel": int(base_tensor.numel()),
                    "instruct_delta_l2": math.sqrt(instruct_sq),
                    "coder_delta_l2": math.sqrt(coder_sq),
                    "delta_l2_ratio_instruct_over_coder": safe_ratio(math.sqrt(instruct_sq), math.sqrt(coder_sq)),
                    "delta_dot": dot,
                    "delta_cosine": cosine,
                    "sign_conflict_rate": sign_conflict,
                }
            )
            del base_fp, instruct_delta, coder_delta, base_tensor
    return pd.DataFrame(rows)


def weighted_mean(df: pd.DataFrame, value: str, weight: str = "numel") -> float | None:
    valid = df[df[value].notna()]
    if valid.empty:
        return None
    weights = valid[weight].astype(float)
    if float(weights.sum()) == 0.0:
        return None
    return float((valid[value].astype(float) * weights).sum() / weights.sum())


def summarize_modules(tensor_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, group_df in tensor_stats.groupby("group", sort=True):
        instruct_l2 = math.sqrt(float((group_df["instruct_delta_l2"] ** 2).sum()))
        coder_l2 = math.sqrt(float((group_df["coder_delta_l2"] ** 2).sum()))
        dot = float(group_df["delta_dot"].sum())
        cosine = safe_ratio(dot, instruct_l2 * coder_l2)
        rows.append(
            {
                "group": group,
                "tensor_count": int(len(group_df)),
                "numel": int(group_df["numel"].sum()),
                "instruct_delta_l2": instruct_l2,
                "coder_delta_l2": coder_l2,
                "delta_l2_ratio_instruct_over_coder": safe_ratio(instruct_l2, coder_l2),
                "delta_cosine": cosine,
                "sign_conflict_rate": weighted_mean(group_df, "sign_conflict_rate"),
                "mean_tensor_cosine": weighted_mean(group_df, "delta_cosine"),
            }
        )
    return pd.DataFrame(rows).sort_values("numel", ascending=False)


def build_rules(module_summary: pd.DataFrame, tensor_stats: pd.DataFrame, variant: str) -> list[GroupRule]:
    # These are explicit engineering hypotheses derived from the probe.
    # module_guarded is the aggressive variant; norm_only isolates whether the highest-conflict norm anchors are the real issue.
    mlp = module_summary[module_summary["group"] == "mlp"]
    mlp_conflict = None if mlp.empty else float(mlp.iloc[0]["sign_conflict_rate"])
    mlp_coder_weight = 0.75 if mlp_conflict is None or mlp_conflict >= 0.25 else 1.0
    if variant == "norm_only":
        return [
            GroupRule(
                group="norm_anchor",
                regex=r"(model\.norm\.weight|model\.layers\.[0-9]+\.(input_layernorm|post_attention_layernorm)\.weight)",
                instruct_weight=0.0,
                coder_weight=0.0,
                reason="freeze only the highest-conflict normalization anchors",
            ),
            GroupRule(
                group="embedding_anchor",
                regex=r"model\.embed_tokens\.weight",
                instruct_weight=0.25,
                coder_weight=1.0,
                reason="keep the global bridge on token embeddings for lexical/task adaptation",
            ),
            GroupRule(
                group="attention",
                regex=r"model\.layers\.[0-9]+\.self_attn\..*",
                instruct_weight=0.25,
                coder_weight=1.0,
                reason="keep the NLL-selected bridge on attention tensors",
            ),
            GroupRule(
                group="mlp",
                regex=r"model\.layers\.[0-9]+\.mlp\..*",
                instruct_weight=0.25,
                coder_weight=1.0,
                reason="keep the global bridge on MLP tensors to isolate norm freezing",
            ),
        ]
    if variant == "selective_norm":
        high_conflict_norms = tensor_stats[
            (tensor_stats["group"] == "norm_anchor") & (tensor_stats["sign_conflict_rate"] >= 0.75)
        ].sort_values("sign_conflict_rate", ascending=False)
        rules = [
            GroupRule(
                group=f"selective_norm:{row['tensor']}",
                regex=re.escape(str(row["tensor"])),
                instruct_weight=0.0,
                coder_weight=0.0,
                reason="freeze only extreme-conflict norm tensor selected by sign-conflict probe",
            )
            for _, row in high_conflict_norms.iterrows()
        ]
        rules.extend(
            [
                GroupRule(
                    group="norm_anchor",
                    regex=r"(model\.norm\.weight|model\.layers\.[0-9]+\.(input_layernorm|post_attention_layernorm)\.weight)",
                    instruct_weight=0.25,
                    coder_weight=1.0,
                    reason="keep the global bridge on non-extreme norm anchors",
                ),
                GroupRule(
                    group="embedding_anchor",
                    regex=r"model\.embed_tokens\.weight",
                    instruct_weight=0.25,
                    coder_weight=1.0,
                    reason="keep the global bridge on token embeddings for lexical/task adaptation",
                ),
                GroupRule(
                    group="attention",
                    regex=r"model\.layers\.[0-9]+\.self_attn\..*",
                    instruct_weight=0.25,
                    coder_weight=1.0,
                    reason="keep the NLL-selected bridge on attention tensors",
                ),
                GroupRule(
                    group="mlp",
                    regex=r"model\.layers\.[0-9]+\.mlp\..*",
                    instruct_weight=0.25,
                    coder_weight=1.0,
                    reason="keep the global bridge on MLP tensors while isolating only extreme norm conflict",
                ),
            ]
        )
        return rules
    return [
        GroupRule(
            group="embedding_anchor",
            regex=r"model\.embed_tokens\.weight",
            instruct_weight=0.0,
            coder_weight=0.0,
            reason="freeze token embedding distribution anchor to base",
        ),
        GroupRule(
            group="norm_anchor",
            regex=r"(model\.norm\.weight|model\.layers\.[0-9]+\.(input_layernorm|post_attention_layernorm)\.weight)",
            instruct_weight=0.0,
            coder_weight=0.0,
            reason="freeze normalization anchors to avoid global activation scale drift",
        ),
        GroupRule(
            group="attention",
            regex=r"model\.layers\.[0-9]+\.self_attn\..*",
            instruct_weight=0.25,
            coder_weight=1.0,
            reason="keep the NLL-selected bridge on attention tensors",
        ),
        GroupRule(
            group="mlp",
            regex=r"model\.layers\.[0-9]+\.mlp\..*",
            instruct_weight=0.25,
            coder_weight=mlp_coder_weight,
            reason="dampen MLP coder delta when module conflict is nontrivial",
        ),
    ]


def write_rules(path: Path, rules: list[GroupRule]) -> None:
    lines = [
        "# PATTERN::instruct=WEIGHT,coder=WEIGHT",
        "# First matching rule wins in write_same_shape_average_checkpoint.py.",
    ]
    for rule in rules:
        lines.append(f"# group={rule.group}; reason={rule.reason}")
        lines.append(f"{rule.regex}::instruct={rule.instruct_weight},coder={rule.coder_weight}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_writer_command(qwen_summary: dict[str, Any], output_checkpoint_dir: str, rule_file: str) -> str:
    sources = qwen_summary["experts"]
    parts = [
        "python",
        "scripts/write_same_shape_average_checkpoint.py",
        "--base",
        qwen_summary["base"],
        "--source",
        f"instruct={sources['instruct']}",
        "--source",
        f"coder={sources['coder']}",
        "--source-weight",
        "instruct=0.0",
        "--source-weight",
        "coder=0.0",
        "--tensor-rule-file",
        rule_file,
        "--output-dir",
        output_checkpoint_dir,
    ]
    return " ".join(shell_quote(part) for part in parts)


def build_vllm_commands(output_checkpoint_dir: str, eval_dir: str, served_model: str) -> dict[str, str]:
    return {
        "serve": (
            "CUDA_VISIBLE_DEVICES=1 /srv/home/bohanlyu/miniconda3/envs/cogdoc/bin/vllm "
            f"serve {shell_quote(output_checkpoint_dir)} --served-model-name {served_model} "
            "--host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 1"
        ),
        "eval": (
            "python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 "
            f"--models {served_model} --tasks gsm8k,mmlu,safety,humaneval_compile "
            "--example-source datasets --max-examples 64 "
            f"--output-dir {shell_quote(eval_dir)}"
        ),
    }


def primary_metric_for_task(task: str) -> str:
    return {
        "gsm8k": "strict_exact",
        "mmlu": "accuracy",
        "safety": "policy_accuracy",
        "humaneval_compile": "compile_rate",
    }[task]


def optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def load_eval_summary(eval_dir: str) -> dict[str, Any] | None:
    eval_path = repo_path(eval_dir)
    model_summary_path = eval_path / "model_summary.csv"
    metrics_path = eval_path / "metrics.csv"
    if not model_summary_path.exists() or not metrics_path.exists():
        return None
    model_summary = pd.read_csv(model_summary_path)
    metrics = pd.read_csv(metrics_path)
    first = model_summary.iloc[0]
    uniform_metrics = pd.read_csv(
        repo_path("results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average/metrics.csv")
    )
    bridge_metrics = pd.read_csv(
        repo_path("results/vllm_checkpoint_eval/qwen_0_5b_probe_guided_bridge_a025_b100/metrics.csv")
    )
    bridge_summary = pd.read_csv(
        repo_path("results/vllm_checkpoint_eval/qwen_0_5b_probe_guided_bridge_a025_b100/model_summary.csv")
    ).iloc[0]
    source_scores = pd.read_csv(repo_path("results/vllm_source_merge_comparison/model_scores.csv"))
    best_source = source_scores[source_scores["role"] == "source"].sort_values(
        ["avg_primary_score", "worst_primary_score"], ascending=False
    ).iloc[0]
    task_rows = []
    for _, row in metrics.iterrows():
        task = str(row["task"])
        metric = primary_metric_for_task(task)
        uniform_row = uniform_metrics[uniform_metrics["task"] == task].iloc[0]
        bridge_row = bridge_metrics[bridge_metrics["task"] == task].iloc[0]
        score = float(row[metric])
        task_rows.append(
            {
                "task": task,
                "primary_metric": metric,
                "primary_score": score,
                "uniform_primary_score": float(uniform_row[metric]),
                "global_bridge_primary_score": float(bridge_row[metric]),
                "delta_vs_uniform": score - float(uniform_row[metric]),
                "delta_vs_global_bridge": score - float(bridge_row[metric]),
                "safe_non_refusal_rate": optional_float(row.get("safe_non_refusal_rate")),
                "unsafe_refusal_rate": optional_float(row.get("unsafe_refusal_rate")),
            }
        )
    return {
        "status": "complete",
        "eval_dir": eval_dir,
        "avg_primary_score": float(first["avg_primary_score"]),
        "worst_primary_score": float(first["worst_primary_score"]),
        "global_bridge_avg_primary_score": float(bridge_summary["avg_primary_score"]),
        "delta_vs_global_bridge_avg_primary": float(first["avg_primary_score"]) - float(bridge_summary["avg_primary_score"]),
        "best_source_model": str(best_source["model_key"]),
        "best_source_display_name": str(best_source["display_name"]),
        "best_source_avg_primary_score": float(best_source["avg_primary_score"]),
        "delta_vs_best_source_avg_primary": float(first["avg_primary_score"]) - float(best_source["avg_primary_score"]),
        "task_metrics": task_rows,
        "report": rel(eval_path / "report.md"),
        "metrics": rel(eval_path / "metrics.csv"),
        "model_summary": rel(eval_path / "model_summary.csv"),
    }


def write_report(
    output_dir: Path,
    module_summary: pd.DataFrame,
    rules: list[GroupRule],
    writer_command: str,
    vllm_commands: dict[str, str],
    eval_summary: dict[str, Any] | None,
    candidate_id: str,
    variant: str,
) -> None:
    title = (
        "Qwen Dense Norm-Guarded Bridge Candidate"
        if variant == "norm_only"
        else "Qwen Dense Selective-Norm Bridge Candidate"
        if variant == "selective_norm"
        else "Qwen Dense Module-Guarded Bridge Candidate"
    )
    variant_text = (
        "只冻结 norm/layernorm，其它模块保持 global bridge"
        if variant == "norm_only"
        else "只冻结 sign conflict >= 0.75 的极端 norm tensor，其它模块保持 global bridge"
        if variant == "selective_norm"
        else "冻结 embedding/norm 分布锚点，并对 MLP delta 做阻尼"
    )
    lines = [
        f"# {title}",
        "",
        "## 结论",
        "",
        (
            f"`{candidate_id}` 是 global bridge 后的模块级 ablation：保留 `alpha=0.25,beta=1.0` 的基本方向，"
            f"{variant_text}。"
        ),
        "",
        "## Module Probe",
        "",
        "| group | tensors | params | cosine | sign conflict | instruct/coder L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in module_summary.iterrows():
        lines.append(
            f"| {row['group']} | {int(row['tensor_count'])} | {int(row['numel'])} | "
            f"{fmt(row['delta_cosine'])} | {fmt(row['sign_conflict_rate'])} | "
            f"{fmt(row['delta_l2_ratio_instruct_over_coder'])} |"
        )
    lines.extend(["", "## Tensor Rules", "", "| group | instruct | coder | reason |", "| --- | ---: | ---: | --- |"])
    for rule in rules:
        lines.append(f"| {rule.group} | {fmt(rule.instruct_weight, 2)} | {fmt(rule.coder_weight, 2)} | {rule.reason} |")
    lines.extend(
        [
            "",
            "## Materialization",
            "",
            "```bash",
            writer_command,
            "```",
            "",
            "## vLLM Eval",
            "",
            "```bash",
            vllm_commands["serve"],
            "```",
            "",
            "```bash",
            vllm_commands["eval"],
            "```",
        ]
    )
    if eval_summary:
        lines.extend(
            [
                "",
                "## vLLM Eval Result",
                "",
                (
                    f"真实 endpoint eval 已完成：avg primary `{fmt(eval_summary['avg_primary_score'])}`，"
                    f"相对 global bridge `{fmt(eval_summary['delta_vs_global_bridge_avg_primary'])}`，"
                    f"相对 best source `{fmt(eval_summary['delta_vs_best_source_avg_primary'])}`。"
                ),
                "",
                "| task | score | global bridge | delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in eval_summary["task_metrics"]:
            lines.append(
                f"| {row['task']} | {fmt(row['primary_score'])} | "
                f"{fmt(row['global_bridge_primary_score'])} | {fmt(row['delta_vs_global_bridge'])} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Tensor stats: `{rel(output_dir / 'tensor_conflict.csv')}`",
            f"- Module summary: `{rel(output_dir / 'module_conflict.csv')}`",
            f"- Tensor rules: `{rel(output_dir / 'tensor_rules.txt')}`",
            f"- Summary: `{rel(output_dir / 'summary.json')}`",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-summary", default="results/qwen_multi_expert_merge/summary.json")
    parser.add_argument("--output-dir", default="results/qwen_dense_module_guarded_candidate")
    parser.add_argument(
        "--variant",
        choices=["module_guarded", "norm_only", "selective_norm"],
        default="module_guarded",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen_summary = json.loads(repo_path(args.qwen_summary).read_text(encoding="utf-8"))
    base = resolve_safetensors_file(qwen_summary["base"])
    instruct = resolve_safetensors_file(qwen_summary["experts"]["instruct"])
    coder = resolve_safetensors_file(qwen_summary["experts"]["coder"])
    candidate_id = (
        "qwen_0_5b_norm_guarded_bridge"
        if args.variant == "norm_only"
        else "qwen_0_5b_selective_norm_guarded_bridge"
        if args.variant == "selective_norm"
        else "qwen_0_5b_module_guarded_bridge"
    )
    checkpoint_dir = f"results/checkpoints/{candidate_id}"
    eval_dir = f"results/vllm_checkpoint_eval/{candidate_id}"
    served_model = f"candidate_{candidate_id}"

    tensor_stats = compute_tensor_stats(base, instruct, coder)
    module_summary = summarize_modules(tensor_stats)
    rules = build_rules(module_summary, tensor_stats, args.variant)
    rule_file = output_dir / "tensor_rules.txt"
    write_rules(rule_file, rules)
    writer_command = build_writer_command(qwen_summary, checkpoint_dir, rel(rule_file))
    vllm_commands = build_vllm_commands(checkpoint_dir, eval_dir, served_model)
    eval_summary = load_eval_summary(eval_dir)

    tensor_stats.to_csv(output_dir / "tensor_conflict.csv", index=False)
    module_summary.to_csv(output_dir / "module_conflict.csv", index=False)
    (output_dir / "writer_command.txt").write_text(writer_command + "\n", encoding="utf-8")
    (output_dir / "vllm_commands.json").write_text(json.dumps(vllm_commands, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "status": "evaluated_complete" if eval_summary else "candidate_selected_waiting_for_materialization",
        "candidate_id": candidate_id,
        "variant": args.variant,
        "checkpoint_output_dir": checkpoint_dir,
        "vllm_eval_output_dir": eval_dir,
        "served_model": served_model,
        "module_summary": [row for row in module_summary.to_dict(orient="records")],
        "rules": [
            {
                "group": rule.group,
                "regex": rule.regex,
                "instruct_weight": rule.instruct_weight,
                "coder_weight": rule.coder_weight,
                "reason": rule.reason,
            }
            for rule in rules
        ],
        "writer_command": writer_command,
        "vllm_commands": vllm_commands,
        "vllm_eval": eval_summary,
        "hypothesis": (
            "Freeze only normalization anchors while keeping global bridge elsewhere."
            if args.variant == "norm_only"
            else "Freeze only extreme-conflict normalization tensors selected by the sign-conflict probe."
            if args.variant == "selective_norm"
            else "Freeze distribution anchors and damp high-conflict MLP deltas while keeping the NLL-selected attention bridge."
        ),
        "artifacts": {
            "report": rel(output_dir / "report.md"),
            "tensor_conflict": rel(output_dir / "tensor_conflict.csv"),
            "module_conflict": rel(output_dir / "module_conflict.csv"),
            "tensor_rules": rel(rule_file),
            "writer_command": rel(output_dir / "writer_command.txt"),
            "vllm_commands": rel(output_dir / "vllm_commands.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir, module_summary, rules, writer_command, vllm_commands, eval_summary, candidate_id, args.variant)
    print(f"Wrote {rel(output_dir / 'summary.json')}")
    print(f"Selected {candidate_id}")


if __name__ == "__main__":
    main()
