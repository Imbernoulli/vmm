#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_CAPS = {"humaneval_compile": 164}


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
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    value = clean_value(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_tasks(value: Any) -> list[str]:
    text = str(clean_value(value) or "")
    return [part.strip() for part in text.split(",") if part.strip()]


def round_up(value: int, unit: int) -> int:
    if unit <= 1:
        return int(value)
    return int(math.ceil(value / unit) * unit)


def wilson_interval(score: float, n: int, z: float) -> tuple[float, float]:
    p = min(1.0, max(0.0, float(score)))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def wilson_half_width(n: int, z: float, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    lower, upper = wilson_interval(p, n, z)
    return max(p - lower, upper - p)


def required_examples_for_wilson(target_half_width: float, z: float) -> int:
    if target_half_width <= 0:
        raise ValueError("target_half_width must be positive")
    lo, hi = 1, 2
    while wilson_half_width(hi, z) > target_half_width:
        lo, hi = hi, hi * 2
        if hi > 10_000_000:
            raise ValueError("Wilson sample-size search exceeded limit")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if wilson_half_width(mid, z) <= target_half_width:
            hi = mid
        else:
            lo = mid
    return hi


def exact_paired_source_advantage_pvalue(candidate_only: int, source_only: int) -> float:
    if source_only <= candidate_only:
        return 1.0
    discordant = candidate_only + source_only
    if discordant <= 0:
        return 1.0
    if discordant > 1024:
        mean = discordant / 2.0
        variance = discordant / 4.0
        z = (candidate_only + 0.5 - mean) / math.sqrt(variance)
        return 0.5 * math.erfc(-z / math.sqrt(2.0))
    return sum(math.comb(discordant, i) for i in range(candidate_only + 1)) / (2.0**discordant)


def required_discordant_for_paired_sign_test(
    *,
    target_net_loss_rate: float,
    assumed_discordance_rate: float,
    alpha: float,
    max_discordant: int,
) -> tuple[int, int, int, float, float]:
    if target_net_loss_rate <= 0:
        raise ValueError("target_net_loss_rate must be positive")
    if assumed_discordance_rate <= 0:
        raise ValueError("assumed_discordance_rate must be positive")
    discordant_advantage_rate = min(0.999, target_net_loss_rate / assumed_discordance_rate)
    if discordant_advantage_rate <= 0:
        raise ValueError("discordant advantage must be positive")
    for discordant in range(1, max_discordant + 1):
        candidate_only = int(math.floor((1.0 - discordant_advantage_rate) * discordant / 2.0))
        source_only = discordant - candidate_only
        net_loss_rate = (source_only - candidate_only) * assumed_discordance_rate / discordant
        pvalue = exact_paired_source_advantage_pvalue(candidate_only, source_only)
        if net_loss_rate >= target_net_loss_rate and pvalue <= alpha:
            return discordant, candidate_only, source_only, net_loss_rate, pvalue
    raise ValueError("paired sign-test sample-size search exceeded limit")


def parse_task_caps(raw: list[str]) -> dict[str, int]:
    caps = dict(DEFAULT_TASK_CAPS)
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Task cap must look like task=n, got {item!r}")
        task, value = item.split("=", 1)
        caps[task.strip()] = int(value)
    return caps


def replace_cli_arg(command: str, option: str, value: int) -> str:
    if not command:
        return command
    tokens = shlex.split(command)
    updated = False
    for idx, token in enumerate(tokens):
        if token == option and idx + 1 < len(tokens):
            tokens[idx + 1] = str(value)
            updated = True
            break
        prefix = option + "="
        if token.startswith(prefix):
            tokens[idx] = f"{option}={value}"
            updated = True
            break
    if not updated:
        tokens.extend([option, str(value)])
    return " ".join(shlex.quote(token) for token in tokens)


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def observed_examples_by_task(eval_output_dir: str | Path) -> dict[str, int]:
    metrics = read_csv_if_exists(repo_path(eval_output_dir) / "metrics.csv")
    if metrics.empty or "task" not in metrics or "examples" not in metrics:
        return {}
    rows: dict[str, int] = {}
    for _, row in metrics.iterrows():
        task = str(row.get("task", "")).strip()
        examples = clean_value(row.get("examples"))
        if task and examples is not None:
            rows[task] = int(examples)
    return rows


def build_task_budget(
    gate: pd.DataFrame,
    *,
    target_half_width: float,
    confidence_z: float,
    target_paired_net_loss_rate: float,
    paired_alpha: float,
    assumed_paired_discordance_rate: float,
    rounding_unit: int,
    task_caps: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_tasks = sorted({task for value in gate["tasks"].tolist() for task in parse_tasks(value)})
    current_gate_examples = int(max(int(row.get("max_examples") or 0) for _, row in gate.iterrows()))
    wilson_required = required_examples_for_wilson(target_half_width, confidence_z)
    wilson_rounded = round_up(wilson_required, rounding_unit)
    paired_discordant, paired_candidate_only, paired_source_only, paired_net_loss_rate, paired_pvalue = (
        required_discordant_for_paired_sign_test(
            target_net_loss_rate=target_paired_net_loss_rate,
            assumed_discordance_rate=assumed_paired_discordance_rate,
            alpha=paired_alpha,
            max_discordant=10_000,
        )
    )
    paired_required_examples = int(math.ceil(paired_discordant / assumed_paired_discordance_rate))
    paired_rounded = round_up(paired_required_examples, rounding_unit)
    command_max_examples = max(current_gate_examples, wilson_rounded, paired_rounded)

    observed: dict[str, list[int]] = {task: [] for task in all_tasks}
    for _, row in gate.iterrows():
        for task, examples in observed_examples_by_task(str(row.get("eval_output_dir", ""))).items():
            if task in observed:
                observed[task].append(examples)

    rows = []
    for task in all_tasks:
        cap = task_caps.get(task)
        achievable = min(command_max_examples, cap) if cap is not None else command_max_examples
        half_width = wilson_half_width(achievable, confidence_z)
        if cap is not None and cap < wilson_required:
            status = "target_not_met_dataset_cap"
        elif achievable >= wilson_required and achievable >= paired_required_examples:
            status = "target_met"
        else:
            status = "target_not_met"
        observed_counts = observed.get(task, [])
        rows.append(
            {
                "task": task,
                "current_gate_examples": current_gate_examples,
                "observed_min_examples": min(observed_counts) if observed_counts else 0,
                "observed_max_examples": max(observed_counts) if observed_counts else 0,
                "target_wilson_half_width": target_half_width,
                "confidence_z": confidence_z,
                "wilson_required_examples": wilson_required,
                "wilson_rounded_examples": wilson_rounded,
                "target_paired_net_loss_rate": target_paired_net_loss_rate,
                "assumed_paired_discordance_rate": assumed_paired_discordance_rate,
                "paired_alpha": paired_alpha,
                "paired_required_discordant": paired_discordant,
                "paired_required_examples": paired_required_examples,
                "paired_rounded_examples": paired_rounded,
                "paired_candidate_only_at_threshold": paired_candidate_only,
                "paired_source_only_at_threshold": paired_source_only,
                "paired_observed_net_loss_rate": paired_net_loss_rate,
                "paired_threshold_pvalue": paired_pvalue,
                "dataset_cap": cap,
                "recommended_command_max_examples": command_max_examples,
                "achievable_examples": achievable,
                "achievable_wilson_half_width": half_width,
                "budget_status": status,
            }
        )
    metadata = {
        "current_gate_examples": current_gate_examples,
        "wilson_required_examples": wilson_required,
        "wilson_rounded_examples": wilson_rounded,
        "paired_required_discordant": paired_discordant,
        "paired_required_examples": paired_required_examples,
        "paired_rounded_examples": paired_rounded,
        "recommended_command_max_examples": command_max_examples,
    }
    return pd.DataFrame(rows), metadata


def build_method_budget(gate: pd.DataFrame, task_budget_meta: dict[str, Any]) -> pd.DataFrame:
    recommended = int(task_budget_meta["recommended_command_max_examples"])
    rows = []
    for _, row in gate.iterrows():
        tasks = parse_tasks(row.get("tasks"))
        current = int(row.get("max_examples") or 0)
        eval_command = str(row.get("eval_command") or "")
        recommended_command = replace_cli_arg(eval_command, "--max-examples", recommended)
        current_prompt_budget = current * len(tasks)
        recommended_prompt_budget = recommended * len(tasks)
        rows.append(
            {
                "gate_order": int(row.get("gate_order") or 0),
                "method": row.get("method"),
                "role": row.get("role"),
                "serve_status": row.get("serve_status"),
                "eval_status": row.get("eval_status"),
                "served_model_id": row.get("served_model_id"),
                "base_url": row.get("base_url"),
                "tasks": ",".join(tasks),
                "task_count": len(tasks),
                "current_max_examples": current,
                "recommended_max_examples": recommended,
                "additional_examples_per_task": max(0, recommended - current),
                "current_prompt_budget": current_prompt_budget,
                "recommended_prompt_budget": recommended_prompt_budget,
                "additional_prompt_budget": max(0, recommended_prompt_budget - current_prompt_budget),
                "eval_output_dir": row.get("eval_output_dir"),
                "serve_command": row.get("serve_command"),
                "eval_command_current": eval_command,
                "eval_command_recommended": recommended_command,
                "audit_min_examples_required_by_gate": current,
                "budget_reason": (
                    "raise examples until Wilson score intervals and paired prediction sign-test "
                    "can detect small source-vs-candidate regressions; original gate remains the "
                    "minimum audit floor"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_mechanism_budget(
    tests: pd.DataFrame,
    method_budget: pd.DataFrame,
    *,
    recommended_max_examples: int,
) -> pd.DataFrame:
    if tests.empty:
        return pd.DataFrame()
    task_count_by_method = {
        str(row["method"]): int(row.get("task_count") or 0) for _, row in method_budget.iterrows()
    }
    rows = []
    for _, row in tests.iterrows():
        methods = []
        for column in ("from_method", "to_method"):
            method = str(row.get(column, "")).strip()
            if method and method not in methods:
                methods.append(method)
        prompt_budget = sum(task_count_by_method.get(method, 0) * recommended_max_examples for method in methods)
        rows.append(
            {
                "test": row.get("test"),
                "from_method": row.get("from_method"),
                "to_method": row.get("to_method"),
                "current_status": row.get("current_status"),
                "mechanism_question": row.get("mechanism_question"),
                "why_it_matters": row.get("why_it_matters"),
                "required_methods": ",".join(methods),
                "recommended_max_examples_per_task": recommended_max_examples,
                "recommended_prompt_budget": prompt_budget,
            }
        )
    return pd.DataFrame(rows)


def build_shell_script(method_budget: pd.DataFrame, output_dir: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run from the repository root on a GPU host with vLLM installed.",
        f"# Usage: {rel(output_dir / 'run_eval_budget.sh')} [all|method_name]",
        "# This budgeted runner keeps the same endpoints as the mechanism gate but raises --max-examples.",
        "",
        "requested=\"${1:-all}\"",
        f"mkdir -p {shell_quote(rel(output_dir / 'logs'))}",
        "",
        "run_one() {",
        "  local method=\"$1\"",
        "  local base_url=\"$2\"",
        "  local serve_cmd=\"$3\"",
        "  local eval_cmd=\"$4\"",
        "  if [[ \"$requested\" != \"all\" && \"$requested\" != \"$method\" ]]; then",
        "    return 0",
        "  fi",
        f"  local log_path=\"{rel(output_dir / 'logs')}/${{method}}.serve.log\"",
        "  echo \"[serve] ${method}\"",
        "  bash -lc \"$serve_cmd\" >\"$log_path\" 2>&1 &",
        "  local server_pid=$!",
        "  cleanup_server() {",
        "    if kill -0 \"$server_pid\" >/dev/null 2>&1; then",
        "      kill \"$server_pid\" >/dev/null 2>&1 || true",
        "      wait \"$server_pid\" >/dev/null 2>&1 || true",
        "    fi",
        "  }",
        "  trap cleanup_server RETURN",
        "  local ready=0",
        "  for _ in $(seq 1 \"${VLLM_WAIT_ATTEMPTS:-240}\"); do",
        "    if curl -sf \"${base_url}/models\" >/dev/null; then",
        "      ready=1",
        "      break",
        "    fi",
        "    sleep \"${VLLM_WAIT_SECONDS:-5}\"",
        "  done",
        "  if [[ \"$ready\" != \"1\" ]]; then",
        "    echo \"vLLM did not become ready for ${method}. See ${log_path}\" >&2",
        "    return 1",
        "  fi",
        "  echo \"[eval] ${method}\"",
        "  bash -lc \"$eval_cmd\"",
        "}",
        "",
    ]
    for _, row in method_budget.iterrows():
        if str(row.get("serve_status")) != "ready_to_host":
            lines.append(f"# Skipping {row['method']}: serve_status={row.get('serve_status')}")
            continue
        lines.append(
            "run_one "
            f"{shell_quote(str(row['method']))} "
            f"{shell_quote(str(row['base_url']))} "
            f"{shell_quote(str(row['serve_command']))} "
            f"{shell_quote(str(row['eval_command_recommended']))}"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    value = clean_value(value)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_report(
    summary: dict[str, Any],
    task_budget: pd.DataFrame,
    method_budget: pd.DataFrame,
    mechanism_budget: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen3 MoE vLLM Eval Budget Plan",
        "",
        "这份计划解决的是评测强度问题：现在 Qwen3 MoE gate 的 `64` examples 只适合 smoke，不足以支撑 final selector 的 Wilson confidence gate 和 paired prediction gate。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Methods to evaluate: `{summary['method_count']}`",
        f"- Current gate max examples: `{summary['current_gate_examples']}`",
        f"- Recommended command max examples: `{summary['recommended_max_examples']}`",
        f"- Total current prompt budget: `{summary['total_current_prompt_budget']}`",
        f"- Total recommended prompt budget: `{summary['total_recommended_prompt_budget']}`",
        f"- Additional prompt budget: `{summary['total_additional_prompt_budget']}`",
        "",
        "## Why This Budget",
        "",
        (
            "Wilson gate: for a binary task score near the worst case `p=0.5`, choose `n` so the 95% Wilson half-width "
            f"is at most `{summary['target_wilson_half_width']}`. This gives `{summary['wilson_required_examples']}` raw examples, rounded to "
            f"`{summary['wilson_rounded_examples']}` for batch-friendly execution."
        ),
        "",
        (
            "Paired gate: final selection compares source and candidate predictions on the same examples. The planner asks for enough shared examples to make a "
            f"`{summary['target_paired_net_loss_rate']}` net source advantage significant at alpha `{summary['paired_alpha']}`, assuming "
            f"`{summary['assumed_paired_discordance_rate']}` paired discordance. This requires `{summary['paired_required_discordant']}` discordant examples, "
            f"about `{summary['paired_required_examples']}` total shared examples before rounding."
        ),
        "",
        "因此这里推荐的不是“静态多跑一点”，而是让下游 eval 能真正支持 source dominance、task regression、score confidence 和 paired-prediction regression 这些机制判断。",
        "",
        "## Task Budget",
        "",
        "| task | current | Wilson n | paired n | recommended max | achievable | half-width | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in task_budget.iterrows():
        lines.append(
            f"| `{row['task']}` | {int(row['current_gate_examples'])} | "
            f"{int(row['wilson_required_examples'])} | {int(row['paired_required_examples'])} | "
            f"{int(row['recommended_command_max_examples'])} | {int(row['achievable_examples'])} | "
            f"{fmt(row['achievable_wilson_half_width'])} | `{row['budget_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Method Budget",
            "",
            "| order | method | role | current | recommended | extra prompts | eval status |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in method_budget.iterrows():
        lines.append(
            f"| {int(row['gate_order'])} | `{row['method']}` | `{row['role']}` | "
            f"{int(row['current_max_examples'])} | {int(row['recommended_max_examples'])} | "
            f"{int(row['additional_prompt_budget'])} | `{row['eval_status']}` |"
        )
    if not mechanism_budget.empty:
        lines.extend(
            [
                "",
                "## Mechanism Budget",
                "",
                "| test | comparison | prompt budget | question |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for _, row in mechanism_budget.iterrows():
            lines.append(
                f"| `{row['test']}` | `{row['from_method']}` -> `{row['to_method']}` | "
                f"{int(row['recommended_prompt_budget'])} | {row['mechanism_question']} |"
            )
    lines.extend(
        [
            "",
            "## How To Run",
            "",
            "在 GPU host 上从仓库根目录运行：",
            "",
            "```bash",
            f"{summary['outputs']['run_script']} all",
            "python scripts/audit_qwen3_moe_eval_bundle.py --output-dir results/qwen3_moe_eval_bundle_audit",
            "python scripts/refresh_qwen3_moe_post_eval.py",
            "```",
            "",
            "也可以只跑一个方法：",
            "",
            "```bash",
            f"{summary['outputs']['run_script']} qwen3_moe_tail_trimmed_expert_only_candidate",
            "```",
            "",
            "注意：原始 gate 里的 `max_examples=64` 仍是 audit floor；预算版 runner 会用更高的 `--max-examples` 覆盖 eval 命令。HumanEval 数据集上限低于推荐值时，selector 会使用实际落盘的样本数计算区间。",
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['task_budget']}`",
            f"- `{summary['outputs']['method_budget']}`",
            f"- `{summary['outputs']['mechanism_budget']}`",
            f"- `{summary['outputs']['run_script']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    task_budget: pd.DataFrame,
    method_budget: pd.DataFrame,
    mechanism_budget: pd.DataFrame,
    *,
    target_half_width: float,
    confidence_z: float,
    target_paired_net_loss_rate: float,
    paired_alpha: float,
    assumed_paired_discordance_rate: float,
    task_budget_meta: dict[str, Any],
) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_path = output_dir / "task_budget.csv"
    method_path = output_dir / "method_budget.csv"
    mechanism_path = output_dir / "mechanism_budget.csv"
    run_script_path = output_dir / "run_eval_budget.sh"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    task_budget.to_csv(task_path, index=False)
    method_budget.to_csv(method_path, index=False)
    mechanism_budget.to_csv(mechanism_path, index=False)
    run_script_path.write_text(build_shell_script(method_budget, output_dir), encoding="utf-8")
    os.chmod(run_script_path, 0o755)

    total_current = int(method_budget["current_prompt_budget"].sum())
    total_recommended = int(method_budget["recommended_prompt_budget"].sum())
    total_additional = int(method_budget["additional_prompt_budget"].sum())
    summary = {
        "schema_version": 1,
        "status": "ready_for_budgeted_remote_vllm_eval",
        "method_count": int(len(method_budget)),
        "source_count": int((method_budget["role"] == "source").sum()),
        "candidate_count": int((method_budget["role"] == "candidate").sum()),
        "current_gate_examples": int(task_budget_meta["current_gate_examples"]),
        "recommended_max_examples": int(task_budget_meta["recommended_command_max_examples"]),
        "target_wilson_half_width": target_half_width,
        "confidence_z": confidence_z,
        "wilson_required_examples": int(task_budget_meta["wilson_required_examples"]),
        "wilson_rounded_examples": int(task_budget_meta["wilson_rounded_examples"]),
        "target_paired_net_loss_rate": target_paired_net_loss_rate,
        "paired_alpha": paired_alpha,
        "assumed_paired_discordance_rate": assumed_paired_discordance_rate,
        "paired_required_discordant": int(task_budget_meta["paired_required_discordant"]),
        "paired_required_examples": int(task_budget_meta["paired_required_examples"]),
        "paired_rounded_examples": int(task_budget_meta["paired_rounded_examples"]),
        "total_current_prompt_budget": total_current,
        "total_recommended_prompt_budget": total_recommended,
        "total_additional_prompt_budget": total_additional,
        "outputs": {
            "task_budget": rel(task_path),
            "method_budget": rel(method_path),
            "mechanism_budget": rel(mechanism_path),
            "run_script": rel(run_script_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, task_budget, method_budget, mechanism_budget), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan statistically powered Qwen3 MoE vLLM eval budgets.")
    parser.add_argument("--gate-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_eval_budget_plan"))
    parser.add_argument("--target-wilson-half-width", type=float, default=0.05)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--target-paired-net-loss-rate", type=float, default=0.05)
    parser.add_argument("--paired-alpha", type=float, default=0.05)
    parser.add_argument("--assumed-paired-discordance-rate", type=float, default=0.25)
    parser.add_argument("--rounding-unit", type=int, default=64)
    parser.add_argument("--task-cap", action="append", default=[], help="Optional task cap, e.g. humaneval_compile=164")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate_dir = repo_path(args.gate_dir)
    gate = read_csv_if_exists(gate_dir / "eval_gate_plan.csv")
    if gate.empty:
        raise SystemExit(f"Missing eval gate plan: {gate_dir / 'eval_gate_plan.csv'}")
    tests = read_csv_if_exists(gate_dir / "mechanism_tests.csv")
    task_caps = parse_task_caps(args.task_cap)
    task_budget, task_budget_meta = build_task_budget(
        gate,
        target_half_width=args.target_wilson_half_width,
        confidence_z=args.confidence_z,
        target_paired_net_loss_rate=args.target_paired_net_loss_rate,
        paired_alpha=args.paired_alpha,
        assumed_paired_discordance_rate=args.assumed_paired_discordance_rate,
        rounding_unit=args.rounding_unit,
        task_caps=task_caps,
    )
    method_budget = build_method_budget(gate, task_budget_meta)
    mechanism_budget = build_mechanism_budget(
        tests,
        method_budget,
        recommended_max_examples=int(task_budget_meta["recommended_command_max_examples"]),
    )
    summary = write_outputs(
        args.output_dir,
        task_budget,
        method_budget,
        mechanism_budget,
        target_half_width=args.target_wilson_half_width,
        confidence_z=args.confidence_z,
        target_paired_net_loss_rate=args.target_paired_net_loss_rate,
        paired_alpha=args.paired_alpha,
        assumed_paired_discordance_rate=args.assumed_paired_discordance_rate,
        task_budget_meta=task_budget_meta,
    )
    print(f"Wrote Qwen3 MoE eval budget plan to {repo_path(args.output_dir).resolve()}")
    print(
        "Recommended max examples: "
        f"{summary['recommended_max_examples']}; additional prompts: {summary['total_additional_prompt_budget']}"
    )


if __name__ == "__main__":
    main()
