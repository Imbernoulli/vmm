#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


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


def fnum(value: Any) -> float | None:
    value = clean_value(value)
    return None if value is None else float(value)


def fint(value: Any) -> int | None:
    value = clean_value(value)
    return None if value is None else int(float(value))


def fbool(value: Any) -> bool | None:
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def structural_by_method(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if table.empty:
        return {}
    out = {}
    for _, row in table.iterrows():
        method = str(row.get("method", "")).strip()
        if not method:
            continue
        out[method] = {str(key): clean_value(value) for key, value in row.to_dict().items()}
    return out


def gate_row(row: dict[str, Any], structural: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    method = str(row.get("method", ""))
    role = str(row.get("role", ""))
    if role == "source":
        return {
            "method": method,
            "role": role,
            "trust_region_category": "source_control",
            "final_selectable_by_trust_region": True,
            "trust_region_rejection_reasons": "",
            "audit_passed": fbool(row.get("audit_passed")),
            "structural_frontier_member": None,
            "structurally_dominated": None,
            "structural_safety_score": None,
            "routed_max_tensor_relative_delta": None,
            "routed_tensors_gt_0_65": None,
            "attention_changed_tensors": None,
            "router_changed_tensors": None,
        }

    reasons = []
    audit_passed = fbool(row.get("audit_passed"))
    router_changed = fint(row.get("router_changed_tensors"))
    attention_changed = fint(row.get("attention_changed_tensors"))
    routed_gt_065 = fint(row.get("routed_tensors_gt_0_65"))
    routed_max = fnum(row.get("routed_max_tensor_relative_delta"))
    dominated = fbool(structural.get("structurally_dominated"))
    frontier = None if dominated is None else not dominated
    safety = fnum(structural.get("structural_safety_score"))

    if audit_passed is not True:
        reasons.append("checkpoint_audit_not_passed")
    if router_changed is None or router_changed > args.max_router_changed_tensors:
        reasons.append("router_changed")
    if attention_changed is None or attention_changed > args.max_attention_changed_tensors:
        reasons.append("shared_attention_changed")
    if routed_max is None or routed_max > args.strict_routed_max_relative_delta:
        reasons.append("routed_max_delta_over_strict_cap")
    if routed_gt_065 is None or routed_gt_065 > args.max_routed_tensors_gt_065:
        reasons.append("routed_tail_over_065")
    if args.require_structural_frontier and frontier is not True:
        reasons.append("structurally_dominated")

    final_selectable = not reasons
    if final_selectable:
        category = "final_selectable_trust_region"
    elif audit_passed is True and router_changed == 0:
        category = "ablation_only"
    else:
        category = "structural_reject"
    return {
        "method": method,
        "role": role,
        "trust_region_category": category,
        "final_selectable_by_trust_region": final_selectable,
        "trust_region_rejection_reasons": ",".join(reasons),
        "audit_passed": audit_passed,
        "structural_frontier_member": frontier,
        "structurally_dominated": dominated,
        "structural_safety_score": safety,
        "routed_max_tensor_relative_delta": routed_max,
        "routed_tensors_gt_0_65": routed_gt_065,
        "attention_changed_tensors": attention_changed,
        "router_changed_tensors": router_changed,
    }


def build_report(summary: dict[str, Any], table: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Candidate Trust-Region Gate",
        "",
        "这个 gate 把结构 probe 转成 final selector 可消费的候选级约束：旧 candidate 仍可作为机制 ablation 跑 vLLM，但只有满足 router freeze、attention freeze、strict routed tail cap 和结构 frontier 的 candidate 才能进入最终默认 average 选择。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Candidate final-selectable: `{summary['final_selectable_candidate_count']}/{summary['candidate_count']}`",
        f"- Ablation-only candidates: `{summary['ablation_only_candidate_count']}`",
        f"- Structural rejects: `{summary['structural_reject_candidate_count']}`",
        f"- Strict routed max cap: `{summary['strict_routed_max_relative_delta']}`",
        "",
        "| method | category | selectable | reasons | frontier | safety | routed max | routed >0.65 | attn changed | router changed |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        if row["role"] == "source":
            continue
        lines.append(
            f"| `{row['method']}` | `{row['trust_region_category']}` | "
            f"`{bool(row['final_selectable_by_trust_region'])}` | "
            f"`{row['trust_region_rejection_reasons']}` | "
            f"`{row['structural_frontier_member']}` | {fmt(row['structural_safety_score'])} | "
            f"{fmt(row['routed_max_tensor_relative_delta'])} | "
            f"{int(row['routed_tensors_gt_0_65']) if clean_value(row['routed_tensors_gt_0_65']) is not None else 'n/a'} | "
            f"{int(row['attention_changed_tensors']) if clean_value(row['attention_changed_tensors']) is not None else 'n/a'} | "
            f"{int(row['router_changed_tensors']) if clean_value(row['router_changed_tensors']) is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['gate']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate-level trust-region gate for Qwen3 MoE final selection.")
    parser.add_argument("--gate-plan", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate/eval_gate_plan.csv"))
    parser.add_argument(
        "--structural-dominance",
        type=Path,
        default=Path("results/qwen3_moe_delta_frontier/structural_dominance.csv"),
    )
    parser.add_argument("--strict-routed-max-relative-delta", type=float, default=0.6505)
    parser.add_argument("--max-routed-tensors-gt-065", type=int, default=0)
    parser.add_argument("--max-attention-changed-tensors", type=int, default=0)
    parser.add_argument("--max-router-changed-tensors", type=int, default=0)
    parser.add_argument("--require-structural-frontier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_candidate_trust_region_gate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate_plan = read_csv(args.gate_plan)
    structural = structural_by_method(read_csv(args.structural_dominance))
    rows = [gate_row(row.to_dict(), structural.get(str(row.get("method")), {}), args) for _, row in gate_plan.iterrows()]
    table = pd.DataFrame(rows)
    candidates = table[table["role"].astype(str) == "candidate"]
    final_selectable = candidates[candidates["final_selectable_by_trust_region"].astype(bool)]
    ablation_only = candidates[candidates["trust_region_category"].astype(str) == "ablation_only"]
    structural_reject = candidates[candidates["trust_region_category"].astype(str) == "structural_reject"]
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "candidate_trust_region_gate.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    table.to_csv(gate_path, index=False)
    summary = {
        "schema_version": 1,
        "status": "candidate_trust_region_gate_ready",
        "candidate_count": int(len(candidates)),
        "source_count": int((table["role"].astype(str) == "source").sum()),
        "final_selectable_candidate_count": int(len(final_selectable)),
        "ablation_only_candidate_count": int(len(ablation_only)),
        "structural_reject_candidate_count": int(len(structural_reject)),
        "strict_routed_max_relative_delta": args.strict_routed_max_relative_delta,
        "max_routed_tensors_gt_065": args.max_routed_tensors_gt_065,
        "max_attention_changed_tensors": args.max_attention_changed_tensors,
        "max_router_changed_tensors": args.max_router_changed_tensors,
        "require_structural_frontier": bool(args.require_structural_frontier),
        "final_selectable_methods": [str(method) for method in final_selectable["method"].tolist()],
        "outputs": {
            "gate": rel(gate_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, table), encoding="utf-8")
    print(f"Wrote Qwen3 MoE candidate trust-region gate to {output_dir.resolve()}")
    print(
        f"Status: {summary['status']}; final-selectable "
        f"{summary['final_selectable_candidate_count']}/{summary['candidate_count']}"
    )


if __name__ == "__main__":
    main()
