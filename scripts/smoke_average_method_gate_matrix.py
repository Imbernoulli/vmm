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


def probe_row(probe_table: pd.DataFrame, gate: str) -> dict[str, Any]:
    rows = probe_table[probe_table["gate"].astype(str) == gate]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def build_report(summary: dict[str, Any], matrix: pd.DataFrame) -> str:
    lines = [
        "# Average Method Gate Matrix Consistency Smoke",
        "",
        "This smoke verifies that the method gate matrix is synchronized with the current unified optimizer and final selector state.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Assertions: `{summary['passed_assertion_count']}/{summary['assertion_count']}`",
        "",
        "| assertion | expected | actual | passed |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['assertion']}` | `{row['expected']}` | `{row['actual']}` | `{bool(row['passed'])}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['matrix']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test average method gate matrix consistency.")
    parser.add_argument(
        "--optimizer-summary",
        type=Path,
        default=Path("results/unified_average_optimizer/summary.json"),
    )
    parser.add_argument(
        "--method-gate-dir",
        type=Path,
        default=Path("results/average_method_gate_matrix"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/average_method_gate_matrix_consistency_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer = read_json(args.optimizer_summary)
    method_summary = read_json(repo_path(args.method_gate_dir) / "summary.json")
    probe_table = read_csv(repo_path(args.method_gate_dir) / "probe_gate_matrix.csv")
    method_table = read_csv(repo_path(args.method_gate_dir) / "method_gate_matrix.csv")
    moe = optimizer.get("moe") or {}
    final_gate = probe_row(probe_table, "final_downstream_acceptance")
    current_gates = method_table["current_gate"].astype(str) if not method_table.empty else pd.Series(dtype=str)
    rows = [
        {
            "assertion": "final_gate_threshold_matches_optimizer_candidate_count",
            "expected": moe.get("qwen3_candidate_count"),
            "actual": clean_value(final_gate.get("threshold")),
            "passed": clean_value(final_gate.get("threshold")) == moe.get("qwen3_candidate_count"),
        },
        {
            "assertion": "final_gate_value_matches_optimizer_eligible_count",
            "expected": moe.get("qwen3_eligible_candidates"),
            "actual": clean_value(final_gate.get("value")),
            "passed": clean_value(final_gate.get("value")) == moe.get("qwen3_eligible_candidates"),
        },
        {
            "assertion": "final_gate_status_matches_optimizer_status",
            "expected": moe.get("qwen3_final_selection_status"),
            "actual": clean_value(final_gate.get("status")),
            "passed": clean_value(final_gate.get("status")) == moe.get("qwen3_final_selection_status"),
        },
        {
            "assertion": "no_method_family_accepted_by_default",
            "expected": 0,
            "actual": int(method_summary.get("accepted_by_default_count", -1)),
            "passed": int(method_summary.get("accepted_by_default_count", -1)) == 0,
        },
        {
            "assertion": "default_rejected_count_matches_table",
            "expected": int((current_gates == "rejected_as_default").sum()),
            "actual": int(method_summary.get("default_rejected_count", -1)),
            "passed": int(method_summary.get("default_rejected_count", -1))
            == int((current_gates == "rejected_as_default").sum()),
        },
    ]
    matrix = pd.DataFrame(rows)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "consistency_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    matrix.to_csv(matrix_path, index=False)
    passed = int(matrix["passed"].astype(bool).sum())
    summary = {
        "schema_version": 1,
        "status": "passed" if passed == len(matrix) else "failed",
        "assertion_count": int(len(matrix)),
        "passed_assertion_count": passed,
        "failed_assertion_count": int(len(matrix) - passed),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, matrix), encoding="utf-8")
    print(f"Wrote average method gate matrix consistency smoke to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; assertions {passed}/{len(matrix)}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
