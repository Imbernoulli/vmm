#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any


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


def count_rule_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def count_delta_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_writer_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "scripts/write_same_shape_average_checkpoint.py",
        "--base",
        args.base,
    ]
    for source in args.source:
        source_name = source.split("=", 1)[0]
        parts.extend(["--source", source])
        parts.extend(["--source-weight", f"{source_name}=0.0"])
    if args.freeze_router:
        parts.append("--freeze-router")
    parts.extend(["--tensor-rule-file", rel(args.tensor_rule_file)])
    if args.source_tensor_alias_file:
        parts.extend(["--source-tensor-alias-file", rel(args.source_tensor_alias_file)])
    parts.extend(["--tensor-add-csv", rel(args.tensor_add_csv)])
    parts.extend(["--output-dir", rel(args.checkpoint_output_dir)])
    if args.dry_run:
        parts.append("--dry-run")
    return quote_command(parts)


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# MoE Combined Materialization Recipe",
        "",
        "这个 recipe 把 confidence-blended expert tensor rules、expert-output alias remap 和 router-bias capacity delta 合成同一个 same-shape writer command。它不扩展 experts，不改 router shape，也不改输出 tensor names；所有动作都发生在已有 tensor 上。",
        "",
        f"- Status: `{summary['recipe_status']}`",
        f"- Tensor rules: `{summary['tensor_rule_count']}`",
        f"- Alias rules: `{summary['alias_rule_count']}`",
        f"- Router-bias delta rows: `{summary['router_bias_delta_rows']}`",
        f"- Freeze router: `{summary['freeze_router']}`",
        f"- Dry run command: `{summary['dry_run']}`",
        "",
        "## 组合逻辑",
        "",
        "- 默认 source delta weight 设为 0，避免未覆盖 tensor 被无意平均。",
        "- `tensor_rules.txt` 决定共享 attention 和每个 expert FFN 的 source 权重。",
        "- `source_tensor_aliases.txt` 只改变从某个 source 读取哪个 expert tensor，不改变输出 checkpoint 的 expert index。",
        "- `router_bias_deltas.csv` 在 frozen/base router bias 上叠加 capacity correction。",
        "",
        "## Writer Command",
        "",
        "```bash",
        summary["writer_command"],
        "```",
        "",
        "## Files",
        "",
        f"- `{summary['inputs']['tensor_rule_file']}`",
        f"- `{summary['inputs']['source_tensor_alias_file']}`",
        f"- `{summary['inputs']['tensor_add_csv']}`",
        f"- `{summary['outputs']['writer_command']}`",
        f"- `{summary['outputs']['summary']}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose same-shape MoE materialization rules into one writer command.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_confidence_blended_combined_recipe"))
    parser.add_argument("--base", default="MOE_BASE_OR_ANCHOR_PATH")
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument(
        "--tensor-rule-file",
        type=Path,
        default=Path("results/toy_moe_confidence_blended_recipes/tensor_rules.txt"),
    )
    parser.add_argument(
        "--source-tensor-alias-file",
        type=Path,
        default=Path("results/toy_moe_expert_remap_plan/source_tensor_aliases.txt"),
    )
    parser.add_argument(
        "--tensor-add-csv",
        type=Path,
        default=Path("results/moe_confidence_blended_router_bias_plan/router_bias_deltas.csv"),
    )
    parser.add_argument(
        "--checkpoint-output-dir",
        type=Path,
        default=Path("results/checkpoints/toy_moe_confidence_blended_combined_candidate"),
    )
    parser.add_argument("--no-freeze-router", dest="freeze_router", action="store_false")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(freeze_router=True, dry_run=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source:
        args.source = ["general=GENERAL_MODEL_PATH", "code=CODE_MODEL_PATH"]
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_rule_file = repo_path(args.tensor_rule_file)
    source_tensor_alias_file = repo_path(args.source_tensor_alias_file) if args.source_tensor_alias_file else None
    tensor_add_csv = repo_path(args.tensor_add_csv)
    tensor_rule_count = count_rule_lines(tensor_rule_file)
    alias_rule_count = count_rule_lines(source_tensor_alias_file) if source_tensor_alias_file else 0
    router_bias_delta_rows = count_delta_rows(tensor_add_csv)
    missing_inputs = [
        rel(path)
        for path in (tensor_rule_file, source_tensor_alias_file, tensor_add_csv)
        if path is not None and not path.exists()
    ]
    recipe_status = "combined_writer_command_ready" if not missing_inputs and tensor_rule_count and router_bias_delta_rows else "missing_inputs"

    writer_command = build_writer_command(args)
    writer_command_path = output_dir / "writer_command.txt"
    writer_command_path.write_text(writer_command + "\n", encoding="utf-8")

    summary = {
        "schema_version": 1,
        "recipe_status": recipe_status,
        "candidate": "toy_moe_confidence_blended_combined_candidate",
        "same_shape_constraint": "Output tensor names and shapes stay unchanged; rules only select source deltas, source tensor aliases, and additive router-bias values.",
        "missing_inputs": missing_inputs,
        "sources": args.source,
        "default_source_delta_weight": 0.0,
        "tensor_rule_count": tensor_rule_count,
        "alias_rule_count": alias_rule_count,
        "router_bias_delta_rows": router_bias_delta_rows,
        "freeze_router": bool(args.freeze_router),
        "dry_run": bool(args.dry_run),
        "writer_command": writer_command,
        "inputs": {
            "tensor_rule_file": rel(tensor_rule_file),
            "source_tensor_alias_file": rel(source_tensor_alias_file) if source_tensor_alias_file else None,
            "tensor_add_csv": rel(tensor_add_csv),
        },
        "outputs": {
            "writer_command": rel(writer_command_path),
            "checkpoint_output_dir": rel(args.checkpoint_output_dir),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(f"Wrote MoE combined materialization recipe to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
