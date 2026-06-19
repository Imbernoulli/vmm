#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from write_same_shape_average_checkpoint import write_average_checkpoint


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


def make_checkpoint(root: Path, tensors: dict[str, torch.Tensor]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "tiny_moe_writer_smoke",
                "num_hidden_layers": 1,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "hidden_size": 2,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})


def source_tensors(offset: float) -> dict[str, torch.Tensor]:
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), offset + 1.0),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.full((2, 2), offset + 10.0),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((2, 2), offset + 20.0),
        "model.layers.0.router.weight": torch.full((2, 2), offset + 30.0),
        "model.embed_tokens.weight": torch.full((2, 2), offset + 40.0),
        "model.int_buffer": torch.tensor([1, 2], dtype=torch.int64),
    }


def expected_tensors(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    general = source_tensors(100.0)
    code = source_tensors(200.0)
    expected: dict[str, torch.Tensor] = {}
    for name, base_tensor in base.items():
        if not torch.is_floating_point(base_tensor):
            expected[name] = base_tensor
        elif "self_attn" in name:
            expected[name] = base_tensor + 0.25 * (general[name] - base_tensor) + 0.75 * (code[name] - base_tensor)
        elif "experts.0." in name:
            expected[name] = base_tensor + 0.75 * (general[name] - base_tensor) + 0.25 * (code[name] - base_tensor)
        elif "experts.1." in name:
            expected[name] = base_tensor + 0.25 * (general[name] - base_tensor) + 0.75 * (code[name] - base_tensor)
        elif "router" in name:
            expected[name] = base_tensor
        else:
            expected[name] = base_tensor
    return expected


def build_rule_file(path: Path) -> None:
    lines = [
        "# Shared module rule.",
        ".*self_attn.*::general=0.25,code=0.75",
        "# Per-expert rules.",
        ".*experts\\.0\\..*::general=0.75,code=0.25",
        ".*experts\\.1\\..*::general=0.25,code=0.75",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moe_tensor_rule_writer_") as tmp_raw:
        tmp = Path(tmp_raw)
        base_dir = tmp / "base"
        general_dir = tmp / "general"
        code_dir = tmp / "code"
        written_dir = tmp / "written"
        rule_file = tmp / "tensor_rules.txt"

        base = source_tensors(0.0)
        make_checkpoint(base_dir, base)
        make_checkpoint(general_dir, source_tensors(100.0))
        make_checkpoint(code_dir, source_tensors(200.0))
        build_rule_file(rule_file)

        args = argparse.Namespace(
            base=str(base_dir),
            source=[f"general={general_dir}", f"code={code_dir}"],
            source_weight=["general=0.0", "code=0.0"],
            tensor_rule=[],
            tensor_rule_file=[str(rule_file)],
            source_tensor_alias=[],
            source_tensor_alias_file=[],
            freeze_regex=[],
            freeze_router=True,
            allow_missing_source_tensors=False,
            output_dtype="base",
            output_dir=str(written_dir),
            copy_metadata=True,
            dry_run=False,
        )
        manifest = write_average_checkpoint(args)
        actual = load_file(str(written_dir / "model.safetensors"))
        expected = expected_tensors(base)
        rows = []
        all_passed = True
        for name in sorted(expected):
            expected_tensor = expected[name]
            actual_tensor = actual[name]
            if torch.is_floating_point(expected_tensor):
                max_abs_error = float((actual_tensor - expected_tensor).abs().max().item())
                passed = max_abs_error < 1e-6
                expected_mean = float(expected_tensor.mean().item())
                actual_mean = float(actual_tensor.mean().item())
            else:
                max_abs_error = 0.0 if torch.equal(actual_tensor, expected_tensor) else float("inf")
                passed = torch.equal(actual_tensor, expected_tensor)
                expected_mean = None
                actual_mean = None
            all_passed = all_passed and passed
            rows.append(
                {
                    "tensor": name,
                    "expected_mean": expected_mean,
                    "actual_mean": actual_mean,
                    "max_abs_error": max_abs_error,
                    "passed": passed,
                }
            )

        stored_manifest = json.loads((written_dir / "merge_manifest.json").read_text(encoding="utf-8"))
        stored_manifest["base"] = "TMP/base"
        stored_manifest["sources"] = {"general": "TMP/general", "code": "TMP/code"}
        for source_name, source_summary in stored_manifest.get("source_summaries", {}).items():
            source_summary["root"] = f"TMP/{source_name}"
        (output_dir / "merge_manifest.json").write_text(
            json.dumps(stored_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    checks = pd.DataFrame(rows)
    checks.to_csv(output_dir / "tensor_checks.csv", index=False)
    summary = {
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        "checked_tensors": int(len(checks)),
        "failed_tensors": int((~checks["passed"]).sum()),
        "rule_counts": manifest.get("rule_counts", {}),
        "floating_tensors": int(manifest.get("floating_tensors", 0)),
        "frozen_tensors": int(manifest.get("frozen_tensors", 0)),
        "copied_nonfloating_tensors": int(manifest.get("copied_nonfloating_tensors", 0)),
        "outputs": {
            "tensor_checks": rel(output_dir / "tensor_checks.csv"),
            "merge_manifest": rel(output_dir / "merge_manifest.json"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# MoE Tensor-Rule Writer Smoke",
        "",
        "这个 smoke 构造 tiny MoE-like safetensors checkpoint，调用真实 same-shape writer 写出权重，再逐张量检查 tensor-rule、freeze-router 和非浮点复制是否生效。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Checked tensors: `{summary['checked_tensors']}`",
        f"- Failed tensors: `{summary['failed_tensors']}`",
        f"- Rule counts: `{json.dumps(summary['rule_counts'], sort_keys=True)}`",
        "",
        "## Files",
        "",
        f"- `{summary['outputs']['tensor_checks']}`",
        f"- `{summary['outputs']['merge_manifest']}`",
        f"- `{summary['outputs']['summary']}`",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test MoE tensor-rule materialization in the same-shape checkpoint writer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_tensor_rule_writer_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(repo_path(args.output_dir))
    print(f"Wrote MoE tensor-rule writer smoke to {repo_path(args.output_dir).resolve()}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
