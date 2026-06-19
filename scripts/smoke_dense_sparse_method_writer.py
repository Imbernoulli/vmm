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
        json.dumps({"model_type": "tiny_sparse_method_writer_smoke"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})


def base_tensors() -> dict[str, torch.Tensor]:
    return {
        "model.layers.0.mlp.up_proj.weight": torch.zeros((2, 2), dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros((2, 2), dtype=torch.float32),
        "model.int_buffer": torch.tensor([1, 2], dtype=torch.int64),
    }


def source_tensors(source: str) -> dict[str, torch.Tensor]:
    if source == "left":
        sparse = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32)
        linear = torch.full((2, 2), 2.0, dtype=torch.float32)
    elif source == "right":
        sparse = torch.tensor([[2.0, -5.0], [-1.0, -4.0]], dtype=torch.float32)
        linear = torch.full((2, 2), 4.0, dtype=torch.float32)
    else:
        raise ValueError(source)
    return {
        "model.layers.0.mlp.up_proj.weight": sparse,
        "model.layers.0.self_attn.q_proj.weight": linear,
        "model.int_buffer": torch.tensor([1, 2], dtype=torch.int64),
    }


def expected_tensors() -> dict[str, torch.Tensor]:
    return {
        # Weighted sign election with 0.5/0.5 weights:
        # [[same sign keep both, conflict keeps right], [same sign keep both, exact tie drops both]]
        "model.layers.0.mlp.up_proj.weight": torch.tensor([[1.5, -2.5], [-2.0, 0.0]], dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 3.0, dtype=torch.float32),
        "model.int_buffer": torch.tensor([1, 2], dtype=torch.int64),
    }


def run_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dense_sparse_method_writer_") as tmp_raw:
        tmp = Path(tmp_raw)
        base_dir = tmp / "base"
        left_dir = tmp / "left"
        right_dir = tmp / "right"
        written_dir = tmp / "written"
        method_file = tmp / "tensor_method_rules.txt"

        make_checkpoint(base_dir, base_tensors())
        make_checkpoint(left_dir, source_tensors("left"))
        make_checkpoint(right_dir, source_tensors("right"))
        method_file.write_text(".*mlp\\.up_proj.*::ties,density=1.0\n", encoding="utf-8")

        args = argparse.Namespace(
            allow_missing_source_tensors=False,
            base=str(base_dir),
            copy_metadata=True,
            dry_run=False,
            freeze_regex=[],
            freeze_router=False,
            output_dir=str(written_dir),
            output_dtype="base",
            packed_expert_rule_csv=[],
            source=[f"left={left_dir}", f"right={right_dir}"],
            source_tensor_alias=[],
            source_tensor_alias_file=[],
            source_weight=["left=0.5", "right=0.5"],
            tensor_add_csv=[],
            tensor_method_rule=[],
            tensor_method_rule_file=[str(method_file)],
            tensor_rule=[],
            tensor_rule_file=[],
        )
        manifest = write_average_checkpoint(args)
        actual = load_file(str(written_dir / "model.safetensors"))
        expected = expected_tensors()

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
                passed = torch.equal(actual_tensor, expected_tensor)
                max_abs_error = 0.0 if passed else float("inf")
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
        stored_manifest["sources"] = {"left": "TMP/left", "right": "TMP/right"}
        for source_summary in stored_manifest.get("source_summaries", {}).values():
            source_summary["root"] = source_summary["root"].replace(str(tmp), "TMP")
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
        "method_counts": manifest.get("method_counts", {}),
        "tensor_method_rules": stored_manifest.get("tensor_method_rules", []),
        "outputs": {
            "merge_manifest": rel(output_dir / "merge_manifest.json"),
            "report": rel(output_dir / "report.md"),
            "summary": rel(output_dir / "summary.json"),
            "tensor_checks": rel(output_dir / "tensor_checks.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Dense Sparse-Method Writer Smoke",
        "",
        "这个 smoke 验证 same-shape writer 的 coordinate-wise sparse merge method。`mlp.up_proj` 使用 TIES-style trim/sign-elect/merge；`self_attn.q_proj` 保持普通线性平均。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Checked tensors: `{summary['checked_tensors']}`",
        f"- Failed tensors: `{summary['failed_tensors']}`",
        f"- Method counts: `{json.dumps(summary['method_counts'], sort_keys=True)}`",
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
    parser = argparse.ArgumentParser(description="Smoke-test coordinate-wise sparse tensor methods in the same-shape writer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dense_sparse_method_writer_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(repo_path(args.output_dir))
    print(f"Wrote dense sparse-method writer smoke to {repo_path(args.output_dir).resolve()}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
