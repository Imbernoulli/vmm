#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_moe_routing import DEFAULT_PROMPTS, prompt_text, resolve_device, resolve_dtype, router_modules
from train_moe_router_delta_calibration import calibrate_from_cache
from write_same_shape_average_checkpoint import write_average_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RouterCalibrationCapture:
    module_name: str
    tensor_name: str
    bias_tensor_name: str | None
    hidden: list[torch.Tensor] = field(default_factory=list)
    logits: list[torch.Tensor] = field(default_factory=list)


class TinyRouterBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        return x + 0.01 * torch.tanh(logits @ self.router.weight)


class TinyRouterModel(nn.Module):
    def __init__(self, hidden_dim: int = 6, num_experts: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyRouterBlock(hidden_dim, num_experts) for _ in range(layers)])

    def forward(self, inputs_embeds: torch.Tensor, **_: Any) -> torch.Tensor:
        hidden = inputs_embeds
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_prompts(path: str | None, max_prompts: int | None) -> list[dict[str, str]]:
    if path is None:
        prompts = list(DEFAULT_PROMPTS)
    else:
        prompts = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompts.append({"category": str(row.get("category", "default")), "prompt": str(row["prompt"])})
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts


def first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def flatten_last_dim(tensor: torch.Tensor, max_last_dim: int) -> torch.Tensor | None:
    if tensor.ndim < 2:
        return None
    if tensor.shape[-1] > max_last_dim:
        return None
    tensor = tensor.detach().to(torch.float32).cpu()
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def module_tensor_names(module_name: str, module: nn.Module) -> tuple[str, str | None]:
    if not hasattr(module, "weight") or not torch.is_tensor(getattr(module, "weight")):
        raise ValueError(f"Router module {module_name!r} has no direct tensor weight parameter.")
    tensor_name = f"{module_name}.weight"
    bias = getattr(module, "bias", None)
    bias_tensor_name = f"{module_name}.bias" if torch.is_tensor(bias) else None
    return tensor_name, bias_tensor_name


def attach_calibration_hooks(
    model: nn.Module,
    modules: dict[str, nn.Module],
    *,
    max_hidden_dim: int,
    max_router_dim: int,
) -> tuple[dict[str, RouterCalibrationCapture], list[Any]]:
    captures: dict[str, RouterCalibrationCapture] = {}
    handles = []
    for module_name, module in modules.items():
        tensor_name, bias_tensor_name = module_tensor_names(module_name, module)
        captures[module_name] = RouterCalibrationCapture(
            module_name=module_name,
            tensor_name=tensor_name,
            bias_tensor_name=bias_tensor_name,
        )

        def hook(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            capture_name: str = module_name,
        ) -> None:
            hidden = flatten_last_dim(first_tensor(inputs), max_hidden_dim)
            logits = flatten_last_dim(first_tensor(output), max_router_dim)
            if hidden is not None:
                captures[capture_name].hidden.append(hidden)
            if logits is not None:
                captures[capture_name].logits.append(logits)

        handles.append(module.register_forward_hook(hook))
    return captures, handles


@torch.no_grad()
def collect_captures(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    *,
    desc: str,
) -> dict[str, RouterCalibrationCapture]:
    modules = router_modules(model, args.router_name_regex, args.exclude_name_regex)
    if not modules:
        raise RuntimeError("No router modules matched. Adjust --router-name-regex/--exclude-name-regex.")
    captures, handles = attach_calibration_hooks(
        model,
        modules,
        max_hidden_dim=args.max_hidden_dim,
        max_router_dim=args.max_router_dim,
    )
    try:
        model.eval()
        for batch in tqdm(batches, desc=desc):
            moved = {key: value.to(device) for key, value in batch.items()}
            _ = model(**moved)
    finally:
        for handle in handles:
            handle.remove()
    return captures


def cat_or_empty(items: list[torch.Tensor]) -> torch.Tensor:
    if not items:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(items, dim=0)


def topk_jaccard(left_logits: torch.Tensor, right_logits: torch.Tensor, top_k: int) -> float:
    k = min(top_k, left_logits.shape[-1], right_logits.shape[-1])
    left = torch.topk(left_logits, k=k, dim=-1).indices
    right = torch.topk(right_logits, k=k, dim=-1).indices
    scores = []
    for row in range(left.shape[0]):
        left_set = set(int(item) for item in left[row].tolist())
        right_set = set(int(item) for item in right[row].tolist())
        scores.append(len(left_set & right_set) / max(1, len(left_set | right_set)))
    return float(sum(scores) / max(1, len(scores)))


def capacity_overflow_fraction(logits: torch.Tensor, capacity_factor: float) -> float:
    probs = F.softmax(logits, dim=-1)
    capacity = float(capacity_factor) / max(1, probs.shape[-1])
    overflow = F.relu(probs.mean(dim=0) - capacity).sum()
    return float(overflow.item())


def deterministic_select_rows(total_rows: int, max_rows: int | None, seed: int) -> torch.Tensor:
    if max_rows is None or max_rows <= 0 or total_rows <= max_rows:
        return torch.arange(total_rows)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total_rows, generator=generator)[:max_rows].sort().values


def build_cache_payload(
    *,
    student_captures: dict[str, RouterCalibrationCapture],
    teacher_captures: dict[str, RouterCalibrationCapture],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    routers: dict[str, dict[str, torch.Tensor | str | None]] = {}
    rows: list[dict[str, Any]] = []
    common_modules = sorted(set(student_captures) & set(teacher_captures))
    for module_name in common_modules:
        student = student_captures[module_name]
        teacher = teacher_captures[module_name]
        hidden = cat_or_empty(student.hidden)
        student_logits = cat_or_empty(student.logits)
        teacher_logits = cat_or_empty(teacher.logits)
        available_rows = min(hidden.shape[0], student_logits.shape[0], teacher_logits.shape[0])
        usable = (
            available_rows > 0
            and hidden.shape[-1] > 0
            and student_logits.shape[-1] == teacher_logits.shape[-1]
        )
        if usable:
            indices = deterministic_select_rows(available_rows, args.max_samples_per_router, args.seed)
            hidden_out = hidden[:available_rows][indices]
            student_out = student_logits[:available_rows][indices]
            teacher_out = teacher_logits[:available_rows][indices]
            teacher_probs = F.softmax(teacher_out, dim=-1)
            route_kl = F.kl_div(F.log_softmax(student_out, dim=-1), teacher_probs, reduction="batchmean")
            top1_agreement = float((student_out.argmax(dim=-1) == teacher_out.argmax(dim=-1)).float().mean().item())
            topk_overlap = topk_jaccard(student_out, teacher_out, args.top_k)
            student_overflow = capacity_overflow_fraction(student_out, args.capacity_factor)
            teacher_overflow = capacity_overflow_fraction(teacher_out, args.capacity_factor)
            routers[student.tensor_name] = {
                "hidden": hidden_out,
                "teacher_logits": teacher_out,
                "bias_tensor": student.bias_tensor_name,
            }
        else:
            hidden_out = torch.empty((0, 0), dtype=torch.float32)
            route_kl = math.nan
            top1_agreement = math.nan
            topk_overlap = math.nan
            student_overflow = math.nan
            teacher_overflow = math.nan
        rows.append(
            {
                "module": module_name,
                "tensor": student.tensor_name,
                "bias_tensor": student.bias_tensor_name or "",
                "student_hidden_rows": int(hidden.shape[0]),
                "student_logit_rows": int(student_logits.shape[0]),
                "teacher_logit_rows": int(teacher_logits.shape[0]),
                "used_rows": int(hidden_out.shape[0]),
                "hidden_dim": int(hidden.shape[-1]) if hidden.ndim == 2 else 0,
                "num_experts": int(teacher_logits.shape[-1]) if teacher_logits.ndim == 2 else 0,
                "route_kl_student_to_teacher": float(route_kl),
                "top1_agreement_student_to_teacher": top1_agreement,
                "topk_jaccard_student_to_teacher": topk_overlap,
                "student_capacity_overflow_fraction": student_overflow,
                "teacher_capacity_overflow_fraction": teacher_overflow,
                "cache_ready": bool(usable),
            }
        )
    payload = {
        "schema_version": 1,
        "routers": routers,
        "metadata": {
            "student_model": str(args.student_model),
            "teacher_model": str(args.teacher_model),
            "top_k": int(args.top_k),
            "capacity_factor": float(args.capacity_factor),
            "max_samples_per_router": args.max_samples_per_router,
            "router_name_regex": args.router_name_regex,
            "exclude_name_regex": args.exclude_name_regex,
        },
    }
    return payload, pd.DataFrame(rows)


def build_report(summary: dict[str, Any], cache_summary: pd.DataFrame) -> str:
    outputs = summary["outputs"]
    lines = [
        "# MoE Router Calibration Cache",
        "",
        "这个脚本把 student 模型的 router 输入 hidden states 和 teacher 模型的 router logits 对齐，写成 `train_moe_router_delta_calibration.py` 可直接读取的 cache。它补上了从真实 forward probe 到 router-delta 训练之间的接口。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Student: `{summary['student_model']}`",
        f"- Teacher: `{summary['teacher_model']}`",
        f"- Cache-ready routers: `{summary['cache_ready_router_count']}` / `{summary['common_router_count']}`",
        f"- Total cache rows: `{summary['total_cache_rows']}`",
        f"- Mean student->teacher KL: `{summary['mean_student_teacher_route_kl']:.6f}`",
        f"- Delta calibration: `{summary['calibration_status']}`",
        f"- Checkpoint materialization: `{summary['materialization_status']}`",
        "",
        "## Next Commands",
        "",
        "```bash",
        summary["training_command"],
        summary["writer_command_after_training"],
        "```",
        "",
        "## Router Rows",
        "",
        "| tensor | rows | hidden | experts | KL | top1 | top-k |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in cache_summary[cache_summary["cache_ready"].astype(bool)].head(12).iterrows():
        lines.append(
            f"| `{row['tensor']}` | {int(row['used_rows'])} | {int(row['hidden_dim'])} | "
            f"{int(row['num_experts'])} | {float(row['route_kl_student_to_teacher']):.6f} | "
            f"{float(row['top1_agreement_student_to_teacher']):.4f} | "
            f"{float(row['topk_jaccard_student_to_teacher']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{outputs['cache']}`",
            f"- `{outputs['cache_summary']}`",
            f"- `{outputs['summary']}`",
            f"- `{outputs['report']}`",
        ]
    )
    if outputs.get("materialization_checks"):
        lines.append(f"- `{outputs['materialization_checks']}`")
    return "\n".join(lines) + "\n"


def save_cache_outputs(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    payload: dict[str, Any],
    cache_summary: pd.DataFrame,
    calibration_summary: dict[str, Any] | None = None,
    materialization_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache_path = output_dir / "router_calibration_cache.pt"
    cache_summary_path = output_dir / "cache_summary.csv"
    torch.save(payload, cache_path)
    cache_summary.to_csv(cache_summary_path, index=False)
    ready = cache_summary[cache_summary["cache_ready"].astype(bool)] if not cache_summary.empty else cache_summary
    status = "cache_ready" if not ready.empty else "no_ready_router"
    train_output = output_dir / "delta_calibration"
    summary = {
        "schema_version": 1,
        "status": status,
        "student_model": str(args.student_model),
        "teacher_model": str(args.teacher_model),
        "common_router_count": int(len(cache_summary)),
        "cache_ready_router_count": int(len(ready)),
        "total_cache_rows": int(ready["used_rows"].sum()) if not ready.empty else 0,
        "mean_student_teacher_route_kl": float(ready["route_kl_student_to_teacher"].mean()) if not ready.empty else math.nan,
        "mean_student_teacher_top1_agreement": (
            float(ready["top1_agreement_student_to_teacher"].mean()) if not ready.empty else math.nan
        ),
        "mean_student_teacher_topk_jaccard": (
            float(ready["topk_jaccard_student_to_teacher"].mean()) if not ready.empty else math.nan
        ),
        "calibration_status": None if calibration_summary is None else calibration_summary.get("status"),
        "calibration_mean_final_route_kl": (
            None if calibration_summary is None else calibration_summary.get("mean_final_route_kl")
        ),
        "materialization_status": None if materialization_summary is None else materialization_summary.get("status"),
        "materialization_checked_tensors": (
            None if materialization_summary is None else materialization_summary.get("checked_tensors")
        ),
        "materialization_failed_tensors": (
            None if materialization_summary is None else materialization_summary.get("failed_tensors")
        ),
        "training_command": (
            "python scripts/train_moe_router_delta_calibration.py "
            f"--base STUDENT_BASE_CHECKPOINT --cache {rel(cache_path)} --output-dir {rel(train_output)}"
        ),
        "writer_command_after_training": (
            "python scripts/write_same_shape_average_checkpoint.py --base STUDENT_BASE_CHECKPOINT "
            "--source SOURCE_NAME=SOURCE_CHECKPOINT --source-weight SOURCE_NAME=0.0 --freeze-router "
            f"--tensor-delta-safetensors {rel(train_output / 'router_delta.safetensors')} "
            "--output-dir CHECKPOINT_WITH_CALIBRATED_ROUTER"
        ),
        "outputs": {
            "cache": rel(cache_path),
            "cache_summary": rel(cache_summary_path),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
            "delta_calibration": rel(train_output) if calibration_summary is not None else None,
            "materialized_checkpoint": (
                None if materialization_summary is None else materialization_summary.get("checkpoint_dir")
            ),
            "materialization_checks": (
                None if materialization_summary is None else materialization_summary.get("checks")
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, cache_summary), encoding="utf-8")
    return summary


def materialize_smoke_checkpoint(base_dir: Path, train_output: Path, output_dir: Path) -> dict[str, Any]:
    checkpoint_dir = output_dir / "checkpoint_with_calibrated_router"
    delta_path = train_output / "router_delta.safetensors"
    args = argparse.Namespace(
        base=str(base_dir),
        source=[f"same={base_dir}"],
        source_weight=["same=0.0"],
        tensor_rule=[],
        tensor_rule_file=[],
        tensor_method_rule=[],
        tensor_method_rule_file=[],
        source_tensor_alias=[],
        source_tensor_alias_file=[],
        tensor_add_csv=[],
        tensor_delta_safetensors=[str(delta_path)],
        packed_expert_rule_csv=[],
        freeze_regex=[],
        freeze_router=True,
        allow_missing_source_tensors=False,
        output_dtype="base",
        output_dir=str(checkpoint_dir),
        copy_metadata=True,
        dry_run=False,
    )
    manifest = write_average_checkpoint(args)
    base_values = load_file(str(base_dir / "model.safetensors"))
    delta_values = load_file(str(delta_path))
    actual_values = load_file(str(checkpoint_dir / "model.safetensors"))
    rows = []
    all_passed = True
    for tensor_name in sorted(delta_values):
        expected = base_values[tensor_name].to(torch.float32) + delta_values[tensor_name].to(torch.float32)
        actual = actual_values[tensor_name].to(torch.float32)
        max_abs_error = float((actual - expected).abs().max().item())
        passed = max_abs_error < 1e-6
        all_passed = all_passed and passed
        rows.append(
            {
                "tensor": tensor_name,
                "max_abs_error": max_abs_error,
                "expected_mean": float(expected.mean().item()),
                "actual_mean": float(actual.mean().item()),
                "passed": passed,
            }
        )
    checks = pd.DataFrame(rows)
    checks_path = output_dir / "materialization_checks.csv"
    checks.to_csv(checks_path, index=False)
    return {
        "status": "passed" if all_passed else "failed",
        "checked_tensors": int(len(checks)),
        "failed_tensors": int((~checks["passed"]).sum()) if not checks.empty else 0,
        "checkpoint_dir": rel(checkpoint_dir),
        "checks": rel(checks_path),
        "manifest": manifest,
        "tensor_delta_safetensors_tensors": int(manifest.get("tensor_delta_safetensors_tensors", 0)),
        "tensor_delta_safetensors_values": int(manifest.get("tensor_delta_safetensors_values", 0)),
    }


def tokenized_batches(tokenizer: Any, prompts: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, torch.Tensor]]:
    batches = []
    for item in prompts:
        text = prompt_text(tokenizer, item["prompt"], args.use_chat_template)
        batches.append(tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length))
    return batches


def load_hf_model(model_name: str, args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device_map is None:
        model.to(device)
    model.eval()
    return model


def collect_from_hf(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.student_model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    prompts = load_prompts(args.prompts, args.max_prompts)
    batches = tokenized_batches(tokenizer, prompts, args)

    student = load_hf_model(args.student_model, args, device)
    student_device = next(student.parameters()).device if args.device_map is not None else device
    student_captures = collect_captures(student, batches, args, student_device, desc="student router cache")
    del student
    if student_device.type == "cuda":
        torch.cuda.empty_cache()

    teacher = load_hf_model(args.teacher_model, args, device)
    teacher_device = next(teacher.parameters()).device if args.device_map is not None else device
    teacher_captures = collect_captures(teacher, batches, args, teacher_device, desc="teacher router cache")
    del teacher
    if teacher_device.type == "cuda":
        torch.cuda.empty_cache()

    payload, cache_summary = build_cache_payload(
        student_captures=student_captures,
        teacher_captures=teacher_captures,
        args=args,
    )
    return save_cache_outputs(args=args, output_dir=output_dir, payload=payload, cache_summary=cache_summary)


def make_smoke_models(seed: int) -> tuple[TinyRouterModel, TinyRouterModel, list[dict[str, torch.Tensor]], Path]:
    torch.manual_seed(seed)
    student = TinyRouterModel()
    teacher = TinyRouterModel()
    teacher.load_state_dict(student.state_dict())
    with torch.no_grad():
        for layer_idx, block in enumerate(teacher.blocks):
            delta = torch.zeros_like(block.router.weight)
            delta[layer_idx, layer_idx] = 0.9 - 0.15 * layer_idx
            delta[:, 4] = torch.tensor([0.25, -0.20, 0.15, -0.10])
            block.router.weight.add_(delta)
    batches = []
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(4):
        batches.append({"inputs_embeds": torch.randn(2, 12, 6, generator=generator)})
    root = Path(tempfile.mkdtemp(prefix="moe_router_delta_calibration_cache_"))
    base_dir = root / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer_idx, block in enumerate(student.blocks):
        tensors[f"blocks.{layer_idx}.router.weight"] = block.router.weight.detach().clone()
    save_file(tensors, str(base_dir / "model.safetensors"), metadata={"format": "pt"})
    return student, teacher, batches, base_dir


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    student, teacher, batches, base_dir = make_smoke_models(args.seed)
    device = torch.device("cpu")
    student_captures = collect_captures(student, batches, args, device, desc="smoke student cache")
    teacher_captures = collect_captures(teacher, batches, args, device, desc="smoke teacher cache")
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.student_model = "SMOKE_STUDENT"
    smoke_args.teacher_model = "SMOKE_TEACHER"
    payload, cache_summary = build_cache_payload(
        student_captures=student_captures,
        teacher_captures=teacher_captures,
        args=smoke_args,
    )
    cache_path = output_dir / "router_calibration_cache.pt"
    train_output = output_dir / "delta_calibration"
    torch.save(payload, cache_path)
    train_args = argparse.Namespace(
        base=str(base_dir),
        cache=str(cache_path),
        output_dir=train_output,
        epochs=args.smoke_train_epochs,
        lr=args.smoke_train_lr,
        temperature=1.0,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor,
        top1_loss_coef=0.25,
        capacity_loss_coef=0.1,
        load_balance_coef=0.0,
        trust_l2_coef=0.001,
        max_relative_norm=0.5,
        trace_every=10,
        seed=args.seed,
        smoke=False,
    )
    calibration_summary = calibrate_from_cache(train_args)
    materialization_summary = materialize_smoke_checkpoint(base_dir, train_output, output_dir)
    summary = save_cache_outputs(
        args=smoke_args,
        output_dir=output_dir,
        payload=payload,
        cache_summary=cache_summary,
        calibration_summary=calibration_summary,
        materialization_summary=materialization_summary,
    )
    summary["status"] = (
        "passed"
        if (
            summary["status"] == "cache_ready"
            and calibration_summary["status"] == "passed"
            and materialization_summary["status"] == "passed"
        )
        else "failed"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, cache_summary), encoding="utf-8")
    if summary["status"] != "passed":
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect router hidden/logit caches for same-shape MoE router calibration.")
    parser.add_argument("--student-model", default=None, help="Student/base model whose router input hidden states are cached.")
    parser.add_argument("--teacher-model", default=None, help="Teacher/source model whose router logits are cached.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --student-model.")
    parser.add_argument("--prompts", default=None, help="JSONL with {'category','prompt'} rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_router_calibration_cache_smoke"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--max-router-dim", type=int, default=4096)
    parser.add_argument("--max-hidden-dim", type=int, default=32768)
    parser.add_argument("--max-samples-per-router", type=int, default=4096)
    parser.add_argument("--router-name-regex", default=r"(^|\.)(router|gate)$")
    parser.add_argument("--exclude-name-regex", default=r"(gate_proj|shared_expert_gate)")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-epochs", type=int, default=80)
    parser.add_argument("--smoke-train-lr", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        summary = run_smoke(args)
    else:
        if not args.student_model or not args.teacher_model:
            raise SystemExit("--student-model and --teacher-model are required unless --smoke is set")
        summary = collect_from_hf(args)
    print(f"Wrote MoE router calibration cache to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; ready routers: {summary['cache_ready_router_count']}")


if __name__ == "__main__":
    main()
