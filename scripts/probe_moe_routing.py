#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    {"category": "general", "prompt": "Explain model merging in two concise paragraphs."},
    {"category": "math", "prompt": "Solve step by step: if 3x + 7 = 31, what is x?"},
    {"category": "code", "prompt": "Write a Python function that checks whether a string is a palindrome."},
    {"category": "safety", "prompt": "How should a lab safely dispose of expired chemical reagents?"},
]


@dataclass
class RouterCapture:
    name: str
    logits: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MoE router top-k, entropy, load-balance, and optional route-overlap probes."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--compare-model", default=None, help="Optional second model for route-overlap comparison.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --model.")
    parser.add_argument("--prompts", default=None, help="JSONL with {'category','prompt'} rows. Uses built-ins if omitted.")
    parser.add_argument("--output-dir", default="results/moe_routing_probe", help="Directory for CSV/JSON outputs.")
    parser.add_argument("--device", default=None, help="Device such as cuda:0 or cpu. Defaults automatically.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device-map", default=None, help="Optional transformers device_map, e.g. 'auto'.")
    parser.add_argument("--router-name-regex", default=r"(^|\.)(router|gate)$")
    parser.add_argument("--exclude-name-regex", default=r"(gate_proj|shared_expert_gate)")
    parser.add_argument("--max-router-dim", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--write-token-routes", action="store_true")
    return parser.parse_args()


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def load_prompts(path: str | None, max_prompts: int | None) -> list[dict[str, str]]:
    if path is None:
        prompts = DEFAULT_PROMPTS
    else:
        prompts = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompts.append(
                    {
                        "category": str(row.get("category", "default")),
                        "prompt": str(row["prompt"]),
                    }
                )
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts


def prompt_text(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def router_modules(
    model: torch.nn.Module,
    include_pattern: str,
    exclude_pattern: str,
) -> dict[str, torch.nn.Module]:
    include = re.compile(include_pattern)
    exclude = re.compile(exclude_pattern) if exclude_pattern else None
    found: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        if include.search(name) and not (exclude and exclude.search(name)):
            found[name] = module
    return found


def normalize_router_logits(output: Any, max_router_dim: int) -> torch.Tensor | None:
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output) or output.ndim < 2:
        return None
    if output.shape[-1] > max_router_dim:
        return None
    logits = output.detach().to(torch.float32).cpu()
    if logits.ndim == 2:
        return logits
    return logits.reshape(-1, logits.shape[-1])


def attach_hooks(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    max_router_dim: int,
) -> tuple[dict[str, RouterCapture], list[Any]]:
    captures = {name: RouterCapture(name=name, logits=[]) for name in modules}
    handles = []

    for name, module in modules.items():
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, module_name: str = name) -> None:
            logits = normalize_router_logits(output, max_router_dim=max_router_dim)
            if logits is not None:
                captures[module_name].logits.append(logits)

        handles.append(module.register_forward_hook(hook))
    return captures, handles


@torch.no_grad()
def collect_routes(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, int, str], dict[str, torch.Tensor]],
]:
    modules = router_modules(model, args.router_name_regex, args.exclude_name_regex)
    if not modules:
        raise RuntimeError(
            "No router modules matched. Adjust --router-name-regex/--exclude-name-regex for this architecture."
        )

    captures, handles = attach_hooks(model, modules, max_router_dim=args.max_router_dim)
    summary_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    token_route_rows: list[dict[str, Any]] = []
    route_cache: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}

    try:
        for prompt_idx, item in enumerate(tqdm(prompts, desc="routing prompts")):
            for capture in captures.values():
                capture.logits.clear()

            text = prompt_text(tokenizer, item["prompt"], args.use_chat_template)
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            _ = model(**encoded, use_cache=False)

            for router_name, capture in captures.items():
                if not capture.logits:
                    continue
                logits = torch.cat(capture.logits, dim=0)
                probs = F.softmax(logits, dim=-1)
                k = min(args.top_k, logits.shape[-1])
                top_values, top_indices = torch.topk(probs, k=k, dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                margin = top_values[:, 0] - top_values[:, 1] if k > 1 else torch.ones_like(top_values[:, 0])
                top1 = top_indices[:, 0]
                top1_counts = Counter(top1.tolist())
                topk_counts = Counter(top_indices.reshape(-1).tolist())
                total_tokens = int(top1.numel())
                top1_probs = torch.tensor(
                    [top1_counts.get(idx, 0) / max(1, total_tokens) for idx in range(logits.shape[-1])],
                    dtype=torch.float64,
                )
                top1_dist_entropy = -float(
                    (top1_probs[top1_probs > 0] * torch.log(top1_probs[top1_probs > 0])).sum().item()
                )

                summary_rows.append(
                    {
                        "model": str(args.model),
                        "category": item["category"],
                        "prompt_idx": prompt_idx,
                        "router": router_name,
                        "num_experts": logits.shape[-1],
                        "tokens": total_tokens,
                        "top_k": k,
                        "router_entropy_mean": float(entropy.mean().item()),
                        "top1_margin_mean": float(margin.mean().item()),
                        "unique_top1_experts": len(top1_counts),
                        "unique_topk_experts": len(topk_counts),
                        "max_top1_fraction": max(top1_counts.values(), default=0) / max(1, total_tokens),
                        "effective_top1_experts": math.exp(top1_dist_entropy),
                    }
                )

                route_cache[(item["category"], prompt_idx, router_name)] = {
                    "top1": top1,
                    "topk": top_indices,
                }

                for expert_id in range(logits.shape[-1]):
                    expert_rows.append(
                        {
                            "model": str(args.model),
                            "category": item["category"],
                            "prompt_idx": prompt_idx,
                            "router": router_name,
                            "expert_id": expert_id,
                            "top1_count": top1_counts.get(expert_id, 0),
                            "top1_fraction": top1_counts.get(expert_id, 0) / max(1, total_tokens),
                            "topk_count": topk_counts.get(expert_id, 0),
                            "topk_fraction": topk_counts.get(expert_id, 0) / max(1, total_tokens * k),
                        }
                    )

                if args.write_token_routes:
                    for token_idx in range(total_tokens):
                        token_route_rows.append(
                            {
                                "model": str(args.model),
                                "category": item["category"],
                                "prompt_idx": prompt_idx,
                                "router": router_name,
                                "token_idx": token_idx,
                                "top_experts": "|".join(str(value) for value in top_indices[token_idx].tolist()),
                                "top_probs": "|".join(f"{value:.8f}" for value in top_values[token_idx].tolist()),
                            }
                        )
    finally:
        for handle in handles:
            handle.remove()

    return summary_rows, expert_rows, token_route_rows, route_cache


def route_overlap_rows(
    left: dict[tuple[str, int, str], dict[str, torch.Tensor]],
    right: dict[tuple[str, int, str], dict[str, torch.Tensor]],
    left_model: str,
    right_model: str,
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(left) & set(right)):
        category, prompt_idx, router = key
        left_top1 = left[key]["top1"]
        right_top1 = right[key]["top1"]
        left_topk = left[key]["topk"]
        right_topk = right[key]["topk"]
        n = min(left_top1.numel(), right_top1.numel())
        if n == 0:
            continue
        top1_agreement = float((left_top1[:n] == right_top1[:n]).to(torch.float32).mean().item())
        jaccards = []
        for idx in range(n):
            a = set(left_topk[idx].tolist())
            b = set(right_topk[idx].tolist())
            jaccards.append(len(a & b) / max(1, len(a | b)))
        rows.append(
            {
                "left_model": left_model,
                "right_model": right_model,
                "category": category,
                "prompt_idx": prompt_idx,
                "router": router,
                "tokens_compared": n,
                "top1_agreement": top1_agreement,
                "topk_jaccard": float(sum(jaccards) / len(jaccards)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_model(model_name: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device_map is None:
        model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    prompts = load_prompts(args.prompts, args.max_prompts)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )

    model = load_model(args.model, args, device)
    if args.device_map is not None:
        device = next(model.parameters()).device
    summary_rows, expert_rows, token_route_rows, route_cache = collect_routes(model, tokenizer, prompts, args, device)
    write_csv(output_dir / "router_summary.csv", summary_rows)
    write_csv(output_dir / "expert_load.csv", expert_rows)
    if args.write_token_routes:
        write_csv(output_dir / "token_routes.csv", token_route_rows)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    overlap_rows: list[dict[str, Any]] = []
    if args.compare_model:
        compare_model = load_model(args.compare_model, args, device)
        compare_args = argparse.Namespace(**{**vars(args), "model": args.compare_model})
        compare_device = next(compare_model.parameters()).device if args.device_map is not None else device
        compare_summary, compare_expert_rows, compare_token_rows, compare_cache = collect_routes(
            compare_model,
            tokenizer,
            prompts,
            compare_args,
            compare_device,
        )
        write_csv(output_dir / "compare_router_summary.csv", compare_summary)
        write_csv(output_dir / "compare_expert_load.csv", compare_expert_rows)
        if args.write_token_routes:
            write_csv(output_dir / "compare_token_routes.csv", compare_token_rows)
        overlap_rows = route_overlap_rows(route_cache, compare_cache, str(args.model), str(args.compare_model))
        write_csv(output_dir / "route_overlap.csv", overlap_rows)

    manifest = {
        "model": args.model,
        "compare_model": args.compare_model,
        "prompts": len(prompts),
        "top_k": args.top_k,
        "router_name_regex": args.router_name_regex,
        "exclude_name_regex": args.exclude_name_regex,
        "outputs": {
            "router_summary": "router_summary.csv",
            "expert_load": "expert_load.csv",
            "token_routes": "token_routes.csv" if args.write_token_routes else None,
            "route_overlap": "route_overlap.csv" if overlap_rows else None,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
