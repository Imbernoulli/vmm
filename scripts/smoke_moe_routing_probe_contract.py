#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from probe_moe_routing import build_report, build_summary, collect_routes, repo_path, write_csv


class TinyTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, truncation
        tokens = [min(255, ord(ch)) for ch in text][:max_length]
        if not tokens:
            tokens = [0]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class TinyRouter(torch.nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, offset: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, num_experts) * 0.1)
        self.offset = offset

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = hidden_states @ self.weight + self.offset
        routing_weights, selected_experts = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1)
        return logits, routing_weights, selected_experts


class TinyMlp(torch.nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, offset: float) -> None:
        super().__init__()
        self.gate = TinyRouter(hidden_size, num_experts, offset=offset)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.gate(hidden_states)
        return hidden_states + logits.mean(dim=-1, keepdim=True)


class TinyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, offset: float) -> None:
        super().__init__()
        self.mlp = TinyMlp(hidden_size, num_experts, offset=offset)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class TinyMoE(torch.nn.Module):
    def __init__(self, *, hidden_size: int = 4, num_experts: int = 4, offset: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = torch.nn.ModuleList(
            [
                TinyLayer(hidden_size, num_experts, offset=offset),
                TinyLayer(hidden_size, num_experts, offset=offset + 0.05),
            ]
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False) -> Any:
        del attention_mask, use_cache
        values = (input_ids.float() % 17) / 17.0
        features = torch.stack(
            [
                values,
                torch.sin(values),
                torch.cos(values),
                values.square(),
            ],
            dim=-1,
        ).reshape(-1, self.hidden_size)
        hidden_states = features
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test scripts/probe_moe_routing.py on a tiny local MoE.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_routing_probe_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        {"category": "general", "prompt": "explain averaging"},
        {"category": "code", "prompt": "write code"},
        {"category": "math", "prompt": "solve 2+2"},
    ]
    namespace = argparse.Namespace(
        model="tiny_moe_left",
        compare_model="tiny_moe_right",
        top_k=2,
        max_length=32,
        use_chat_template=False,
        write_token_routes=True,
        router_name_regex=r"(^|\.)gate$",
        exclude_name_regex=r"(gate_proj|shared_expert_gate)",
        max_router_dim=16,
    )
    tokenizer = TinyTokenizer()
    left = TinyMoE(offset=0.0)
    right = TinyMoE(offset=0.15)
    device = torch.device("cpu")
    summary_rows, expert_rows, token_route_rows, route_cache = collect_routes(
        left,
        tokenizer,
        prompts,
        namespace,
        device,
    )
    compare_args = argparse.Namespace(**{**vars(namespace), "model": namespace.compare_model})
    compare_summary, compare_expert_rows, compare_token_rows, compare_cache = collect_routes(
        right,
        tokenizer,
        prompts,
        compare_args,
        device,
    )
    from probe_moe_routing import route_overlap_rows

    overlap_rows = route_overlap_rows(route_cache, compare_cache, namespace.model, namespace.compare_model)
    write_csv(output_dir / "router_summary.csv", summary_rows)
    write_csv(output_dir / "expert_load.csv", expert_rows)
    write_csv(output_dir / "token_routes.csv", token_route_rows)
    write_csv(output_dir / "compare_router_summary.csv", compare_summary)
    write_csv(output_dir / "compare_expert_load.csv", compare_expert_rows)
    write_csv(output_dir / "compare_token_routes.csv", compare_token_rows)
    write_csv(output_dir / "route_overlap.csv", overlap_rows)
    summary = build_summary(
        args=namespace,
        output_dir=output_dir,
        prompts=prompts,
        summary_rows=summary_rows,
        expert_rows=expert_rows,
        token_route_rows=token_route_rows,
        compare_summary=compare_summary,
        compare_expert_rows=compare_expert_rows,
        compare_token_rows=compare_token_rows,
        overlap_rows=overlap_rows,
    )
    if summary["router_count"] != 2:
        raise RuntimeError(f"Expected 2 tiny routers, got {summary['router_count']}")
    if summary["row_counts"]["route_overlap"] != len(prompts) * summary["router_count"]:
        raise RuntimeError("Route overlap rows do not match prompts x routers.")
    manifest = {
        "model": namespace.model,
        "compare_model": namespace.compare_model,
        "prompts": len(prompts),
        "top_k": namespace.top_k,
        "router_name_regex": namespace.router_name_regex,
        "exclude_name_regex": namespace.exclude_name_regex,
        "outputs": {
            "router_summary": "router_summary.csv",
            "expert_load": "expert_load.csv",
            "token_routes": "token_routes.csv",
            "compare_router_summary": "compare_router_summary.csv",
            "compare_expert_load": "compare_expert_load.csv",
            "compare_token_routes": "compare_token_routes.csv",
            "route_overlap": "route_overlap.csv",
            "summary": "summary.json",
            "report": "report.md",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(f"Wrote tiny MoE routing probe smoke to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
