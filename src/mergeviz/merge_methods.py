from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MergeResult:
    name: str
    vector: Tensor
    details: dict[str, float | int | str]


def linear_average(base: Tensor, deltas: list[Tensor]) -> Tensor:
    return base + torch.stack(deltas, dim=0).mean(dim=0)


def task_arithmetic(base: Tensor, deltas: list[Tensor], scale: float) -> Tensor:
    return base + scale * torch.stack(deltas, dim=0).sum(dim=0)


def slerp(a: Tensor, b: Tensor, t: float = 0.5, eps: float = 1e-8) -> Tensor:
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    an = torch.linalg.norm(a64)
    bn = torch.linalg.norm(b64)
    if float(an) < eps or float(bn) < eps:
        return ((1.0 - t) * a + t * b).to(a.dtype)
    a_unit = a64 / an
    b_unit = b64 / bn
    dot = torch.clamp(a_unit @ b_unit, -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if float(torch.abs(sin_omega)) < eps:
        return ((1.0 - t) * a + t * b).to(a.dtype)
    out = (torch.sin((1.0 - t) * omega) / sin_omega) * a64
    out = out + (torch.sin(t * omega) / sin_omega) * b64
    return out.to(a.dtype)


def _topk_mask(delta: Tensor, density: float) -> Tensor:
    if density >= 1.0:
        return torch.ones_like(delta, dtype=torch.bool)
    if density <= 0.0:
        return torch.zeros_like(delta, dtype=torch.bool)
    k = max(1, int(math.ceil(delta.numel() * density)))
    if k >= delta.numel():
        return torch.ones_like(delta, dtype=torch.bool)
    threshold = torch.topk(delta.abs(), k=k, largest=True).values[-1]
    return delta.abs() >= threshold


def ties_merge(base: Tensor, deltas: list[Tensor], density: float = 0.5) -> Tensor:
    """A compact TIES-style trim, sign-elect, and disjoint merge."""
    trimmed = []
    for delta in deltas:
        trimmed.append(delta * _topk_mask(delta, density).to(delta.dtype))
    stacked = torch.stack(trimmed, dim=0)
    elected_sign = torch.sign(stacked.sum(dim=0))
    sign_match = torch.sign(stacked) == elected_sign.unsqueeze(0)
    nonzero_elected = elected_sign != 0
    mask = sign_match & nonzero_elected.unsqueeze(0)
    numerator = (stacked * mask.to(stacked.dtype)).sum(dim=0)
    denom = mask.sum(dim=0).clamp_min(1).to(stacked.dtype)
    merged_delta = numerator / denom
    merged_delta = torch.where(nonzero_elected, merged_delta, torch.zeros_like(merged_delta))
    return base + merged_delta


def dare_delta(delta: Tensor, drop_rate: float = 0.5, seed: int = 0) -> Tensor:
    if drop_rate <= 0.0:
        return delta.clone()
    if drop_rate >= 1.0:
        return torch.zeros_like(delta)
    generator = torch.Generator(device=delta.device)
    generator.manual_seed(seed)
    keep = torch.rand(delta.shape, generator=generator, device=delta.device) >= drop_rate
    return delta * keep.to(delta.dtype) / (1.0 - drop_rate)


def dare_average(base: Tensor, deltas: list[Tensor], drop_rate: float = 0.5, seed: int = 0) -> Tensor:
    dropped = [dare_delta(delta, drop_rate=drop_rate, seed=seed + idx) for idx, delta in enumerate(deltas)]
    return linear_average(base, dropped)


def ties_dare_merge(
    base: Tensor,
    deltas: list[Tensor],
    density: float = 0.5,
    drop_rate: float = 0.5,
    seed: int = 0,
) -> Tensor:
    dropped = [dare_delta(delta, drop_rate=drop_rate, seed=seed + idx) for idx, delta in enumerate(deltas)]
    return ties_merge(base, dropped, density=density)


def fisher_weighted_average(thetas: list[Tensor], fishers: list[Tensor], eps: float = 1e-8) -> Tensor:
    fisher_stack = torch.stack(fishers, dim=0).to(torch.float32)
    theta_stack = torch.stack(thetas, dim=0).to(torch.float32)
    numerator = (fisher_stack * theta_stack).sum(dim=0)
    denom = fisher_stack.sum(dim=0)
    average = theta_stack.mean(dim=0)
    weighted = numerator / denom.clamp_min(eps)
    return torch.where(denom > eps, weighted, average)
