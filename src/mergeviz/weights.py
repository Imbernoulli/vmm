from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    numel: int
    start: int
    end: int


@dataclass(frozen=True)
class VectorSpec:
    tensors: tuple[TensorSpec, ...]

    @property
    def numel(self) -> int:
        return self.tensors[-1].end if self.tensors else 0


def vectorize_state_dict(state_dict: dict[str, Tensor], names: Iterable[str] | None = None) -> tuple[Tensor, VectorSpec]:
    """Flatten floating tensors from a state dict into one CPU float32 vector."""
    selected = list(names) if names is not None else [
        name for name, value in state_dict.items() if torch.is_floating_point(value)
    ]
    pieces: list[Tensor] = []
    specs: list[TensorSpec] = []
    offset = 0
    for name in selected:
        value = state_dict[name].detach().cpu()
        if not torch.is_floating_point(value):
            continue
        flat = value.to(torch.float32).reshape(-1)
        next_offset = offset + flat.numel()
        specs.append(TensorSpec(name, value.shape, value.dtype, flat.numel(), offset, next_offset))
        pieces.append(flat)
        offset = next_offset
    if not pieces:
        raise ValueError("No floating tensors found to vectorize.")
    return torch.cat(pieces), VectorSpec(tuple(specs))


def vectorize_model(model: nn.Module) -> tuple[Tensor, VectorSpec]:
    return vectorize_state_dict(model.state_dict())


def state_dict_from_vector(vector: Tensor, spec: VectorSpec, reference: dict[str, Tensor]) -> dict[str, Tensor]:
    """Create a full state dict by replacing floating tensors from a vector."""
    out: dict[str, Tensor] = {}
    vector = vector.detach().cpu()
    for name, value in reference.items():
        out[name] = value.detach().clone()
    for tensor_spec in spec.tensors:
        piece = vector[tensor_spec.start:tensor_spec.end].reshape(tensor_spec.shape)
        out[tensor_spec.name] = piece.to(dtype=tensor_spec.dtype)
    return out


def load_vector_into_model(model: nn.Module, vector: Tensor, spec: VectorSpec, reference: dict[str, Tensor]) -> None:
    model.load_state_dict(state_dict_from_vector(vector, spec, reference), strict=True)


def layer_slices(spec: VectorSpec) -> dict[str, slice]:
    """Return slices keyed by tensor name.

    The first version keeps one state-dict tensor as one "layer" for maximal
    transparency in small controlled experiments.
    """
    return {item.name: slice(item.start, item.end) for item in spec.tensors}


def project_to_plane(point: Tensor, origin: Tensor, tau_a: Tensor, tau_b: Tensor) -> tuple[float, float, float]:
    """Project an arbitrary point onto span(tau_a, tau_b).

    Returns alpha, beta, and relative residual norm. Points produced by raw
    task arithmetic have near-zero residuals; masked methods such as TIES need
    projection because they leave the raw two-vector plane.
    """
    delta = (point - origin).to(torch.float64)
    basis = torch.stack([tau_a.to(torch.float64), tau_b.to(torch.float64)], dim=1)
    gram = basis.T @ basis
    rhs = basis.T @ delta
    try:
        coeff = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        coeff = torch.linalg.pinv(gram) @ rhs
    recon = basis @ coeff
    denom = torch.linalg.norm(delta).clamp_min(1e-12)
    residual = torch.linalg.norm(delta - recon) / denom
    return float(coeff[0]), float(coeff[1]), float(residual)


def cosine(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    a64 = a.detach().to(torch.float64).reshape(-1)
    b64 = b.detach().to(torch.float64).reshape(-1)
    denom = torch.linalg.norm(a64) * torch.linalg.norm(b64)
    if float(denom) <= eps:
        return 0.0
    return float((a64 @ b64) / denom)


def interpolation_barrier(losses: list[float]) -> float:
    if not losses:
        return 0.0
    endpoints = max(losses[0], losses[-1])
    return float(max(losses) - endpoints)
