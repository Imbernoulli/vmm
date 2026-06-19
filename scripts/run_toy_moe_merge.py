#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


CATEGORY_TO_SOURCE = {"general": "general", "code": "code"}


class TinyMoEClassifier(nn.Module):
    def __init__(self, input_dim: int = 4, hidden: int = 24, n_experts: int = 4, n_classes: int = 2) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(input_dim, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, n_classes),
                )
                for _ in range(n_experts)
            ]
        )

    def expert_logits(self, x: Tensor) -> Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=1)

    def router_probs(self, x: Tensor) -> Tensor:
        return F.softmax(self.router(x), dim=-1)

    def dispatch_probs(self, x: Tensor, dispatch_mode: str = "soft_all") -> Tensor:
        probs = self.router_probs(x)
        if dispatch_mode == "soft_all":
            return probs
        if dispatch_mode == "hard_top1":
            k = 1
        elif dispatch_mode == "hard_top2":
            k = min(2, self.n_experts)
        else:
            raise ValueError(f"Unknown dispatch mode: {dispatch_mode}")
        top_values, top_indices = torch.topk(probs, k=k, dim=-1)
        masked = torch.zeros_like(probs)
        masked.scatter_(1, top_indices, top_values)
        return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward_dispatch(self, x: Tensor, dispatch_mode: str = "soft_all") -> Tensor:
        probs = self.dispatch_probs(x, dispatch_mode)
        expert_logits = self.expert_logits(x)
        return torch.einsum("be,bec->bc", probs, expert_logits)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_dispatch(x)


@dataclass(frozen=True)
class MethodState:
    name: str
    state: dict[str, Tensor]
    description: str


@dataclass(frozen=True)
class ConnectivityPath:
    name: str
    left: dict[str, Tensor]
    right: dict[str, Tensor]
    candidate: dict[str, Tensor] | None
    kind: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_category_data(category: str, n: int, seed: int) -> TensorDataset:
    rng = np.random.default_rng(seed)
    xy = rng.normal(size=(n, 2)).astype("float32")
    if category == "general":
        domain = np.tile(np.array([1.0, 0.0], dtype="float32"), (n, 1))
        score = xy[:, 0] * xy[:, 1] + 0.35 * xy[:, 0] - 0.15 * xy[:, 1]
        y = (score > 0.0).astype("int64")
    elif category == "code":
        domain = np.tile(np.array([0.0, 1.0], dtype="float32"), (n, 1))
        radius = xy[:, 0] ** 2 + 0.7 * xy[:, 1] ** 2 + 0.25 * xy[:, 0]
        y = (radius > 1.15).astype("int64")
    else:
        raise ValueError(f"Unknown category: {category}")
    x = np.concatenate([xy, domain], axis=1).astype("float32")
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def concat_datasets(left: TensorDataset, right: TensorDataset) -> TensorDataset:
    lx, ly = left.tensors
    rx, ry = right.tensors
    return TensorDataset(torch.cat([lx, rx], dim=0), torch.cat([ly, ry], dim=0))


def prepare_data(seed: int, n_train_per_category: int, n_test_per_category: int, batch_size: int) -> dict[str, Any]:
    general_train = make_category_data("general", n_train_per_category, seed + 1)
    code_train = make_category_data("code", n_train_per_category, seed + 2)
    general_test = make_category_data("general", n_test_per_category, seed + 3)
    code_test = make_category_data("code", n_test_per_category, seed + 4)
    mixed_train = concat_datasets(general_train, code_train)
    mixed_test = concat_datasets(general_test, code_test)

    def loader(dataset: TensorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return {
        "general_train": loader(general_train, True),
        "code_train": loader(code_train, True),
        "mixed_train": loader(mixed_train, True),
        "general_calib": loader(general_train, False),
        "code_calib": loader(code_train, False),
        "mixed_calib": loader(mixed_train, False),
        "general_test": loader(general_test, False),
        "code_test": loader(code_test, False),
        "mixed_test": loader(mixed_test, False),
    }


def load_balance_loss(model: TinyMoEClassifier, x: Tensor) -> Tensor:
    probs = model.router_probs(x)
    mean_probs = probs.mean(dim=0)
    return model.n_experts * mean_probs.pow(2).sum()


def capacity_overflow_loss(
    model: TinyMoEClassifier,
    x: Tensor,
    *,
    dispatch_mode: str,
    capacity_factor: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if dispatch_mode == "hard_top1":
        top_k = 1
    elif dispatch_mode == "hard_top2":
        top_k = min(2, model.n_experts)
    else:
        top_k = model.n_experts
    probs = model.dispatch_probs(x, dispatch_mode)
    expected_assignments = probs.sum(dim=0) * float(top_k)
    capacity = float(capacity_factor) * float(x.shape[0] * top_k) / max(1, model.n_experts)
    overflow = F.relu(expected_assignments - capacity)
    overflow_fraction = overflow.sum() / max(1.0, float(x.shape[0] * top_k))
    max_capacity_ratio = expected_assignments.max() / max(1e-12, capacity)
    return model.n_experts * overflow_fraction.pow(2), overflow_fraction, max_capacity_ratio


def train_model(
    model: TinyMoEClassifier,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    aux_coef: float,
    device: torch.device,
    desc: str,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + aux_coef * load_balance_loss(model, x)
            loss.backward()
            optimizer.step()


def calibrate_router_only_state(
    template: TinyMoEClassifier,
    initial_state: dict[str, Tensor],
    reference_router: TinyMoEClassifier,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    kl_coef: float,
    aux_coef: float,
    device: torch.device,
    desc: str,
) -> dict[str, Tensor]:
    model = deepcopy(template)
    model.load_state_dict(initial_state)
    model.to(device)
    reference_router.to(device)
    reference_router.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.router.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=lr, weight_decay=0.0)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            router_probs = model.router_probs(x)
            with torch.no_grad():
                reference_probs = reference_router.router_probs(x)
            router_kl = F.kl_div(torch.log(router_probs.clamp_min(1e-12)), reference_probs, reduction="batchmean")
            loss = F.cross_entropy(logits, y) + kl_coef * router_kl + aux_coef * load_balance_loss(model, x)
            loss.backward()
            optimizer.step()
    return cpu_state(model)


def calibrate_router_dispatch_aware_state(
    template: TinyMoEClassifier,
    initial_state: dict[str, Tensor],
    reference_router: TinyMoEClassifier,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    dispatch_mode: str,
    dispatch_loss_coef: float,
    soft_loss_coef: float,
    kl_coef: float,
    aux_coef: float,
    device: torch.device,
    desc: str,
) -> dict[str, Tensor]:
    model = deepcopy(template)
    model.load_state_dict(initial_state)
    model.to(device)
    reference_router.to(device)
    reference_router.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.router.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=lr, weight_decay=0.0)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            soft_logits = model.forward_dispatch(x, "soft_all")
            dispatch_logits = model.forward_dispatch(x, dispatch_mode)
            router_probs = model.router_probs(x)
            with torch.no_grad():
                reference_probs = reference_router.router_probs(x)
            router_kl = F.kl_div(torch.log(router_probs.clamp_min(1e-12)), reference_probs, reduction="batchmean")
            loss = dispatch_loss_coef * F.cross_entropy(dispatch_logits, y)
            loss = loss + soft_loss_coef * F.cross_entropy(soft_logits, y)
            loss = loss + kl_coef * router_kl + aux_coef * load_balance_loss(model, x)
            loss.backward()
            optimizer.step()
    return cpu_state(model)


def calibrate_router_kd_state(
    template: TinyMoEClassifier,
    initial_state: dict[str, Tensor],
    reference_router: TinyMoEClassifier,
    teacher_loaders: list[tuple[str, TinyMoEClassifier, DataLoader, float]],
    *,
    epochs: int,
    lr: float,
    temperature: float,
    router_kl_coef: float,
    aux_coef: float,
    device: torch.device,
    desc: str,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    model = deepcopy(template)
    model.load_state_dict(initial_state)
    model.to(device)
    reference_router.to(device)
    reference_router.eval()
    teachers = []
    for source_name, teacher, loader, source_weight in teacher_loaders:
        teacher.to(device)
        teacher.eval()
        teachers.append((source_name, teacher, loader, source_weight))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.router.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=lr, weight_decay=0.0)
    rows: list[dict[str, Any]] = []
    for epoch in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        source_totals = {
            source_name: {"examples": 0, "kd_loss": 0.0, "router_kl": 0.0, "aux_loss": 0.0}
            for source_name, _teacher, _loader, _weight in teachers
        }
        for source_name, teacher, loader, source_weight in teachers:
            for x, _ in loader:
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                student_logits = model.forward_dispatch(x, "soft_all")
                with torch.no_grad():
                    teacher_logits = teacher.forward_dispatch(x, "soft_all")
                    reference_probs = reference_router.router_probs(x)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature**2)
                router_probs = model.router_probs(x)
                router_kl = F.kl_div(torch.log(router_probs.clamp_min(1e-12)), reference_probs, reduction="batchmean")
                aux_loss = load_balance_loss(model, x)
                loss = float(source_weight) * kd_loss + router_kl_coef * router_kl + aux_coef * aux_loss
                loss.backward()
                optimizer.step()
                n = int(x.shape[0])
                source_totals[source_name]["examples"] += n
                source_totals[source_name]["kd_loss"] += float(kd_loss.detach().cpu()) * n
                source_totals[source_name]["router_kl"] += float(router_kl.detach().cpu()) * n
                source_totals[source_name]["aux_loss"] += float(aux_loss.detach().cpu()) * n
        for source_name, totals in source_totals.items():
            examples = max(1, int(totals["examples"]))
            rows.append(
                {
                    "epoch": epoch,
                    "source": source_name,
                    "examples": int(totals["examples"]),
                    "temperature": float(temperature),
                    "source_weight": float(
                        next(weight for name, _teacher, _loader, weight in teachers if name == source_name)
                    ),
                    "kd_loss": totals["kd_loss"] / examples,
                    "router_kl": totals["router_kl"] / examples,
                    "aux_loss": totals["aux_loss"] / examples,
                    "same_shape_action": "router_only_label_free_kd",
                }
            )
    return cpu_state(model), pd.DataFrame(rows)


def calibrate_router_route_kd_state(
    template: TinyMoEClassifier,
    initial_state: dict[str, Tensor],
    teacher_loaders: list[tuple[str, TinyMoEClassifier, DataLoader, float]],
    *,
    epochs: int,
    lr: float,
    temperature: float,
    top1_loss_coef: float,
    aux_coef: float,
    device: torch.device,
    desc: str,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    model = deepcopy(template)
    model.load_state_dict(initial_state)
    model.to(device)
    teachers = []
    for source_name, teacher, loader, source_weight in teacher_loaders:
        teacher.to(device)
        teacher.eval()
        teachers.append((source_name, teacher, loader, source_weight))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.router.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=lr, weight_decay=0.0)
    rows: list[dict[str, Any]] = []
    for epoch in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        source_totals = {
            source_name: {"examples": 0, "route_kl": 0.0, "top1_ce": 0.0, "aux_loss": 0.0}
            for source_name, _teacher, _loader, _weight in teachers
        }
        for source_name, teacher, loader, source_weight in teachers:
            for x, _ in loader:
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                student_router_logits = model.router(x)
                with torch.no_grad():
                    teacher_router_logits = teacher.router(x)
                    teacher_top1 = teacher_router_logits.argmax(dim=-1)
                route_kl = F.kl_div(
                    F.log_softmax(student_router_logits / temperature, dim=-1),
                    F.softmax(teacher_router_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature**2)
                top1_ce = F.cross_entropy(student_router_logits, teacher_top1)
                aux_loss = load_balance_loss(model, x)
                loss = float(source_weight) * (route_kl + top1_loss_coef * top1_ce) + aux_coef * aux_loss
                loss.backward()
                optimizer.step()
                n = int(x.shape[0])
                source_totals[source_name]["examples"] += n
                source_totals[source_name]["route_kl"] += float(route_kl.detach().cpu()) * n
                source_totals[source_name]["top1_ce"] += float(top1_ce.detach().cpu()) * n
                source_totals[source_name]["aux_loss"] += float(aux_loss.detach().cpu()) * n
        for source_name, totals in source_totals.items():
            examples = max(1, int(totals["examples"]))
            rows.append(
                {
                    "epoch": epoch,
                    "source": source_name,
                    "examples": int(totals["examples"]),
                    "temperature": float(temperature),
                    "top1_loss_coef": float(top1_loss_coef),
                    "source_weight": float(
                        next(weight for name, _teacher, _loader, weight in teachers if name == source_name)
                    ),
                    "route_kl": totals["route_kl"] / examples,
                    "top1_ce": totals["top1_ce"] / examples,
                    "aux_loss": totals["aux_loss"] / examples,
                    "same_shape_action": "router_only_label_free_route_kd",
                }
            )
    return cpu_state(model), pd.DataFrame(rows)


def calibrate_unified_moe_router_state(
    template: TinyMoEClassifier,
    initial_state: dict[str, Tensor],
    reference_router: TinyMoEClassifier,
    teacher_loaders: list[tuple[str, TinyMoEClassifier, DataLoader, float]],
    *,
    epochs: int,
    lr: float,
    dispatch_mode: str,
    soft_loss_coef: float,
    dispatch_loss_coef: float,
    route_kd_coef: float,
    output_kd_coef: float,
    top1_loss_coef: float,
    base_kl_coef: float,
    load_balance_coef: float,
    capacity_loss_coef: float,
    capacity_factor: float,
    temperature: float,
    device: torch.device,
    desc: str,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    model = deepcopy(template)
    model.load_state_dict(initial_state)
    model.to(device)
    reference_router.to(device)
    reference_router.eval()
    teachers = []
    for source_name, teacher, loader, source_weight in teacher_loaders:
        teacher.to(device)
        teacher.eval()
        teachers.append((source_name, teacher, loader, source_weight))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.router.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=lr, weight_decay=0.0)
    rows: list[dict[str, Any]] = []
    for epoch in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        source_totals = {
            source_name: {
                "examples": 0,
                "soft_ce": 0.0,
                "dispatch_ce": 0.0,
                "route_kl": 0.0,
                "output_kd": 0.0,
                "top1_ce": 0.0,
                "base_kl": 0.0,
                "load_balance": 0.0,
                "capacity_overflow": 0.0,
                "capacity_ratio": 0.0,
                "total_loss": 0.0,
            }
            for source_name, _teacher, _loader, _weight in teachers
        }
        for source_name, teacher, loader, source_weight in teachers:
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                soft_logits = model.forward_dispatch(x, "soft_all")
                dispatch_logits = model.forward_dispatch(x, dispatch_mode)
                student_router_logits = model.router(x)
                student_router_probs = F.softmax(student_router_logits, dim=-1)
                with torch.no_grad():
                    teacher_router_logits = teacher.router(x)
                    teacher_router_probs = F.softmax(teacher_router_logits, dim=-1)
                    teacher_top1 = teacher_router_logits.argmax(dim=-1)
                    teacher_logits = teacher.forward_dispatch(x, "soft_all")
                    reference_probs = reference_router.router_probs(x)

                soft_ce = F.cross_entropy(soft_logits, y)
                dispatch_ce = F.cross_entropy(dispatch_logits, y)
                route_kl = F.kl_div(
                    F.log_softmax(student_router_logits / temperature, dim=-1),
                    F.softmax(teacher_router_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature**2)
                output_kd = F.kl_div(
                    F.log_softmax(soft_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature**2)
                top1_ce = F.cross_entropy(student_router_logits, teacher_top1)
                base_kl = F.kl_div(torch.log(student_router_probs.clamp_min(1e-12)), reference_probs, reduction="batchmean")
                load_balance = load_balance_loss(model, x)
                capacity_loss, capacity_overflow, capacity_ratio = capacity_overflow_loss(
                    model,
                    x,
                    dispatch_mode=dispatch_mode,
                    capacity_factor=capacity_factor,
                )
                task_loss = soft_loss_coef * soft_ce + dispatch_loss_coef * dispatch_ce
                teacher_loss = route_kd_coef * route_kl + output_kd_coef * output_kd + top1_loss_coef * top1_ce
                guard_loss = base_kl_coef * base_kl + load_balance_coef * load_balance + capacity_loss_coef * capacity_loss
                loss = float(source_weight) * (task_loss + teacher_loss) + guard_loss
                loss.backward()
                optimizer.step()

                n = int(x.shape[0])
                totals = source_totals[source_name]
                totals["examples"] += n
                totals["soft_ce"] += float(soft_ce.detach().cpu()) * n
                totals["dispatch_ce"] += float(dispatch_ce.detach().cpu()) * n
                totals["route_kl"] += float(route_kl.detach().cpu()) * n
                totals["output_kd"] += float(output_kd.detach().cpu()) * n
                totals["top1_ce"] += float(top1_ce.detach().cpu()) * n
                totals["base_kl"] += float(base_kl.detach().cpu()) * n
                totals["load_balance"] += float(load_balance.detach().cpu()) * n
                totals["capacity_overflow"] += float(capacity_overflow.detach().cpu()) * n
                totals["capacity_ratio"] += float(capacity_ratio.detach().cpu()) * n
                totals["total_loss"] += float(loss.detach().cpu()) * n
        for source_name, totals in source_totals.items():
            examples = max(1, int(totals["examples"]))
            rows.append(
                {
                    "epoch": epoch,
                    "source": source_name,
                    "examples": int(totals["examples"]),
                    "source_weight": float(
                        next(weight for name, _teacher, _loader, weight in teachers if name == source_name)
                    ),
                    "dispatch_mode": dispatch_mode,
                    "soft_loss_coef": soft_loss_coef,
                    "dispatch_loss_coef": dispatch_loss_coef,
                    "route_kd_coef": route_kd_coef,
                    "output_kd_coef": output_kd_coef,
                    "top1_loss_coef": top1_loss_coef,
                    "base_kl_coef": base_kl_coef,
                    "load_balance_coef": load_balance_coef,
                    "capacity_loss_coef": capacity_loss_coef,
                    "capacity_factor": capacity_factor,
                    "temperature": temperature,
                    "soft_ce": totals["soft_ce"] / examples,
                    "dispatch_ce": totals["dispatch_ce"] / examples,
                    "route_kl": totals["route_kl"] / examples,
                    "output_kd": totals["output_kd"] / examples,
                    "top1_ce": totals["top1_ce"] / examples,
                    "base_kl": totals["base_kl"] / examples,
                    "load_balance": totals["load_balance"] / examples,
                    "capacity_overflow": totals["capacity_overflow"] / examples,
                    "capacity_ratio": totals["capacity_ratio"] / examples,
                    "total_loss": totals["total_loss"] / examples,
                    "same_shape_action": "capacity_aware_unified_router_objective_expert_aligned_same_shape",
                }
            )
    return cpu_state(model), pd.DataFrame(rows)


@torch.no_grad()
def evaluate(
    model: TinyMoEClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    dispatch_mode: str = "soft_all",
) -> dict[str, float]:
    model.to(device)
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model.forward_dispatch(x, dispatch_mode)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").detach().cpu())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return {"loss": loss_sum / total, "acc": correct / total, "n": total}


def cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def average_states(*states: dict[str, Tensor], weights: list[float] | None = None) -> dict[str, Tensor]:
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    out: dict[str, Tensor] = {}
    for name in states[0]:
        value = torch.zeros_like(states[0][name], dtype=torch.float32)
        for state, weight in zip(states, weights):
            value = value + float(weight) * state[name].to(torch.float32)
        out[name] = value.to(dtype=states[0][name].dtype)
    return out


def task_vector_average(base: dict[str, Tensor], sources: list[dict[str, Tensor]], weights: list[float]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name in base:
        value = base[name].to(torch.float32)
        for source, weight in zip(sources, weights):
            value = value + float(weight) * (source[name].to(torch.float32) - base[name].to(torch.float32))
        out[name] = value.to(dtype=base[name].dtype)
    return out


def interpolate_states(left: dict[str, Tensor], right: dict[str, Tensor], t: float) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, left_value in left.items():
        right_value = right[name]
        if torch.is_floating_point(left_value):
            value = (1.0 - t) * left_value.to(torch.float32) + t * right_value.to(torch.float32)
            out[name] = value.to(dtype=left_value.dtype)
        else:
            out[name] = left_value.clone()
    return out


def permute_experts_and_router(model: TinyMoEClassifier, order: list[int]) -> TinyMoEClassifier:
    permuted = deepcopy(model).cpu()
    with torch.no_grad():
        for new_idx, old_idx in enumerate(order):
            permuted.experts[new_idx].load_state_dict(model.experts[old_idx].state_dict())
        permuted.router.weight.copy_(model.router.weight.detach().cpu()[order])
        permuted.router.bias.copy_(model.router.bias.detach().cpu()[order])
    return permuted


@torch.no_grad()
def expert_output_features(model: TinyMoEClassifier, loader: DataLoader, device: torch.device, max_batches: int) -> Tensor:
    model.to(device)
    model.eval()
    chunks: list[Tensor] = []
    batches = 0
    for x, _ in loader:
        x = x.to(device)
        chunks.append(model.expert_logits(x).detach().cpu())
        batches += 1
        if batches >= max_batches:
            break
    logits = torch.cat(chunks, dim=0)
    features = logits.transpose(0, 1).reshape(model.n_experts, -1)
    return F.normalize(features, dim=1)


def match_experts(
    reference: TinyMoEClassifier,
    target: TinyMoEClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[TinyMoEClassifier, pd.DataFrame]:
    ref_features = expert_output_features(reference, loader, device, max_batches)
    target_features = expert_output_features(target, loader, device, max_batches)
    similarity = ref_features @ target_features.T
    rows, cols = linear_sum_assignment((-similarity).numpy())
    order = [0] * target.n_experts
    match_rows = []
    for ref_idx, target_idx in zip(rows, cols):
        order[int(ref_idx)] = int(target_idx)
        match_rows.append(
            {
                "reference_expert": int(ref_idx),
                "target_expert_before_alignment": int(target_idx),
                "output_cosine": float(similarity[ref_idx, target_idx].item()),
            }
        )
    return permute_experts_and_router(target, order), pd.DataFrame(match_rows)


def route_mass_weights(
    base_model: TinyMoEClassifier,
    loaders: dict[str, DataLoader],
    device: torch.device,
    anchor_floor: float,
) -> pd.DataFrame:
    rows = []
    per_expert: dict[int, dict[str, float]] = {
        expert_id: {"general": 0.0, "code": 0.0} for expert_id in range(base_model.n_experts)
    }
    for category in ("general", "code"):
        stats = router_stats(base_model, loaders[f"{category}_test"], device, method="base", category=category)
        for row in stats["expert_rows"]:
            per_expert[int(row["expert_id"])][CATEGORY_TO_SOURCE[category]] += float(row["topk_fraction"])
    for expert_id, masses in per_expert.items():
        total = masses["general"] + masses["code"]
        if total <= 0:
            general_weight = 0.0
            code_weight = 0.0
            action = "anchor_heavy_or_freeze"
        else:
            scale = 1.0 - anchor_floor
            general_weight = scale * masses["general"] / total
            code_weight = scale * masses["code"] / total
            action = "route_frequency_weighted_average"
        rows.append(
            {
                "expert_id": expert_id,
                "route_mass_general": masses["general"],
                "route_mass_code": masses["code"],
                "weight_general": general_weight,
                "weight_code": code_weight,
                "anchor_floor": anchor_floor,
                "same_shape_action": action,
            }
        )
    return pd.DataFrame(rows)


def route_aware_state(
    base: dict[str, Tensor],
    general: dict[str, Tensor],
    code: dict[str, Tensor],
    weights: pd.DataFrame,
    n_experts: int,
) -> dict[str, Tensor]:
    out = {name: value.clone() for name, value in base.items()}
    weight_by_expert = {
        int(row["expert_id"]): (float(row["weight_general"]), float(row["weight_code"]))
        for _, row in weights.iterrows()
    }
    for name in base:
        if name.startswith("router."):
            out[name] = base[name].clone()
            continue
        expert_id = None
        for idx in range(n_experts):
            if name.startswith(f"experts.{idx}."):
                expert_id = idx
                break
        if expert_id is None:
            out[name] = task_vector_average(base, [general, code], [0.5, 0.5])[name]
            continue
        general_weight, code_weight = weight_by_expert.get(expert_id, (0.0, 0.0))
        value = base[name].to(torch.float32)
        value = value + general_weight * (general[name].to(torch.float32) - base[name].to(torch.float32))
        value = value + code_weight * (code[name].to(torch.float32) - base[name].to(torch.float32))
        out[name] = value.to(dtype=base[name].dtype)
    return out


def parse_grid(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("expert search grid must contain at least one value")
    return values


def parse_dispatch_modes(raw: str) -> list[str]:
    allowed = {"soft_all", "hard_top1", "hard_top2"}
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    if not modes:
        raise ValueError("dispatch eval modes must contain at least one mode")
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown dispatch eval modes: {unknown}; allowed={sorted(allowed)}")
    return modes


def expert_id_for_tensor(name: str, n_experts: int) -> int | None:
    for idx in range(n_experts):
        if name.startswith(f"experts.{idx}."):
            return idx
    return None


def candidate_source_pairs(grid: list[float], max_delta_sum: float) -> list[tuple[float, float]]:
    pairs = [
        (general_weight, code_weight)
        for general_weight in grid
        for code_weight in grid
        if general_weight + code_weight <= max_delta_sum + 1e-9
    ]
    if not pairs:
        raise ValueError("expert search grid and max-delta-sum produced no valid candidate pairs")
    return sorted(pairs, key=lambda item: (item[0] + item[1], item[0], item[1]))


def expert_coeff_state(
    base: dict[str, Tensor],
    general: dict[str, Tensor],
    code: dict[str, Tensor],
    *,
    n_experts: int,
    expert_weights: dict[int, tuple[float, float]],
    shared_weights: tuple[float, float],
) -> dict[str, Tensor]:
    shared_general, shared_code = shared_weights
    out: dict[str, Tensor] = {}
    for name, base_value in base.items():
        if name.startswith("router."):
            out[name] = base_value.clone()
            continue
        expert_id = expert_id_for_tensor(name, n_experts)
        if expert_id is None:
            general_weight, code_weight = shared_general, shared_code
        else:
            general_weight, code_weight = expert_weights.get(expert_id, (0.0, 0.0))
        value = base_value.to(torch.float32)
        value = value + general_weight * (general[name].to(torch.float32) - base_value.to(torch.float32))
        value = value + code_weight * (code[name].to(torch.float32) - base_value.to(torch.float32))
        out[name] = value.to(dtype=base_value.dtype)
    return out


def project_two_source_weights(solution: Tensor, max_delta_sum: float) -> tuple[float, float]:
    x0 = float(solution[0].item())
    x1 = float(solution[1].item())
    if x0 >= 0.0 and x1 >= 0.0 and x0 + x1 <= max_delta_sum:
        return x0, x1
    clipped0 = max(0.0, x0)
    clipped1 = max(0.0, x1)
    if clipped0 + clipped1 <= max_delta_sum:
        return clipped0, clipped1
    projected0 = 0.5 * (max_delta_sum + x0 - x1)
    projected0 = min(max(projected0, 0.0), max_delta_sum)
    return projected0, max_delta_sum - projected0


@torch.no_grad()
def output_space_expert_weight_state(
    base_model: TinyMoEClassifier,
    general_model: TinyMoEClassifier,
    code_model: TinyMoEClassifier,
    base: dict[str, Tensor],
    general: dict[str, Tensor],
    code: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    *,
    n_experts: int,
    shared_weights: tuple[float, float],
    dispatch_mode: str,
    ridge: float,
    max_delta_sum: float,
    device: torch.device,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    source_specs = [
        ("general", general_model, loaders["general_calib"], 0.5),
        ("code", code_model, loaders["code_calib"], 0.5),
    ]
    models = [base_model, general_model, code_model]
    for model in models:
        model.to(device)
        model.eval()
    weights: dict[int, tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    for expert_id in range(n_experts):
        ata = torch.zeros((2, 2), dtype=torch.float64)
        atb = torch.zeros(2, dtype=torch.float64)
        btb = torch.tensor(0.0, dtype=torch.float64)
        route_mass = 0.0
        tokens = 0
        for category, target_model, loader, source_weight in source_specs:
            category_ata = torch.zeros((2, 2), dtype=torch.float64)
            category_atb = torch.zeros(2, dtype=torch.float64)
            category_btb = torch.tensor(0.0, dtype=torch.float64)
            category_route_mass = 0.0
            category_tokens = 0
            for x, _y in loader:
                x = x.to(device)
                base_logits = base_model.expert_logits(x)[:, expert_id, :]
                general_logits = general_model.expert_logits(x)[:, expert_id, :]
                code_logits = code_model.expert_logits(x)[:, expert_id, :]
                target_logits = target_model.expert_logits(x)[:, expert_id, :]
                route_probs = base_model.dispatch_probs(x, dispatch_mode)[:, expert_id].detach()
                sample_weight = (float(source_weight) * route_probs).clamp_min(0.0)
                design = torch.stack(
                    [
                        general_logits - base_logits,
                        code_logits - base_logits,
                    ],
                    dim=-1,
                ).detach().to(torch.float64)
                target = (target_logits - base_logits).detach().to(torch.float64)
                weighted_design = design * sample_weight.to(torch.float64).view(-1, 1, 1)
                category_ata = category_ata + torch.einsum("nci,ncj->ij", weighted_design, design)
                category_atb = category_atb + torch.einsum("nci,nc->i", weighted_design, target)
                category_btb = category_btb + (target.pow(2) * sample_weight.to(torch.float64).view(-1, 1)).sum()
                category_route_mass += float(route_probs.sum().item())
                category_tokens += int(x.shape[0])
            ata = ata + category_ata
            atb = atb + category_atb
            btb = btb + category_btb
            route_mass += category_route_mass
            tokens += category_tokens
            rows.append(
                {
                    "expert_id": expert_id,
                    "category": category,
                    "tokens": category_tokens,
                    "route_mass": category_route_mass,
                    "target_delta_energy": float(category_btb.item()),
                    "same_shape_action": "route_conditioned_output_space_probe",
                }
            )
        ridge_value = max(float(ridge), float(ridge) * float(torch.trace(ata).item()) / max(1, ata.shape[0]))
        solution = torch.linalg.solve(ata + torch.eye(2, dtype=torch.float64) * ridge_value, atb)
        general_weight, code_weight = project_two_source_weights(solution, max_delta_sum)
        weight_vector = torch.tensor([general_weight, code_weight], dtype=torch.float64)
        residual_energy = float((btb - 2.0 * (weight_vector @ atb) + weight_vector @ ata @ weight_vector).clamp_min(0.0).item())
        target_energy = float(btb.item())
        captured_fraction = 1.0 - residual_energy / max(1e-12, target_energy)
        weights[expert_id] = (general_weight, code_weight)
        rows.append(
            {
                "expert_id": expert_id,
                "category": "combined",
                "tokens": tokens,
                "route_mass": route_mass,
                "target_delta_energy": target_energy,
                "residual_energy": residual_energy,
                "captured_fraction": captured_fraction,
                "unconstrained_weight_general": float(solution[0].item()),
                "unconstrained_weight_code": float(solution[1].item()),
                "weight_general": general_weight,
                "weight_code": code_weight,
                "anchor_weight": 1.0 - general_weight - code_weight,
                "ridge_value": ridge_value,
                "dispatch_mode": dispatch_mode,
                "same_shape_action": "route_conditioned_output_space_expert_delta",
            }
        )
    state = expert_coeff_state(
        base,
        general,
        code,
        n_experts=n_experts,
        expert_weights=weights,
        shared_weights=shared_weights,
    )
    return state, pd.DataFrame(rows)


@torch.no_grad()
def collect_linear_covariances(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    *,
    layer_prefix: str,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    model.to(device)
    model.eval()
    covariances: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def hook_for(name: str):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float64).cpu()
            ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
            augmented = torch.cat([x, ones], dim=1)
            if name not in covariances:
                covariances[name] = torch.zeros((augmented.shape[1], augmented.shape[1]), dtype=torch.float64)
            covariances[name] = covariances[name] + augmented.T @ augmented
            counts[name] = counts.get(name, 0) + int(augmented.shape[0])

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.startswith(layer_prefix):
            handles.append(module.register_forward_hook(hook_for(name)))

    batches = 0
    for x, _y in loader:
        model(x.to(device))
        batches += 1
        if batches >= max_batches:
            break

    for handle in handles:
        handle.remove()

    rows = []
    for name, cov in sorted(covariances.items()):
        rows.append(
            {
                "layer": name,
                "examples": counts[name],
                "augmented_dim": int(cov.shape[0]),
                "trace": float(torch.trace(cov)),
                "condition": float(torch.linalg.cond(cov + torch.eye(cov.shape[0], dtype=cov.dtype) * 1e-8)),
            }
        )
    return covariances, pd.DataFrame(rows)


def regmean_expert_state(
    initial_state: dict[str, Tensor],
    source_states: list[dict[str, Tensor]],
    covariances: list[dict[str, Tensor]],
    *,
    ridge: float,
    layer_prefix: str,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    merged = {name: value.detach().clone() for name, value in initial_state.items()}
    rows = []
    linear_layers = sorted(
        {
            name.rsplit(".", 1)[0]
            for name in initial_state
            if name.startswith(layer_prefix) and name.endswith(".weight")
        }
    )
    for layer in linear_layers:
        weight_name = f"{layer}.weight"
        bias_name = f"{layer}.bias"
        if bias_name not in initial_state:
            continue
        if not all(layer in cov for cov in covariances):
            continue
        if not all(weight_name in state and bias_name in state for state in source_states):
            continue

        denom = None
        numerator = None
        for state, cov_by_layer in zip(source_states, covariances, strict=True):
            cov = cov_by_layer[layer].to(torch.float64)
            weight = state[weight_name].detach().cpu().to(torch.float64)
            bias = state[bias_name].detach().cpu().to(torch.float64).reshape(-1, 1)
            augmented_weight = torch.cat([weight, bias], dim=1)
            denom = cov if denom is None else denom + cov
            part = augmented_weight @ cov
            numerator = part if numerator is None else numerator + part

        assert denom is not None and numerator is not None
        dim = denom.shape[0]
        ridge_value = ridge * float(torch.trace(denom)) / max(1, dim)
        system = denom + torch.eye(dim, dtype=torch.float64) * max(ridge_value, ridge)
        merged_augmented = torch.linalg.solve(system.T, numerator.T).T
        merged[weight_name] = merged_augmented[:, :-1].to(dtype=initial_state[weight_name].dtype)
        merged[bias_name] = merged_augmented[:, -1].to(dtype=initial_state[bias_name].dtype)
        rows.append(
            {
                "layer": layer,
                "out_features": int(merged[weight_name].shape[0]),
                "in_features": int(merged[weight_name].shape[1]),
                "ridge_value": float(max(ridge_value, ridge)),
                "same_shape_action": "regmean_expert_linear_layer",
            }
        )
    return merged, pd.DataFrame(rows)


def topk_mask(delta: Tensor, density: float) -> Tensor:
    if density >= 1.0:
        return torch.ones_like(delta, dtype=torch.bool)
    if density <= 0.0:
        return torch.zeros_like(delta, dtype=torch.bool)
    flat = delta.abs().reshape(-1)
    k = max(1, int(math.ceil(flat.numel() * density)))
    if k >= flat.numel():
        return torch.ones_like(delta, dtype=torch.bool)
    threshold = torch.topk(flat, k=k, largest=True).values[-1]
    return delta.abs() >= threshold


def stable_tensor_seed(seed: int, name: str, source_idx: int) -> int:
    return int(seed + source_idx * 1_000_003 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(name)))


def dare_delta(delta: Tensor, drop_rate: float, seed: int) -> Tensor:
    if drop_rate <= 0.0:
        return delta.clone()
    if drop_rate >= 1.0:
        return torch.zeros_like(delta)
    generator = torch.Generator(device=delta.device)
    generator.manual_seed(seed)
    keep = torch.rand(delta.shape, generator=generator, device=delta.device) >= drop_rate
    return delta * keep.to(delta.dtype) / (1.0 - drop_rate)


def ties_delta(deltas: list[Tensor], density: float) -> Tensor:
    trimmed = [delta * topk_mask(delta, density).to(delta.dtype) for delta in deltas]
    stacked = torch.stack(trimmed, dim=0)
    elected_sign = torch.sign(stacked.sum(dim=0))
    nonzero_elected = elected_sign != 0
    sign_match = torch.sign(stacked) == elected_sign.unsqueeze(0)
    mask = sign_match & nonzero_elected.unsqueeze(0)
    numerator = (stacked * mask.to(stacked.dtype)).sum(dim=0)
    denom = mask.sum(dim=0).clamp_min(1).to(stacked.dtype)
    merged = numerator / denom
    return torch.where(nonzero_elected, merged, torch.zeros_like(merged))


def expert_sparse_task_vector_state(
    base: dict[str, Tensor],
    general: dict[str, Tensor],
    code: dict[str, Tensor],
    *,
    mode: str,
    density: float,
    drop_rate: float,
    seed: int,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    out = {name: value.detach().clone() for name, value in base.items()}
    rows = []
    for name, base_value in base.items():
        if name.startswith("router.") or not name.startswith("experts.") or not torch.is_floating_point(base_value):
            continue
        source_deltas = [
            general[name].detach().cpu().to(torch.float32) - base_value.detach().cpu().to(torch.float32),
            code[name].detach().cpu().to(torch.float32) - base_value.detach().cpu().to(torch.float32),
        ]
        deltas = source_deltas
        if mode in {"dare", "ties_dare"}:
            deltas = [
                dare_delta(delta, drop_rate, stable_tensor_seed(seed, name, idx))
                for idx, delta in enumerate(source_deltas)
            ]
        if mode == "ties":
            merged_delta = ties_delta(deltas, density)
        elif mode == "dare":
            merged_delta = torch.stack(deltas, dim=0).mean(dim=0)
        elif mode == "ties_dare":
            merged_delta = ties_delta(deltas, density)
        else:
            raise ValueError(f"Unknown sparse expert mode: {mode}")
        out[name] = (base_value.detach().cpu().to(torch.float32) + merged_delta).to(dtype=base_value.dtype)
        rows.append(
            {
                "method": f"expert_matched_{mode}_average",
                "tensor": name,
                "density": density if mode in {"ties", "ties_dare"} else 1.0,
                "drop_rate": drop_rate if mode in {"dare", "ties_dare"} else 0.0,
                "source_delta_norm_mean": float(torch.stack([delta.norm() for delta in source_deltas]).mean()),
                "merged_delta_norm": float(merged_delta.norm()),
                "merged_delta_nonzero_fraction": float((merged_delta != 0).to(torch.float32).mean()),
                "same_shape_action": "expert_local_sparse_task_vector_merge",
            }
        )
    return out, pd.DataFrame(rows)


def evaluate_state(
    template: TinyMoEClassifier,
    state: dict[str, Tensor],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model = deepcopy(template)
    model.load_state_dict(state)
    return evaluate(model, loader, device)


def evaluate_state_by_category(
    template: TinyMoEClassifier,
    state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    split_prefix: str,
) -> dict[str, float]:
    model = deepcopy(template)
    model.load_state_dict(state)
    general = evaluate(model, loaders[f"general_{split_prefix}"], device)
    code = evaluate(model, loaders[f"code_{split_prefix}"], device)
    return {
        f"{split_prefix}_general_loss": general["loss"],
        f"{split_prefix}_code_loss": code["loss"],
        f"{split_prefix}_avg_loss": 0.5 * (general["loss"] + code["loss"]),
        f"{split_prefix}_worst_loss": max(general["loss"], code["loss"]),
        f"{split_prefix}_general_acc": general["acc"],
        f"{split_prefix}_code_acc": code["acc"],
        f"{split_prefix}_avg_acc": 0.5 * (general["acc"] + code["acc"]),
        f"{split_prefix}_worst_acc": min(general["acc"], code["acc"]),
    }


def search_expert_source_weights(
    template: TinyMoEClassifier,
    base: dict[str, Tensor],
    general: dict[str, Tensor],
    code: dict[str, Tensor],
    route_weight_prior: pd.DataFrame,
    loaders: dict[str, DataLoader],
    *,
    n_experts: int,
    candidate_pairs: list[tuple[float, float]],
    passes: int,
    prior_penalty: float,
    shared_weights: tuple[float, float],
    objective: str,
    device: torch.device,
) -> tuple[dict[str, Tensor], pd.DataFrame, pd.DataFrame]:
    prior = {
        int(row["expert_id"]): (float(row["weight_general"]), float(row["weight_code"]))
        for _, row in route_weight_prior.iterrows()
    }
    weights = {expert_id: prior.get(expert_id, (0.0, 0.0)) for expert_id in range(n_experts)}

    def objective_for(candidate_weights: dict[int, tuple[float, float]]) -> tuple[float, dict[str, float]]:
        state = expert_coeff_state(
            base,
            general,
            code,
            n_experts=n_experts,
            expert_weights=candidate_weights,
            shared_weights=shared_weights,
        )
        general_metrics = evaluate_state(template, state, loaders["general_calib"], device)
        code_metrics = evaluate_state(template, state, loaders["code_calib"], device)
        metrics = {
            "general_loss": general_metrics["loss"],
            "code_loss": code_metrics["loss"],
            "avg_loss": 0.5 * (general_metrics["loss"] + code_metrics["loss"]),
            "worst_loss": max(general_metrics["loss"], code_metrics["loss"]),
            "general_acc": general_metrics["acc"],
            "code_acc": code_metrics["acc"],
            "avg_acc": 0.5 * (general_metrics["acc"] + code_metrics["acc"]),
            "worst_acc": min(general_metrics["acc"], code_metrics["acc"]),
        }
        if objective not in metrics:
            raise ValueError(f"Unknown expert search objective: {objective}")
        penalty = 0.0
        for expert_id, (general_weight, code_weight) in candidate_weights.items():
            prior_general, prior_code = prior.get(expert_id, (0.0, 0.0))
            penalty += (general_weight - prior_general) ** 2 + (code_weight - prior_code) ** 2
        metrics["prior_penalty"] = penalty
        metrics["objective"] = metrics[objective] + prior_penalty * penalty
        return float(metrics["objective"]), metrics

    trace_rows: list[dict[str, Any]] = []
    for pass_idx in range(passes):
        for expert_id in range(n_experts):
            previous_general, previous_code = weights[expert_id]
            best_weights = dict(weights)
            best_objective, best_metrics = objective_for(best_weights)
            for general_weight, code_weight in candidate_pairs:
                trial_weights = dict(weights)
                trial_weights[expert_id] = (general_weight, code_weight)
                trial_objective, metrics = objective_for(trial_weights)
                if trial_objective < best_objective:
                    best_objective = trial_objective
                    best_metrics = metrics
                    best_weights = trial_weights
            weights = best_weights
            selected_general, selected_code = weights[expert_id]
            prior_general, prior_code = prior.get(expert_id, (0.0, 0.0))
            trace_rows.append(
                {
                    "pass": pass_idx,
                    "expert_id": expert_id,
                    "previous_weight_general": previous_general,
                    "previous_weight_code": previous_code,
                    "selected_weight_general": selected_general,
                    "selected_weight_code": selected_code,
                    "prior_weight_general": prior_general,
                    "prior_weight_code": prior_code,
                    "calibration_general_loss": best_metrics["general_loss"],
                    "calibration_code_loss": best_metrics["code_loss"],
                    "calibration_avg_loss": best_metrics["avg_loss"],
                    "calibration_worst_loss": best_metrics["worst_loss"],
                    "calibration_general_acc": best_metrics["general_acc"],
                    "calibration_code_acc": best_metrics["code_acc"],
                    "calibration_worst_acc": best_metrics["worst_acc"],
                    "prior_penalty": best_metrics["prior_penalty"],
                    "objective_name": objective,
                    "objective": best_metrics["objective"],
                    "changed": (selected_general, selected_code) != (previous_general, previous_code),
                }
            )

    state = expert_coeff_state(
        base,
        general,
        code,
        n_experts=n_experts,
        expert_weights=weights,
        shared_weights=shared_weights,
    )
    rows = []
    for expert_id in range(n_experts):
        general_weight, code_weight = weights[expert_id]
        prior_general, prior_code = prior.get(expert_id, (0.0, 0.0))
        rows.append(
            {
                "expert_id": expert_id,
                "weight_general": general_weight,
                "weight_code": code_weight,
                "anchor_weight": 1.0 - general_weight - code_weight,
                "prior_weight_general": prior_general,
                "prior_weight_code": prior_code,
                "prior_anchor_weight": 1.0 - prior_general - prior_code,
                "same_shape_action": "calibration_searched_expert_delta",
            }
        )
    return state, pd.DataFrame(rows), pd.DataFrame(trace_rows)


def sweep_router_calibration(
    template: TinyMoEClassifier,
    initial_state: dict[str, Tensor],
    reference_router: TinyMoEClassifier,
    base_model: TinyMoEClassifier,
    loaders: dict[str, DataLoader],
    *,
    kl_values: list[float],
    epochs: int,
    lr: float,
    aux_coef: float,
    min_topk_jaccard: float,
    min_top1_agreement: float,
    device: torch.device,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, Tensor]] = {}
    for kl_coef in kl_values:
        kl_key = f"{kl_coef:g}"
        state = calibrate_router_only_state(
            template,
            initial_state,
            reference_router,
            loaders["mixed_calib"],
            epochs=epochs,
            lr=lr,
            kl_coef=kl_coef,
            aux_coef=aux_coef,
            device=device,
            desc=f"sweep router kl={kl_coef:g}",
        )
        states[kl_key] = state
        metrics = {
            "kl_coef": kl_coef,
            **evaluate_state_by_category(template, state, loaders, device, split_prefix="calib"),
            **evaluate_state_by_category(template, state, loaders, device, split_prefix="test"),
        }
        model = deepcopy(template)
        model.load_state_dict(state)
        overlaps = [
            route_overlap(
                base_model,
                model,
                loaders[f"{category}_test"],
                device,
                left_name="base",
                right_name=f"router_sweep_kl_{kl_key}",
                category=category,
            )
            for category in ("general", "code")
        ]
        router_summaries = [
            router_stats(model, loaders[f"{category}_test"], device, method=f"router_sweep_kl_{kl_key}", category=category)[
                "summary_row"
            ]
            for category in ("general", "code")
        ]
        metrics.update(
            {
                "min_test_top1_agreement": min(float(item["top1_agreement"]) for item in overlaps),
                "min_test_topk_jaccard": min(float(item["topk_jaccard"]) for item in overlaps),
                "max_test_top1_fraction": max(float(item["max_top1_fraction"]) for item in router_summaries),
                "mean_test_router_entropy": float(np.mean([item["router_entropy_mean"] for item in router_summaries])),
            }
        )
        rows.append(metrics)
    sweep = pd.DataFrame(rows)
    sweep["eligible_by_route_guard"] = (
        (sweep["min_test_topk_jaccard"] >= min_topk_jaccard)
        & (sweep["min_test_top1_agreement"] >= min_top1_agreement)
    )
    candidate_rows = sweep[sweep["eligible_by_route_guard"]]
    if candidate_rows.empty:
        candidate_rows = sweep
    candidate_rows = candidate_rows.sort_values(["calib_worst_loss", "calib_avg_loss", "kl_coef"], ascending=[True, True, True])
    selected_key = f"{float(candidate_rows.iloc[0]['kl_coef']):g}"
    sweep["selected_by_guarded_calib_worst_loss"] = sweep["kl_coef"].map(lambda value: f"{float(value):g}" == selected_key)
    return states[selected_key], sweep.sort_values("kl_coef")


def router_source_weight_state(
    initial_state: dict[str, Tensor],
    base: dict[str, Tensor],
    general: dict[str, Tensor],
    code: dict[str, Tensor],
    *,
    general_weight: float,
    code_weight: float,
) -> dict[str, Tensor]:
    state = {name: value.clone() for name, value in initial_state.items()}
    for name, base_value in base.items():
        if not name.startswith("router."):
            continue
        value = base_value.to(torch.float32)
        value = value + general_weight * (general[name].to(torch.float32) - base_value.to(torch.float32))
        value = value + code_weight * (code[name].to(torch.float32) - base_value.to(torch.float32))
        state[name] = value.to(dtype=base_value.dtype)
    return state


@torch.no_grad()
def hessian_router_average_state(
    initial_state: dict[str, Tensor],
    sources: list[tuple[str, TinyMoEClassifier, DataLoader, float]],
    *,
    ridge: float,
    device: torch.device,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    state = {name: value.clone() for name, value in initial_state.items()}
    n_experts = int(initial_state["router.bias"].numel())
    input_dim = int(initial_state["router.weight"].shape[1])
    aug_dim = input_dim + 1
    system_dim = n_experts * aug_dim
    precision = torch.zeros((system_dim, system_dim), dtype=torch.float64)
    rhs = torch.zeros(system_dim, dtype=torch.float64)
    source_rows: list[dict[str, Any]] = []

    for source_name, model, loader, source_weight in sources:
        model.to(device)
        model.eval()
        router_aug = torch.cat(
            [
                model.router.weight.detach().cpu().to(torch.float64),
                model.router.bias.detach().cpu().to(torch.float64).unsqueeze(1),
            ],
            dim=1,
        )
        router_vec = router_aug.reshape(-1)
        source_precision = torch.zeros_like(precision)
        tokens = 0
        entropy_sum = 0.0
        hessian_trace_sum = 0.0
        for x, _ in loader:
            x = x.to(device)
            probs = model.router_probs(x).detach().cpu().to(torch.float64)
            x_aug = torch.cat(
                [
                    x.detach().cpu().to(torch.float64),
                    torch.ones((x.shape[0], 1), dtype=torch.float64),
                ],
                dim=1,
            )
            for sample_probs, sample_x in zip(probs, x_aug, strict=True):
                hessian = torch.diag(sample_probs) - torch.outer(sample_probs, sample_probs)
                source_precision = source_precision + float(source_weight) * torch.kron(
                    hessian,
                    torch.outer(sample_x, sample_x),
                )
                entropy_sum += float((-(sample_probs * torch.log(sample_probs.clamp_min(1e-12))).sum()).item())
                hessian_trace_sum += float(torch.trace(hessian).item())
                tokens += 1
        precision = precision + source_precision
        rhs = rhs + source_precision @ router_vec
        source_rows.append(
            {
                "source": source_name,
                "tokens": tokens,
                "source_weight": float(source_weight),
                "router_entropy_mean": entropy_sum / max(1, tokens),
                "router_hessian_trace_mean": hessian_trace_sum / max(1, tokens),
                "precision_trace": float(torch.trace(source_precision).item()),
                "same_shape_action": "hessian_router_distribution_average",
            }
        )

    scale = float(torch.trace(precision).item()) / max(1, system_dim)
    ridge_value = max(float(ridge), float(ridge) * scale)
    solved = torch.linalg.solve(
        precision + torch.eye(system_dim, dtype=torch.float64) * ridge_value,
        rhs,
    ).reshape(n_experts, aug_dim)
    state["router.weight"] = solved[:, :-1].to(dtype=initial_state["router.weight"].dtype)
    state["router.bias"] = solved[:, -1].to(dtype=initial_state["router.bias"].dtype)
    source_rows.append(
        {
            "source": "solved_router",
            "tokens": sum(int(row["tokens"]) for row in source_rows),
            "source_weight": sum(float(row["source_weight"]) for row in source_rows),
            "router_entropy_mean": None,
            "router_hessian_trace_mean": None,
            "precision_trace": float(torch.trace(precision).item()),
            "ridge_value": ridge_value,
            "system_dim": system_dim,
            "same_shape_action": "closed_form_hessian_router_average",
        }
    )
    return state, pd.DataFrame(source_rows)


def search_router_source_weights(
    template: TinyMoEClassifier,
    initial_state: dict[str, Tensor],
    base: dict[str, Tensor],
    general: dict[str, Tensor],
    code: dict[str, Tensor],
    base_model: TinyMoEClassifier,
    loaders: dict[str, DataLoader],
    *,
    candidate_pairs: list[tuple[float, float]],
    min_topk_jaccard: float,
    min_top1_agreement: float,
    device: torch.device,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    states: dict[tuple[float, float], dict[str, Tensor]] = {}
    for general_weight, code_weight in candidate_pairs:
        state = router_source_weight_state(
            initial_state,
            base,
            general,
            code,
            general_weight=general_weight,
            code_weight=code_weight,
        )
        states[(general_weight, code_weight)] = state
        metrics = {
            "router_weight_general": general_weight,
            "router_weight_code": code_weight,
            "router_anchor_weight": 1.0 - general_weight - code_weight,
            **evaluate_state_by_category(template, state, loaders, device, split_prefix="calib"),
            **evaluate_state_by_category(template, state, loaders, device, split_prefix="test"),
        }
        model = deepcopy(template)
        model.load_state_dict(state)
        method_name = f"router_weight_g{general_weight:g}_c{code_weight:g}"
        overlaps = [
            route_overlap(
                base_model,
                model,
                loaders[f"{category}_test"],
                device,
                left_name="base",
                right_name=method_name,
                category=category,
            )
            for category in ("general", "code")
        ]
        router_summaries = [
            router_stats(model, loaders[f"{category}_test"], device, method=method_name, category=category)[
                "summary_row"
            ]
            for category in ("general", "code")
        ]
        metrics.update(
            {
                "min_test_top1_agreement": min(float(item["top1_agreement"]) for item in overlaps),
                "min_test_topk_jaccard": min(float(item["topk_jaccard"]) for item in overlaps),
                "max_test_top1_fraction": max(float(item["max_top1_fraction"]) for item in router_summaries),
                "mean_test_router_entropy": float(np.mean([item["router_entropy_mean"] for item in router_summaries])),
            }
        )
        rows.append(metrics)
    search = pd.DataFrame(rows)
    search["eligible_by_route_guard"] = (
        (search["min_test_topk_jaccard"] >= min_topk_jaccard)
        & (search["min_test_top1_agreement"] >= min_top1_agreement)
    )
    candidate_rows = search[search["eligible_by_route_guard"]]
    if candidate_rows.empty:
        candidate_rows = search
    candidate_rows = candidate_rows.sort_values(
        ["calib_worst_loss", "calib_avg_loss", "router_weight_general", "router_weight_code"],
        ascending=[True, True, True, True],
    )
    selected = candidate_rows.iloc[0]
    selected_pair = (float(selected["router_weight_general"]), float(selected["router_weight_code"]))
    search["selected_by_guarded_calib_worst_loss"] = search.apply(
        lambda row: (
            float(row["router_weight_general"]) == selected_pair[0]
            and float(row["router_weight_code"]) == selected_pair[1]
        ),
        axis=1,
    )
    return states[selected_pair], search.sort_values(["router_weight_general", "router_weight_code"])


def state_on_connectivity_path(path: ConnectivityPath, t: float) -> tuple[dict[str, Tensor], str, float]:
    if path.candidate is None:
        return interpolate_states(path.left, path.right, t), "direct", t
    if t <= 0.5:
        local_t = 2.0 * t
        return interpolate_states(path.left, path.candidate, local_t), "left_to_candidate", local_t
    local_t = 2.0 * t - 1.0
    return interpolate_states(path.candidate, path.right, local_t), "candidate_to_right", local_t


def evaluate_connectivity_paths(
    template: TinyMoEClassifier,
    paths: list[ConnectivityPath],
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    grid = np.linspace(0.0, 1.0, steps)
    for path in paths:
        for t in grid:
            state, segment, local_t = state_on_connectivity_path(path, float(t))
            metrics = evaluate_state_by_category(template, state, loaders, device, split_prefix="test")
            metric_rows.append(
                {
                    "path": path.name,
                    "kind": path.kind,
                    "t": float(t),
                    "segment": segment,
                    "local_t": float(local_t),
                    **metrics,
                }
            )
    metrics_df = pd.DataFrame(metric_rows)
    summary_rows = []
    for path_name, group in metrics_df.groupby("path", sort=False):
        group = group.sort_values("t")
        start = group.iloc[0]
        end = group.iloc[-1]
        midpoint = group.iloc[(group["t"] - 0.5).abs().argmin()]
        endpoint_worst_loss = max(float(start["test_worst_loss"]), float(end["test_worst_loss"]))
        max_worst_loss = float(group["test_worst_loss"].max())
        summary_rows.append(
            {
                "path": path_name,
                "kind": str(group.iloc[0]["kind"]),
                "steps": int(len(group)),
                "endpoint_worst_loss": endpoint_worst_loss,
                "max_worst_loss": max_worst_loss,
                "barrier_worst_loss": max_worst_loss - endpoint_worst_loss,
                "midpoint_worst_loss": float(midpoint["test_worst_loss"]),
                "midpoint_worst_acc": float(midpoint["test_worst_acc"]),
                "min_worst_acc": float(group["test_worst_acc"].min()),
                "max_avg_loss": float(group["test_avg_loss"].max()),
            }
        )
    return metrics_df, pd.DataFrame(summary_rows).sort_values("barrier_worst_loss")


@torch.no_grad()
def router_stats(
    model: TinyMoEClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    method: str,
    category: str,
    top_k: int = 2,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    top1_values: list[Tensor] = []
    topk_values: list[Tensor] = []
    entropy_values: list[Tensor] = []
    margin_values: list[Tensor] = []
    for x, _ in loader:
        x = x.to(device)
        probs = model.router_probs(x)
        k = min(top_k, model.n_experts)
        top_values, top_indices = torch.topk(probs, k=k, dim=-1)
        top1_values.append(top_indices[:, 0].cpu())
        topk_values.append(top_indices.cpu())
        entropy_values.append((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)).cpu())
        if k > 1:
            margin_values.append((top_values[:, 0] - top_values[:, 1]).cpu())
        else:
            margin_values.append(torch.ones_like(top_values[:, 0]).cpu())
    top1 = torch.cat(top1_values)
    topk = torch.cat(topk_values, dim=0)
    entropy = torch.cat(entropy_values)
    margin = torch.cat(margin_values)
    total = int(top1.numel())
    top1_counts = torch.bincount(top1, minlength=model.n_experts).to(torch.float64)
    topk_counts = torch.bincount(topk.reshape(-1), minlength=model.n_experts).to(torch.float64)
    top1_probs = top1_counts / max(1, total)
    positive = top1_probs[top1_probs > 0]
    top1_dist_entropy = -float((positive * torch.log(positive)).sum().item()) if len(positive) else 0.0
    summary_row = {
        "model": method,
        "method": method,
        "category": category,
        "prompt_idx": 0,
        "router": "toy_router",
        "num_experts": model.n_experts,
        "tokens": total,
        "top_k": min(top_k, model.n_experts),
        "router_entropy_mean": float(entropy.mean().item()),
        "top1_margin_mean": float(margin.mean().item()),
        "unique_top1_experts": int((top1_counts > 0).sum().item()),
        "unique_topk_experts": int((topk_counts > 0).sum().item()),
        "max_top1_fraction": float((top1_counts.max() / max(1, total)).item()),
        "effective_top1_experts": math.exp(top1_dist_entropy),
    }
    expert_rows = []
    for expert_id in range(model.n_experts):
        expert_rows.append(
            {
                "model": method,
                "method": method,
                "category": category,
                "prompt_idx": 0,
                "router": "toy_router",
                "expert_id": expert_id,
                "top1_count": int(top1_counts[expert_id].item()),
                "top1_fraction": float((top1_counts[expert_id] / max(1, total)).item()),
                "topk_count": int(topk_counts[expert_id].item()),
                "topk_fraction": float((topk_counts[expert_id] / max(1, total * min(top_k, model.n_experts))).item()),
            }
        )
    return {"summary_row": summary_row, "expert_rows": expert_rows, "top1": top1, "topk": topk}


def route_overlap(
    left: TinyMoEClassifier,
    right: TinyMoEClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    left_name: str,
    right_name: str,
    category: str,
) -> dict[str, Any]:
    left_stats = router_stats(left, loader, device, method=left_name, category=category)
    right_stats = router_stats(right, loader, device, method=right_name, category=category)
    left_top1 = left_stats["top1"]
    right_top1 = right_stats["top1"]
    left_topk = left_stats["topk"]
    right_topk = right_stats["topk"]
    n = min(left_top1.numel(), right_top1.numel())
    top1_agreement = float((left_top1[:n] == right_top1[:n]).to(torch.float32).mean().item())
    jaccards = []
    for idx in range(n):
        a = set(left_topk[idx].tolist())
        b = set(right_topk[idx].tolist())
        jaccards.append(len(a & b) / max(1, len(a | b)))
    return {
        "left_model": left_name,
        "right_model": right_name,
        "category": category,
        "prompt_idx": 0,
        "router": "toy_router",
        "tokens_compared": n,
        "top1_agreement": top1_agreement,
        "topk_jaccard": float(sum(jaccards) / len(jaccards)),
    }


def evaluate_method(
    template: TinyMoEClassifier,
    method: MethodState,
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model = deepcopy(template)
    model.load_state_dict(method.state)
    general = evaluate(model, loaders["general_test"], device)
    code = evaluate(model, loaders["code_test"], device)
    method_row = {
        "method": method.name,
        "description": method.description,
        "general_loss": general["loss"],
        "code_loss": code["loss"],
        "general_acc": general["acc"],
        "code_acc": code["acc"],
        "avg_loss": 0.5 * (general["loss"] + code["loss"]),
        "worst_loss": max(general["loss"], code["loss"]),
        "avg_acc": 0.5 * (general["acc"] + code["acc"]),
        "worst_acc": min(general["acc"], code["acc"]),
    }
    router_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    for category in ("general", "code"):
        stats = router_stats(model, loaders[f"{category}_test"], device, method=method.name, category=category)
        router_rows.append(stats["summary_row"])
        expert_rows.extend(stats["expert_rows"])
    return method_row, router_rows, expert_rows


def evaluate_dispatch_modes(
    template: TinyMoEClassifier,
    methods: list[MethodState],
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    dispatch_modes: list[str],
) -> pd.DataFrame:
    rows = []
    for method in methods:
        model = deepcopy(template)
        model.load_state_dict(method.state)
        for dispatch_mode in dispatch_modes:
            general = evaluate(model, loaders["general_test"], device, dispatch_mode=dispatch_mode)
            code = evaluate(model, loaders["code_test"], device, dispatch_mode=dispatch_mode)
            rows.append(
                {
                    "method": method.name,
                    "dispatch_mode": dispatch_mode,
                    "general_loss": general["loss"],
                    "code_loss": code["loss"],
                    "general_acc": general["acc"],
                    "code_acc": code["acc"],
                    "avg_loss": 0.5 * (general["loss"] + code["loss"]),
                    "worst_loss": max(general["loss"], code["loss"]),
                    "avg_acc": 0.5 * (general["acc"] + code["acc"]),
                    "worst_acc": min(general["acc"], code["acc"]),
                }
            )
    return pd.DataFrame(rows)


def router_capacity_metrics(
    router_summary: pd.DataFrame,
    expert_load: pd.DataFrame,
    *,
    capacity_factor: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary_index = router_summary.set_index(["method", "category"])
    for (method, category), group in expert_load.groupby(["method", "category"], sort=False):
        if (method, category) not in summary_index.index:
            continue
        summary = summary_index.loc[(method, category)]
        if isinstance(summary, pd.DataFrame):
            summary = summary.iloc[0]
        tokens = int(summary["tokens"])
        top_k = int(summary["top_k"])
        n_experts = int(summary["num_experts"])
        top1_capacity = int(math.ceil(capacity_factor * tokens / max(1, n_experts)))
        topk_capacity = int(math.ceil(capacity_factor * tokens * top_k / max(1, n_experts)))
        top1_counts = group["top1_count"].astype(float)
        topk_counts = group["topk_count"].astype(float)
        top1_overflow = float((top1_counts - top1_capacity).clip(lower=0).sum())
        topk_overflow = float((topk_counts - topk_capacity).clip(lower=0).sum())
        rows.append(
            {
                "method": str(method),
                "category": str(category),
                "router": str(summary["router"]),
                "tokens": tokens,
                "num_experts": n_experts,
                "top_k": top_k,
                "capacity_factor": float(capacity_factor),
                "top1_capacity_per_expert": top1_capacity,
                "topk_capacity_per_expert": topk_capacity,
                "top1_overflow_tokens": int(top1_overflow),
                "top1_overflow_fraction": top1_overflow / max(1, tokens),
                "topk_overflow_assignments": int(topk_overflow),
                "topk_overflow_fraction": topk_overflow / max(1, tokens * top_k),
                "max_top1_capacity_ratio": float(top1_counts.max() / max(1, top1_capacity)),
                "max_topk_capacity_ratio": float(topk_counts.max() / max(1, topk_capacity)),
                "capacity_action": "capacity_overflow_risk" if topk_overflow > 0 else "capacity_ok",
            }
        )
    return pd.DataFrame(rows)


def plot_results(method_metrics: pd.DataFrame, router_summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    order = method_metrics.sort_values("worst_acc", ascending=False)["method"].tolist()
    ax = axes[0]
    values = method_metrics.set_index("method").loc[order, "worst_acc"]
    ax.barh(range(len(order)), values, color="#2a9d8f")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("worst-category accuracy")
    ax.set_title("MoE average methods")

    ax = axes[1]
    pivot = router_summary.pivot_table(index="method", columns="category", values="max_top1_fraction", aggfunc="mean")
    pivot = pivot.reindex(order)
    pivot.plot(kind="barh", ax=ax, color=["#264653", "#e76f51"])
    ax.invert_yaxis()
    ax.set_xlabel("max top-1 expert fraction")
    ax.set_title("Router load concentration")
    ax.legend(title="category", fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_connectivity_paths(path_metrics: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    order = (
        path_metrics.groupby("path")["test_worst_loss"]
        .max()
        .sort_values()
        .index.tolist()
    )
    for path in order:
        group = path_metrics[path_metrics["path"] == path].sort_values("t")
        ax.plot(group["t"], group["test_worst_loss"], marker="o", markersize=2.5, linewidth=1.2, label=path)
    ax.set_xlabel("path coordinate t")
    ax.set_ylabel("worst-category loss")
    ax.set_title("MoE source/candidate connectivity")
    ax.legend(fontsize=7, ncols=2)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, summary: dict[str, Any], method_metrics: pd.DataFrame) -> None:
    best = method_metrics.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    all_avg = method_metrics[method_metrics["method"] == "all_weight_average"].iloc[0]
    matched = method_metrics[method_metrics["method"] == "expert_matched_average"].iloc[0]
    matched_router_frozen = method_metrics[method_metrics["method"] == "matched_router_frozen_average"].iloc[0]
    expert_regmean = method_metrics[method_metrics["method"] == "expert_matched_regmean_average"].iloc[0]
    expert_ties = method_metrics[method_metrics["method"] == "expert_matched_ties_average"].iloc[0]
    expert_dare = method_metrics[method_metrics["method"] == "expert_matched_dare_average"].iloc[0]
    expert_ties_dare = method_metrics[method_metrics["method"] == "expert_matched_ties_dare_average"].iloc[0]
    router_weight_search = method_metrics[method_metrics["method"] == "matched_router_weight_search_average"].iloc[0]
    router_hessian = method_metrics[method_metrics["method"] == "matched_router_hessian_average"].iloc[0]
    router_kd = method_metrics[method_metrics["method"] == "matched_router_kd_average"].iloc[0]
    router_route_kd = method_metrics[method_metrics["method"] == "matched_router_route_kd_average"].iloc[0]
    calibrated = method_metrics[method_metrics["method"] == "matched_router_calibrated_average"].iloc[0]
    topk_calibrated = method_metrics[method_metrics["method"] == "matched_router_topk_calibrated_average"].iloc[0]
    sweep_selected = method_metrics[method_metrics["method"] == "matched_router_sweep_selected_average"].iloc[0]
    expert_search = method_metrics[method_metrics["method"] == "expert_weight_search_average"].iloc[0]
    expert_search_calibrated = method_metrics[
        method_metrics["method"] == "expert_weight_search_router_calibrated_average"
    ].iloc[0]
    expert_output_projection = method_metrics[method_metrics["method"] == "expert_output_projection_average"].iloc[0]
    expert_output_projection_calibrated = method_metrics[
        method_metrics["method"] == "expert_output_projection_router_calibrated_average"
    ].iloc[0]
    unified_moe = method_metrics[method_metrics["method"] == "unified_moe_average"].iloc[0]
    route_aware = method_metrics[method_metrics["method"] == "route_aware_expert_average"].iloc[0]
    connectivity = summary["connectivity"]
    dispatch = summary["dispatch_robustness"]
    capacity = summary["router_capacity"]
    lines = [
        "# Toy MoE Route-Aware Merge",
        "",
        "这个实验用一个很小的 soft-router MoE 做可控验证：base 先在 general/code 两类合成任务上训练，然后从同一 base fine-tune 两个同构 source。为了模拟 MoE 中常见的 expert-index 语义漂移，code source 在保持函数等价的前提下被 permute experts 和 router rows。",
        "",
        "它验证的点很具体：直接 all-weight average 会把不同语义的 expert index 相加；expert matching 和 route-frequency expert weights 可以缓解这个问题；router 是否能开放平均要看 route overlap、load concentration 和 top-k margin。",
        "",
        "## 关键结果",
        "",
        f"- Best method by worst accuracy: `{best['method']}` = `{best['worst_acc']:.3f}`.",
        f"- All-weight average worst accuracy: `{all_avg['worst_acc']:.3f}`.",
        f"- Expert-matched average worst accuracy: `{matched['worst_acc']:.3f}`.",
        f"- Matched + router-frozen average worst accuracy: `{matched_router_frozen['worst_acc']:.3f}`.",
        f"- Expert-matched RegMean average worst accuracy: `{expert_regmean['worst_acc']:.3f}`.",
        f"- Expert-matched TIES average worst accuracy: `{expert_ties['worst_acc']:.3f}`.",
        f"- Expert-matched DARE average worst accuracy: `{expert_dare['worst_acc']:.3f}`.",
        f"- Expert-matched TIES+DARE average worst accuracy: `{expert_ties_dare['worst_acc']:.3f}`.",
        f"- Matched + router-weight-search average worst accuracy: `{router_weight_search['worst_acc']:.3f}`.",
        f"- Matched + Hessian-router average worst accuracy: `{router_hessian['worst_acc']:.3f}`.",
        f"- Matched + Router-KD average worst accuracy: `{router_kd['worst_acc']:.3f}`.",
        f"- Matched + route-KD average worst accuracy: `{router_route_kd['worst_acc']:.3f}`.",
        f"- Matched + router-calibrated average worst accuracy: `{calibrated['worst_acc']:.3f}`.",
        f"- Matched + router-topk-calibrated average worst accuracy: `{topk_calibrated['worst_acc']:.3f}`.",
        f"- Matched + router-sweep-selected average worst accuracy: `{sweep_selected['worst_acc']:.3f}`.",
        f"- Expert-weight search average worst accuracy: `{expert_search['worst_acc']:.3f}`.",
        f"- Expert-weight search + router-calibrated worst accuracy: `{expert_search_calibrated['worst_acc']:.3f}`.",
        f"- Expert output-projection average worst accuracy: `{expert_output_projection['worst_acc']:.3f}`.",
        f"- Expert output-projection + router-calibrated worst accuracy: `{expert_output_projection_calibrated['worst_acc']:.3f}`.",
        f"- Unified expert/router objective worst accuracy: `{unified_moe['worst_acc']:.3f}`.",
        f"- Route-aware expert average worst accuracy: `{route_aware['worst_acc']:.3f}`.",
        f"- Lowest MoE connectivity barrier: `{connectivity['best_path']}` = `{connectivity['best_barrier_worst_loss']:.4f}` worst-loss barrier.",
        f"- Direct unmatched source barrier: `{connectivity['direct_unmatched_barrier_worst_loss']:.4f}`.",
        f"- Direct matched source barrier: `{connectivity['direct_matched_barrier_worst_loss']:.4f}`.",
        f"- Matched + router-calibrated hard top-1 worst accuracy: `{dispatch['matched_router_calibrated_hard_top1_worst_acc']:.3f}`.",
        f"- Matched + router-calibrated hard top-2 worst accuracy: `{dispatch['matched_router_calibrated_hard_top2_worst_acc']:.3f}`.",
        f"- Matched + Hessian-router hard top-2 worst accuracy: `{dispatch['matched_router_hessian_hard_top2_worst_acc']:.3f}`.",
        f"- Matched + Router-KD hard top-2 worst accuracy: `{dispatch['matched_router_kd_hard_top2_worst_acc']:.3f}`.",
        f"- Matched + route-KD hard top-2 worst accuracy: `{dispatch['matched_router_route_kd_hard_top2_worst_acc']:.3f}`.",
        f"- Route-KD hard top-2 delta vs router-calibrated: `{dispatch['route_kd_minus_calibrated_hard_top2_worst_acc']:.3f}`.",
        f"- Matched + router-topk-calibrated hard top-2 worst accuracy: `{dispatch['matched_router_topk_calibrated_hard_top2_worst_acc']:.3f}`.",
        f"- Capacity factor `{capacity['capacity_factor']:.2f}` max top-k overflow fraction: `{capacity['max_topk_overflow_fraction']:.3f}`.",
        f"- Top-k router calibration delta vs soft router calibration under hard top-2: `{dispatch['topk_calibrated_minus_soft_calibrated_hard_top2_worst_acc']:.3f}`.",
        f"- Recovered expert matching mean cosine: `{summary['expert_match_mean_cosine']:.3f}`.",
        f"- Code source permutation: `{summary['code_source_permutation']}`.",
        "",
        "## Method Table",
        "",
        "| method | general acc | code acc | worst acc | avg loss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in method_metrics.sort_values("worst_acc", ascending=False).iterrows():
        lines.append(
            f"| {row['method']} | {row['general_acc']:.3f} | {row['code_acc']:.3f} | "
            f"{row['worst_acc']:.3f} | {row['avg_loss']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `all_weight_average` 是朴素 baseline：router 和 expert tensors 都按同名 index 平均，因此在 expert permutation 后会暴露 MoE index-alignment 风险。",
            "- `expert_matched_average` 先用 unlabeled calibration input 的 expert-output cosine 做 Hungarian matching，再平均；这对应 Sub-MoE / Expert Merging 里强调的 function-aware expert alignment。",
            "- `matched_router_frozen_average` 直接验证 MoE 特有假设：先对齐 expert 功能，再固定 token-to-expert dispatch，只平均非 router 权重。",
            "- `expert_matched_regmean_average` 在 expert matching 后只对 expert Linear 层做 activation-covariance RegMean，router 仍固定为 base；这把 Dense RegMean 转成了 MoE expert-local 版本。",
            "- `expert_matched_ties_average` / `expert_matched_dare_average` / `expert_matched_ties_dare_average` 把 Dense sparse task-vector merging 迁移到 MoE expert 子网；router 不参与稀疏合并。",
            "- `matched_router_weight_search_average` 不做梯度训练，只对 router tensor 的 general/code task-vector 系数做 guarded search；这是 checkpoint-only 的 MoE router probe。",
            "- `matched_router_hessian_average` 只解 router：用 source router softmax Hessian 和输入协方差做二阶加权最小二乘，检验 routing breakdown 是否来自线性 router averaging 的非线性 mismatch。",
            "- `matched_router_kd_average` 不用标签，只让 router 蒸馏 general/code source logits；这对应 Router KD 的轻量 router-expert mismatch 修复假设。",
            "- `matched_router_route_kd_average` 不用标签，直接蒸馏 source router 的 full route distribution 和 top-1 route；它检验 route-level signal 是否比 output-level KD 更适合 MoE merging。",
            "- `matched_router_calibrated_average` 冻结 matched experts，只用小校准集更新 router，并用 base-router KL 约束防止 dispatch 漂移。",
            "- `matched_router_topk_calibrated_average` 在 router-only calibration 里显式加入 hard top-2 dispatch loss，用来检验 soft-router 优化是否能迁移到真实 sparse dispatch。",
            "- `matched_router_sweep_selected_average` 对 router calibration 的 KL 系数做 sweep，先过 route-overlap guard，再按 calibration worst-loss 选择候选；它把 router overlap/load 和任务精度放到同一个 probe 里。",
            "- `expert_weight_search_average` 在同一个 expert 数和 tensor shape 内，对每个 expert 的 general/code delta 系数做校准集 min-max 坐标搜索；router 仍固定为 base。",
            "- `expert_weight_search_router_calibrated_average` 在 per-expert 系数搜索后，只开放 router 做 guarded calibration。",
            "- `expert_output_projection_average` 不用标签分数搜索，而是用 route-conditioned expert output residual 解每个 expert 的 source-delta 权重；它检验 output-space projection 是否能解释 expert merging。",
            "- `expert_output_projection_router_calibrated_average` 在 output-space expert 权重后只校准 router，用来区分 expert 输出拟合和 router dispatch 校准的贡献。",
            "- `unified_moe_average` 先用 per-expert source weight search 处理 expert 语义和重要性，再只更新 router；目标同时包含 soft/hard task loss、source route KD、source output KD、base-router KL、load-balance 和 differentiable capacity-overflow surrogate，用来检验这些 probe 能否合成一个统一方法。",
            "- `route_aware_expert_average` 冻结 base router，并按 base router 在 general/code prompt 上的 route mass 给每个 expert 设置 source delta 权重；这对应 route-weight recipes 的 toy 版本。",
            "- 这个实验不是 Qwen3 结果，但它把 MoE merging 的特质从报告落成了可跑的 probe：expert index、router overlap、expert load 和 category route mass 都会影响 average 是否安全。",
            "",
            "## Files",
            "",
            "- `method_metrics.csv`",
            "- `dispatch_mode_metrics.csv`",
            "- `router_summary.csv`",
            "- `expert_load.csv`",
            "- `router_capacity_metrics.csv`",
            "- `route_overlap.csv`",
            "- `expert_match.csv`",
            "- `route_weights_by_expert.csv`",
            "- `connectivity_path_metrics.csv`",
            "- `connectivity_summary.csv`",
            "- `connectivity_paths.png`",
            "- `expert_regmean_covariances.csv`",
            "- `expert_regmean_layers.csv`",
            "- `expert_sparse_task_vectors.csv`",
            "- `expert_search_weights_by_expert.csv`",
            "- `expert_weight_search_trace.csv`",
            "- `expert_output_projection_weights_by_expert.csv`",
            "- `router_weight_search.csv`",
            "- `router_hessian_average.csv`",
            "- `router_kd_trace.csv`",
            "- `router_route_kd_trace.csv`",
            "- `unified_moe_trace.csv`",
            "- `router_calibration_sweep.csv`",
            "- `toy_moe_merge.png`",
            "- `summary.json`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small route-aware MoE model averaging experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_moe_merge"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--train-per-category", type=int, default=500)
    parser.add_argument("--test-per-category", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--router-calibration-epochs", type=int, default=8)
    parser.add_argument("--router-calibration-lr", type=float, default=5e-3)
    parser.add_argument("--router-calibration-kl-coef", type=float, default=0.25)
    parser.add_argument("--router-calibration-kl-sweep", default="0,0.05,0.25,1.0,4.0")
    parser.add_argument("--router-calibration-sweep-min-topk-jaccard", type=float, default=0.80)
    parser.add_argument("--router-calibration-sweep-min-top1-agreement", type=float, default=0.75)
    parser.add_argument("--dispatch-router-calibration-mode", choices=["hard_top2"], default="hard_top2")
    parser.add_argument("--dispatch-router-calibration-loss-coef", type=float, default=1.0)
    parser.add_argument("--dispatch-router-calibration-soft-loss-coef", type=float, default=0.5)
    parser.add_argument("--dispatch-router-calibration-kl-coef", type=float, default=0.25)
    parser.add_argument("--router-hessian-ridge", type=float, default=1e-4)
    parser.add_argument("--router-kd-epochs", type=int, default=8)
    parser.add_argument("--router-kd-lr", type=float, default=5e-3)
    parser.add_argument("--router-kd-temperature", type=float, default=2.0)
    parser.add_argument("--router-kd-kl-coef", type=float, default=0.10)
    parser.add_argument("--router-route-kd-epochs", type=int, default=8)
    parser.add_argument("--router-route-kd-lr", type=float, default=5e-3)
    parser.add_argument("--router-route-kd-temperature", type=float, default=1.0)
    parser.add_argument("--router-route-kd-top1-loss-coef", type=float, default=0.25)
    parser.add_argument("--unified-router-epochs", type=int, default=8)
    parser.add_argument("--unified-router-lr", type=float, default=5e-3)
    parser.add_argument("--unified-router-temperature", type=float, default=1.0)
    parser.add_argument("--unified-router-dispatch-mode", choices=["hard_top2"], default="hard_top2")
    parser.add_argument("--unified-router-soft-loss-coef", type=float, default=0.5)
    parser.add_argument("--unified-router-dispatch-loss-coef", type=float, default=1.0)
    parser.add_argument("--unified-router-route-kd-coef", type=float, default=1.0)
    parser.add_argument("--unified-router-output-kd-coef", type=float, default=0.0)
    parser.add_argument("--unified-router-top1-loss-coef", type=float, default=0.5)
    parser.add_argument("--unified-router-base-kl-coef", type=float, default=0.25)
    parser.add_argument("--unified-router-load-balance-coef", type=float, default=0.10)
    parser.add_argument("--unified-router-capacity-loss-coef", type=float, default=0.05)
    parser.add_argument("--router-weight-search-grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--router-weight-search-max-delta-sum", type=float, default=1.0)
    parser.add_argument("--router-weight-search-min-topk-jaccard", type=float, default=0.80)
    parser.add_argument("--router-weight-search-min-top1-agreement", type=float, default=0.75)
    parser.add_argument("--expert-regmean-batches", type=int, default=4)
    parser.add_argument("--expert-regmean-ridge", type=float, default=1e-4)
    parser.add_argument("--expert-ties-density", type=float, default=0.5)
    parser.add_argument("--expert-dare-drop-rate", type=float, default=0.25)
    parser.add_argument("--expert-dare-seed", type=int, default=123)
    parser.add_argument("--expert-search-grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--expert-search-passes", type=int, default=2)
    parser.add_argument("--expert-search-prior-penalty", type=float, default=0.02)
    parser.add_argument("--expert-search-objective", choices=["avg_loss", "worst_loss"], default="worst_loss")
    parser.add_argument("--expert-search-max-delta-sum", type=float, default=1.0)
    parser.add_argument("--expert-search-shared-general-weight", type=float, default=0.5)
    parser.add_argument("--expert-search-shared-code-weight", type=float, default=0.5)
    parser.add_argument("--output-projection-ridge", type=float, default=1e-4)
    parser.add_argument("--output-projection-max-delta-sum", type=float, default=1.0)
    parser.add_argument("--output-projection-dispatch-mode", choices=["soft_all", "hard_top1", "hard_top2"], default="hard_top2")
    parser.add_argument("--output-projection-shared-general-weight", type=float, default=0.5)
    parser.add_argument("--output-projection-shared-code-weight", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-coef", type=float, default=0.02)
    parser.add_argument("--anchor-floor", type=float, default=0.15)
    parser.add_argument("--match-batches", type=int, default=6)
    parser.add_argument("--connectivity-steps", type=int, default=21)
    parser.add_argument("--dispatch-eval-modes", default="soft_all,hard_top1,hard_top2")
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    dispatch_eval_modes = parse_dispatch_modes(args.dispatch_eval_modes)
    loaders = prepare_data(args.seed, args.train_per_category, args.test_per_category, args.batch_size)
    template = TinyMoEClassifier(hidden=args.hidden, n_experts=args.experts)

    base = deepcopy(template)
    train_model(
        base,
        loaders["mixed_train"],
        epochs=args.base_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aux_coef=args.aux_coef,
        device=device,
        desc="train toy MoE base",
    )
    base.cpu()
    base_state = cpu_state(base)

    general_model = deepcopy(base)
    train_model(
        general_model,
        loaders["general_train"],
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        aux_coef=args.aux_coef,
        device=device,
        desc="fine-tune general source",
    )
    general_model.cpu()
    general_state = cpu_state(general_model)

    code_model = deepcopy(base)
    train_model(
        code_model,
        loaders["code_train"],
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        aux_coef=args.aux_coef,
        device=device,
        desc="fine-tune code source",
    )
    code_model.cpu()
    permutation = list(range(args.experts))
    if args.experts >= 4:
        permutation = [2, 0, 3, 1] + list(range(4, args.experts))
    else:
        permutation = list(reversed(permutation))
    code_permuted = permute_experts_and_router(code_model, permutation)
    code_permuted_state = cpu_state(code_permuted)

    matched_code, expert_match = match_experts(general_model, code_permuted, loaders["mixed_test"], device, args.match_batches)
    matched_code.cpu()
    matched_code_state = cpu_state(matched_code)

    route_weights = route_mass_weights(base, loaders, device, args.anchor_floor)
    route_aware = route_aware_state(base_state, general_state, matched_code_state, route_weights, args.experts)
    expert_search_grid = parse_grid(args.expert_search_grid)
    expert_search_pairs = candidate_source_pairs(expert_search_grid, args.expert_search_max_delta_sum)
    expert_search, expert_search_weights, expert_search_trace = search_expert_source_weights(
        template,
        base_state,
        general_state,
        matched_code_state,
        route_weights,
        loaders,
        n_experts=args.experts,
        candidate_pairs=expert_search_pairs,
        passes=args.expert_search_passes,
        prior_penalty=args.expert_search_prior_penalty,
        shared_weights=(args.expert_search_shared_general_weight, args.expert_search_shared_code_weight),
        objective=args.expert_search_objective,
        device=device,
    )
    expert_output_projection, expert_output_projection_weights = output_space_expert_weight_state(
        base,
        general_model,
        matched_code,
        base_state,
        general_state,
        matched_code_state,
        loaders,
        n_experts=args.experts,
        shared_weights=(args.output_projection_shared_general_weight, args.output_projection_shared_code_weight),
        dispatch_mode=args.output_projection_dispatch_mode,
        ridge=args.output_projection_ridge,
        max_delta_sum=args.output_projection_max_delta_sum,
        device=device,
    )

    all_average = task_vector_average(base_state, [general_state, code_permuted_state], [0.5, 0.5])
    router_frozen = {name: value.clone() for name, value in all_average.items()}
    for name in router_frozen:
        if name.startswith("router."):
            router_frozen[name] = base_state[name].clone()
    expert_matched = task_vector_average(base_state, [general_state, matched_code_state], [0.5, 0.5])
    matched_router_frozen = {name: value.clone() for name, value in expert_matched.items()}
    for name in matched_router_frozen:
        if name.startswith("router."):
            matched_router_frozen[name] = base_state[name].clone()
    general_cov, general_cov_rows = collect_linear_covariances(
        general_model,
        loaders["general_calib"],
        device,
        args.expert_regmean_batches,
        layer_prefix="experts.",
    )
    code_cov, code_cov_rows = collect_linear_covariances(
        matched_code,
        loaders["code_calib"],
        device,
        args.expert_regmean_batches,
        layer_prefix="experts.",
    )
    general_cov_rows.insert(0, "source", "general")
    code_cov_rows.insert(0, "source", "code_matched")
    regmean_covariances = pd.concat([general_cov_rows, code_cov_rows], ignore_index=True)
    expert_matched_regmean, expert_regmean_layers = regmean_expert_state(
        matched_router_frozen,
        [general_state, matched_code_state],
        [general_cov, code_cov],
        ridge=args.expert_regmean_ridge,
        layer_prefix="experts.",
    )
    expert_matched_ties, expert_ties_rows = expert_sparse_task_vector_state(
        base_state,
        general_state,
        matched_code_state,
        mode="ties",
        density=args.expert_ties_density,
        drop_rate=args.expert_dare_drop_rate,
        seed=args.expert_dare_seed,
    )
    expert_matched_dare, expert_dare_rows = expert_sparse_task_vector_state(
        base_state,
        general_state,
        matched_code_state,
        mode="dare",
        density=args.expert_ties_density,
        drop_rate=args.expert_dare_drop_rate,
        seed=args.expert_dare_seed,
    )
    expert_matched_ties_dare, expert_ties_dare_rows = expert_sparse_task_vector_state(
        base_state,
        general_state,
        matched_code_state,
        mode="ties_dare",
        density=args.expert_ties_density,
        drop_rate=args.expert_dare_drop_rate,
        seed=args.expert_dare_seed,
    )
    expert_sparse_task_vectors = pd.concat(
        [expert_ties_rows, expert_dare_rows, expert_ties_dare_rows],
        ignore_index=True,
    )
    router_weight_search_grid = parse_grid(args.router_weight_search_grid)
    router_weight_search_pairs = candidate_source_pairs(
        router_weight_search_grid,
        args.router_weight_search_max_delta_sum,
    )
    matched_router_weight_search, router_weight_search = search_router_source_weights(
        template,
        matched_router_frozen,
        base_state,
        general_state,
        matched_code_state,
        base,
        loaders,
        candidate_pairs=router_weight_search_pairs,
        min_topk_jaccard=args.router_weight_search_min_topk_jaccard,
        min_top1_agreement=args.router_weight_search_min_top1_agreement,
        device=device,
    )
    matched_router_hessian, router_hessian_average = hessian_router_average_state(
        expert_matched,
        [
            ("general_source", general_model, loaders["general_calib"], 0.5),
            ("matched_code_source", matched_code, loaders["code_calib"], 0.5),
        ],
        ridge=args.router_hessian_ridge,
        device=device,
    )
    matched_router_kd, router_kd_trace = calibrate_router_kd_state(
        template,
        matched_router_frozen,
        base,
        [
            ("general_source", general_model, loaders["general_calib"], 0.5),
            ("matched_code_source", matched_code, loaders["code_calib"], 0.5),
        ],
        epochs=args.router_kd_epochs,
        lr=args.router_kd_lr,
        temperature=args.router_kd_temperature,
        router_kl_coef=args.router_kd_kl_coef,
        aux_coef=args.aux_coef,
        device=device,
        desc="distill matched router from source logits",
    )
    matched_router_route_kd, router_route_kd_trace = calibrate_router_route_kd_state(
        template,
        matched_router_frozen,
        [
            ("general_source", general_model, loaders["general_calib"], 0.5),
            ("matched_code_source", matched_code, loaders["code_calib"], 0.5),
        ],
        epochs=args.router_route_kd_epochs,
        lr=args.router_route_kd_lr,
        temperature=args.router_route_kd_temperature,
        top1_loss_coef=args.router_route_kd_top1_loss_coef,
        aux_coef=args.aux_coef,
        device=device,
        desc="distill matched router from source routes",
    )
    matched_router_calibrated = calibrate_router_only_state(
        template,
        matched_router_frozen,
        base,
        loaders["mixed_calib"],
        epochs=args.router_calibration_epochs,
        lr=args.router_calibration_lr,
        kl_coef=args.router_calibration_kl_coef,
        aux_coef=args.aux_coef,
        device=device,
        desc="calibrate matched router",
    )
    matched_router_topk_calibrated = calibrate_router_dispatch_aware_state(
        template,
        matched_router_frozen,
        base,
        loaders["mixed_calib"],
        epochs=args.router_calibration_epochs,
        lr=args.router_calibration_lr,
        dispatch_mode=args.dispatch_router_calibration_mode,
        dispatch_loss_coef=args.dispatch_router_calibration_loss_coef,
        soft_loss_coef=args.dispatch_router_calibration_soft_loss_coef,
        kl_coef=args.dispatch_router_calibration_kl_coef,
        aux_coef=args.aux_coef,
        device=device,
        desc=f"calibrate matched router for {args.dispatch_router_calibration_mode}",
    )
    router_calibration_kl_values = parse_grid(args.router_calibration_kl_sweep)
    matched_router_sweep_selected, router_calibration_sweep = sweep_router_calibration(
        template,
        matched_router_frozen,
        base,
        base,
        loaders,
        kl_values=router_calibration_kl_values,
        epochs=args.router_calibration_epochs,
        lr=args.router_calibration_lr,
        aux_coef=args.aux_coef,
        min_topk_jaccard=args.router_calibration_sweep_min_topk_jaccard,
        min_top1_agreement=args.router_calibration_sweep_min_top1_agreement,
        device=device,
    )
    expert_search_router_calibrated = calibrate_router_only_state(
        template,
        expert_search,
        base,
        loaders["mixed_calib"],
        epochs=args.router_calibration_epochs,
        lr=args.router_calibration_lr,
        kl_coef=args.router_calibration_kl_coef,
        aux_coef=args.aux_coef,
        device=device,
        desc="calibrate searched expert router",
    )
    expert_output_projection_router_calibrated = calibrate_router_only_state(
        template,
        expert_output_projection,
        base,
        loaders["mixed_calib"],
        epochs=args.router_calibration_epochs,
        lr=args.router_calibration_lr,
        kl_coef=args.router_calibration_kl_coef,
        aux_coef=args.aux_coef,
        device=device,
        desc="calibrate output-projected expert router",
    )
    unified_moe_average, unified_moe_trace = calibrate_unified_moe_router_state(
        template,
        expert_search,
        base,
        [
            ("general_source", general_model, loaders["general_calib"], 0.5),
            ("matched_code_source", matched_code, loaders["code_calib"], 0.5),
        ],
        epochs=args.unified_router_epochs,
        lr=args.unified_router_lr,
        dispatch_mode=args.unified_router_dispatch_mode,
        soft_loss_coef=args.unified_router_soft_loss_coef,
        dispatch_loss_coef=args.unified_router_dispatch_loss_coef,
        route_kd_coef=args.unified_router_route_kd_coef,
        output_kd_coef=args.unified_router_output_kd_coef,
        top1_loss_coef=args.unified_router_top1_loss_coef,
        base_kl_coef=args.unified_router_base_kl_coef,
        load_balance_coef=args.unified_router_load_balance_coef,
        capacity_loss_coef=args.unified_router_capacity_loss_coef,
        capacity_factor=args.capacity_factor,
        temperature=args.unified_router_temperature,
        device=device,
        desc="unified expert/router objective",
    )

    methods = [
        MethodState("base", base_state, "mixed-task base before fine-tuning"),
        MethodState("general_endpoint", general_state, "general source endpoint"),
        MethodState("code_endpoint_permuted", code_permuted_state, "code source endpoint with function-preserving expert permutation"),
        MethodState("all_weight_average", all_average, "average same-name router and expert tensors without expert matching"),
        MethodState("router_frozen_average", router_frozen, "all-weight average but router tensors reset to base"),
        MethodState("expert_matched_average", expert_matched, "align code experts to general source by output cosine before averaging"),
        MethodState(
            "matched_router_frozen_average",
            matched_router_frozen,
            "align code experts by output cosine and keep the base router fixed",
        ),
        MethodState(
            "expert_matched_regmean_average",
            expert_matched_regmean,
            "align code experts and RegMean-merge only expert linear layers with the base router fixed",
        ),
        MethodState(
            "expert_matched_ties_average",
            expert_matched_ties,
            "align code experts and apply TIES to expert task vectors with the base router fixed",
        ),
        MethodState(
            "expert_matched_dare_average",
            expert_matched_dare,
            "align code experts and apply DARE to expert task vectors with the base router fixed",
        ),
        MethodState(
            "expert_matched_ties_dare_average",
            expert_matched_ties_dare,
            "align code experts and apply DARE+TIES to expert task vectors with the base router fixed",
        ),
        MethodState(
            "matched_router_weight_search_average",
            matched_router_weight_search,
            "align code experts and search router source-delta weights with route guards",
        ),
        MethodState(
            "matched_router_hessian_average",
            matched_router_hessian,
            "align code experts and solve a Hessian-aware router average from source router distributions",
        ),
        MethodState(
            "matched_router_kd_average",
            matched_router_kd,
            "align code experts and distill only the router from source model logits on unlabeled calibration inputs",
        ),
        MethodState(
            "matched_router_route_kd_average",
            matched_router_route_kd,
            "align code experts and distill only the router from source route distributions on unlabeled calibration inputs",
        ),
        MethodState(
            "matched_router_calibrated_average",
            matched_router_calibrated,
            "align code experts, freeze non-router tensors, and calibrate only the router",
        ),
        MethodState(
            "matched_router_topk_calibrated_average",
            matched_router_topk_calibrated,
            "align code experts and calibrate the router with hard top-k dispatch loss",
        ),
        MethodState(
            "matched_router_sweep_selected_average",
            matched_router_sweep_selected,
            "align code experts, then select router calibration KL by calibration worst loss",
        ),
        MethodState(
            "expert_weight_search_average",
            expert_search,
            "search per-expert source delta weights on the calibration set with the base router fixed",
        ),
        MethodState(
            "expert_weight_search_router_calibrated_average",
            expert_search_router_calibrated,
            "search per-expert source delta weights, then calibrate only the router",
        ),
        MethodState(
            "expert_output_projection_average",
            expert_output_projection,
            "solve per-expert source delta weights from route-conditioned output-space residuals with the base router fixed",
        ),
        MethodState(
            "expert_output_projection_router_calibrated_average",
            expert_output_projection_router_calibrated,
            "solve per-expert output-space source delta weights, then calibrate only the router",
        ),
        MethodState(
            "unified_moe_average",
            unified_moe_average,
            "search expert source weights, then learn one router objective combining task loss, hard top-k dispatch, route KD, output KD, base-router KL, load balance, and capacity overflow",
        ),
        MethodState("route_aware_expert_average", route_aware, "freeze base router and use route-frequency expert source weights"),
    ]

    connectivity_paths = [
        ConnectivityPath(
            "direct_unmatched_general_to_code",
            general_state,
            code_permuted_state,
            None,
            "direct_unmatched",
        ),
        ConnectivityPath(
            "direct_matched_general_to_code",
            general_state,
            matched_code_state,
            None,
            "direct_matched",
        ),
        ConnectivityPath(
            "via_expert_matched_average",
            general_state,
            matched_code_state,
            expert_matched,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_matched_router_weight_search",
            general_state,
            matched_code_state,
            matched_router_weight_search,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_matched_router_hessian",
            general_state,
            matched_code_state,
            matched_router_hessian,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_matched_router_kd",
            general_state,
            matched_code_state,
            matched_router_kd,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_matched_router_route_kd",
            general_state,
            matched_code_state,
            matched_router_route_kd,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_matched_router_calibrated",
            general_state,
            matched_code_state,
            matched_router_calibrated,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_matched_router_topk_calibrated",
            general_state,
            matched_code_state,
            matched_router_topk_calibrated,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_expert_weight_search_router_calibrated",
            general_state,
            matched_code_state,
            expert_search_router_calibrated,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_expert_output_projection_router_calibrated",
            general_state,
            matched_code_state,
            expert_output_projection_router_calibrated,
            "candidate_two_segment",
        ),
        ConnectivityPath(
            "via_unified_moe_average",
            general_state,
            matched_code_state,
            unified_moe_average,
            "candidate_two_segment",
        ),
    ]
    connectivity_path_metrics, connectivity_summary = evaluate_connectivity_paths(
        template,
        connectivity_paths,
        loaders,
        device,
        steps=args.connectivity_steps,
    )
    dispatch_mode_metrics = evaluate_dispatch_modes(
        template,
        methods,
        loaders,
        device,
        dispatch_modes=dispatch_eval_modes,
    )

    method_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    models_for_overlap: dict[str, TinyMoEClassifier] = {}
    for method in methods:
        row, router_stats_rows, expert_stats_rows = evaluate_method(template, method, loaders, device)
        method_rows.append(row)
        router_rows.extend(router_stats_rows)
        expert_rows.extend(expert_stats_rows)
        model = deepcopy(template)
        model.load_state_dict(method.state)
        models_for_overlap[method.name] = model

    overlap_rows = []
    for method_name, model in models_for_overlap.items():
        if method_name == "base":
            continue
        for category in ("general", "code"):
            overlap_rows.append(
                route_overlap(
                    models_for_overlap["base"],
                    model,
                    loaders[f"{category}_test"],
                    device,
                    left_name="base",
                    right_name=method_name,
                    category=category,
                )
            )

    method_metrics = pd.DataFrame(method_rows)
    router_summary = pd.DataFrame(router_rows)
    expert_load = pd.DataFrame(expert_rows)
    route_overlap_df = pd.DataFrame(overlap_rows)
    router_capacity = router_capacity_metrics(router_summary, expert_load, capacity_factor=args.capacity_factor)

    method_metrics.to_csv(args.output_dir / "method_metrics.csv", index=False)
    dispatch_mode_metrics.to_csv(args.output_dir / "dispatch_mode_metrics.csv", index=False)
    connectivity_path_metrics.to_csv(args.output_dir / "connectivity_path_metrics.csv", index=False)
    connectivity_summary.to_csv(args.output_dir / "connectivity_summary.csv", index=False)
    router_summary.to_csv(args.output_dir / "router_summary.csv", index=False)
    expert_load.to_csv(args.output_dir / "expert_load.csv", index=False)
    router_capacity.to_csv(args.output_dir / "router_capacity_metrics.csv", index=False)
    route_overlap_df.to_csv(args.output_dir / "route_overlap.csv", index=False)
    expert_match.to_csv(args.output_dir / "expert_match.csv", index=False)
    route_weights.to_csv(args.output_dir / "route_weights_by_expert.csv", index=False)
    regmean_covariances.to_csv(args.output_dir / "expert_regmean_covariances.csv", index=False)
    expert_regmean_layers.to_csv(args.output_dir / "expert_regmean_layers.csv", index=False)
    expert_sparse_task_vectors.to_csv(args.output_dir / "expert_sparse_task_vectors.csv", index=False)
    expert_search_weights.to_csv(args.output_dir / "expert_search_weights_by_expert.csv", index=False)
    expert_search_trace.to_csv(args.output_dir / "expert_weight_search_trace.csv", index=False)
    expert_output_projection_weights.to_csv(args.output_dir / "expert_output_projection_weights_by_expert.csv", index=False)
    router_weight_search.to_csv(args.output_dir / "router_weight_search.csv", index=False)
    router_hessian_average.to_csv(args.output_dir / "router_hessian_average.csv", index=False)
    router_kd_trace.to_csv(args.output_dir / "router_kd_trace.csv", index=False)
    router_route_kd_trace.to_csv(args.output_dir / "router_route_kd_trace.csv", index=False)
    unified_moe_trace.to_csv(args.output_dir / "unified_moe_trace.csv", index=False)
    router_calibration_sweep.to_csv(args.output_dir / "router_calibration_sweep.csv", index=False)
    plot_results(method_metrics, router_summary, args.output_dir / "toy_moe_merge.png")
    plot_connectivity_paths(connectivity_path_metrics, args.output_dir / "connectivity_paths.png")

    best = method_metrics.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    best_connectivity = connectivity_summary.iloc[0]
    connectivity_by_path = connectivity_summary.set_index("path")
    dispatch_best_rows = {}
    for mode in dispatch_eval_modes:
        mode_rows = dispatch_mode_metrics[dispatch_mode_metrics["dispatch_mode"] == mode]
        if not mode_rows.empty:
            best_dispatch = mode_rows.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
            dispatch_best_rows[mode] = {
                "method": str(best_dispatch["method"]),
                "worst_acc": float(best_dispatch["worst_acc"]),
                "avg_acc": float(best_dispatch["avg_acc"]),
                "worst_loss": float(best_dispatch["worst_loss"]),
            }
    dispatch_index = dispatch_mode_metrics.set_index(["method", "dispatch_mode"])
    dispatch_index_keys = set(dispatch_index.index)
    all_avg = method_metrics[method_metrics["method"] == "all_weight_average"].iloc[0]
    expert_matched_row = method_metrics[method_metrics["method"] == "expert_matched_average"].iloc[0]
    matched_router_frozen_row = method_metrics[method_metrics["method"] == "matched_router_frozen_average"].iloc[0]
    expert_matched_regmean_row = method_metrics[method_metrics["method"] == "expert_matched_regmean_average"].iloc[0]
    expert_matched_ties_row = method_metrics[method_metrics["method"] == "expert_matched_ties_average"].iloc[0]
    expert_matched_dare_row = method_metrics[method_metrics["method"] == "expert_matched_dare_average"].iloc[0]
    expert_matched_ties_dare_row = method_metrics[
        method_metrics["method"] == "expert_matched_ties_dare_average"
    ].iloc[0]
    matched_router_weight_search_row = method_metrics[
        method_metrics["method"] == "matched_router_weight_search_average"
    ].iloc[0]
    matched_router_hessian_row = method_metrics[method_metrics["method"] == "matched_router_hessian_average"].iloc[0]
    matched_router_kd_row = method_metrics[method_metrics["method"] == "matched_router_kd_average"].iloc[0]
    matched_router_route_kd_row = method_metrics[
        method_metrics["method"] == "matched_router_route_kd_average"
    ].iloc[0]
    matched_router_calibrated_row = method_metrics[method_metrics["method"] == "matched_router_calibrated_average"].iloc[0]
    matched_router_topk_calibrated_row = method_metrics[
        method_metrics["method"] == "matched_router_topk_calibrated_average"
    ].iloc[0]
    matched_router_sweep_selected_row = method_metrics[method_metrics["method"] == "matched_router_sweep_selected_average"].iloc[0]
    expert_search_row = method_metrics[method_metrics["method"] == "expert_weight_search_average"].iloc[0]
    expert_search_router_calibrated_row = method_metrics[
        method_metrics["method"] == "expert_weight_search_router_calibrated_average"
    ].iloc[0]
    expert_output_projection_row = method_metrics[method_metrics["method"] == "expert_output_projection_average"].iloc[0]
    expert_output_projection_router_calibrated_row = method_metrics[
        method_metrics["method"] == "expert_output_projection_router_calibrated_average"
    ].iloc[0]
    unified_moe_row = method_metrics[method_metrics["method"] == "unified_moe_average"].iloc[0]
    route_aware_row = method_metrics[method_metrics["method"] == "route_aware_expert_average"].iloc[0]
    capacity_route_kd = router_capacity[router_capacity["method"] == "matched_router_route_kd_average"]
    capacity_calibrated = router_capacity[router_capacity["method"] == "matched_router_calibrated_average"]
    capacity_unified = router_capacity[router_capacity["method"] == "unified_moe_average"]
    worst_capacity_row = router_capacity.sort_values("topk_overflow_fraction", ascending=False).iloc[0]
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "n_experts": args.experts,
        "code_source_permutation": permutation,
        "expert_match_mean_cosine": float(expert_match["output_cosine"].mean()),
        "best_method": str(best["method"]),
        "best_worst_acc": float(best["worst_acc"]),
        "connectivity": {
            "steps": args.connectivity_steps,
            "best_path": str(best_connectivity["path"]),
            "best_barrier_worst_loss": float(best_connectivity["barrier_worst_loss"]),
            "direct_unmatched_barrier_worst_loss": float(
                connectivity_by_path.loc["direct_unmatched_general_to_code", "barrier_worst_loss"]
            ),
            "direct_matched_barrier_worst_loss": float(
                connectivity_by_path.loc["direct_matched_general_to_code", "barrier_worst_loss"]
            ),
            "matched_minus_unmatched_barrier_worst_loss": float(
                connectivity_by_path.loc["direct_matched_general_to_code", "barrier_worst_loss"]
                - connectivity_by_path.loc["direct_unmatched_general_to_code", "barrier_worst_loss"]
            ),
            "path_count": int(len(connectivity_summary)),
        },
        "dispatch_robustness": {
            "modes": dispatch_eval_modes,
            "row_count": int(len(dispatch_mode_metrics)),
            "best_by_mode": dispatch_best_rows,
            "matched_router_calibrated_hard_top1_worst_acc": float(
                dispatch_index.loc[("matched_router_calibrated_average", "hard_top1"), "worst_acc"]
            )
            if ("matched_router_calibrated_average", "hard_top1") in dispatch_index_keys
            else None,
            "matched_router_calibrated_hard_top2_worst_acc": float(
                dispatch_index.loc[("matched_router_calibrated_average", "hard_top2"), "worst_acc"]
            )
            if ("matched_router_calibrated_average", "hard_top2") in dispatch_index_keys
            else None,
            "matched_router_topk_calibrated_hard_top1_worst_acc": float(
                dispatch_index.loc[("matched_router_topk_calibrated_average", "hard_top1"), "worst_acc"]
            )
            if ("matched_router_topk_calibrated_average", "hard_top1") in dispatch_index_keys
            else None,
            "matched_router_topk_calibrated_hard_top2_worst_acc": float(
                dispatch_index.loc[("matched_router_topk_calibrated_average", "hard_top2"), "worst_acc"]
            )
            if ("matched_router_topk_calibrated_average", "hard_top2") in dispatch_index_keys
            else None,
            "matched_router_hessian_hard_top1_worst_acc": float(
                dispatch_index.loc[("matched_router_hessian_average", "hard_top1"), "worst_acc"]
            )
            if ("matched_router_hessian_average", "hard_top1") in dispatch_index_keys
            else None,
            "matched_router_hessian_hard_top2_worst_acc": float(
                dispatch_index.loc[("matched_router_hessian_average", "hard_top2"), "worst_acc"]
            )
            if ("matched_router_hessian_average", "hard_top2") in dispatch_index_keys
            else None,
            "matched_router_kd_hard_top1_worst_acc": float(
                dispatch_index.loc[("matched_router_kd_average", "hard_top1"), "worst_acc"]
            )
            if ("matched_router_kd_average", "hard_top1") in dispatch_index_keys
            else None,
            "matched_router_kd_hard_top2_worst_acc": float(
                dispatch_index.loc[("matched_router_kd_average", "hard_top2"), "worst_acc"]
            )
            if ("matched_router_kd_average", "hard_top2") in dispatch_index_keys
            else None,
            "matched_router_route_kd_hard_top1_worst_acc": float(
                dispatch_index.loc[("matched_router_route_kd_average", "hard_top1"), "worst_acc"]
            )
            if ("matched_router_route_kd_average", "hard_top1") in dispatch_index_keys
            else None,
            "matched_router_route_kd_hard_top2_worst_acc": float(
                dispatch_index.loc[("matched_router_route_kd_average", "hard_top2"), "worst_acc"]
            )
            if ("matched_router_route_kd_average", "hard_top2") in dispatch_index_keys
            else None,
            "unified_moe_hard_top1_worst_acc": float(
                dispatch_index.loc[("unified_moe_average", "hard_top1"), "worst_acc"]
            )
            if ("unified_moe_average", "hard_top1") in dispatch_index_keys
            else None,
            "unified_moe_hard_top2_worst_acc": float(
                dispatch_index.loc[("unified_moe_average", "hard_top2"), "worst_acc"]
            )
            if ("unified_moe_average", "hard_top2") in dispatch_index_keys
            else None,
            "matched_router_calibrated_soft_to_hard_top1_worst_acc_delta": float(
                dispatch_index.loc[("matched_router_calibrated_average", "hard_top1"), "worst_acc"]
                - dispatch_index.loc[("matched_router_calibrated_average", "soft_all"), "worst_acc"]
            )
            if {
                ("matched_router_calibrated_average", "hard_top1"),
                ("matched_router_calibrated_average", "soft_all"),
            }.issubset(dispatch_index_keys)
            else None,
            "matched_router_calibrated_soft_to_hard_top2_worst_acc_delta": float(
                dispatch_index.loc[("matched_router_calibrated_average", "hard_top2"), "worst_acc"]
                - dispatch_index.loc[("matched_router_calibrated_average", "soft_all"), "worst_acc"]
            )
            if {
                ("matched_router_calibrated_average", "hard_top2"),
                ("matched_router_calibrated_average", "soft_all"),
            }.issubset(dispatch_index_keys)
            else None,
            "topk_calibrated_minus_soft_calibrated_hard_top1_worst_acc": float(
                dispatch_index.loc[("matched_router_topk_calibrated_average", "hard_top1"), "worst_acc"]
                - dispatch_index.loc[("matched_router_calibrated_average", "hard_top1"), "worst_acc"]
            )
            if {
                ("matched_router_topk_calibrated_average", "hard_top1"),
                ("matched_router_calibrated_average", "hard_top1"),
            }.issubset(dispatch_index_keys)
            else None,
            "topk_calibrated_minus_soft_calibrated_hard_top2_worst_acc": float(
                dispatch_index.loc[("matched_router_topk_calibrated_average", "hard_top2"), "worst_acc"]
                - dispatch_index.loc[("matched_router_calibrated_average", "hard_top2"), "worst_acc"]
            )
            if {
                ("matched_router_topk_calibrated_average", "hard_top2"),
                ("matched_router_calibrated_average", "hard_top2"),
            }.issubset(dispatch_index_keys)
            else None,
            "route_kd_minus_calibrated_hard_top2_worst_acc": float(
                dispatch_index.loc[("matched_router_route_kd_average", "hard_top2"), "worst_acc"]
                - dispatch_index.loc[("matched_router_calibrated_average", "hard_top2"), "worst_acc"]
            )
            if {
                ("matched_router_route_kd_average", "hard_top2"),
                ("matched_router_calibrated_average", "hard_top2"),
            }.issubset(dispatch_index_keys)
            else None,
            "route_kd_minus_output_kd_hard_top2_worst_acc": float(
                dispatch_index.loc[("matched_router_route_kd_average", "hard_top2"), "worst_acc"]
                - dispatch_index.loc[("matched_router_kd_average", "hard_top2"), "worst_acc"]
            )
            if {
                ("matched_router_route_kd_average", "hard_top2"),
                ("matched_router_kd_average", "hard_top2"),
            }.issubset(dispatch_index_keys)
            else None,
            "unified_moe_minus_route_kd_hard_top2_worst_acc": float(
                dispatch_index.loc[("unified_moe_average", "hard_top2"), "worst_acc"]
                - dispatch_index.loc[("matched_router_route_kd_average", "hard_top2"), "worst_acc"]
            )
            if {
                ("unified_moe_average", "hard_top2"),
                ("matched_router_route_kd_average", "hard_top2"),
            }.issubset(dispatch_index_keys)
            else None,
            "unified_moe_minus_calibrated_hard_top2_worst_acc": float(
                dispatch_index.loc[("unified_moe_average", "hard_top2"), "worst_acc"]
                - dispatch_index.loc[("matched_router_calibrated_average", "hard_top2"), "worst_acc"]
            )
            if {
                ("unified_moe_average", "hard_top2"),
                ("matched_router_calibrated_average", "hard_top2"),
            }.issubset(dispatch_index_keys)
            else None,
        },
        "router_capacity": {
            "capacity_factor": args.capacity_factor,
            "row_count": int(len(router_capacity)),
            "max_top1_overflow_fraction": float(router_capacity["top1_overflow_fraction"].max()),
            "max_topk_overflow_fraction": float(router_capacity["topk_overflow_fraction"].max()),
            "worst_topk_overflow_method": str(worst_capacity_row["method"]),
            "worst_topk_overflow_category": str(worst_capacity_row["category"]),
            "matched_router_route_kd_max_topk_overflow_fraction": float(
                capacity_route_kd["topk_overflow_fraction"].max()
            ),
            "matched_router_calibrated_max_topk_overflow_fraction": float(
                capacity_calibrated["topk_overflow_fraction"].max()
            ),
            "route_kd_minus_calibrated_max_topk_overflow_fraction": float(
                capacity_route_kd["topk_overflow_fraction"].max()
                - capacity_calibrated["topk_overflow_fraction"].max()
            ),
            "unified_moe_max_topk_overflow_fraction": float(capacity_unified["topk_overflow_fraction"].max()),
            "unified_moe_minus_route_kd_max_topk_overflow_fraction": float(
                capacity_unified["topk_overflow_fraction"].max()
                - capacity_route_kd["topk_overflow_fraction"].max()
            ),
        },
        "all_weight_average_worst_acc": float(all_avg["worst_acc"]),
        "matched_router_frozen_worst_acc": float(matched_router_frozen_row["worst_acc"]),
        "matched_router_frozen_minus_all_weight_worst_acc": float(
            matched_router_frozen_row["worst_acc"] - all_avg["worst_acc"]
        ),
        "matched_router_calibrated_worst_acc": float(matched_router_calibrated_row["worst_acc"]),
        "matched_router_calibrated_minus_all_weight_worst_acc": float(
            matched_router_calibrated_row["worst_acc"] - all_avg["worst_acc"]
        ),
        "matched_router_calibrated_minus_frozen_worst_acc": float(
            matched_router_calibrated_row["worst_acc"] - matched_router_frozen_row["worst_acc"]
        ),
        "matched_router_topk_calibrated_worst_acc": float(matched_router_topk_calibrated_row["worst_acc"]),
        "matched_router_topk_calibrated_minus_matched_calibrated_worst_acc": float(
            matched_router_topk_calibrated_row["worst_acc"] - matched_router_calibrated_row["worst_acc"]
        ),
        "expert_matched_regmean_worst_acc": float(expert_matched_regmean_row["worst_acc"]),
        "expert_matched_regmean_minus_matched_frozen_worst_acc": float(
            expert_matched_regmean_row["worst_acc"] - matched_router_frozen_row["worst_acc"]
        ),
        "expert_regmean": {
            "batches": args.expert_regmean_batches,
            "ridge": args.expert_regmean_ridge,
            "linear_layers": int(len(expert_regmean_layers)),
            "covariance_rows": int(len(regmean_covariances)),
        },
        "expert_sparse_task_vector": {
            "ties_density": args.expert_ties_density,
            "dare_drop_rate": args.expert_dare_drop_rate,
            "dare_seed": args.expert_dare_seed,
            "tensor_rows": int(len(expert_sparse_task_vectors)),
            "ties_worst_acc": float(expert_matched_ties_row["worst_acc"]),
            "dare_worst_acc": float(expert_matched_dare_row["worst_acc"]),
            "ties_dare_worst_acc": float(expert_matched_ties_dare_row["worst_acc"]),
        },
        "expert_matched_ties_worst_acc": float(expert_matched_ties_row["worst_acc"]),
        "expert_matched_dare_worst_acc": float(expert_matched_dare_row["worst_acc"]),
        "expert_matched_ties_dare_worst_acc": float(expert_matched_ties_dare_row["worst_acc"]),
        "expert_matched_ties_minus_matched_average_worst_acc": float(
            expert_matched_ties_row["worst_acc"] - expert_matched_row["worst_acc"]
        ),
        "expert_matched_dare_minus_matched_average_worst_acc": float(
            expert_matched_dare_row["worst_acc"] - expert_matched_row["worst_acc"]
        ),
        "expert_matched_ties_dare_minus_matched_average_worst_acc": float(
            expert_matched_ties_dare_row["worst_acc"] - expert_matched_row["worst_acc"]
        ),
        "router_weight_search": {
            "grid": router_weight_search_grid,
            "candidate_pair_count": len(router_weight_search_pairs),
            "max_delta_sum": args.router_weight_search_max_delta_sum,
            "min_topk_jaccard": args.router_weight_search_min_topk_jaccard,
            "min_top1_agreement": args.router_weight_search_min_top1_agreement,
            "eligible_count": int(router_weight_search["eligible_by_route_guard"].sum()),
            "selected_weight_general": float(
                router_weight_search[router_weight_search["selected_by_guarded_calib_worst_loss"]].iloc[0][
                    "router_weight_general"
                ]
            ),
            "selected_weight_code": float(
                router_weight_search[router_weight_search["selected_by_guarded_calib_worst_loss"]].iloc[0][
                    "router_weight_code"
                ]
            ),
            "selected_calib_worst_loss": float(
                router_weight_search[router_weight_search["selected_by_guarded_calib_worst_loss"]].iloc[0][
                    "calib_worst_loss"
                ]
            ),
            "selected_test_worst_acc": float(matched_router_weight_search_row["worst_acc"]),
            "selected_minus_frozen_worst_acc": float(
                matched_router_weight_search_row["worst_acc"] - matched_router_frozen_row["worst_acc"]
            ),
        },
        "matched_router_weight_search_worst_acc": float(matched_router_weight_search_row["worst_acc"]),
        "matched_router_hessian_worst_acc": float(matched_router_hessian_row["worst_acc"]),
        "matched_router_hessian_minus_expert_matched_worst_acc": float(
            matched_router_hessian_row["worst_acc"] - expert_matched_row["worst_acc"]
        ),
        "matched_router_hessian_minus_matched_calibrated_worst_acc": float(
            matched_router_hessian_row["worst_acc"] - matched_router_calibrated_row["worst_acc"]
        ),
        "router_hessian": {
            "ridge": args.router_hessian_ridge,
            "rows": int(len(router_hessian_average)),
            "precision_trace": float(
                router_hessian_average[router_hessian_average["source"] == "solved_router"].iloc[0]["precision_trace"]
            ),
            "ridge_value": float(
                router_hessian_average[router_hessian_average["source"] == "solved_router"].iloc[0]["ridge_value"]
            ),
        },
        "matched_router_kd_worst_acc": float(matched_router_kd_row["worst_acc"]),
        "matched_router_kd_minus_expert_matched_worst_acc": float(
            matched_router_kd_row["worst_acc"] - expert_matched_row["worst_acc"]
        ),
        "matched_router_kd_minus_matched_calibrated_worst_acc": float(
            matched_router_kd_row["worst_acc"] - matched_router_calibrated_row["worst_acc"]
        ),
        "router_kd": {
            "epochs": args.router_kd_epochs,
            "lr": args.router_kd_lr,
            "temperature": args.router_kd_temperature,
            "router_kl_coef": args.router_kd_kl_coef,
            "rows": int(len(router_kd_trace)),
            "final_mean_kd_loss": float(router_kd_trace[router_kd_trace["epoch"] == router_kd_trace["epoch"].max()]["kd_loss"].mean()),
        },
        "matched_router_route_kd_worst_acc": float(matched_router_route_kd_row["worst_acc"]),
        "matched_router_route_kd_minus_expert_matched_worst_acc": float(
            matched_router_route_kd_row["worst_acc"] - expert_matched_row["worst_acc"]
        ),
        "matched_router_route_kd_minus_matched_calibrated_worst_acc": float(
            matched_router_route_kd_row["worst_acc"] - matched_router_calibrated_row["worst_acc"]
        ),
        "matched_router_route_kd_minus_router_kd_worst_acc": float(
            matched_router_route_kd_row["worst_acc"] - matched_router_kd_row["worst_acc"]
        ),
        "router_route_kd": {
            "epochs": args.router_route_kd_epochs,
            "lr": args.router_route_kd_lr,
            "temperature": args.router_route_kd_temperature,
            "top1_loss_coef": args.router_route_kd_top1_loss_coef,
            "rows": int(len(router_route_kd_trace)),
            "final_mean_route_kl": float(
                router_route_kd_trace[router_route_kd_trace["epoch"] == router_route_kd_trace["epoch"].max()][
                    "route_kl"
                ].mean()
            ),
            "final_mean_top1_ce": float(
                router_route_kd_trace[router_route_kd_trace["epoch"] == router_route_kd_trace["epoch"].max()][
                    "top1_ce"
                ].mean()
            ),
        },
        "router_calibration": {
            "epochs": args.router_calibration_epochs,
            "lr": args.router_calibration_lr,
            "kl_coef": args.router_calibration_kl_coef,
            "aux_coef": args.aux_coef,
        },
        "router_calibration_sweep": {
            "kl_values": router_calibration_kl_values,
            "min_topk_jaccard": args.router_calibration_sweep_min_topk_jaccard,
            "min_top1_agreement": args.router_calibration_sweep_min_top1_agreement,
            "eligible_count": int(router_calibration_sweep["eligible_by_route_guard"].sum()),
            "selected_kl_coef": float(
                router_calibration_sweep[router_calibration_sweep["selected_by_guarded_calib_worst_loss"]].iloc[0][
                    "kl_coef"
                ]
            ),
            "selected_calib_worst_loss": float(
                router_calibration_sweep[router_calibration_sweep["selected_by_guarded_calib_worst_loss"]].iloc[0][
                    "calib_worst_loss"
                ]
            ),
            "selected_test_worst_acc": float(matched_router_sweep_selected_row["worst_acc"]),
            "selected_minus_fixed_kl_worst_acc": float(
                matched_router_sweep_selected_row["worst_acc"] - matched_router_calibrated_row["worst_acc"]
            ),
        },
        "matched_router_sweep_selected_worst_acc": float(matched_router_sweep_selected_row["worst_acc"]),
        "expert_weight_search_worst_acc": float(expert_search_row["worst_acc"]),
        "expert_weight_search_router_calibrated_worst_acc": float(expert_search_router_calibrated_row["worst_acc"]),
        "expert_weight_search_router_calibrated_minus_all_weight_worst_acc": float(
            expert_search_router_calibrated_row["worst_acc"] - all_avg["worst_acc"]
        ),
        "expert_weight_search_router_calibrated_minus_matched_calibrated_worst_acc": float(
            expert_search_router_calibrated_row["worst_acc"] - matched_router_calibrated_row["worst_acc"]
        ),
        "expert_output_projection_worst_acc": float(expert_output_projection_row["worst_acc"]),
        "expert_output_projection_router_calibrated_worst_acc": float(
            expert_output_projection_router_calibrated_row["worst_acc"]
        ),
        "expert_output_projection_router_calibrated_minus_all_weight_worst_acc": float(
            expert_output_projection_router_calibrated_row["worst_acc"] - all_avg["worst_acc"]
        ),
        "expert_output_projection_router_calibrated_minus_matched_calibrated_worst_acc": float(
            expert_output_projection_router_calibrated_row["worst_acc"] - matched_router_calibrated_row["worst_acc"]
        ),
        "expert_output_projection": {
            "ridge": args.output_projection_ridge,
            "max_delta_sum": args.output_projection_max_delta_sum,
            "dispatch_mode": args.output_projection_dispatch_mode,
            "shared_weights": {
                "general": args.output_projection_shared_general_weight,
                "code": args.output_projection_shared_code_weight,
            },
            "rows": int(len(expert_output_projection_weights)),
            "mean_captured_fraction": float(
                expert_output_projection_weights[expert_output_projection_weights["category"] == "combined"][
                    "captured_fraction"
                ].mean()
            ),
            "mean_residual_energy": float(
                expert_output_projection_weights[expert_output_projection_weights["category"] == "combined"][
                    "residual_energy"
                ].mean()
            ),
        },
        "unified_moe_worst_acc": float(unified_moe_row["worst_acc"]),
        "unified_moe_minus_expert_search_worst_acc": float(
            unified_moe_row["worst_acc"] - expert_search_row["worst_acc"]
        ),
        "unified_moe_minus_expert_search_router_calibrated_worst_acc": float(
            unified_moe_row["worst_acc"] - expert_search_router_calibrated_row["worst_acc"]
        ),
        "unified_moe_minus_route_kd_worst_acc": float(
            unified_moe_row["worst_acc"] - matched_router_route_kd_row["worst_acc"]
        ),
        "unified_moe": {
            "epochs": args.unified_router_epochs,
            "lr": args.unified_router_lr,
            "temperature": args.unified_router_temperature,
            "dispatch_mode": args.unified_router_dispatch_mode,
            "soft_loss_coef": args.unified_router_soft_loss_coef,
            "dispatch_loss_coef": args.unified_router_dispatch_loss_coef,
            "route_kd_coef": args.unified_router_route_kd_coef,
            "output_kd_coef": args.unified_router_output_kd_coef,
            "top1_loss_coef": args.unified_router_top1_loss_coef,
            "base_kl_coef": args.unified_router_base_kl_coef,
            "load_balance_coef": args.unified_router_load_balance_coef,
            "capacity_loss_coef": args.unified_router_capacity_loss_coef,
            "capacity_factor": args.capacity_factor,
            "rows": int(len(unified_moe_trace)),
            "final_mean_total_loss": float(
                unified_moe_trace[unified_moe_trace["epoch"] == unified_moe_trace["epoch"].max()][
                    "total_loss"
                ].mean()
            ),
            "final_mean_route_kl": float(
                unified_moe_trace[unified_moe_trace["epoch"] == unified_moe_trace["epoch"].max()][
                    "route_kl"
                ].mean()
            ),
            "final_mean_output_kd": float(
                unified_moe_trace[unified_moe_trace["epoch"] == unified_moe_trace["epoch"].max()][
                    "output_kd"
                ].mean()
            ),
            "final_mean_capacity_overflow": float(
                unified_moe_trace[unified_moe_trace["epoch"] == unified_moe_trace["epoch"].max()][
                    "capacity_overflow"
                ].mean()
            ),
            "final_mean_capacity_ratio": float(
                unified_moe_trace[unified_moe_trace["epoch"] == unified_moe_trace["epoch"].max()][
                    "capacity_ratio"
                ].mean()
            ),
        },
        "expert_weight_search": {
            "grid": expert_search_grid,
            "candidate_pair_count": len(expert_search_pairs),
            "passes": args.expert_search_passes,
            "prior_penalty": args.expert_search_prior_penalty,
            "objective": args.expert_search_objective,
            "max_delta_sum": args.expert_search_max_delta_sum,
            "shared_weights": {
                "general": args.expert_search_shared_general_weight,
                "code": args.expert_search_shared_code_weight,
            },
        },
        "route_aware_worst_acc": float(route_aware_row["worst_acc"]),
        "route_aware_minus_all_weight_worst_acc": float(route_aware_row["worst_acc"] - all_avg["worst_acc"]),
        "same_shape_constraint": "All methods keep the same TinyMoEClassifier architecture, expert count, router shape, and output classes.",
        "outputs": {
            "method_metrics": "method_metrics.csv",
            "dispatch_mode_metrics": "dispatch_mode_metrics.csv",
            "router_summary": "router_summary.csv",
            "expert_load": "expert_load.csv",
            "router_capacity_metrics": "router_capacity_metrics.csv",
            "route_overlap": "route_overlap.csv",
            "expert_match": "expert_match.csv",
            "route_weights_by_expert": "route_weights_by_expert.csv",
            "expert_regmean_covariances": "expert_regmean_covariances.csv",
            "expert_regmean_layers": "expert_regmean_layers.csv",
            "expert_sparse_task_vectors": "expert_sparse_task_vectors.csv",
            "expert_search_weights_by_expert": "expert_search_weights_by_expert.csv",
            "expert_weight_search_trace": "expert_weight_search_trace.csv",
            "expert_output_projection_weights_by_expert": "expert_output_projection_weights_by_expert.csv",
            "router_weight_search": "router_weight_search.csv",
            "router_hessian_average": "router_hessian_average.csv",
            "router_kd_trace": "router_kd_trace.csv",
            "router_route_kd_trace": "router_route_kd_trace.csv",
            "unified_moe_trace": "unified_moe_trace.csv",
            "router_calibration_sweep": "router_calibration_sweep.csv",
            "figure": "toy_moe_merge.png",
            "connectivity_path_metrics": "connectivity_path_metrics.csv",
            "connectivity_summary": "connectivity_summary.csv",
            "connectivity_figure": "connectivity_paths.png",
            "report": "report.md",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_dir, summary, method_metrics)
    print(f"Wrote toy MoE merge results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
