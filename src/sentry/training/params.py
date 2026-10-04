from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch.nn as nn
from torch import Tensor

from sentry.models.lora import GatedAdapter

__all__ = ["AdapterGroups", "split_adapters", "freeze_all_but", "audit", "AuditReport"]


@dataclass
class AdapterGroups:

    delta_V: list[tuple[str, nn.Parameter]]
    delta_B: list[tuple[str, nn.Parameter]]

    def params_V(self) -> list[nn.Parameter]:
        return [p for _, p in self.delta_V]

    def params_B(self) -> list[nn.Parameter]:
        return [p for _, p in self.delta_B]

    def summary(self) -> str:
        nv = sum(p.numel() for p in self.params_V())
        nb = sum(p.numel() for p in self.params_B())
        return (
            f"Delta_V: {len(self.delta_V):3d} tensors, {nv:,} params\n"
            f"Delta_B: {len(self.delta_B):3d} tensors, {nb:,} params"
        )


def split_adapters(
    model: nn.Module,
    encoder_prefixes: Sequence[str] = ("encoder",),
    backbone_prefixes: Sequence[str] = ("backbone", "readout"),
) -> AdapterGroups:
    delta_V: list[tuple[str, nn.Parameter]] = []
    delta_B: list[tuple[str, nn.Parameter]] = []
    unclaimed: list[str] = []

    lora_paths = {
        name for name, mod in model.named_modules() if isinstance(mod, GatedAdapter)
    }

    for name, p in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        owner = name.rsplit(".lora_", 1)[0]
        if owner not in lora_paths:
            unclaimed.append(name)
            continue
        if any(owner.startswith(pre) for pre in encoder_prefixes):
            delta_V.append((name, p))
        elif any(owner.startswith(pre) for pre in backbone_prefixes):
            delta_B.append((name, p))
        else:
            unclaimed.append(name)

    if unclaimed:
        raise ValueError(
            f"{len(unclaimed)} adapter tensors match neither Delta_V nor Delta_B "
            f"(first: {unclaimed[0]}).  An unclassified adapter is trained by no "
            "stage and stays at its zero initialisation -- silently inert."
        )
    return AdapterGroups(delta_V=delta_V, delta_B=delta_B)


def freeze_all_but(model: nn.Module, trainable: Iterable[nn.Parameter]) -> None:
    keep = {id(p) for p in trainable}
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)


@dataclass
class AuditReport:
    stage: str
    expected: int
    trainable: int
    connected: int
    leaked: list[str]
    disconnected: list[str]

    @property
    def ok(self) -> bool:
        return not self.leaked and self.connected > 0 and self.trainable == self.expected

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        out = (
            f"[{status}] {self.stage}: {self.trainable}/{self.expected} trainable, "
            f"{self.connected} in the graph"
        )
        if self.leaked:
            out += f" | LEAK({len(self.leaked)}): {self.leaked[:2]}"
        if self.disconnected:
            out += f" | inactive-at-this-depth: {len(self.disconnected)}"
        return out


def audit(
    model: nn.Module,
    stage: str,
    expected_trainable: Sequence[tuple[str, nn.Parameter]],
) -> AuditReport:
    expected_ids = {id(p) for _, p in expected_trainable}
    leaked, disconnected, connected = [], [], 0

    for name, p in model.named_parameters():
        in_graph = p.grad is not None
        if id(p) in expected_ids:
            if in_graph:
                connected += 1
            else:
                disconnected.append(name)
        elif in_graph:
            leaked.append(name)

    return AuditReport(
        stage=stage,
        expected=len(expected_trainable),
        trainable=sum(1 for p in model.parameters() if p.requires_grad),
        connected=connected,
        leaked=leaked,
        disconnected=disconnected,
    )
