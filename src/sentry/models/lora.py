from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "GatedAdapter",
    "LoRALinear",
    "MultiLoRALinear",
    "set_adapters",
    "adapters",
    "set_adapter_slot",
    "adapter_slot",
    "lora_parameters",
    "wrap_linear",
    "wrap_linear_multi",
    "count_lora_parameters",
]


def _low_rank(
    x: Tensor,
    w_a: Tensor,
    w_b: Tensor,
    dropout: nn.Module,
    out_dtype: torch.dtype,
) -> Tensor:
    a = w_a.to(x.dtype)
    b = w_b.to(x.dtype)
    return F.linear(F.linear(dropout(x), a), b).to(out_dtype)


class GatedAdapter(nn.Module):

    gate: bool

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError


class LoRALinear(GatedAdapter):

    def __init__(
        self,
        base: nn.Linear,
        r: int = 16,
        alpha: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if r < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {r}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.alpha = alpha if alpha is not None else 2 * r
        self.scaling = self.alpha / self.r

        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.gate: bool = False


    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if not self.gate:
            return out
        return out + self.scaling * _low_rank(
            x, self.lora_A.weight, self.lora_B.weight, self.lora_dropout, out.dtype
        )

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, gate={self.gate}"

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()


class MultiLoRALinear(GatedAdapter):

    def __init__(
        self,
        base: nn.Linear,
        n_slots: int,
        r: int = 32,
        alpha: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if r < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {r}")
        if n_slots < 1:
            raise ValueError(f"need at least one rung, got {n_slots}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.alpha = alpha if alpha is not None else 2 * r
        self.scaling = self.alpha / self.r
        self.n_slots = n_slots

        self.lora_A = nn.ModuleList(
            [nn.Linear(base.in_features, r, bias=False) for _ in range(n_slots)]
        )
        self.lora_B = nn.ModuleList(
            [nn.Linear(r, base.out_features, bias=False) for _ in range(n_slots)]
        )
        self.lora_dropout = nn.Dropout(dropout)

        for a, b in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(a.weight, a=5**0.5)
            nn.init.zeros_(b.weight)

        self.gate: bool = False
        self.slot: int = 0


    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if not self.gate:
            return out
        if not (0 <= self.slot < self.n_slots):
            raise IndexError(
                f"adapter slot {self.slot} outside [0, {self.n_slots}) -- the "
                "cascade must select a rung before evaluating check mode"
            )
        a, b = self.lora_A[self.slot], self.lora_B[self.slot]
        return out + self.scaling * _low_rank(
            x, a.weight, b.weight, self.lora_dropout, out.dtype
        )

    def extra_repr(self) -> str:
        return f"r={self.r}, slots={self.n_slots}, slot={self.slot}, gate={self.gate}"

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()


def set_adapters(module: nn.Module, enabled: bool) -> int:
    n = 0
    for m in module.modules():
        if isinstance(m, GatedAdapter):
            m.gate = enabled
            n += 1
    return n


@contextmanager
def adapters(module: nn.Module, enabled: bool) -> Iterator[None]:
    previous = [(m, m.gate) for m in module.modules() if isinstance(m, GatedAdapter)]
    try:
        for m, _ in previous:
            m.gate = enabled
        yield
    finally:
        for m, was in previous:
            m.gate = was


def set_adapter_slot(module: nn.Module, j: int) -> int:
    n = 0
    for m in module.modules():
        if isinstance(m, MultiLoRALinear):
            if not (0 <= j < m.n_slots):
                raise IndexError(f"rung {j} outside [0, {m.n_slots})")
            m.slot = j
            n += 1
    return n


@contextmanager
def adapter_slot(module: nn.Module, j: int) -> Iterator[None]:
    previous = [(m, m.slot) for m in module.modules() if isinstance(m, MultiLoRALinear)]
    try:
        set_adapter_slot(module, j)
        yield
    finally:
        for m, was in previous:
            m.slot = was


def wrap_linear(
    parent: nn.Module,
    attr: str,
    r: int = 16,
    alpha: int | None = None,
    dropout: float = 0.05,
) -> LoRALinear:
    base = getattr(parent, attr)
    if isinstance(base, LoRALinear):
        return base
    if not isinstance(base, nn.Linear):
        raise TypeError(f"{type(parent).__name__}.{attr} is {type(base).__name__}, not nn.Linear")
    wrapped = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
    setattr(parent, attr, wrapped)
    return wrapped


def wrap_linear_multi(
    parent: nn.Module,
    attr: str,
    n_slots: int,
    r: int = 32,
    alpha: int | None = None,
    dropout: float = 0.05,
) -> MultiLoRALinear:
    base = getattr(parent, attr)
    if isinstance(base, MultiLoRALinear):
        return base
    if isinstance(base, LoRALinear):
        base = base.base
    if not isinstance(base, nn.Linear):
        raise TypeError(
            f"{type(parent).__name__}.{attr} is {type(base).__name__}, not nn.Linear"
        )
    wrapped = MultiLoRALinear(base, n_slots=n_slots, r=r, alpha=alpha, dropout=dropout)
    setattr(parent, attr, wrapped)
    return wrapped


def lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    for m in module.modules():
        if isinstance(m, GatedAdapter):
            yield from m.adapter_parameters()


def count_lora_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(module))
