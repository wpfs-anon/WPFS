"""Unmerged, boolean-gated LoRA (SS2.9, *Where the adapters go*).

"Adapters are kept unmerged and gated by a boolean, since plan mode and check
mode alternate within a single episode; merging would require a weight copy per
switch."

That constraint is why this is hand-rolled rather than delegated to ``peft``:
the merge/unmerge dance that libraries optimise for is exactly what we must
avoid, and we need the gate to be a per-forward flag rather than a state
transition.

**Proposition 4 depends on this module.**  ``B`` is zero-initialised and the
gated-off path returns ``base(x)`` untouched -- not ``base(x) + 0``, which
would still perturb the last bit through floating-point addition.  With
adapters off the network is *bit-identical* to the pretrained policy, which
:mod:`sentry.tests.test_lora` asserts by exact tensor equality.
"""

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
    """``B(A(dropout(x)))``, computed in ``x``'s dtype with fp32 master weights.

    The adapter keeps its own precision: a real pi_0 checkpoint runs the frozen
    trunk in bf16, but a rank-``r`` update trained by AdamW wants fp32 master
    weights, since 8 bits of mantissa cannot accumulate small updates and the
    adapter is the *only* thing Stage A/B trains.

    Which side of that gap gets cast is a performance decision, not a numerical
    one, and the obvious choice is the wrong one.  Casting the **activations**
    up to fp32 materialises a full copy of them per adapted projection -- for
    SigLIP that is 768 tokens x 1152 channels, roughly 3.5 MB, forty-eight times
    per check.  Measured on pi_0 that overhead *exceeded* the saving from
    skipping nineteen of twenty-seven encoder layers, so truncation came out
    slower than full depth.  Casting the **weights** down instead moves 36k
    values rather than 884k, and the cast is differentiable, so gradients still
    land on the fp32 parameters.
    """
    a = w_a.to(x.dtype)
    b = w_b.to(x.dtype)
    return F.linear(F.linear(dropout(x), a), b).to(out_dtype)


class GatedAdapter(nn.Module):
    """Common surface for everything that wraps a frozen ``nn.Linear``.

    Exists so the gate protocol lives in one place: with the gate off, *every*
    adapter must return ``base(x)`` untouched, and Proposition 4 is a statement
    about that invariant rather than about any particular adapter shape.
    """

    gate: bool

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        """The parameters a training stage may update."""
        raise NotImplementedError


class LoRALinear(GatedAdapter):
    """``nn.Linear`` plus a gated low-rank update.

    ``y = base(x) + gate * (alpha/r) * B(A(dropout(x)))``

    Defaults follow SS2.9: rank ``r = 16`` (32 for the action read-out),
    ``alpha_LoRA = 2r``, dropout 0.05.
    """

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
            p.requires_grad_(False)  # theta is frozen throughout (SS2.5)

        self.r = r
        self.alpha = alpha if alpha is not None else 2 * r  # alpha_LoRA = 2r
        self.scaling = self.alpha / self.r

        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Kaiming on A, zeros on B: Delta contributes exactly zero at init, so
        # even a freshly constructed adapter cannot perturb plan mode.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Not an nn.Parameter or buffer: it is control flow, not state, and
        # must not travel in state_dict or move with .to().
        self.gate: bool = False

    # -- transparency to attribute inspection -----------------------------
    #
    # A wrapper that hides the base layer's attributes breaks any host code
    # that *introspects* rather than calls.  openpi does exactly that in three
    # places -- ``modeling_siglip`` and ``modeling_gemma`` both sniff the run
    # dtype off ``layers[0].self_attn.q_proj.weight.dtype``, and so does
    # ``PI0Pytorch.forward`` -- so wrapping q_proj without these raises
    # ``'LoRALinear' object has no attribute 'weight'`` from inside the model.
    #
    # These are read-only views on the base layer, not parameters of the
    # wrapper: ``base`` is a submodule, so its weight already appears in
    # ``state_dict`` as ``base.weight`` and is not duplicated here.

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
            # Return the base result untouched -- see the module docstring.
            return out
        return out + self.scaling * _low_rank(
            x, self.lora_A.weight, self.lora_B.weight, self.lora_dropout, out.dtype
        )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"r={self.r}, alpha={self.alpha}, gate={self.gate}"

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()


class MultiLoRALinear(GatedAdapter):
    r"""One frozen projection carrying ``J`` selectable low-rank updates.

    This is the *per-rung read-out adapter* of SS3.5.2:

        "a per-rung read-out adapter :math:`\Delta^{(j)}_{\mathrm{out}}` on the
        action read-out projection, one for each :math:`E^{(j)} \in \mathcal{E}`"

    and SS3.9 pins down that they share a projection rather than duplicating it:
    they "attach to the **single** action read-out projection and are selected by
    which rung is being evaluated".  So this is *not* ``J`` heads, and not a
    separate path from layer :math:`E^{(j)}` to the head.  The path is always
    the same one projection; what changes per rung is which rank-``r`` update is
    added to it:

    .. math::
        y^{(j)} = W_{\mathrm{out}} h + \tfrac{\alpha}{r} B_j A_j h

    Why not LayerSkip's single shared head: that "imposes the constraint that
    the residual streams at layer :math:`E^{(1)}` and layer :math:`E^{(J)}`
    inhabit a common space; we have no use for that constraint, since only
    :math:`J = 3` depths are ever queried."  And the price is small -- a rank-32
    update on one projection, so ``J`` of them cost less than the trunk adapter
    of a single layer.

    ``lora_A`` and ``lora_B`` are ``ModuleList``s on purpose: their parameters
    are then named ``...lora_A.<j>.weight``, which
    :func:`sentry.training.params.split_adapters` already classifies correctly
    without knowing this class exists.
    """

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
        """Which rung is being evaluated.  Set by :func:`set_adapter_slot`."""

    # -- transparency, for the same reason LoRALinear needs it ------------

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

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"r={self.r}, slots={self.n_slots}, slot={self.slot}, gate={self.gate}"

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()


# --------------------------------------------------------------------------
# Gating
# --------------------------------------------------------------------------


def set_adapters(module: nn.Module, enabled: bool) -> int:
    """Gate every adapter beneath ``module``.  Returns the count."""
    n = 0
    for m in module.modules():
        if isinstance(m, GatedAdapter):
            m.gate = enabled
            n += 1
    return n


@contextmanager
def adapters(module: nn.Module, enabled: bool) -> Iterator[None]:
    """Temporarily gate adapters, restoring the previous state on exit.

    Plan mode and check mode alternate within a single episode, so the gate is
    flipped constantly; a context manager keeps an exception from leaving the
    model in check mode and silently corrupting the next plan.
    """
    previous = [(m, m.gate) for m in module.modules() if isinstance(m, GatedAdapter)]
    try:
        for m, _ in previous:
            m.gate = enabled
        yield
    finally:
        for m, was in previous:
            m.gate = was


def set_adapter_slot(module: nn.Module, j: int) -> int:
    """Select rung ``j`` on every :class:`MultiLoRALinear`.  Returns the count."""
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
    """Temporarily select rung ``j``, restoring the previous selection on exit.

    The cascade escalates within a single check, so the slot changes as often as
    the gate does; leaving a stale slot behind would silently evaluate the next
    rung's read-out against this rung's hidden state.
    """
    previous = [(m, m.slot) for m in module.modules() if isinstance(m, MultiLoRALinear)]
    try:
        set_adapter_slot(module, j)
        yield
    finally:
        for m, was in previous:
            m.slot = was


# --------------------------------------------------------------------------
# Construction and inspection
# --------------------------------------------------------------------------


def wrap_linear(
    parent: nn.Module,
    attr: str,
    r: int = 16,
    alpha: int | None = None,
    dropout: float = 0.05,
) -> LoRALinear:
    """Replace ``parent.<attr>`` (an ``nn.Linear``) with a :class:`LoRALinear`."""
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
    """Replace ``parent.<attr>`` with a :class:`MultiLoRALinear` of ``n_slots``.

    Use this for the action read-out and nothing else: SS3.5.2 gives *one*
    adapter per rung to the read-out, while the trunk adapter is shared across
    rungs by design -- "the trunk, by contrast, is deliberately shared across
    rungs: it is what allows one training run to serve the whole ladder."
    """
    base = getattr(parent, attr)
    if isinstance(base, MultiLoRALinear):
        return base
    if isinstance(base, LoRALinear):
        # Unwrap a single-slot adapter rather than nesting one inside another,
        # which would leave the inner one permanently gated and invisible.
        base = base.base
    if not isinstance(base, nn.Linear):
        raise TypeError(
            f"{type(parent).__name__}.{attr} is {type(base).__name__}, not nn.Linear"
        )
    wrapped = MultiLoRALinear(base, n_slots=n_slots, r=r, alpha=alpha, dropout=dropout)
    setattr(parent, attr, wrapped)
    return wrapped


def lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """Yield only the adapter parameters -- everything Stage A/B may train."""
    for m in module.modules():
        if isinstance(m, GatedAdapter):
            yield from m.adapter_parameters()


def count_lora_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(module))
