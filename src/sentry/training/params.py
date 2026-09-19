"""Partitioning adapters into ``Delta_V`` and ``Delta_B`` (SS2.5).

Each stage "freezes everything except a small adapter set", and the sets are
different:

- **Stage A** trains ``Delta_V`` -- "LoRA adapters on the attention and MLP
  projections of encoder layers ``0..E_V-1``.  **Nothing else.**"
- **Stage B** trains ``Delta_B`` on backbone layers plus the action read-out,
  with "``Delta_V`` from Stage A ... frozen".

Getting this wrong produces a model that trains and whose loss goes down. The
damage is silent: Stage B quietly re-tuning ``Delta_V`` undoes the perception
warm-start that Stage A paid 20k steps for, and nothing in the loss curve says
so.  Hence :func:`audit` -- assert the partition rather than trust it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch.nn as nn
from torch import Tensor

from sentry.models.lora import GatedAdapter

__all__ = ["AdapterGroups", "split_adapters", "freeze_all_but", "audit", "AuditReport"]


@dataclass
class AdapterGroups:
    """Named adapter parameters, split by which stage owns them."""

    delta_V: list[tuple[str, nn.Parameter]]
    """Encoder adapters -- trained in Stage A, frozen in Stage B."""
    delta_B: list[tuple[str, nn.Parameter]]
    """Backbone + read-out adapters -- trained in Stage B only."""

    def params_V(self) -> list[nn.Parameter]:
        return [p for _, p in self.delta_V]

    def params_B(self) -> list[nn.Parameter]:
        return [p for _, p in self.delta_B]

    def summary(self) -> str:  # pragma: no cover - cosmetic
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
    """Partition every LoRA parameter into ``Delta_V`` or ``Delta_B`` by module path.

    The read-out counts as ``Delta_B``: SS2.5.2 lists it alongside the backbone
    adapters, because "it was trained to consume features from layer ``L_B``,
    and feeding it features from layer ``E_B`` is the principal source of
    mismatch".

    Raises if any adapter matches neither group -- an unclassified adapter
    would be trained by no stage and stay at its zero initialisation, which is
    invisible unless you go looking.
    """
    delta_V: list[tuple[str, nn.Parameter]] = []
    delta_B: list[tuple[str, nn.Parameter]] = []
    unclaimed: list[str] = []

    lora_paths = {
        name for name, mod in model.named_modules() if isinstance(mod, GatedAdapter)
    }

    for name, p in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        # ``rsplit`` on the *last* occurrence handles both shapes: a single
        # adapter names its tensors ``<owner>.lora_A.weight``, and a per-rung
        # read-out names them ``<owner>.lora_A.<j>.weight``.  Both resolve to
        # the same owner path, so the per-rung adapters of SS3.5.2 classify as
        # Delta_B without this function needing to know they exist.
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
    """Set ``requires_grad`` on exactly ``trainable`` and nothing else.

    ``theta`` is frozen throughout (SS2.5), so this is the only mechanism that
    decides what a stage may change.
    """
    keep = {id(p) for p in trainable}
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)


@dataclass
class AuditReport:
    stage: str
    expected: int
    trainable: int
    connected: int
    """Expected parameters that received a gradient *tensor* -- i.e. that are
    in the autograd graph at all."""
    leaked: list[str]
    """Frozen parameters that received a gradient.  Always a bug."""
    disconnected: list[str]
    """Expected parameters that received no gradient tensor.

    **Not automatically a bug.**  An adapter attached to a layer beyond the
    current truncation depth never executes, so it legitimately has no
    gradient this step -- and under the Stage-B depth curriculum
    (``E_B ~ p_depth``) which layers those are changes every step by design.
    """

    @property
    def ok(self) -> bool:
        # The hard invariant is that nothing frozen moved.  Everything else is
        # a judgement call about depth, reported rather than enforced.
        return not self.leaked and self.connected > 0 and self.trainable == self.expected

    def __str__(self) -> str:  # pragma: no cover - cosmetic
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
    """Check, after a backward pass, that exactly the right tensors moved.

    Membership in the autograd graph is tested by ``p.grad is not None``, not
    by gradient magnitude.  A LoRA pair initialises ``B`` at zero, and
    ``dL/dA = scaling * B^T * dL/dout * x^T`` is therefore **exactly zero on
    the first backward** -- ``A`` only starts moving once ``B`` has left zero.
    Treating a zero-magnitude gradient as evidence of disconnection would flag
    every ``lora_A`` tensor in the model on step 0, which is a property of
    LoRA rather than a defect.  For a sharper reading, run a couple of steps
    before auditing.

    The invariant enforced is the one that is always a bug:

    - **leak** -- a frozen parameter received a gradient.  Stage B re-tuning
      ``Delta_V`` undoes the perception warm-start Stage A paid 20k steps for;
      anything touching ``theta`` breaks Proposition 4 outright.

    Disconnection is reported but not enforced, because at a truncated depth
    the adapters above that depth *should* be idle.
    """
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
