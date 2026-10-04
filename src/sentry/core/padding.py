from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from sentry.config import PaddingScheme
from sentry.core.types import ChannelSpec

__all__ = ["PlanExhausted", "repad"]


class PlanExhausted(RuntimeError):
    pass


def repad(
    live_suffix: Tensor,
    H: int,
    scheme: PaddingScheme,
    spec: ChannelSpec,
    learned_pad: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    if live_suffix.ndim != 2:
        raise ValueError(f"live_suffix must be (H_k, d_a), got {tuple(live_suffix.shape)}")

    H_k, d_a = live_suffix.shape
    if H_k == 0:
        raise PlanExhausted(
            "H_k = 0: the plan is exhausted and a full replan is mandatory "
            "(Section 2.4.1)."
        )
    if H_k > H:
        raise ValueError(f"live suffix longer than the chunk width: {H_k} > {H}")
    if d_a != spec.d_a:
        raise ValueError(f"action dim mismatch: suffix has {d_a}, spec has {spec.d_a}")

    A_hat = torch.empty(H, d_a, dtype=live_suffix.dtype, device=live_suffix.device)
    A_hat[:H_k] = live_suffix

    n_tail = H - H_k
    if n_tail > 0:
        A_hat[H_k:] = _pad_value(live_suffix, scheme, spec, learned_pad).expand(n_tail, d_a)

    return A_hat, H_k


def _pad_value(
    live_suffix: Tensor,
    scheme: PaddingScheme,
    spec: ChannelSpec,
    learned_pad: Optional[Tensor],
) -> Tensor:
    last = live_suffix[-1]

    if scheme == "hold":
        return last.unsqueeze(0)

    if scheme == "zero_delta":
        return spec.zero_delta_action(last[spec.grip]).unsqueeze(0)

    if scheme == "learned":
        if learned_pad is None:
            raise ValueError(
                "padding='learned' requires a learned_pad embedding; it is a "
                "trainable parameter of the system (Section 2.4.1)."
            )
        if learned_pad.shape != (spec.d_a,):
            raise ValueError(
                f"learned_pad must be ({spec.d_a},), got {tuple(learned_pad.shape)}"
            )
        return learned_pad.unsqueeze(0).to(dtype=live_suffix.dtype, device=live_suffix.device)

    raise ValueError(f"unknown padding scheme {scheme!r}")
