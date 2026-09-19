"""Candidate construction -- equation 7 (SS2.4.1).

The target's velocity field is trained on chunks whose index 0 denotes "the
action to execute now, given the conditioning observation".  Since the
conditioning observation at check time is ``o_{t+k}``, the live suffix must be
**re-indexed so that** ``a_{t+k}`` **occupies index 0**.  Getting this wrong is
silent: the model still returns a plausible velocity, just for the wrong
question.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from sentry.config import PaddingScheme
from sentry.core.types import ChannelSpec

__all__ = ["PlanExhausted", "repad"]


class PlanExhausted(RuntimeError):
    """Raised when ``H_k`` reaches zero.

    SS2.4.1: "when ``H_k`` reaches zero the plan is exhausted and a full replan
    is mandatory."  Algorithm 1 guards this with ``if k >= H: break`` before
    calling ``REPAD``; this exception exists so a caller that forgets cannot
    silently verify an empty suffix.
    """


def repad(
    live_suffix: Tensor,
    H: int,
    scheme: PaddingScheme,
    spec: ChannelSpec,
    learned_pad: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    r"""Build the full-width candidate ``A_hat`` of equation 7.

    .. math::
        \hat{A}[h] = \begin{cases}
            \tilde{A}_{t,k}[h] = a_{t+k+h} & 0 \le h < H_k \\
            \mathrm{pad}(\tilde{A}_{t,k})  & H_k \le h < H
        \end{cases}

    Args:
        live_suffix: ``(H_k, d_a)`` the surviving actions ``[a_{t+k}, ...,
            a_{t+H-1}]``, **already re-indexed** so index 0 is "act now", in
            the policy's normalised action space.
        H: full chunk width the velocity field expects.
        scheme: one of ``hold`` / ``zero_delta`` / ``learned``.
        spec: channel semantics, used by ``zero_delta``.
        learned_pad: ``(d_a,)`` trainable embedding, required by ``learned``.

    Returns:
        ``(A_hat, H_k)`` with ``A_hat`` of shape ``(H, d_a)``.

    The three schemes (SS2.4.1):

    - ``hold`` -- repeat the final surviving action.
    - ``zero_delta`` -- a null incremental motion with the gripper channel
      **held** (not zeroed); see :meth:`ChannelSpec.zero_delta_action`.
    - ``learned`` -- a single trainable embedding broadcast over the tail.

    The scheme is "a property of the system, not of the call: whichever is
    chosen must be used identically when generating the training data of
    Sec. 2.5, since it is the training distribution that teaches the shallow
    mode to ignore the tail."  Callers should therefore pass
    ``config.padding``, never a local choice -- and note that agreement is only
    ever evaluated on real entries ``h < H_k`` anyway
    (see :mod:`sentry.core.acceptance`).
    """
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
    """Return the ``(1, d_a)`` row broadcast over the tail."""
    last = live_suffix[-1]

    if scheme == "hold":
        return last.unsqueeze(0)

    if scheme == "zero_delta":
        # Null incremental motion, gripper held at the last surviving value.
        # "Null" is zero in *raw* delta space, which is generally not zero
        # after normalisation -- ChannelSpec owns that round trip.
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
