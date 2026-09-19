r"""The acceptance rule -- equations 10 and 11, and the margin (SS2.4.3).

"Actions carry channels with different semantics, and we treat them
differently."  The continuous translation/rotation channels are compared by
thresholded Euclidean distance; the gripper channel is *discrete in intent* and
is handled by sign agreement.

.. math::
    d^\tau_h = \max\left\{
        \frac{\|R^\tau[h]_{\mathcal{C}_{pos}} - \hat{A}[h]_{\mathcal{C}_{pos}}\|_2}{\delta_{pos}},
        \frac{\|R^\tau[h]_{\mathcal{C}_{rot}} - \hat{A}[h]_{\mathcal{C}_{rot}}\|_2}{\delta_{rot}}
    \right\}

Folding the thresholds into the statistic is what makes ``d^tau_h <= 1``
*be* the acceptance condition rather than merely imply it.

All distances are computed in the policy's **normalised** action space
(SS2.9, convention (ii)), so that per-channel thresholds are commensurable.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from sentry.core.types import ChannelSpec, Thresholds

__all__ = [
    "group_distances",
    "raw_distances",
    "normalised_distances",
    "sign_agreement",
    "accepted_length",
    "margin",
    "first_violation",
]


# --------------------------------------------------------------------------
# Distances (equation 10)
# --------------------------------------------------------------------------


def group_distances(R: Tensor, A_hat: Tensor, idx: tuple[int, ...]) -> Tensor:
    """L2 distance between ``R`` and ``A_hat`` over one channel group.

    Args:
        R: ``(K, L, d_a)`` reconstructed endpoints.
        A_hat: ``(L, d_a)`` candidate.
        idx: channel indices forming the group.

    Returns ``(K, L)``.  An empty group yields zeros, so a spec without
    rotation channels degrades gracefully instead of producing NaNs.
    """
    if not idx:
        return torch.zeros(R.shape[:2], dtype=R.dtype, device=R.device)
    sel = torch.as_tensor(idx, dtype=torch.long, device=R.device)
    diff = R.index_select(-1, sel) - A_hat.index_select(-1, sel).unsqueeze(0)
    return torch.linalg.vector_norm(diff, ord=2, dim=-1)


def raw_distances(
    R: Tensor, A_hat: Tensor, spec: ChannelSpec, H_k: int
) -> tuple[Tensor, Tensor]:
    """Equation 10 evaluated at ``delta_pos = delta_rot = 1`` -- the raw distances.

    Returns ``(d_pos, d_rot)``, each ``(K, H_k)``.

    This is the statistic conformal calibration consumes (SS2.4.4: "let
    ``d^(i)`` denote the statistic equation 10 evaluated at
    ``delta_pos = delta_rot = 1``, i.e. the raw distance").

    Note that "Agreement is only ever evaluated on real entries, ``h < H_k``"
    (SS2.4.1) -- the padded tail never reaches the acceptance rule, which is
    why the padding scheme only has to teach the shallow mode to *ignore* the
    tail rather than to reproduce it.
    """
    _validate(R, A_hat, spec, H_k)
    R_real = R[:, :H_k, :]
    A_real = A_hat[:H_k, :]
    return (
        group_distances(R_real, A_real, spec.pos),
        group_distances(R_real, A_real, spec.rot),
    )


def normalised_distances(
    R: Tensor, A_hat: Tensor, spec: ChannelSpec, thresholds: Thresholds, H_k: int
) -> Tensor:
    """Equation 10 in full: ``(K, H_k)`` with ``d <= 1`` the acceptance condition."""
    d_pos, d_rot = raw_distances(R, A_hat, spec, H_k)
    return torch.maximum(d_pos / thresholds.pos, d_rot / thresholds.rot)


def sign_agreement(R: Tensor, A_hat: Tensor, spec: ChannelSpec, H_k: int) -> Tensor:
    """Gripper sign test: ``sign(R^tau[h]_grip) == sign(A_hat[h]_grip)``.

    Returns ``(K, H_k)`` bool.

    Implemented as ``R_grip * A_grip > 0``, which treats an exact zero on
    either side as *disagreement*.  That is the conservative reading: a
    gripper command sitting exactly on the decision boundary carries no
    intent, and accepting it would be the one place where the sign test could
    pass vacuously.  :class:`ChannelSpec` separately asserts that the two raw
    gripper modes standardise to opposite signs (SS2.9, convention (iii)), so
    a well-formed spec never lands here by construction.
    """
    _validate(R, A_hat, spec, H_k)
    r_grip = R[:, :H_k, spec.grip]
    a_grip = A_hat[:H_k, spec.grip].unsqueeze(0)
    return (r_grip * a_grip) > 0


# --------------------------------------------------------------------------
# Accepted length (equation 11) and margin
# --------------------------------------------------------------------------


def accepted_length(d: Tensor, sign_ok: Tensor) -> int:
    r"""Equation 11 -- the longest agreeing prefix, taken conservatively over ``T``.

    .. math::
        N = \min_{\tau \in \mathcal{T}} \sum_{h=0}^{H_k-1} \prod_{j=0}^{h}
            \mathbf{1}\left[d^\tau_j \le 1 \wedge \text{sign agrees at } j\right]

    Args:
        d: ``(K, H_k)`` normalised distances.
        sign_ok: ``(K, H_k)`` gripper sign agreement.

    Returns the accepted prefix length ``N``, with ``0 <= N <= H_k``.

    The cumulative product is what makes this a *prefix* length rather than a
    count of agreeing positions: one failure at ``j`` zeroes every term from
    ``j`` onward.  The ``min`` over ``T`` means a single verification timestep
    that objects is enough to truncate the prefix.
    """
    if d.shape != sign_ok.shape:
        raise ValueError(f"shape mismatch: d {tuple(d.shape)}, sign_ok {tuple(sign_ok.shape)}")
    if d.numel() == 0:
        return 0

    ok = (d <= 1.0) & sign_ok                       # (K, H_k)
    prefix = torch.cumprod(ok.to(torch.int64), dim=-1)   # (K, H_k)
    per_tau = prefix.sum(dim=-1)                    # (K,)
    return int(per_tau.min().item())


def margin(d: Tensor, N: int) -> Optional[float]:
    r"""``mu = 1 - \max_{\tau} d^\tau_{N-1}`` -- how comfortably the prefix passed.

    Returns ``None`` when ``N == 0``.  SS2.4.3 defines the margin only "when
    ``N >= 1``"; returning ``None`` forces :mod:`sentry.core.cascade` to handle
    rejection explicitly instead of comparing a fabricated ``0.0`` against
    ``mu_esc``.

    The margin is measured at the **last accepted** position ``N-1``, not the
    first rejected one: it asks how close the accepted prefix came to failing,
    which is what "drives the depth cascade of Sec. 2.6".
    """
    if N <= 0:
        return None
    if N > d.shape[-1]:
        raise ValueError(f"N={N} exceeds available positions {d.shape[-1]}")
    return float(1.0 - d[:, N - 1].max().item())


def first_violation(d: Tensor, sign_ok: Tensor) -> Optional[int]:
    """Index of the first position rejected by *any* verification timestep.

    Returns ``None`` if every position passes.  Used by calibration to locate
    where to read the raw distance, and by the deferred Stage B training to
    supply ``h_star``.
    """
    if d.numel() == 0:
        return None
    ok = (d <= 1.0) & sign_ok
    all_ok = ok.all(dim=0)              # (H_k,) -- conservative across T
    bad = (~all_ok).nonzero(as_tuple=False)
    return None if bad.numel() == 0 else int(bad[0].item())


# --------------------------------------------------------------------------


def _validate(R: Tensor, A_hat: Tensor, spec: ChannelSpec, H_k: int) -> None:
    if R.ndim != 3:
        raise ValueError(f"R must be (K, H, d_a), got {tuple(R.shape)}")
    if A_hat.ndim != 2:
        raise ValueError(f"A_hat must be (H, d_a), got {tuple(A_hat.shape)}")
    if R.shape[1] != A_hat.shape[0] or R.shape[2] != A_hat.shape[1]:
        raise ValueError(
            f"R {tuple(R.shape)} incompatible with A_hat {tuple(A_hat.shape)}"
        )
    if A_hat.shape[1] != spec.d_a:
        raise ValueError(f"action dim mismatch: {A_hat.shape[1]} vs spec {spec.d_a}")
    if not (0 < H_k <= A_hat.shape[0]):
        raise ValueError(f"H_k={H_k} outside (0, {A_hat.shape[0]}]")
