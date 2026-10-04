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


def group_distances(R: Tensor, A_hat: Tensor, idx: tuple[int, ...]) -> Tensor:
    if not idx:
        return torch.zeros(R.shape[:2], dtype=R.dtype, device=R.device)
    sel = torch.as_tensor(idx, dtype=torch.long, device=R.device)
    diff = R.index_select(-1, sel) - A_hat.index_select(-1, sel).unsqueeze(0)
    return torch.linalg.vector_norm(diff, ord=2, dim=-1)


def raw_distances(
    R: Tensor, A_hat: Tensor, spec: ChannelSpec, H_k: int
) -> tuple[Tensor, Tensor]:
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
    d_pos, d_rot = raw_distances(R, A_hat, spec, H_k)
    return torch.maximum(d_pos / thresholds.pos, d_rot / thresholds.rot)


def sign_agreement(R: Tensor, A_hat: Tensor, spec: ChannelSpec, H_k: int) -> Tensor:
    _validate(R, A_hat, spec, H_k)
    r_grip = R[:, :H_k, spec.grip]
    a_grip = A_hat[:H_k, spec.grip].unsqueeze(0)
    return (r_grip * a_grip) > 0


def accepted_length(d: Tensor, sign_ok: Tensor) -> int:
    if d.shape != sign_ok.shape:
        raise ValueError(f"shape mismatch: d {tuple(d.shape)}, sign_ok {tuple(sign_ok.shape)}")
    if d.numel() == 0:
        return 0

    ok = (d <= 1.0) & sign_ok
    prefix = torch.cumprod(ok.to(torch.int64), dim=-1)
    per_tau = prefix.sum(dim=-1)
    return int(per_tau.min().item())


def margin(d: Tensor, N: int) -> Optional[float]:
    if N <= 0:
        return None
    if N > d.shape[-1]:
        raise ValueError(f"N={N} exceeds available positions {d.shape[-1]}")
    return float(1.0 - d[:, N - 1].max().item())


def first_violation(d: Tensor, sign_ok: Tensor) -> Optional[int]:
    if d.numel() == 0:
        return None
    ok = (d <= 1.0) & sign_ok
    all_ok = ok.all(dim=0)
    bad = (~all_ok).nonzero(as_tuple=False)
    return None if bad.numel() == 0 else int(bad[0].item())


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
