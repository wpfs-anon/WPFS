from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from sentry.config import TauConvention
from sentry.core.interfaces import VLABackend
from sentry.core.types import DepthRung, Observation

__all__ = ["interpolate", "reconstruct", "to_model_time", "renoise_and_reconstruct"]


def to_model_time(s: Tensor, convention: TauConvention) -> Tensor:
    if convention == "one_is_clean":
        return s
    if convention == "zero_is_clean":
        return 1.0 - s
    raise ValueError(f"unknown tau convention {convention!r}")


def _compute_dtype(*tensors: Tensor) -> torch.dtype:
    dt = torch.float32
    for t in tensors:
        dt = torch.promote_types(dt, t.dtype)
    return dt


def _velocity_sign(convention: TauConvention) -> float:
    return 1.0 if convention == "one_is_clean" else -1.0


def interpolate(A_hat: Tensor, s: Tensor, eps: Tensor) -> Tensor:
    if A_hat.shape != eps.shape:
        raise ValueError(
            f"eps must match A_hat: {tuple(eps.shape)} vs {tuple(A_hat.shape)}"
        )
    dt = _compute_dtype(A_hat, eps, s)
    s_ = s.to(dt).view(-1, 1, 1)
    return s_ * A_hat.to(dt).unsqueeze(0) + (1.0 - s_) * eps.to(dt).unsqueeze(0)


def reconstruct(
    A_s: Tensor,
    s: Tensor,
    v_model: Tensor,
    convention: TauConvention = "one_is_clean",
) -> Tensor:
    dt = _compute_dtype(A_s, v_model, s)
    s_ = s.to(dt).view(-1, 1, 1)
    return A_s.to(dt) + (1.0 - s_) * _velocity_sign(convention) * v_model.to(dt)


def renoise_and_reconstruct(
    backend: VLABackend,
    A_hat: Tensor,
    obs: Observation,
    rung: DepthRung,
    taus: Sequence[float],
    convention: TauConvention = "one_is_clean",
    eps: Optional[Tensor] = None,
    adapters: bool = True,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if not taus:
        raise ValueError("T must contain at least one verification timestep")
    if any(not (0.0 < t < 1.0) for t in taus):
        raise ValueError(f"verification timesteps must lie in (0,1), got {tuple(taus)}")

    device = A_hat.device
    dtype = _compute_dtype(A_hat)
    s = torch.as_tensor(tuple(taus), dtype=dtype, device=device)

    if eps is None:
        gen_device = generator.device if generator is not None else device
        eps = torch.randn(
            A_hat.shape, dtype=dtype, device=gen_device, generator=generator
        ).to(device)

    A_s = interpolate(A_hat, s, eps)
    tau_model = to_model_time(s, convention)

    v_model = backend.velocity(
        A_s, tau_model, obs, E_V=rung.E_V, E_B=rung.E_B, adapters=adapters
    )
    if v_model.shape != A_s.shape:
        raise ValueError(
            f"backend returned velocity of shape {tuple(v_model.shape)}, "
            f"expected {tuple(A_s.shape)}"
        )

    R = reconstruct(A_s, s, v_model, convention)
    return R, A_s, eps
