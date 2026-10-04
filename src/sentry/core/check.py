from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from sentry.config import TauConvention
from sentry.core import acceptance
from sentry.core.interfaces import VLABackend
from sentry.core.renoise import renoise_and_reconstruct
from sentry.core.types import ChannelSpec, CheckResult, DepthRung, Observation, Thresholds

__all__ = ["check"]


def check(
    backend: VLABackend,
    A_hat: Tensor,
    H_k: int,
    obs: Observation,
    rung: DepthRung,
    thresholds: Thresholds,
    taus: Sequence[float],
    spec: ChannelSpec,
    convention: TauConvention = "one_is_clean",
    eps: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> CheckResult:
    if rung.E_V > backend.L_V:
        raise ValueError(f"E_V={rung.E_V} exceeds encoder depth L_V={backend.L_V}")
    if rung.E_B > backend.L_B:
        raise ValueError(f"E_B={rung.E_B} exceeds backbone depth L_B={backend.L_B}")
    if not (0 < H_k <= A_hat.shape[0]):
        raise ValueError(f"H_k={H_k} outside (0, {A_hat.shape[0]}]")

    R, _A_s, _eps = renoise_and_reconstruct(
        backend=backend,
        A_hat=A_hat,
        obs=obs,
        rung=rung,
        taus=taus,
        convention=convention,
        eps=eps,
        adapters=True,
        generator=generator,
    )

    d = acceptance.normalised_distances(R, A_hat, spec, thresholds, H_k)
    sign_ok = acceptance.sign_agreement(R, A_hat, spec, H_k)

    N = acceptance.accepted_length(d, sign_ok)
    N = min(N, H_k)
    mu = acceptance.margin(d, N)

    return CheckResult(N=N, margin=mu, d=d, sign_ok=sign_ok, H_k=H_k, rung=rung)
