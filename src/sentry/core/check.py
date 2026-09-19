"""The shallow verification operator ``Check`` (SS2.4).

"The operator ``Check`` takes the live suffix, the fresh observation, and a
depth budget, and returns an accepted prefix length."

The four steps of SS2.4 are split across modules -- candidate construction in
:mod:`sentry.core.padding` (eq. 7), re-noising and reconstruction in
:mod:`sentry.core.renoise` (eqs. 8-9), the acceptance rule in
:mod:`sentry.core.acceptance` (eqs. 10-11), calibration in
:mod:`sentry.core.calibration` (eq. 12) -- and composed here.

**Exactly one backend call per invocation.**  That is the whole cost model.
"""

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
    """Run one shallow check and return the accepted prefix length and margin.

    Args:
        backend: the target policy, exposed at two depths.
        A_hat: ``(H, d_a)`` full-width candidate from equation 7.
        H_k: number of real (non-padded) entries in ``A_hat``.
        obs: the **fresh** observation ``o_{t+k}``.  Corollary 3: "A verifier
            able to detect exogenous change must consume fresh exteroceptive
            input."  Passing a cached observation here is exactly the
            Proposition 2 failure mode.
        rung: the depth budget ``(E_V, E_B)``.
        thresholds: ``delta_pos``, ``delta_rot`` from conformal calibration.
        taus: ``T``, the verification timesteps.
        spec: channel semantics and normalisation stats.
        convention: tau direction (SS2.9, convention (i)).
        eps: optional shared noise draw; sampled if omitted.  The cascade
            passes the same ``eps`` to every rung so that escalating depth
            re-answers *the same* query.
        generator: RNG for reproducible ``eps``.

    Returns:
        A :class:`CheckResult`.  Its ``N`` is the only quantity check mode is
        permitted to produce -- "This mode never produces an action that is
        executed" (SS2.3).

    Note the depth budgets are validated against the backend's *own*
    ``L_V``/``L_B``, which SS2.9 (iv) requires be read off the checkpoint since
    "VLM backbones in this family vary between 18 and 32 layers".
    """
    if rung.E_V > backend.L_V:
        raise ValueError(f"E_V={rung.E_V} exceeds encoder depth L_V={backend.L_V}")
    if rung.E_B > backend.L_B:
        raise ValueError(f"E_B={rung.E_B} exceeds backbone depth L_B={backend.L_B}")
    if not (0 < H_k <= A_hat.shape[0]):
        raise ValueError(f"H_k={H_k} outside (0, {A_hat.shape[0]}]")

    # Steps 2.4.2 -- eqs. 8 and 9, one batched forward over all K timesteps.
    # Adapters are gated ON: this is check mode.
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

    # Step 2.4.3 -- eqs. 10 and 11.
    d = acceptance.normalised_distances(R, A_hat, spec, thresholds, H_k)
    sign_ok = acceptance.sign_agreement(R, A_hat, spec, H_k)

    N = acceptance.accepted_length(d, sign_ok)
    # "the accepted length is capped at H_k" (SS2.4.1).
    N = min(N, H_k)
    mu = acceptance.margin(d, N)

    return CheckResult(N=N, margin=mu, d=d, sign_ok=sign_ok, H_k=H_k, rung=rung)
