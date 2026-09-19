r"""Re-noising and one-step endpoint reconstruction -- equations 8 and 9 (SS2.4.2).

We exploit the fact that flow matching supervises the velocity along the
*entire* linear path between noise and data, so any point on that path is an
admissible query:

.. math::
    \hat{A}^\tau &= \tau \hat{A} + (1-\tau)\epsilon                        \\
    R^\tau       &= \hat{A}^\tau + (1-\tau)\, v^{(E_V,E_B)}_{\theta\oplus\Delta}
                    (\hat{A}^\tau, \tau \mid o_{t+k})

Equation 9 asks a well-posed question: *starting from a slightly corrupted
version of the plan, and looking at the scene as it is now, where does the
policy's flow send it?*  If the plan is still the policy's intent under the
present observation, the flow returns it; if the scene has changed in a way
that matters, the flow pulls it elsewhere.

Two design points distinguish this from the superficially similar cached-context
test of prior work: the conditioning ``o_{t+k}`` is **recomputed from the
current images**, so by Corollary 3 the test is sensitive to exogenous change;
and no KV cache is shared with plan mode.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from sentry.config import TauConvention
from sentry.core.interfaces import VLABackend
from sentry.core.types import DepthRung, Observation

__all__ = ["interpolate", "reconstruct", "to_model_time", "renoise_and_reconstruct"]


# --------------------------------------------------------------------------
# Tau convention (SS2.9, convention (i))
# --------------------------------------------------------------------------
#
# Internally we always parametrise by *cleanliness* ``s in (0,1)``, with
# ``s = 1`` clean data and ``s = 0`` pure noise -- the convention the paper
# adopts throughout ("we adopt throughout the convention that tau = 0 is pure
# noise and tau = 1 is the clean action").
#
# A backend that integrates in the opposite direction needs two corrections,
# not one.  Its time argument is ``1 - s``, and because its velocity is
# ``dA/dtau_model`` with ``tau_model = 1 - s``, the chain rule gives
# ``dA/ds = -dA/dtau_model``.  So the reconstruction picks up a sign flip as
# well.  The paper's "the obvious substitution tau -> 1 - tau" covers both once
# eqs. 8 and 9 are read together; we spell it out because getting only the time
# argument right yields a reconstruction that is wrong by ``2(1-s)v`` and still
# looks plausible.


def to_model_time(s: Tensor, convention: TauConvention) -> Tensor:
    """Map internal cleanliness ``s`` to the backend's own time argument."""
    if convention == "one_is_clean":
        return s
    if convention == "zero_is_clean":
        return 1.0 - s
    raise ValueError(f"unknown tau convention {convention!r}")


# --------------------------------------------------------------------------
# Numerical precision of eqs. 8 and 9
# --------------------------------------------------------------------------
#
# The arithmetic below runs in **at least float32**, regardless of the dtype the
# backend's weights carry.  A real pi_0 checkpoint runs bf16, which has 8 bits
# of mantissa: tau = 0.6 is stored as 0.6015625, and eq. 9 is a difference of
# two nearby quantities, ``R - A_hat``.  The rounding error that survives that
# cancellation lands on the same order of magnitude as the thresholds eq. 12
# calibrates, so the acceptance statistic would be reading its own noise.
#
# Weights stay bf16 -- only this arithmetic is promoted.  The backend receives a
# float32 interpolant and casts it internally, exactly as openpi's own
# ``pi0_pytorch`` casts its embeddings at the model boundary.


def _compute_dtype(*tensors: Tensor) -> torch.dtype:
    """float32, or wider if a caller is deliberately working in float64."""
    dt = torch.float32
    for t in tensors:
        dt = torch.promote_types(dt, t.dtype)
    return dt


def _velocity_sign(convention: TauConvention) -> float:
    """``dA/ds`` in units of the backend's ``dA/dtau_model``."""
    return 1.0 if convention == "one_is_clean" else -1.0


# --------------------------------------------------------------------------
# Equations 8 and 9
# --------------------------------------------------------------------------


def interpolate(A_hat: Tensor, s: Tensor, eps: Tensor) -> Tensor:
    r"""Equation 8: :math:`\hat{A}^\tau = \tau\hat{A} + (1-\tau)\epsilon`.

    Args:
        A_hat: ``(H, d_a)`` full-width candidate from equation 7.
        s: ``(K,)`` cleanliness values, one per verification timestep.
        eps: ``(H, d_a)`` a **single shared** noise draw.

    Returns ``(K, H, d_a)``.

    SS2.4.2 draws "a single shared ``epsilon ~ N(0, I)``" for the whole of
    ``T``.  Sharing it means the K queries differ *only* in where they sit on
    the same noise-to-data line, which is what makes the ``min`` over ``T`` in
    equation 11 a conservative reading of one situation rather than an average
    over K unrelated ones.
    """
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
    r"""Equation 9: :math:`R^\tau = \hat{A}^\tau + (1-\tau)\, v(\hat{A}^\tau, \tau \mid o)`.

    Args:
        A_s: ``(K, H, d_a)`` interpolants from :func:`interpolate`.
        s: ``(K,)`` cleanliness values.
        v_model: ``(K, H, d_a)`` the backend's velocity, in *its* time direction.
        convention: see :func:`to_model_time`.

    Returns ``(K, H, d_a)`` reconstructed endpoints.

    This is exact under a constant velocity field: with ``v = A - eps``,

        ``A^s + (1-s)v = sA + (1-s)eps + (1-s)(A - eps) = A``

    which :mod:`sentry.tests.test_renoise` asserts to float tolerance.  That
    test is the cheapest possible tripwire for a tau-direction slip.
    """
    dt = _compute_dtype(A_s, v_model, s)
    s_ = s.to(dt).view(-1, 1, 1)
    return A_s.to(dt) + (1.0 - s_) * _velocity_sign(convention) * v_model.to(dt)


# --------------------------------------------------------------------------
# Composed: one batched forward covering all K timesteps
# --------------------------------------------------------------------------


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
    """Apply eqs. 8 and 9 in **one** batched backend call.

    Returns ``(R, A_s, eps)`` with ``R``, ``A_s`` of shape ``(K, H, d_a)``.

    SS2.4.2: "The K interpolants differ only in the leading batch dimension and
    are evaluated in a single batched forward pass, so the cost of ``T`` is
    that of one evaluation, not ``K``."  :mod:`sentry.core.cost` relies on
    this -- equation 17's last term is a single ``L_den``.
    """
    if not taus:
        raise ValueError("T must contain at least one verification timestep")
    if any(not (0.0 < t < 1.0) for t in taus):
        raise ValueError(f"verification timesteps must lie in (0,1), got {tuple(taus)}")

    # eqs. 8 and 9 run in float32 even when the checkpoint is bf16 -- see the
    # note above _compute_dtype.  ``eps`` is drawn at the same precision so a
    # shared draw means the same numbers at every rung of the cascade.
    device = A_hat.device
    dtype = _compute_dtype(A_hat)
    s = torch.as_tensor(tuple(taus), dtype=dtype, device=device)

    if eps is None:
        # Draw on the generator's own device, then move.  A caller that seeds a
        # CPU generator for reproducibility should not have to know that the
        # model happens to live on an accelerator -- and drawing on CPU keeps
        # the noise stream identical across machines, which is what makes a
        # cascade's shared eps and a re-run of the same seed comparable.
        gen_device = generator.device if generator is not None else device
        eps = torch.randn(
            A_hat.shape, dtype=dtype, device=gen_device, generator=generator
        ).to(device)

    A_s = interpolate(A_hat, s, eps)
    tau_model = to_model_time(s, convention)

    # The one and only backend call per check.  No cache crosses this boundary.
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
