"""An analytic backend with a *controllable* flow field.

An untrained :class:`TinyPi0` validates the plumbing but has no meaningful
velocity field, so it cannot say anything about whether staleness is detected.
This backend closes that gap: its flow is defined to land exactly on whatever
the supplied ``planner`` would plan *from the observation it is given*, plus a
depth-dependent perturbation standing in for truncation error.

That single property makes the whole of SS2.8's Diagnostic D1 directly
controllable:

- ``d_prune`` -- disagreement at reduced depth under the *same* observation --
  is set by :attr:`OracleConfig.truncation_sigma`;
- ``d_stale`` -- disagreement at full depth under a *fresh* observation -- is
  set by how much the environment actually changed;
- "The method is viable only in the regime ``d_prune << d_stale``", which
  becomes something we can dial to either side of and confirm the algorithm
  behaves as claimed.

**The planner must be a pure function of its ``Observation``.**  That is what
makes the Proposition 2 demonstration honest: feeding the cached-context
baseline a stale observation must genuinely blind it, not merely inconvenience
it.  If the planner could reach around the observation to the environment's
true state, the demo would prove nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from sentry.core.types import Observation, assert_shared_depth

__all__ = ["OracleConfig", "OracleBackend", "exponential_truncation"]

Planner = Callable[[Observation], Tensor]
"""``Observation -> (H, d_a)`` in normalised action space.  Must be pure."""


def exponential_truncation(sigma_0: float = 0.5, decay: float = 4.0) -> Callable[[float, float], float]:
    """Truncation error that decays with fractional depth.

    Returns ``sigma(f_V, f_B) = sigma_0 * exp(-decay * min(f_V, f_B))`` where
    ``f = E/L`` is fractional depth.  The ``min`` reflects SS2.9's *Depth is
    shared*: the shallower of the two pathways bounds what the model can know,
    because action tokens at layer ``l`` attend to the visual keys of layer
    ``l``.
    """

    def sigma(f_V: float, f_B: float) -> float:
        return sigma_0 * math.exp(-decay * min(f_V, f_B))

    return sigma


@dataclass
class OracleConfig:
    L_V: int = 27
    L_B: int = 18
    H: int = 50
    d_a: int = 7
    M: int = 10
    truncation_sigma: Callable[[float, float], float] = field(
        default_factory=exponential_truncation
    )
    """``(E_V/L_V, E_B/L_B) -> sigma``.  Governs ``d_prune``."""
    horizon_growth: float = 3.0
    """How much reconstruction error grows across the chunk.

    Error at position ``h`` is scaled by ``1 + growth * h / H``.  A flat error
    profile is unrealistic and has a sharp consequence for the algorithm: it
    lets equation 11's agreeing prefix run the entire width of the chunk, so a
    single check licenses ``H - k`` open-loop actions.  Real velocity fields
    are less certain further out, which bounds ``N`` naturally.
    """
    seed: Optional[int] = 0
    """Fixes the truncation perturbation so runs are reproducible."""


class OracleBackend:
    """A :class:`VLABackend` whose reconstruction is analytically known.

    Under equation 9 with no truncation error, ``R^tau`` equals
    ``planner(obs)`` exactly -- for every ``tau``, and regardless of the
    candidate it was handed.  That is the *correct* behaviour for a verifier:
    "If the plan is still the policy's intent under the present observation,
    the flow returns it; if the scene has changed in a way that matters, the
    flow pulls it elsewhere."
    """

    def __init__(self, planner: Planner, cfg: OracleConfig = OracleConfig()) -> None:
        self.planner = planner
        self.cfg = cfg
        self.L_V, self.L_B = cfg.L_V, cfg.L_B
        self.H, self.d_a, self.M = cfg.H, cfg.d_a, cfg.M
        self._gen = torch.Generator()
        if cfg.seed is not None:
            self._gen.manual_seed(cfg.seed)

        self.calls: int = 0
        """Velocity evaluations, for cost accounting in tests."""

    # -- plan mode --------------------------------------------------------

    def plan(self, obs: Observation, noise: Optional[Tensor] = None) -> Tensor:
        """Full-depth plan: exactly what the planner intends from ``obs``.

        ``noise`` is accepted for protocol conformance and ignored: the planner
        is required to be a pure function of its observation, so this backend
        is already deterministic and the common-random-number contract of
        paper defect D5 holds vacuously.
        """
        A = self.planner(obs)
        if A.shape != (self.H, self.d_a):
            raise ValueError(
                f"planner returned {tuple(A.shape)}, expected {(self.H, self.d_a)}"
            )
        return A

    # -- check mode -------------------------------------------------------

    def velocity(
        self,
        A_tau: Tensor,
        tau: Tensor,
        obs: Observation,
        E_V: int,
        E_B: int,
        adapters: bool,
    ) -> Tensor:
        r"""Return the velocity that sends ``A^tau`` to ``planner(obs)``.

        Solving equation 9 for ``v`` given a desired endpoint ``A^*``:

        .. math::
            R^\tau = \hat{A}^\tau + (1-\tau)v = A^* \implies
            v = \frac{A^* - \hat{A}^\tau}{1-\tau}

        A depth-dependent perturbation is then added to ``v`` so that the
        reconstruction lands at ``A^* + (1-tau)\sigma\xi`` -- i.e. truncation
        error enters the *reconstruction* at a scale independent of ``tau``,
        which is the behaviour a real truncated network would show.

        ``adapters`` is accepted for protocol conformance and ignored: this
        backend has no weights, so Proposition 4 holds vacuously.
        """
        if not (1 <= E_V <= self.L_V):
            raise ValueError(f"E_V={E_V} outside [1, {self.L_V}]")
        if not (1 <= E_B <= self.L_B):
            raise ValueError(f"E_B={E_B} outside [1, {self.L_B}]")
        assert_shared_depth(E_B, E_B)

        self.calls += 1

        A_star = self.planner(obs)                     # (H, d_a) -- pure in obs
        K = A_tau.shape[0]
        s = tau.view(K, 1, 1).to(A_tau.dtype)

        target = A_star.unsqueeze(0).expand(K, -1, -1)

        sigma = self.cfg.truncation_sigma(E_V / self.L_V, E_B / self.L_B)
        if sigma > 0.0:
            xi = torch.randn(
                A_tau.shape, dtype=A_tau.dtype, device=A_tau.device, generator=self._gen
            )
            H = A_tau.shape[1]
            h = torch.arange(H, dtype=A_tau.dtype, device=A_tau.device)
            growth = (1.0 + self.cfg.horizon_growth * h / max(H - 1, 1)).view(1, H, 1)
            target = target + sigma * growth * xi

        # v = (A* - A^tau) / (1 - tau).  Guarded: tau = 1 is pure data and is
        # excluded from T by construction (T subset of the open interval (0,1)).
        denom = (1.0 - s).clamp_min(1e-6)
        return (target - A_tau) / denom
