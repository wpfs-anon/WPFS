"""Algorithm 1 -- SENTRY: self-verified adaptive-length execution (SS2.6).

The loop's two structural commitments, both of which the implementation makes
explicit where the pseudocode leaves them implicit:

1. **The observation is re-read from the environment at every check.**
   Algorithm 1 line 12 writes ``CHECK(A_hat, o_t, ...)`` and means it: line 21
   advances ``t <- t + N`` inside the speculative loop, so ``o_t`` there *is*
   the current observation.  The subscript is easy to misread as the anchor's
   observation, though, and getting it wrong is the one bug that cannot be
   caught downstream -- a stale ``o`` reduces SENTRY to the Proposition 2 straw
   man (SS2.4.2: "the conditioning ``o_{t+k}`` is recomputed from the current
   images, so by Corollary 3 the test is sensitive to exogenous change") while
   still producing entirely plausible numbers, slightly better speedup for
   slightly worse success.  So freshness is asserted, not assumed.

2. **``m_min`` is committed unconditionally, for liveness rather than speed.**
   Those actions "were produced by the target from the current observation, so
   verifying them against the target's own plan would be vacuous, and without a
   commit the loop could reject at the same ``t`` indefinitely without
   advancing the environment."  Proposition 5 is checked as a runtime
   invariant.

There is **no timer**.  SS2.6: "Algorithm 1 contains no timer.  We regard this
as a falsifiable commitment rather than an omission... If a fresh-observation
verifier still requires a timer to hold up, the central claim of this paper is
wrong."  ``cfg.S_max`` and ``cfg.mu_warn`` exist as ablation axes and are
``None`` by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from sentry.config import SentryConfig
from sentry.core.cascade import CascadeResult, CascadeVerifier, Verifier
from sentry.core.interfaces import Environment, VLABackend
from sentry.core.padding import repad
from sentry.core.types import ChannelSpec, Observation, Thresholds

__all__ = ["EpisodeTrace", "run_episode", "StaleObservationError"]


class StaleObservationError(RuntimeError):
    """Raised when the verifier would be fed a non-current observation.

    This is the Proposition 2 failure mode.  It is a bug class worth its own
    exception because the symptom -- slightly better speedup, slightly worse
    success -- looks like a tuning problem rather than a correctness one.
    """


@dataclass
class EpisodeTrace:
    """Everything eq. 18 and the paper's headline plots need from one episode."""

    chunk_lengths: list[int] = field(default_factory=list)
    """Realised chunk length per full-depth call: ``k`` at speculative exit.

    SS2.6: "It is not a hyperparameter.  Its distribution -- wide in static
    phases, short under perturbation -- is the primary object we report."
    """

    accepted_prefixes: list[int] = field(default_factory=list)
    """Every accepted ``N > 0``.  Mean is ``N_bar`` in eq. 18."""

    rungs_per_check: list[int] = field(default_factory=list)
    """Rungs evaluated per check.  Mean is ``J_bar`` in eq. 18."""

    margins: list[float] = field(default_factory=list)
    """Margin of every accepted verdict.  Feeds the prefetch analysis."""

    checks: int = 0
    rejections: int = 0
    """Checks that ended in ``N = 0``.  ``rho = rejections / checks`` (eq. 18)."""

    target_invocations: int = 0
    """Full-depth ``Pi_deep`` calls.  Bounded by Proposition 5."""

    steps: int = 0
    plan_exhaustions: int = 0
    """Speculative phases that ended because ``k >= H`` rather than by rejection."""

    forced_refreshes: int = 0
    """``S_max`` firings.  Non-zero only in the ablation."""

    prefetch_triggers: int = 0
    """``mu_warn`` firings.  Non-zero only in the ablation."""

    escalated_unresolved: int = 0
    """Cascades that exhausted the ladder while still fragile."""

    # -- eq. 18 inputs ----------------------------------------------------

    @property
    def rho(self) -> float:
        """Fraction of checks that end in rejection."""
        return self.rejections / self.checks if self.checks else 0.0

    @property
    def N_bar(self) -> float:
        """Mean accepted prefix."""
        return (
            sum(self.accepted_prefixes) / len(self.accepted_prefixes)
            if self.accepted_prefixes
            else 0.0
        )

    @property
    def J_bar(self) -> float:
        """Mean number of cascade rungs evaluated per check, ``>= 1``."""
        return (
            sum(self.rungs_per_check) / len(self.rungs_per_check)
            if self.rungs_per_check
            else 1.0
        )

    @property
    def gain_ratio(self) -> float:
        """``rho^-1 N_bar`` -- "where the gain must appear" (SS2.7).

        SS2.7 is unusually candid here: a shallow check is *not* cheaper per
        call than a small dedicated drafter, so "SENTRY is therefore not a
        claim about ``L_check`` but about equation 18: a verifier that sees the
        present should reject less often and accept longer prefixes, and the
        product ``rho^-1 N_bar`` is where the gain must appear.  This is stated
        as a prediction so that it can fail."  Reporting it as a first-class
        metric is what keeps the prediction falsifiable.
        """
        return float("inf") if self.rho == 0.0 else self.N_bar / self.rho

    @property
    def mean_chunk_length(self) -> float:
        return (
            sum(self.chunk_lengths) / len(self.chunk_lengths)
            if self.chunk_lengths
            else 0.0
        )

    def merge(self, other: "EpisodeTrace") -> "EpisodeTrace":
        """Combine traces across episodes (for aggregate reporting)."""
        out = EpisodeTrace(
            chunk_lengths=self.chunk_lengths + other.chunk_lengths,
            accepted_prefixes=self.accepted_prefixes + other.accepted_prefixes,
            rungs_per_check=self.rungs_per_check + other.rungs_per_check,
            margins=self.margins + other.margins,
        )
        for name in (
            "checks", "rejections", "target_invocations", "steps",
            "plan_exhaustions", "forced_refreshes", "prefetch_triggers",
            "escalated_unresolved",
        ):
            setattr(out, name, getattr(self, name) + getattr(other, name))
        return out


def run_episode(
    backend: VLABackend,
    env: Environment,
    cfg: SentryConfig,
    thresholds: Thresholds,
    spec: ChannelSpec,
    T_max: int,
    learned_pad: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    on_prefetch: Optional[Callable[[Observation], None]] = None,
    on_replan: Optional[Callable[[Observation], None]] = None,
    verifier: Optional[Verifier] = None,
) -> EpisodeTrace:
    """Run one episode under Algorithm 1.

    Args:
        backend: the target policy at two depths.  Used for ``plan()``; the
            checking path goes through ``verifier``.
        env: the environment; ``observe()`` must return the state *as of now*.
        cfg: system configuration (Table 2 defaults).
        thresholds: ``delta`` from conformal calibration (eq. 12).
        spec: channel semantics + normalisation stats.
        T_max: episode step budget.
        learned_pad: required iff ``cfg.padding == "learned"``.
        generator: RNG for reproducible verification noise.
        verifier: the checking policy.  Defaults to SENTRY's own depth cascade
            over the fresh observation.  Substituting
            :class:`sentry.baselines.cached_context.CachedContextVerifier`
            turns this loop into the Proposition 2 straw man, which is how the
            two are compared on identical footing.
        on_prefetch: optional hook fired when ``mu < mu_warn``.  The paper's
            optional async prefetch (SS2.6) is "an orthogonal systems
            optimisation... disabled in the main results so that latency
            numbers reflect the algorithm rather than the schedule", so this
            only records an event; it launches nothing.
        on_replan: fired with ``o_t`` at every full-depth replan.  This is the
            moment a cached-context pipeline would recompute its visual KV
            cache, so the Proposition 2 baseline subscribes here.

    Returns:
        An :class:`EpisodeTrace`.
    """
    trace = EpisodeTrace()
    H = cfg.H
    if verifier is None:
        verifier = CascadeVerifier(backend, cfg, thresholds, spec, generator)

    while not env.terminated and trace.steps < T_max:
        # ---- line 3: full-depth replan.  Adapters OFF; M solver steps. -----
        obs = _fresh(env)
        A = backend.plan(obs)
        trace.target_invocations += 1
        if on_replan is not None:
            on_replan(obs)
        if A.shape[0] != H:
            raise ValueError(f"plan returned {A.shape[0]} actions, expected H={H}")

        # ---- line 4: unconditional commit.  Liveness, not performance. -----
        k = _execute(env, A[: cfg.m_min], trace, T_max)
        if k == 0:
            break  # budget exhausted mid-commit

        checks_this_round = 0

        # ---- lines 5-22: speculative phase.  No full-depth call inside. ----
        while True:
            if env.terminated or trace.steps >= T_max:
                break

            # line 6: plan exhausted -> a full replan is mandatory (SS2.4.1).
            if k >= H:
                trace.plan_exhaustions += 1
                break

            # Ablation only: the forced-refresh cap prior work needs to stay
            # stable.  Disabled by default -- see the module docstring.
            if cfg.S_max is not None and checks_this_round >= cfg.S_max:
                trace.forced_refreshes += 1
                break

            # line 9: REPAD -- eq. 7.  Index 0 becomes "act now".
            A_hat, H_k = repad(
                live_suffix=A[k:H],
                H=H,
                scheme=cfg.padding,
                spec=spec,
                learned_pad=learned_pad,
            )

            # FRESH exteroception.  This is the entire point (Corollary 3).
            o_now = _fresh(env)

            # lines 10-17: depth cascade.
            result: CascadeResult = verifier(A_hat, H_k, o_now)
            trace.checks += 1
            checks_this_round += 1
            trace.rungs_per_check.append(result.rungs_used)
            if result.escalated_unresolved:
                trace.escalated_unresolved += 1

            # line 18: no prefix survives at any depth -> replan.
            if result.N == 0:
                trace.rejections += 1
                break

            assert result.margin is not None
            trace.margins.append(result.margin)

            # Paper defect D4 (ablation only, disabled by default): bound the
            # open-loop blindness horizon.  See SentryConfig.max_accept.
            N = result.N if cfg.max_accept is None else min(result.N, cfg.max_accept)
            trace.accepted_prefixes.append(N)

            # Ablation only: a low margin precedes rejection, so the replan
            # *could* be launched asynchronously here (SS2.6).
            if cfg.mu_warn is not None and result.margin < cfg.mu_warn:
                trace.prefetch_triggers += 1
                if on_prefetch is not None:
                    on_prefetch(o_now)

            # line 21: execute the accepted prefix.
            advanced = _execute(env, A_hat[:N], trace, T_max)
            if advanced == 0:
                break
            k += advanced

        # The realised chunk length -- the paper's primary reported object.
        trace.chunk_lengths.append(k)

    _assert_proposition_5(trace, cfg, T_max)
    return trace


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _fresh(env: Environment) -> Observation:
    """Read the observation and assert it is current."""
    obs = env.observe()
    if obs.t != env.t:
        raise StaleObservationError(
            f"environment is at t={env.t} but observe() returned t={obs.t}.  "
            "The verifier must consume fresh exteroceptive input (Corollary 3); "
            "conditioning on a cached context is provably blind to exogenous "
            "change (Proposition 2)."
        )
    return obs


def _execute(env: Environment, actions: Tensor, trace: EpisodeTrace, T_max: int) -> int:
    """Step the environment, respecting termination and the step budget.

    Returns the number of actions actually executed.
    """
    n = 0
    for i in range(actions.shape[0]):
        if env.terminated or trace.steps >= T_max:
            break
        env.step(actions[i])
        trace.steps += 1
        n += 1
    return n


def _assert_proposition_5(trace: EpisodeTrace, cfg: SentryConfig, T_max: int) -> None:
    """Check the liveness bound as a runtime invariant.

    Proposition 5: "on an episode of length ``T_max`` the target is invoked at
    most ``ceil(T_max / m_min)`` times."  Every outer iteration advances ``t``
    by at least ``m_min >= 1`` before entering the speculative phase, and every
    speculative iteration either advances ``t`` by ``N >= 1`` or exits, so no
    infinite sequence of iterations leaves ``t`` fixed.

    The bound can only be violated by a bug -- a commit that did not advance,
    or a speculative loop that accepted ``N = 0`` -- so we check it rather than
    trusting it.
    """
    bound = max(1, math.ceil(T_max / cfg.m_min))
    if trace.target_invocations > bound:
        raise AssertionError(
            f"Proposition 5 violated: {trace.target_invocations} target "
            f"invocations exceeds ceil(T_max/m_min) = {bound}."
        )
