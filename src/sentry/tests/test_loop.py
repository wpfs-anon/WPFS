from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from sentry.config import SentryConfig
from sentry.core.cascade import CascadeResult
from sentry.core.interfaces import assert_no_cache_seam
from sentry.core.loop import StaleObservationError, run_episode
from sentry.core.types import (
    ChannelSpec,
    CheckResult,
    DepthRung,
    Observation,
    Thresholds,
)


def spec() -> ChannelSpec:
    return ChannelSpec(
        d_a=7, pos=(0, 1, 2), rot=(3, 4, 5), grip=6,
        mean=torch.zeros(7), scale=torch.ones(7),
    )


@dataclass
class CountingEnv:

    max_steps: int = 40
    stale_by: int = 0

    def __post_init__(self):
        self._t = 0
        self.observations: list[int] = []

    @property
    def t(self) -> int:
        return self._t

    @property
    def terminated(self) -> bool:
        return self._t >= self.max_steps

    def step(self, action):
        self._t += 1

    def observe(self) -> Observation:
        self.observations.append(self._t)
        return Observation(
            images=torch.zeros(1, 3, 4, 4),
            language=torch.zeros(2, dtype=torch.long),
            state=torch.zeros(4),
            t=self._t - self.stale_by,
        )


@dataclass
class StubBackend:
    H: int = 10
    d_a: int = 7
    L_V: int = 27
    L_B: int = 18
    M: int = 10
    plans: int = 0

    def plan(self, obs):
        self.plans += 1
        return torch.zeros(self.H, self.d_a)

    def velocity(self, A_tau, tau, obs, E_V, E_B, adapters):
        return torch.zeros_like(A_tau)


def verdict(N: int, H_k: int, margin=0.9) -> CascadeResult:
    r = CheckResult(
        N=N,
        margin=None if N == 0 else margin,
        d=torch.zeros(1, H_k),
        sign_ok=torch.ones(1, H_k, dtype=torch.bool),
        H_k=H_k,
        rung=DepthRung(8, 5),
    )
    return CascadeResult(verdict=r, rungs_used=1, per_rung=(r,), escalated_unresolved=False)


def cfg(**kw) -> SentryConfig:
    base = dict(H=10, m_min=4, ladder=(DepthRung(8, 5),))
    base.update(kw)
    return SentryConfig(**base)


def go(env, verifier, c=None, T_max=40, backend=None):
    backend = backend or StubBackend()
    return backend, run_episode(
        backend, env, c or cfg(), Thresholds(1.0, 1.0), spec(), T_max=T_max,
        verifier=verifier,
    )


def test_always_reject_still_makes_progress():
    env = CountingEnv(max_steps=40)
    backend, trace = go(env, lambda A, H_k, o: verdict(0, H_k))

    assert trace.steps == 40
    assert trace.target_invocations == 40 // 4
    assert all(c == 4 for c in trace.chunk_lengths)
    assert trace.rho == 1.0


def test_target_invocation_bound_holds():
    for m_min in (1, 2, 4, 7):
        env = CountingEnv(max_steps=60)
        _, trace = go(env, lambda A, H_k, o: verdict(0, H_k), cfg(m_min=m_min), T_max=60)
        assert trace.target_invocations <= -(-60 // m_min)


def test_accepting_verifier_needs_far_fewer_target_calls():
    env = CountingEnv(max_steps=40)
    _, trace = go(env, lambda A, H_k, o: verdict(H_k, H_k))
    assert trace.chunk_lengths == [10, 10, 10, 10]
    assert trace.target_invocations == 4
    assert trace.rho == 0.0


def test_commit_is_unconditional():
    seen: list[int] = []

    def verifier(A_hat, H_k, obs):
        seen.append(obs.t)
        return verdict(0, H_k)

    env = CountingEnv(max_steps=8)
    go(env, verifier, cfg(m_min=4), T_max=8)
    assert seen and all(t % 4 == 0 and t > 0 for t in seen)


def test_plan_exhaustion_is_not_counted_as_rejection():
    env = CountingEnv(max_steps=40)
    _, trace = go(env, lambda A, H_k, o: verdict(H_k, H_k), T_max=24)
    assert trace.rejections == 0
    assert trace.rho == 0.0
    assert trace.plan_exhaustions == 2
    assert trace.chunk_lengths == [10, 10, 4]


def test_stale_observation_is_rejected_loudly():
    env = CountingEnv(max_steps=20, stale_by=3)
    with pytest.raises(StaleObservationError, match="fresh exteroceptive input"):
        go(env, lambda A, H_k, o: verdict(1, H_k), T_max=20)


def test_observation_is_re_read_at_every_check():
    env = CountingEnv(max_steps=20)
    _, trace = go(env, lambda A, H_k, o: verdict(1, H_k), T_max=20)
    assert len(env.observations) == trace.target_invocations + trace.checks


def test_backend_may_not_expose_a_cache():

    class Leaky(StubBackend):
        def velocity(self, A_tau, tau, obs, E_V, E_B, adapters, kv_cache=None):
            return torch.zeros_like(A_tau)

    assert_no_cache_seam(StubBackend())
    with pytest.raises(TypeError, match="no cache may be shared"):
        assert_no_cache_seam(Leaky())


def test_no_timer_by_default():
    env = CountingEnv(max_steps=40)
    _, trace = go(env, lambda A, H_k, o: verdict(1, H_k), T_max=40)
    assert trace.forced_refreshes == 0
    assert trace.prefetch_triggers == 0


def test_S_max_forces_a_refresh_when_enabled():
    env = CountingEnv(max_steps=40)
    _, trace = go(env, lambda A, H_k, o: verdict(1, H_k), cfg(S_max=2), T_max=40)
    assert trace.forced_refreshes > 0


def test_mu_warn_records_prefetch_opportunities():
    fired: list[int] = []
    env = CountingEnv(max_steps=20)
    backend = StubBackend()
    run_episode(
        backend, env, cfg(mu_warn=0.5), Thresholds(1.0, 1.0), spec(), T_max=20,
        verifier=lambda A, H_k, o: verdict(1, H_k, margin=0.2),
        on_prefetch=lambda o: fired.append(o.t),
    )
    assert fired, "a margin of 0.2 is below mu_warn=0.5 and should have fired"


def test_max_accept_bounds_the_blindness_horizon():
    env = CountingEnv(max_steps=40)
    _, uncapped = go(env, lambda A, H_k, o: verdict(H_k, H_k), T_max=40)

    env2 = CountingEnv(max_steps=40)
    _, capped = go(env2, lambda A, H_k, o: verdict(H_k, H_k), cfg(max_accept=2), T_max=40)

    assert max(uncapped.accepted_prefixes) == 6
    assert max(capped.accepted_prefixes) == 2
    assert capped.checks > uncapped.checks


def test_on_replan_fires_once_per_target_invocation():
    seen: list[int] = []
    env = CountingEnv(max_steps=20)
    backend = StubBackend()
    trace = run_episode(
        backend, env, cfg(), Thresholds(1.0, 1.0), spec(), T_max=20,
        verifier=lambda A, H_k, o: verdict(0, H_k),
        on_replan=lambda o: seen.append(o.t),
    )
    assert len(seen) == trace.target_invocations == backend.plans


def test_trace_reports_eq18_inputs():
    env = CountingEnv(max_steps=40)
    _, trace = go(env, lambda A, H_k, o: verdict(3, H_k), T_max=40)
    assert trace.rho == pytest.approx(0.0)
    assert trace.N_bar == pytest.approx(3.0)
    assert trace.J_bar == pytest.approx(1.0)
    assert trace.gain_ratio == float("inf")


def test_gain_ratio_is_finite_when_rejections_occur():
    calls = {"n": 0}

    def alternating(A_hat, H_k, obs):
        calls["n"] += 1
        return verdict(0 if calls["n"] % 2 else 2, H_k)

    env = CountingEnv(max_steps=40)
    _, trace = go(env, alternating, T_max=40)
    assert 0.0 < trace.rho < 1.0
    assert trace.gain_ratio == pytest.approx(trace.N_bar / trace.rho)


def test_traces_merge():
    env = CountingEnv(max_steps=20)
    _, a = go(env, lambda A, H_k, o: verdict(2, H_k), T_max=20)
    env2 = CountingEnv(max_steps=20)
    _, b = go(env2, lambda A, H_k, o: verdict(2, H_k), T_max=20)
    merged = a.merge(b)
    assert merged.steps == a.steps + b.steps
    assert merged.checks == a.checks + b.checks
    assert len(merged.chunk_lengths) == len(a.chunk_lengths) + len(b.chunk_lengths)
