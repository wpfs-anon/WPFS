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
    pass


@dataclass
class EpisodeTrace:

    chunk_lengths: list[int] = field(default_factory=list)

    accepted_prefixes: list[int] = field(default_factory=list)

    rungs_per_check: list[int] = field(default_factory=list)

    margins: list[float] = field(default_factory=list)

    checks: int = 0
    rejections: int = 0

    target_invocations: int = 0

    steps: int = 0
    plan_exhaustions: int = 0

    forced_refreshes: int = 0

    prefetch_triggers: int = 0

    escalated_unresolved: int = 0


    @property
    def rho(self) -> float:
        return self.rejections / self.checks if self.checks else 0.0

    @property
    def N_bar(self) -> float:
        return (
            sum(self.accepted_prefixes) / len(self.accepted_prefixes)
            if self.accepted_prefixes
            else 0.0
        )

    @property
    def J_bar(self) -> float:
        return (
            sum(self.rungs_per_check) / len(self.rungs_per_check)
            if self.rungs_per_check
            else 1.0
        )

    @property
    def gain_ratio(self) -> float:
        return float("inf") if self.rho == 0.0 else self.N_bar / self.rho

    @property
    def mean_chunk_length(self) -> float:
        return (
            sum(self.chunk_lengths) / len(self.chunk_lengths)
            if self.chunk_lengths
            else 0.0
        )

    def merge(self, other: "EpisodeTrace") -> "EpisodeTrace":
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
    trace = EpisodeTrace()
    H = cfg.H
    if verifier is None:
        verifier = CascadeVerifier(backend, cfg, thresholds, spec, generator)

    while not env.terminated and trace.steps < T_max:
        obs = _fresh(env)
        A = backend.plan(obs)
        trace.target_invocations += 1
        if on_replan is not None:
            on_replan(obs)
        if A.shape[0] != H:
            raise ValueError(f"plan returned {A.shape[0]} actions, expected H={H}")

        k = _execute(env, A[: cfg.m_min], trace, T_max)
        if k == 0:
            break

        checks_this_round = 0

        while True:
            if env.terminated or trace.steps >= T_max:
                break

            if k >= H:
                trace.plan_exhaustions += 1
                break

            if cfg.S_max is not None and checks_this_round >= cfg.S_max:
                trace.forced_refreshes += 1
                break

            A_hat, H_k = repad(
                live_suffix=A[k:H],
                H=H,
                scheme=cfg.padding,
                spec=spec,
                learned_pad=learned_pad,
            )

            o_now = _fresh(env)

            result: CascadeResult = verifier(A_hat, H_k, o_now)
            trace.checks += 1
            checks_this_round += 1
            trace.rungs_per_check.append(result.rungs_used)
            if result.escalated_unresolved:
                trace.escalated_unresolved += 1

            if result.N == 0:
                trace.rejections += 1
                break

            assert result.margin is not None
            trace.margins.append(result.margin)

            N = result.N if cfg.max_accept is None else min(result.N, cfg.max_accept)
            trace.accepted_prefixes.append(N)

            if cfg.mu_warn is not None and result.margin < cfg.mu_warn:
                trace.prefetch_triggers += 1
                if on_prefetch is not None:
                    on_prefetch(o_now)

            advanced = _execute(env, A_hat[:N], trace, T_max)
            if advanced == 0:
                break
            k += advanced

        trace.chunk_lengths.append(k)

    _assert_proposition_5(trace, cfg, T_max)
    return trace


def _fresh(env: Environment) -> Observation:
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
    n = 0
    for i in range(actions.shape[0]):
        if env.terminated or trace.steps >= T_max:
            break
        env.step(actions[i])
        trace.steps += 1
        n += 1
    return n


def _assert_proposition_5(trace: EpisodeTrace, cfg: SentryConfig, T_max: int) -> None:
    bound = max(1, math.ceil(T_max / cfg.m_min))
    if trace.target_invocations > bound:
        raise AssertionError(
            f"Proposition 5 violated: {trace.target_invocations} target "
            f"invocations exceeds ceil(T_max/m_min) = {bound}."
        )
