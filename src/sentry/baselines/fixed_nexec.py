"""The fixed-``N_exec`` baseline SENTRY claims to dominate.

SS2.1: "Under standard chunked execution, a fixed prefix of ``N_exec`` actions
of ``A_t`` is executed open-loop and the model then replans from
``o_{t+N_exec}``.  ... Our method replaces the constant ``N_exec`` by a
stopping rule on ``k``."

SS2.6 states the claim this baseline exists to test: "the claim it substantiates
is that a single adaptive policy dominates any fixed ``N_exec`` on the
success-latency plane."  Dominance is a statement about a *curve*, so the
baseline must be swept over ``N_exec``, not run at one value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sentry.core.interfaces import Environment, VLABackend
from sentry.core.loop import EpisodeTrace

__all__ = ["run_fixed"]


def run_fixed(
    backend: VLABackend,
    env: Environment,
    N_exec: int,
    H: int,
    T_max: int,
) -> EpisodeTrace:
    """Replan every ``N_exec`` steps, open-loop in between.

    Returns an :class:`EpisodeTrace` so the two policies can be compared with
    the same accounting.  There are no checks, so ``rho``, ``N_bar`` and
    ``J_bar`` are vacuous; ``chunk_lengths`` is constant at ``N_exec`` by
    construction, which is exactly the property SENTRY replaces.
    """
    if not (1 <= N_exec <= H):
        raise ValueError(f"N_exec must lie in [1, {H}], got {N_exec}")

    trace = EpisodeTrace()
    while not env.terminated and trace.steps < T_max:
        A = backend.plan(env.observe())
        trace.target_invocations += 1

        executed = 0
        for i in range(min(N_exec, H)):
            if env.terminated or trace.steps >= T_max:
                break
            env.step(A[i])
            trace.steps += 1
            executed += 1

        if executed == 0:
            break
        trace.chunk_lengths.append(executed)

    return trace
