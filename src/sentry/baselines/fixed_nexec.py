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
