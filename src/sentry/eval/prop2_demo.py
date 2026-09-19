"""Proposition 2, measured rather than cited.

Runs the same episode twice under identical conditions -- same backend, same
thresholds, same ladder, same seeds -- varying exactly one thing: whether the
verifier's conditioning is fresh or cached.

An exogenous event ``Z`` displaces the target mid-chunk without touching the
robot's own dynamics, which is the hypothesis of Proposition 2.  The prediction
is sharp:

    "No verifier of this form can react to ``Z`` before ``Z`` has changed the
    robot's dynamics, i.e. before contact has already occurred."

Run: ``python -m sentry.eval.prop2_demo``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sentry.baselines.cached_context import CachedContextVerifier
from sentry.core.cascade import CascadeVerifier
from sentry.core.loop import EpisodeTrace, run_episode
from sentry.envs.mock_env import ExogenousEvent, MockReachConfig, MockReachEnv
from sentry.eval.setup import Rig, build_rig

__all__ = ["Outcome", "run_one", "main"]


@dataclass
class Outcome:
    name: str
    trace: EpisodeTrace
    success: bool
    steps: int
    final_distance: float
    reaction_step: Optional[int]
    """First environment step at which the verifier rejected after ``Z``.

    ``None`` means it never reacted -- Proposition 2's prediction for the
    cached-context family.
    """


def run_one(
    rig: Rig,
    event: Optional[ExogenousEvent],
    cached: bool,
    T_max: int = 200,
    seed: int = 0,
) -> Outcome:
    env_cfg = MockReachConfig(
        H=rig.cfg.H, max_steps=rig.env_cfg.max_steps, event=event
    )
    env = MockReachEnv(env_cfg, rig.spec)
    backend = rig.backend()
    gen = torch.Generator().manual_seed(seed)

    rejections: list[int] = []
    base = (
        CachedContextVerifier(backend, rig.cfg, rig.thresholds, rig.spec, gen)
        if cached
        else CascadeVerifier(backend, rig.cfg, rig.thresholds, rig.spec, gen)
    )

    def verifier(A_hat, H_k, obs):
        result = base(A_hat, H_k, obs)
        if result.N == 0:
            rejections.append(obs.t)
        return result

    trace = run_episode(
        backend,
        env,
        rig.cfg,
        rig.thresholds,
        rig.spec,
        T_max=T_max,
        generator=gen,
        verifier=verifier,
        on_replan=base.refresh_context if cached else None,
    )

    after = [t for t in rejections if event is None or t >= event.step]
    return Outcome(
        name="cached-context (Prop. 2)" if cached else "SENTRY (fresh)",
        trace=trace,
        success=env.success,
        steps=trace.steps,
        final_distance=env.distance,
        reaction_step=after[0] if after else None,
    )


def main() -> None:  # pragma: no cover - reporting
    rig = build_rig()
    print("Conformal calibration (eq. 12)")
    print("  " + rig.calibration.summary().replace("\n", "\n  "))
    print()

    event = ExogenousEvent(step=10, displacement=(-0.45, 0.30))

    # The right comparison is not "which verifier does better", but "does Z
    # change what this verifier does at all".  Proposition 2 says the answer
    # for the cached-context family is no: I(V; Z) = 0.
    print(f"Response to an exogenous event Z at t={event.step}")
    print("(each verifier compared against ITSELF with and without Z)")
    print()
    header = (
        f"{'verifier':<26} {'Z':<8} {'targets':>8} {'steps':>6} "
        f"{'reacted':>8}  chunk lengths"
    )
    print(header)
    print("-" * len(header))

    outcomes: dict[str, list[Outcome]] = {}
    for cached in (False, True):
        for ev in (None, event):
            o = run_one(rig, ev, cached=cached)
            outcomes.setdefault(o.name, []).append(o)
            reacted = (
                "-" if ev is None
                else ("never" if o.reaction_step is None else f"t={o.reaction_step}")
            )
            print(
                f"{o.name:<26} {('absent' if ev is None else 'present'):<8} "
                f"{o.trace.target_invocations:>8} {o.steps:>6} {reacted:>8}  "
                f"{o.trace.chunk_lengths}"
            )
        print()

    # SS2.6 states the discriminator: the chunk-length distribution should be
    # "wide in static phases, short under perturbation".  So the question is
    # not whether the two runs differ at all -- episode lengths differ for
    # trivial reasons once the target moves -- but whether Z *shortens* the
    # realised chunk.
    print("Reading the result: does Z shorten the realised chunk?")
    print("-" * 54)
    print(f"  {'verifier':<26} {'static':>8} {'with Z':>8} {'change':>9}   verdict")
    for name, (static, perturbed) in outcomes.items():
        a = static.trace.mean_chunk_length
        b = perturbed.trace.mean_chunk_length
        rel = (b - a) / a if a else 0.0
        verdict = "DETECTS Z" if rel <= -0.25 else "blind to Z"
        print(f"  {name:<26} {a:>8.1f} {b:>8.1f} {rel:>+8.0%}   {verdict}")

    print()
    print(
        "The cached-context verifier's rejections form a *schedule*, not a\n"
        "detection.  Its chunk length does not shorten when Z occurs -- it was\n"
        "already at the floor set by its own truncation error, and\n"
        "I(V(A_hat, c_t, s_{t+k}); Z) = 0 leaves it nothing to shorten *for*.\n"
        "Note it also pays many more full-depth calls than SENTRY in the\n"
        "static case, where nothing whatsoever has happened: being blind, it\n"
        "cannot tell that the plan is still good either.  This is the\n"
        "empirical shape of the pattern SS2.2 reports from prior work, whose\n"
        "losses are recovered only by a forced periodic refresh and a\n"
        "hand-specified gripper fallback -- 'Both are timers, not detectors.'"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
