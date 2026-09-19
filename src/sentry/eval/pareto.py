"""The success-versus-speedup Pareto curve, traced by sweeping ``alpha``.

SS2.4.4: "The corresponding rejection rate on the live population is not
controlled and is instead the efficiency the system achieves; **sweeping alpha
traces the success-versus-speedup Pareto curve that we report in place of a
single operating point.**"

So ``alpha`` is the dial, and the deliverable is a curve.  Each point is a full
re-calibration: ``alpha`` sets ``delta`` through equation 12, ``delta`` sets how
readily the operator accepts, and that in turn sets both how often the target is
invoked and how often a stale plan slips through.

Run: ``python -m sentry.eval.pareto``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sentry.config import SentryConfig
from sentry.core.cost import RTX4090D, StageLatencies, l_check, l_step, speedup_vs_fixed
from sentry.core.loop import EpisodeTrace, run_episode
from sentry.envs.mock_env import MockReachConfig, MockReachEnv
from sentry.eval.calibrate import build_records, calibrate_split
from sentry.eval.chunk_length import REGIMES, Regime, _episode_configs
from sentry.eval.harness import SampleGenConfig, generate
from sentry.eval.setup import Rig, build_rig

__all__ = ["ParetoPoint", "sweep", "main"]

DEFAULT_ALPHAS: tuple[float, ...] = (0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4)


@dataclass(frozen=True)
class ParetoPoint:
    alpha: float
    delta_pos: float
    delta_rot: float
    success_rate: float
    mean_chunk: float
    rho: float
    N_bar: float
    J_bar: float
    gain_ratio: float
    L_step: float
    speedup: float
    """Against the strongest fixed baseline that still succeeds -- see
    :func:`sweep`."""
    holdout_false_acceptance: float
    holdout_live_rejection: float


def sweep(
    rig: Rig,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    regime: Optional[Regime] = None,
    n_episodes: int = 12,
    T_max: int = 200,
    lat: StageLatencies = RTX4090D,
    baseline_N_exec: int = 8,
    seed: int = 0,
) -> list[ParetoPoint]:
    """Re-calibrate at each ``alpha`` and re-run the suite.

    The calibration *records* are computed once and reused across ``alpha``:
    the underlying statistics do not depend on the threshold, only the quantile
    taken from them does.  This is the same amortisation SS2.9 describes for
    Stage-B data generation -- "we cache one plan per anchor and reuse it
    across all k, tau, eps and depth draws".
    """
    regime = regime or REGIMES[1]

    samples = generate(
        rig.backend(), rig.cfg, rig.spec, n=600,
        gen_cfg=SampleGenConfig(env=rig.env_cfg), seed=seed + 1,
    )
    records = build_records(
        rig.backend(), rig.cfg, rig.spec, samples,
        generator=torch.Generator().manual_seed(seed),
    )

    points: list[ParetoPoint] = []
    for alpha in alphas:
        try:
            cal = calibrate_split(
                records, alpha=alpha, mode=rig.cfg.calibration_mode, seed=seed
            )
        except ValueError:
            # A degenerate threshold at very small alpha means the stale
            # population is not separated from zero there.  Skip rather than
            # report a fabricated operating point.
            continue

        total = EpisodeTrace()
        successes = 0
        for i, ec in enumerate(_episode_configs(rig, regime, n_episodes, seed)):
            env = MockReachEnv(ec, rig.spec)
            trace = run_episode(
                rig.backend(), env, rig.cfg, cal.thresholds, rig.spec, T_max=T_max,
                generator=torch.Generator().manual_seed(seed + i),
            )
            total = total.merge(trace)
            successes += int(env.success)

        L_check_bar = sum(l_check(r, lat) for r in rig.cfg.ladder) / len(rig.cfg.ladder)
        L_step = l_step(
            rho=total.rho, N_bar=max(total.N_bar, 1e-9), J_bar=max(total.J_bar, 1.0),
            m_min=rig.cfg.m_min, lat=lat, L_check_bar=L_check_bar,
        )

        points.append(
            ParetoPoint(
                alpha=alpha,
                delta_pos=cal.thresholds.pos,
                delta_rot=cal.thresholds.rot,
                success_rate=successes / n_episodes,
                mean_chunk=total.mean_chunk_length,
                rho=total.rho,
                N_bar=total.N_bar,
                J_bar=total.J_bar,
                gain_ratio=total.gain_ratio,
                L_step=L_step,
                speedup=speedup_vs_fixed(L_step, baseline_N_exec, lat),
                holdout_false_acceptance=cal.holdout_false_acceptance,
                holdout_live_rejection=cal.holdout_live_rejection,
            )
        )
    return points


def frontier(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Keep only the non-dominated points on the (success, speedup) plane."""
    out: list[ParetoPoint] = []
    for p in sorted(points, key=lambda q: (-q.success_rate, -q.speedup)):
        if not any(
            q.success_rate >= p.success_rate and q.speedup >= p.speedup and q is not p
            for q in out
        ):
            out.append(p)
    return out


def main() -> None:  # pragma: no cover - reporting
    rig = build_rig()
    regime = REGIMES[1]
    points = sweep(rig, regime=regime)

    print(f"Success-vs-speedup Pareto curve, swept over alpha  [regime: {regime.name}]")
    print()
    header = (
        f"{'alpha':>7} {'d_pos':>8} {'d_rot':>8} {'success':>8} {'chunk':>7} "
        f"{'rho':>6} {'N_bar':>7} {'rho^-1 N':>9} {'L_step':>8} {'speedup':>8} "
        f"{'FAR':>7} {'rej|live':>9}"
    )
    print(header)
    print("-" * len(header))
    for p in points:
        gain = "inf" if p.gain_ratio == float("inf") else f"{p.gain_ratio:.1f}"
        print(
            f"{p.alpha:>7.3f} {p.delta_pos:>8.3f} {p.delta_rot:>8.3f} "
            f"{p.success_rate:>7.0%} {p.mean_chunk:>7.1f} {p.rho:>6.2f} "
            f"{p.N_bar:>7.1f} {gain:>9} {p.L_step:>8.2f} {p.speedup:>7.2f}x "
            f"{p.holdout_false_acceptance:>7.3f} {p.holdout_live_rejection:>9.3f}"
        )

    print()
    front = frontier(points)
    print(f"Non-dominated operating points: alpha in {[p.alpha for p in front]}")
    print()
    print(
        "This curve is the deliverable, not any single row.  Reading it:\n"
        "  * alpha bounds P(accept | stale) -- the FAR column should track it,\n"
        "    up to the O(1/|D^-_cal|) slack, and it is a *marginal* bound: the\n"
        "    per-phase breakdown in eval/d1.py is where the conditional gap\n"
        "    shows up.\n"
        "  * The rejection rate on live plans is NOT controlled.  It is the\n"
        "    price paid for that bound, and it is what moves the speedup.\n"
        "  * Tightening alpha shortens chunks and raises rho, so both terms of\n"
        "    rho^-1 N_bar move the same way -- which is why SS2.7 puts the whole\n"
        "    claim on that product rather than on L_check."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
