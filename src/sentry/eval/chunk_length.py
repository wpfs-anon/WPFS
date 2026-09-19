"""The realised chunk-length distribution -- the paper's primary reported object.

SS2.6: "The quantity ``k`` at the moment the speculative loop exits is the
*realised chunk length*: the number of actions executed per full-depth call.
**It is not a hyperparameter.**  Its distribution -- wide in static phases,
short under perturbation -- is the primary object we report, and the claim it
substantiates is that a single adaptive policy dominates any fixed ``N_exec``
on the success-latency plane."

Run: ``python -m sentry.eval.chunk_length``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from sentry.baselines.fixed_nexec import run_fixed
from sentry.core.cost import RTX4090D, StageLatencies, l_check, l_step
from sentry.core.loop import EpisodeTrace, run_episode
from sentry.envs.mock_env import ExogenousEvent, MockReachConfig, MockReachEnv
from sentry.eval.setup import Rig, build_rig

__all__ = ["Regime", "REGIMES", "collect", "Summary", "summarise", "main"]


@dataclass(frozen=True)
class Regime:
    name: str
    event: Optional[ExogenousEvent]


REGIMES: tuple[Regime, ...] = (
    Regime("static", None),
    Regime("single displacement", ExogenousEvent(step=10, displacement=(-0.45, 0.30))),
    Regime("moving target", ExogenousEvent(step=6, drift=(0.012, -0.010))),
)


STEP_PERIOD_MS = 50.0
"""Physical actuation time per environment step, at a 20 Hz control rate.

Not a detail.  ``L_step`` alone measures *compute per step*, and on that metric
a policy that blunders along and eventually arrives looks excellent: a fixed
``N_exec = 50`` replans twice an episode, so its compute amortises to almost
nothing even while it takes four times as many steps to finish the task.  SS2.6
claims dominance "on the success-latency plane", and latency to *complete the
task* is what a robot actually spends -- compute plus actuation, over however
many steps the policy needs.
"""


@dataclass
class Summary:
    """Aggregate accounting for one policy in one regime."""

    label: str
    trace: EpisodeTrace
    successes: int
    episodes: int
    mean_steps: float
    L_step: float
    """Amortised cost per environment step (eq. 18), in ms."""
    step_period_ms: float = STEP_PERIOD_MS

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def throughput(self) -> float:
        """Environment steps per second, from ``L_step``."""
        return 1000.0 / self.L_step if self.L_step > 0 else float("inf")

    @property
    def total_ms(self) -> float:
        """Mean wall-clock to finish the task: steps x (compute + actuation).

        This is the quantity the success-latency plane is about.  Reporting
        ``L_step`` on its own rewards taking more steps, which is backwards.
        """
        return self.mean_steps * (self.L_step + self.step_period_ms)


def _episode_configs(rig: Rig, regime: Regime, n: int, seed: int = 0):
    """Vary the start/target so the distribution has something to be *of*."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        start = (0.10 + 0.25 * torch.rand(1, generator=g).item(),
                 0.10 + 0.25 * torch.rand(1, generator=g).item())
        target = (0.65 + 0.25 * torch.rand(1, generator=g).item(),
                  0.60 + 0.30 * torch.rand(1, generator=g).item())
        yield MockReachConfig(
            H=rig.cfg.H, max_steps=rig.env_cfg.max_steps,
            start=start, target=target, event=regime.event,
        )


def collect(
    rig: Rig,
    regime: Regime,
    n_episodes: int = 12,
    T_max: int = 200,
    lat: StageLatencies = RTX4090D,
    seed: int = 0,
) -> Summary:
    """Run SENTRY across ``n_episodes`` and aggregate."""
    total = EpisodeTrace()
    successes = 0
    steps: list[int] = []

    for i, ec in enumerate(_episode_configs(rig, regime, n_episodes, seed)):
        env = MockReachEnv(ec, rig.spec)
        backend = rig.backend()
        trace = run_episode(
            backend, env, rig.cfg, rig.thresholds, rig.spec, T_max=T_max,
            generator=torch.Generator().manual_seed(seed + i),
        )
        total = total.merge(trace)
        successes += int(env.success)
        steps.append(trace.steps)

    return Summary(
        label="SENTRY (adaptive)",
        trace=total,
        successes=successes,
        episodes=n_episodes,
        mean_steps=sum(steps) / len(steps),
        L_step=_amortised(rig, total, lat),
    )


def collect_fixed(
    rig: Rig,
    regime: Regime,
    N_exec: int,
    n_episodes: int = 12,
    T_max: int = 200,
    lat: StageLatencies = RTX4090D,
    seed: int = 0,
) -> Summary:
    """Run the fixed-``N_exec`` baseline across the same episodes."""
    total = EpisodeTrace()
    successes = 0
    steps: list[int] = []

    for ec in _episode_configs(rig, regime, n_episodes, seed):
        env = MockReachEnv(ec, rig.spec)
        trace = run_fixed(rig.backend(), env, N_exec, rig.cfg.H, T_max)
        total = total.merge(trace)
        successes += int(env.success)
        steps.append(trace.steps)

    # No checks: the baseline pays one full replan per N_exec executed steps.
    executed = sum(total.chunk_lengths) or 1
    return Summary(
        label=f"fixed N_exec={N_exec}",
        trace=total,
        successes=successes,
        episodes=n_episodes,
        mean_steps=sum(steps) / len(steps),
        L_step=lat.L_full * total.target_invocations / executed,
    )


def _amortised(rig: Rig, trace: EpisodeTrace, lat: StageLatencies) -> float:
    """Equation 18, using this run's measured ``rho``, ``N_bar`` and ``J_bar``."""
    L_check_bar = sum(l_check(r, lat) for r in rig.cfg.ladder) / len(rig.cfg.ladder)
    if trace.rho == 0.0 and trace.N_bar == 0.0:
        return float("inf")
    return l_step(
        rho=trace.rho, N_bar=max(trace.N_bar, 1e-9), J_bar=max(trace.J_bar, 1.0),
        m_min=rig.cfg.m_min, lat=lat, L_check_bar=L_check_bar,
    )


def histogram(values: Sequence[int], width: int = 40) -> str:
    """A terminal histogram of the chunk-length distribution."""
    if not values:
        return "  (empty)"
    lo, hi = min(values), max(values)
    n_bins = min(10, max(1, hi - lo + 1))
    span = max(hi - lo + 1, 1)
    bins = [0] * n_bins
    for v in values:
        bins[min(n_bins - 1, (v - lo) * n_bins // span)] += 1
    peak = max(bins) or 1
    out = []
    for i, c in enumerate(bins):
        a = lo + i * span // n_bins
        b = lo + ((i + 1) * span // n_bins) - 1
        label = f"{a}" if a == b else f"{a}-{b}"
        out.append(f"  {label:>7} | {'#' * int(width * c / peak):<{width}} {c}")
    return "\n".join(out)


def main() -> None:  # pragma: no cover - reporting
    rig = build_rig()
    print("Conformal calibration (eq. 12)")
    print("  " + rig.calibration.summary().replace("\n", "\n  "))
    print()

    for regime in REGIMES:
        s = collect(rig, regime)
        lengths = s.trace.chunk_lengths
        print(f"=== regime: {regime.name} " + "=" * (46 - len(regime.name)))
        print(
            f"  episodes={s.episodes}  success={s.success_rate:.0%}  "
            f"mean chunk={s.trace.mean_chunk_length:.1f}  "
            f"min/max={min(lengths)}/{max(lengths)}  "
            f"rho={s.trace.rho:.2f}  N_bar={s.trace.N_bar:.1f}  "
            f"J_bar={s.trace.J_bar:.2f}"
        )
        print(histogram(lengths))
        print()

    # SS2.6's claim: a single adaptive policy dominates any fixed N_exec.
    print("=== success-latency plane: adaptive vs fixed N_exec " + "=" * 16)
    print(f"(actuation assumed at {1000/STEP_PERIOD_MS:.0f} Hz, i.e. "
          f"{STEP_PERIOD_MS:.0f} ms per step)")
    header = (
        f"{'regime':<22} {'policy':<20} {'success':>8} {'steps':>7} "
        f"{'L_step':>8} {'total ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for regime in REGIMES:
        rows = [collect(rig, regime)]
        rows += [collect_fixed(rig, regime, n) for n in (1, 4, 10, 25, 50)]
        best = min(r.total_ms for r in rows)
        for r in rows:
            mark = " <-" if r.total_ms == best else ""
            print(
                f"{regime.name:<22} {r.label:<20} {r.success_rate:>7.0%} "
                f"{r.mean_steps:>7.1f} {r.L_step:>8.2f} {r.total_ms:>10.0f}{mark}"
            )
        print()

    print(
        "The chunk length is an output, not a setting.  A fixed N_exec must be\n"
        "chosen in advance for the worst regime it will meet; the adaptive rule\n"
        "spends long chunks where the scene is static and short ones where it\n"
        "is not.\n\n"
        "Read the two latency columns together, never L_step alone.  A lazy\n"
        "policy scores beautifully on L_step -- fixed N_exec=50 replans twice\n"
        "an episode, so its compute amortises to nearly nothing -- while taking\n"
        "several times as many steps to finish.  Compute per step is not the\n"
        "cost; time to complete the task is.\n\n"
        "And per SS2.7, the gain is not supposed to come from L_check being\n"
        "small: a shallow check is not cheaper per call than a 110M drafter.\n"
        "It has to come from the product rho^-1 N_bar in equation 18."
    )
    for regime in REGIMES:
        s = collect(rig, regime)
        print(f"    rho^-1 N_bar [{regime.name}] = {s.trace.gain_ratio:.1f}")


if __name__ == "__main__":  # pragma: no cover
    main()
