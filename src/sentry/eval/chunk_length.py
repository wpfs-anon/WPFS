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


@dataclass
class Summary:

    label: str
    trace: EpisodeTrace
    successes: int
    episodes: int
    mean_steps: float
    L_step: float
    step_period_ms: float = STEP_PERIOD_MS

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def throughput(self) -> float:
        return 1000.0 / self.L_step if self.L_step > 0 else float("inf")

    @property
    def total_ms(self) -> float:
        return self.mean_steps * (self.L_step + self.step_period_ms)


def _episode_configs(rig: Rig, regime: Regime, n: int, seed: int = 0):
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
    total = EpisodeTrace()
    successes = 0
    steps: list[int] = []

    for ec in _episode_configs(rig, regime, n_episodes, seed):
        env = MockReachEnv(ec, rig.spec)
        trace = run_fixed(rig.backend(), env, N_exec, rig.cfg.H, T_max)
        total = total.merge(trace)
        successes += int(env.success)
        steps.append(trace.steps)

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
    L_check_bar = sum(l_check(r, lat) for r in rig.cfg.ladder) / len(rig.cfg.ladder)
    if trace.rho == 0.0 and trace.N_bar == 0.0:
        return float("inf")
    return l_step(
        rho=trace.rho, N_bar=max(trace.N_bar, 1e-9), J_bar=max(trace.J_bar, 1.0),
        m_min=rig.cfg.m_min, lat=lat, L_check_bar=L_check_bar,
    )


def histogram(values: Sequence[int], width: int = 40) -> str:
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


def main() -> None:
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


if __name__ == "__main__":
    main()
