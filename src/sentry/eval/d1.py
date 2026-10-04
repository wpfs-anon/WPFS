from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sentry.config import SentryConfig
from sentry.core import acceptance
from sentry.core.calibration import liveness
from sentry.core.interfaces import VLABackend
from sentry.core.renoise import renoise_and_reconstruct
from sentry.core.types import ChannelSpec, DepthRung
from sentry.eval.harness import Sample, SampleGenConfig, generate
from sentry.eval.setup import Rig, build_rig

__all__ = ["D1Row", "diagnose", "auc", "recommend_ladder", "main"]


@dataclass(frozen=True)
class D1Row:

    rung: DepthRung
    d_prune: float
    d_stale: float
    auc: float
    per_phase_auc: dict[str, float]
    separation: float

    @property
    def viable(self) -> bool:
        return self.separation >= 3.0 and self.auc >= 0.9


def _statistic(
    backend: VLABackend,
    cfg: SentryConfig,
    spec: ChannelSpec,
    sample: Sample,
    rung: DepthRung,
    eps: torch.Tensor,
    reference: Optional[torch.Tensor] = None,
) -> tuple[float, torch.Tensor]:
    R, _, _ = renoise_and_reconstruct(
        backend=backend, A_hat=sample.A_hat, obs=sample.obs, rung=rung,
        taus=cfg.taus, convention=cfg.tau_convention, eps=eps, adapters=True,
    )
    if reference is None:
        d_pos, d_rot = acceptance.raw_distances(R, sample.A_hat, spec, sample.H_k)
        d = torch.maximum(d_pos, d_rot).max(dim=0).values
        return float(d[0].item()), R

    per_tau = []
    for j in range(R.shape[0]):
        d_pos, d_rot = acceptance.raw_distances(
            R[j : j + 1], reference[j], spec, sample.H_k
        )
        per_tau.append(torch.maximum(d_pos, d_rot)[0])
    d = torch.stack(per_tau).max(dim=0).values
    return float(d[0].item()), R


def diagnose(
    rig: Rig,
    samples: Sequence[Sample],
    ladder: Optional[Sequence[DepthRung]] = None,
) -> list[D1Row]:
    cfg, spec = rig.cfg, rig.spec
    full = DepthRung(E_V=rig.oracle_cfg.L_V, E_B=rig.oracle_cfg.L_B)
    ladder = list(ladder or cfg.ladder)

    rows: list[D1Row] = []
    for rung in ladder:
        prune: list[float] = []
        both: list[float] = []
        labels: list[int] = []
        phases: list[str] = []
        stale_signal: list[float] = []

        for i, s in enumerate(samples):
            backend = rig.backend()
            eps = torch.randn(
                s.A_hat.shape, generator=torch.Generator().manual_seed(i)
            )

            _, R_full = _statistic(backend, cfg, spec, s, full, eps)
            dp, _ = _statistic(backend, cfg, spec, s, rung, eps, reference=R_full)
            prune.append(dp)

            db, _ = _statistic(backend, cfg, spec, s, rung, eps)
            both.append(db)

            ds, _ = _statistic(backend, cfg, spec, s, full, eps)

            is_live, _ = liveness(
                s.fresh_plan, s.A_hat, s.H_k, eps=cfg.liveness_eps, m=cfg.liveness_m
            )
            if not is_live:
                stale_signal.append(ds)

            labels.append(0 if is_live else 1)
            phases.append(s.phase)

        by_phase: dict[str, list[tuple[float, int]]] = {}
        for d, y, p in zip(both, labels, phases):
            by_phase.setdefault(p, []).append((d, y))

        d_prune = float(torch.tensor(prune).median())
        d_stale = float(torch.tensor(stale_signal).median()) if stale_signal else 0.0

        rows.append(
            D1Row(
                rung=rung,
                d_prune=d_prune,
                d_stale=d_stale,
                auc=auc(both, labels),
                per_phase_auc={
                    p: auc([d for d, _ in v], [y for _, y in v])
                    for p, v in sorted(by_phase.items())
                },
                separation=d_stale / d_prune if d_prune > 0 else float("inf"),
            )
        )
    return rows


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1

    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def recommend_ladder(rows: Sequence[D1Row], n: int = 3) -> list[DepthRung]:
    viable = sorted((r for r in rows if r.viable), key=lambda r: r.rung.E_B)
    if not viable:
        return []
    if len(viable) <= n:
        return [r.rung for r in viable]
    step = (len(viable) - 1) / (n - 1)
    return [viable[round(i * step)].rung for i in range(n)]


def main() -> None:
    rig = build_rig()
    samples = generate(
        rig.backend(), rig.cfg, rig.spec, n=200,
        gen_cfg=SampleGenConfig(env=rig.env_cfg), seed=7,
    )

    L_V, L_B = rig.oracle_cfg.L_V, rig.oracle_cfg.L_B
    sweep = [
        DepthRung(E_V=max(1, round(L_V * eb / L_B)), E_B=eb)
        for eb in (2, 3, 5, 7, 9, 12, 15)
    ]
    rows = diagnose(rig, samples, sweep)

    header = (
        f"{'rung':<20} {'d_prune':>9} {'d_stale':>9} {'d_stale/d_prune':>16} "
        f"{'AUC':>7}  viable"
    )
    print("Diagnostic D1 -- is verification easier than planning?")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r.rung):<20} {r.d_prune:>9.4f} {r.d_stale:>9.4f} "
            f"{r.separation:>16.1f} {r.auc:>7.3f}  {'yes' if r.viable else 'NO'}"
        )

    print()
    print("AUC by manipulation phase")
    phases = sorted({p for r in rows for p in r.per_phase_auc})
    print(f"  {'rung':<20} " + " ".join(f"{p:>16}" for p in phases))
    for r in rows:
        cells = " ".join(f"{r.per_phase_auc.get(p, float('nan')):>16.3f}" for p in phases)
        print(f"  {str(r.rung):<20} {cells}")

    print()
    recommended = recommend_ladder(rows)
    print(f"Ladder recommended by D1: {[str(r) for r in recommended]}")
    print(f"Table 2's published ladder: {[str(r) for r in rig.cfg.ladder]}")
    print()
    print(
        "SS2.8 treats this as a gate on the whole approach, not an assumption:\n"
        "if no depth reaches d_prune << d_stale, the premise that a decision\n"
        "needs less depth than a generation is false and no amount of tuning\n"
        "delta will rescue the verifier.  The per-phase columns are where the\n"
        "anticipated limitation should surface -- a plan stale for a reason not\n"
        "visible at the truncated depth will be accepted, and fine alignment is\n"
        "where that is expected to bite."
    )


if __name__ == "__main__":
    main()
