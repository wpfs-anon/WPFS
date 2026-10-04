from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from sentry.config import CalibrationMode
from sentry.core import acceptance
from sentry.core.types import ChannelSpec, Thresholds

__all__ = [
    "liveness",
    "liveness_distance",
    "calibrate_liveness_eps",
    "CalibrationRecord",
    "CalibrationReport",
    "harvest",
    "calibrate",
]


def liveness(
    fresh_plan: Tensor,
    A_hat: Tensor,
    H_k: int,
    eps: float,
    m: int,
) -> tuple[bool, Optional[int]]:
    if m < 1:
        raise ValueError(f"horizon m must be >= 1, got {m}")
    if fresh_plan.shape != A_hat.shape:
        raise ValueError(
            f"fresh_plan {tuple(fresh_plan.shape)} != A_hat {tuple(A_hat.shape)}"
        )
    if not (0 < H_k <= A_hat.shape[0]):
        raise ValueError(f"H_k={H_k} outside (0, {A_hat.shape[0]}]")

    err = torch.linalg.vector_norm(
        fresh_plan[:H_k] - A_hat[:H_k], ord=2, dim=-1
    )

    horizon = min(m, H_k)
    is_live = bool((err[:horizon] <= eps).all().item())

    bad = (err > eps).nonzero(as_tuple=False)
    first = None if bad.numel() == 0 else int(bad[0].item())
    return is_live, first


def liveness_distance(fresh_plan: Tensor, A_hat: Tensor, m: int = 1) -> float:
    if m < 1:
        raise ValueError(f"horizon m must be >= 1, got {m}")
    err = torch.linalg.vector_norm(fresh_plan[:m] - A_hat[:m], ord=2, dim=-1)
    return float(err.max().item())


def calibrate_liveness_eps(
    positive_distances: Sequence[float], quantile: float = 0.9
) -> float:
    if not positive_distances:
        raise ValueError("need at least one positive distance to calibrate epsilon")
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must lie in (0,1), got {quantile}")
    d = torch.tensor(sorted(positive_distances), dtype=torch.float64)
    idx = min(int(torch.ceil(torch.tensor(quantile * (len(d) + 1))).item()) - 1, len(d) - 1)
    return float(d[max(idx, 0)])


@dataclass(frozen=True)
class CalibrationRecord:

    d_pos: float
    d_rot: float
    live: bool
    h_star: Optional[int]
    phase: Optional[str] = None

    @property
    def usable_for_calibration(self) -> bool:
        return (not self.live) and self.h_star is not None


def harvest(
    R: Tensor,
    A_hat: Tensor,
    fresh_plan: Tensor,
    H_k: int,
    spec: ChannelSpec,
    eps: float,
    m: int,
    phase: Optional[str] = None,
) -> CalibrationRecord:
    is_live, h_star = liveness(fresh_plan, A_hat, H_k, eps=eps, m=m)
    d_pos_all, d_rot_all = acceptance.raw_distances(R, A_hat, spec, H_k)

    if h_star is None:
        idx = 0
    else:
        idx = h_star

    return CalibrationRecord(
        d_pos=float(d_pos_all[:, idx].max().item()),
        d_rot=float(d_rot_all[:, idx].max().item()),
        live=is_live,
        h_star=h_star,
        phase=phase,
    )


@dataclass(frozen=True)
class CalibrationReport:

    thresholds: Thresholds
    alpha: float
    mode: CalibrationMode
    n_stale: int
    n_live: int

    empirical_false_acceptance: float

    live_rejection_rate: float

    finite_sample_slack: float

    per_phase_false_acceptance: dict[str, float]

    def caveats(self) -> tuple[str, ...]:
        return (
            "Exchangeability fails under distribution shift between "
            "calibration and deployment scenes; the guarantee is void there.",
            "The guarantee is marginal rather than conditional: it does not "
            "bound false acceptance within a rare but critical phase such as "
            "final insertion.  See per_phase_false_acceptance.",
        )


def calibrate(
    records: Sequence[CalibrationRecord],
    alpha: float,
    mode: CalibrationMode = "per_group",
) -> CalibrationReport:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0,1), got {alpha}")

    stale = [r for r in records if r.usable_for_calibration]
    live = [r for r in records if r.live]
    if not stale:
        raise ValueError(
            "no usable stale records: eq. 12 calibrates on D^-_cal, the stale "
            "subset with a locatable first violation."
        )

    d_pos = torch.tensor([r.d_pos for r in stale], dtype=torch.float64)
    d_rot = torch.tensor([r.d_rot for r in stale], dtype=torch.float64)

    if mode == "per_group":
        thresholds = Thresholds(
            pos=_quantile(d_pos, alpha),
            rot=_quantile(d_rot, alpha),
        )
    elif mode == "joint":
        joint = torch.maximum(d_pos, d_rot)
        delta = _quantile(joint, alpha)
        thresholds = Thresholds(pos=delta, rot=delta)
    else:
        raise ValueError(f"unknown calibration mode {mode!r}")

    return CalibrationReport(
        thresholds=thresholds,
        alpha=alpha,
        mode=mode,
        n_stale=len(stale),
        n_live=len(live),
        empirical_false_acceptance=_accept_rate(stale, thresholds),
        live_rejection_rate=1.0 - _accept_rate(live, thresholds) if live else 0.0,
        finite_sample_slack=1.0 / len(stale),
        per_phase_false_acceptance=_per_phase(stale, thresholds),
    )


def _quantile(x: Tensor, alpha: float) -> float:
    q = float(torch.quantile(x, alpha).item())
    if not (q > 0.0):
        raise ValueError(
            f"calibration produced a non-positive threshold ({q:.6g}) at "
            f"alpha={alpha}.  The stale population is not separated from zero "
            "by the acceptance statistic; check Diagnostic D1 "
            "(d_prune << d_stale) before trusting the verifier."
        )
    return q


def _accept_rate(records: Sequence[CalibrationRecord], th: Thresholds) -> float:
    if not records:
        return 0.0
    hits = sum(1 for r in records if r.d_pos <= th.pos and r.d_rot <= th.rot)
    return hits / len(records)


def _per_phase(
    stale: Sequence[CalibrationRecord], th: Thresholds
) -> dict[str, float]:
    phases: dict[str, list[CalibrationRecord]] = {}
    for r in stale:
        if r.phase is not None:
            phases.setdefault(r.phase, []).append(r)
    return {p: _accept_rate(rs, th) for p, rs in sorted(phases.items())}
