"""Conformal calibration driver -- equation 12 (SS2.4.4).

Turns a set of :class:`sentry.eval.harness.Sample` into a
:class:`sentry.core.calibration.CalibrationReport`.

The split matters and is enforced here: ``delta`` is fitted on one half of the
calibration set and *evaluated* on the other.  Eq. 12's guarantee rests on
exchangeability between calibration and deployment data, and an in-sample
false-acceptance rate is optimistic by construction -- reporting only that
number would quietly convert a distribution-free guarantee into a fitted one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sentry.config import CalibrationMode, SentryConfig
from sentry.core.calibration import (
    CalibrationRecord,
    CalibrationReport,
    calibrate,
    harvest,
)
from sentry.core.interfaces import VLABackend
from sentry.core.renoise import renoise_and_reconstruct
from sentry.core.types import ChannelSpec, DepthRung, Thresholds
from sentry.eval.harness import Sample

__all__ = ["build_records", "SplitCalibration", "calibrate_split"]


def build_records(
    backend: VLABackend,
    cfg: SentryConfig,
    spec: ChannelSpec,
    samples: Sequence[Sample],
    rung: Optional[DepthRung] = None,
    generator: Optional[torch.Generator] = None,
) -> list[CalibrationRecord]:
    """Run the deployed check on each sample and reduce it to a record.

    ``rung`` defaults to the ladder's **first** rung: that is where the cascade
    starts, so it is the depth at which most acceptance decisions are actually
    made, and therefore the depth whose statistic should be calibrated.
    """
    rung = rung or cfg.ladder[0]
    records: list[CalibrationRecord] = []

    for s in samples:
        # Eq. 9 at the deployed depth.  The thresholds are irrelevant here --
        # harvest() reads the *raw* distances, i.e. eq. 10 at delta = 1.
        R, _, _ = renoise_and_reconstruct(
            backend=backend,
            A_hat=s.A_hat,
            obs=s.obs,
            rung=rung,
            taus=cfg.taus,
            convention=cfg.tau_convention,
            adapters=True,
            generator=generator,
        )
        records.append(
            harvest(
                R=R,
                A_hat=s.A_hat,
                fresh_plan=s.fresh_plan,
                H_k=s.H_k,
                spec=spec,
                eps=cfg.liveness_eps,
                m=cfg.liveness_m,
                phase=s.phase,
            )
        )
    return records


@dataclass(frozen=True)
class SplitCalibration:
    """A fitted threshold plus its honest, held-out evaluation."""

    thresholds: Thresholds
    fit: CalibrationReport
    """Report from the fitting half (in-sample; optimistic)."""
    holdout_false_acceptance: float
    """``P(accept | stale)`` on the held-out half.  **This** is the number to
    compare against ``alpha``."""
    holdout_live_rejection: float
    """``P(reject | live)`` held out -- the efficiency, which eq. 12 does not
    control and which sweeping ``alpha`` trades against success."""
    n_fit_stale: int
    n_holdout_stale: int
    n_holdout_live: int

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"alpha={self.fit.alpha:.3f}  {self.thresholds}\n"
            f"  held-out P(accept|stale) = {self.holdout_false_acceptance:.4f} "
            f"(target <= {self.fit.alpha:.3f} + {1.0/max(self.n_fit_stale,1):.4f})\n"
            f"  held-out P(reject|live)  = {self.holdout_live_rejection:.4f} "
            f"(uncontrolled -- this is the efficiency)\n"
            f"  n_stale fit/holdout = {self.n_fit_stale}/{self.n_holdout_stale}, "
            f"n_live holdout = {self.n_holdout_live}"
        )


def calibrate_split(
    records: Sequence[CalibrationRecord],
    alpha: float,
    mode: CalibrationMode = "per_group",
    fit_fraction: float = 0.5,
    seed: int = 0,
) -> SplitCalibration:
    """Fit ``delta`` on one split and evaluate it on the other."""
    idx = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed))
    cut = int(len(records) * fit_fraction)
    fit_set = [records[int(i)] for i in idx[:cut]]
    hold_set = [records[int(i)] for i in idx[cut:]]

    report = calibrate(fit_set, alpha=alpha, mode=mode)
    th = report.thresholds

    hold_stale = [r for r in hold_set if r.usable_for_calibration]
    hold_live = [r for r in hold_set if r.live]

    def accept_rate(rs: Sequence[CalibrationRecord]) -> float:
        if not rs:
            return 0.0
        return sum(1 for r in rs if r.d_pos <= th.pos and r.d_rot <= th.rot) / len(rs)

    return SplitCalibration(
        thresholds=th,
        fit=report,
        holdout_false_acceptance=accept_rate(hold_stale),
        holdout_live_rejection=1.0 - accept_rate(hold_live),
        n_fit_stale=report.n_stale,
        n_holdout_stale=len(hold_stale),
        n_holdout_live=len(hold_live),
    )
