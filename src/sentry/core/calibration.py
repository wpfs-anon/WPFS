r"""Threshold calibration -- Definition 1 and equation 12 (SS2.4.4).

"The thresholds ``delta_pos``, ``delta_rot`` are the system's only
safety-relevant knobs and we set them by conformal calibration rather than by
hand.  The quantity to control is the *false-acceptance rate*: the probability
of accepting a step whose plan is not live in the sense of Definition 1."

.. math::
    \delta = \mathrm{Quantile}_\alpha\left(\{d^{(i)} : i \in \mathcal{D}^-_{cal}\}\right)

yielding :math:`\mathbb{P}(\text{accept} \mid \text{stale}) \le \alpha +
O(1/|\mathcal{D}^-_{cal}|)` by exchangeability of calibration and deployment
data.

Two things the paper is careful about, and so is this module:

- The **rejection rate on the live population is not controlled**.  It "is
  instead the efficiency the system achieves; sweeping ``alpha`` traces the
  success-versus-speedup Pareto curve that we report in place of a single
  operating point."  :class:`CalibrationReport` measures it but never targets it.
- The guarantee's two caveats are returned as **data**, not left in prose:
  exchangeability fails under distribution shift between calibration and
  deployment scenes, and the guarantee is *marginal rather than conditional*,
  so it does not by itself bound false acceptance within a rare but critical
  phase such as final insertion.  SS2.6 addresses the latter structurally, by
  raising depth rather than by tightening ``delta``.
"""

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


# --------------------------------------------------------------------------
# Definition 1 -- live validity (equation 5)
# --------------------------------------------------------------------------


def liveness(
    fresh_plan: Tensor,
    A_hat: Tensor,
    H_k: int,
    eps: float,
    m: int,
) -> tuple[bool, Optional[int]]:
    r"""Definition 1: is the chunk ``(eps, m)``-live at execution step ``t+k``?

    .. math::
        \max_{0 \le h < m} \left\| [\pi_\theta(o_{t+k})]_h - \tilde{A}_{t,k}[h]
        \right\| \le \varepsilon

    Args:
        fresh_plan: ``(H, d_a)`` a **full-depth replan** ``pi_theta(o_{t+k})``,
            already indexed so that position 0 is "act now" -- the same
            indexing convention eq. 7 imposes on ``A_hat``, and drawn under the
            **same** ``A^0`` noise as the plan ``A_hat`` came from (see below).
        A_hat: ``(H, d_a)`` the candidate.
        H_k: number of real entries.
        eps: tolerance ``epsilon``.
        m: horizon ``m >= 1``.

    Returns ``(is_live, first_violation)``, where ``first_violation`` is the
    first position over the **whole** live suffix at which the fresh plan
    disagrees by more than ``eps`` (``None`` if none does).

    Two horizons are deliberately in play.  The binary label uses the paper's
    horizon ``m`` -- liveness is a statement about the *next* ``m`` actions.
    The violation index scans the entire suffix, because eq. 12 reads its
    statistic "on the first genuinely violated position", and that position may
    lie beyond ``m``.

    Liveness "is deliberately defined against the target's own fresh plan
    rather than against a ground-truth action, because the target's fresh plan
    is the best decision available to the system, and because it makes the
    quantity estimable offline".  Computing it requires a full-depth replan --
    "expensive but is done once, offline".

    **Caller contract -- paper defect D5.**  ``fresh_plan`` and the plan that
    produced ``A_hat`` must come from the **same** ``A^0`` draw.  Definition 1
    writes ``pi_theta(o_{t+k})`` as though it were a value, but for a
    flow-matching policy it is a sample from a multi-modal distribution, and two
    independent draws on the *same* observation differ by the policy's own
    sampling spread.  On an untrained ``TinyPi0`` that spread is ~4.0 per
    position against ``liveness_eps = 0.05``, which drives ``first_violation``
    to 0 for every sample -- positives included -- and silently turns eq. 15
    into "reject at position 0" (see
    :func:`sentry.training.losses.margin_loss`).  Nothing here can detect the
    violation, because a mis-drawn ``fresh_plan`` is indistinguishable from a
    genuinely stale one; the fix belongs at the call site, which is why
    :meth:`sentry.core.interfaces.VLABackend.plan` takes a ``noise`` argument.
    :func:`sentry.training.preflight.check_liveness_labels_are_informative`
    guards it empirically.
    """
    if m < 1:
        raise ValueError(f"horizon m must be >= 1, got {m}")
    if fresh_plan.shape != A_hat.shape:
        raise ValueError(
            f"fresh_plan {tuple(fresh_plan.shape)} != A_hat {tuple(A_hat.shape)}"
        )
    if not (0 < H_k <= A_hat.shape[0]):
        raise ValueError(f"H_k={H_k} outside (0, {A_hat.shape[0]}]")

    # Norm over the full normalised action vector.  Eq. 5 writes a bare
    # ||.||; padded channels are identically zero on both sides and so
    # contribute nothing, which makes the full-vector reading the neutral one.
    err = torch.linalg.vector_norm(
        fresh_plan[:H_k] - A_hat[:H_k], ord=2, dim=-1
    )  # (H_k,)

    horizon = min(m, H_k)
    is_live = bool((err[:horizon] <= eps).all().item())

    bad = (err > eps).nonzero(as_tuple=False)
    first = None if bad.numel() == 0 else int(bad[0].item())
    return is_live, first


# --------------------------------------------------------------------------
# Choosing Definition 1's epsilon
# --------------------------------------------------------------------------


def liveness_distance(fresh_plan: Tensor, A_hat: Tensor, m: int = 1) -> float:
    r"""The quantity Definition 1 thresholds: ``max_{h<m} ||fresh[h] - A_hat[h]||``.

    Exposed separately from :func:`liveness` so the *distribution* of this
    distance can be measured before a tolerance is chosen for it.
    """
    if m < 1:
        raise ValueError(f"horizon m must be >= 1, got {m}")
    err = torch.linalg.vector_norm(fresh_plan[:m] - A_hat[:m], ord=2, dim=-1)
    return float(err.max().item())


def calibrate_liveness_eps(
    positive_distances: Sequence[float], quantile: float = 0.9
) -> float:
    r"""Set Definition 1's ``epsilon`` from the *positive* population.

    **Definition 1 does not say how to choose ``epsilon``.**  Equation 5 opens
    with "Fix a tolerance ``eps > 0``" and neither the text nor Table 2 supplies
    a value or a procedure -- yet every downstream quantity depends on it: eq. 12
    calibrates ``delta`` against the stale population that ``epsilon`` defines,
    and eq. 15 reads ``h^star`` from the same label.  This is the same shape of
    gap as defect D3, one level further up.

    A default guessed in normalised action space does not survive contact with a
    real policy.  Measured on ``pi0_libero`` over real LIBERO demonstrations, the
    distance of eq. 5 at position 0 has median ``0.47`` on positives -- chunks
    whose scene evolved only through execution -- against ``2.98`` for a
    plan/observation mismatch and ``3.51`` cross-task.  The signal is there, and
    cleanly: negatives sit about ``4.9x`` higher.  But a tolerance of ``0.05``
    labels **98% of everything stale, positives included**, which collapses the
    label to a constant and takes eq. 12 and eq. 15 down with it.

    The principled choice mirrors what SS3.4.4 already does for ``delta``, one
    population over: ``delta`` is the ``alpha``-quantile of the **stale**
    distances, so ``epsilon`` is the ``quantile``-quantile of the **positive**
    distances.  Positives are identified by *construction* -- ``o* = o_{t+k}``
    from the same demonstration -- not by the label, so there is no circularity.
    ``1 - quantile`` is then the rate at which genuinely live chunks are
    mislabelled stale, and it plays for ``epsilon`` the role ``alpha`` plays for
    ``delta``.

    Args:
        positive_distances: :func:`liveness_distance` over positive samples.
        quantile: fraction of positives that must be labelled live.

    Returns the tolerance.
    """
    if not positive_distances:
        raise ValueError("need at least one positive distance to calibrate epsilon")
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must lie in (0,1), got {quantile}")
    d = torch.tensor(sorted(positive_distances), dtype=torch.float64)
    # Conservative (upper) order statistic, matching eq. 12's finite-sample form.
    idx = min(int(torch.ceil(torch.tensor(quantile * (len(d) + 1))).item()) - 1, len(d) - 1)
    return float(d[max(idx, 0)])


# --------------------------------------------------------------------------
# Calibration records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRecord:
    """One tuple ``(o_{t+k}, A_tilde_{t,k}, y)`` reduced to its statistics."""

    d_pos: float
    """Raw translation distance at the first genuinely violated position."""
    d_rot: float
    """Raw rotation distance at the same position."""
    live: bool
    """``y in {live, stale}`` per Definition 1."""
    h_star: Optional[int]
    """First genuinely violated position, or ``None`` if the plan never diverges."""
    phase: Optional[str] = None
    """Optional manipulation-phase tag, so false acceptance can be broken down
    per phase -- the conditional quantity eq. 12 does *not* control."""

    @property
    def usable_for_calibration(self) -> bool:
        """Stale records with a locatable violation are what eq. 12 consumes."""
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
    """Reduce one calibration sample to a :class:`CalibrationRecord`.

    Args:
        R: ``(K, H, d_a)`` reconstruction from the **deployed** shallow check.
        A_hat: ``(H, d_a)`` candidate.
        fresh_plan: ``(H, d_a)`` full-depth replan, used only for the label.
        H_k: number of real entries.
        spec: channel semantics.
        eps, m: Definition 1 parameters.
        phase: optional phase tag.

    The distances come from the deployed statistic evaluated at
    ``delta_pos = delta_rot = 1``; the *label* and the *position* come from the
    full-depth replan.  Mixing the two is the point: we are calibrating the
    cheap statistic against the expensive ground truth.
    """
    is_live, h_star = liveness(fresh_plan, A_hat, H_k, eps=eps, m=m)
    d_pos_all, d_rot_all = acceptance.raw_distances(R, A_hat, spec, H_k)

    if h_star is None:
        # No genuine violation.  Read the statistic at position 0 -- the step
        # actually under decision.  SS2.4.4 defines the controlled quantity as
        # "the probability of accepting a *step* whose plan is not live", and
        # acceptance is a prefix rule, so a live plan is rejected precisely
        # when position 0 fails.  Reading the worst position instead would
        # report the far tail of the chunk, which no acceptance decision ever
        # rests on, and would inflate the apparent live-rejection rate.
        idx = 0
    else:
        idx = h_star

    # Conservative over T, matching eq. 11's min-over-tau reading.
    return CalibrationRecord(
        d_pos=float(d_pos_all[:, idx].max().item()),
        d_rot=float(d_rot_all[:, idx].max().item()),
        live=is_live,
        h_star=h_star,
        phase=phase,
    )


# --------------------------------------------------------------------------
# Equation 12
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationReport:
    """Thresholds plus the diagnostics needed to read them honestly."""

    thresholds: Thresholds
    alpha: float
    mode: CalibrationMode
    n_stale: int
    n_live: int

    empirical_false_acceptance: float
    """Measured ``P(accept | stale)`` on the calibration set itself.

    Should track ``alpha``.  Being an in-sample quantity it is optimistic;
    the honest number comes from a held-out split.
    """

    live_rejection_rate: float
    """Measured ``P(reject | live)`` -- **not controlled** by eq. 12.

    "The corresponding rejection rate on the live population is not controlled
    and is instead the efficiency the system achieves."
    """

    finite_sample_slack: float
    """The ``O(1/|D^-_cal|)`` term in ``P(accept | stale) <= alpha + O(1/n)``."""

    per_phase_false_acceptance: dict[str, float]
    """Conditional false acceptance by manipulation phase.

    Eq. 12's guarantee is **marginal, not conditional**: it "does not by itself
    bound the false-acceptance rate within a rare but critical phase such as
    final insertion".  Measuring the conditional quantity is the only way to
    see whether that caveat bites; SS2.6 addresses it by raising depth, not by
    tightening ``delta``.
    """

    def caveats(self) -> tuple[str, ...]:
        """The two caveats SS2.4.4 "states plainly", returned as data."""
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
    """Equation 12: set ``delta`` to the ``alpha``-quantile of the stale population.

    ``mode``:

    - ``"per_group"`` -- separate ``alpha``-quantile for the translation and
      rotation populations.  This is what eq. 10's two symbols
      ``delta_pos, delta_rot`` imply, and is the default.
    - ``"joint"`` -- paper-literal.  Eq. 10 evaluated at
      ``delta_pos = delta_rot = 1`` collapses to the single scalar
      ``max(||dpos||, ||drot||)``, so eq. 12 as written yields **one** delta,
      which is then shared by both channel groups.

    The quantile is a **lower-tail** quantile: acceptance is ``d <= delta``, so
    ``P(d <= delta | stale) = alpha`` requires ``delta`` at the ``alpha``
    quantile from below.
    """
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


# --------------------------------------------------------------------------


def _quantile(x: Tensor, alpha: float) -> float:
    """Lower-tail ``alpha``-quantile, guarded against degenerate values."""
    q = float(torch.quantile(x, alpha).item())
    if not (q > 0.0):
        # A non-positive threshold would reject everything.  It means the stale
        # population is not separated from zero by the statistic -- a modelling
        # failure, not a tuning one.
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
