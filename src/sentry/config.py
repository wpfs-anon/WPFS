"""SENTRY configuration -- Table 2 defaults.

Table 2's caption matters: "Depth ladder and thresholds are set by Diagnostic
D1 and the calibration of Sec. 2.4.4 respectively rather than tuned on the
evaluation suites."  So ``ladder`` and ``Thresholds`` are *outputs* of the
offline pipeline, not knobs to twiddle against a benchmark.  The values here
are the paper's published defaults, and the fields that come from D1 /
calibration are marked as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Sequence

from sentry.core.types import DepthRung

__all__ = [
    "PaddingScheme",
    "TauConvention",
    "CascadeGuard",
    "CalibrationMode",
    "SentryConfig",
    "TABLE2",
    "LADDER_TABLE2",
    "LADDER_TABLE1",
]

PaddingScheme = Literal["hold", "zero_delta", "learned"]
TauConvention = Literal["one_is_clean", "zero_is_clean"]
CascadeGuard = Literal["prose", "literal"]
CalibrationMode = Literal["per_group", "joint"]


# Paper defect D2 -- see sentry.core.types.DepthRung.
LADDER_TABLE2: tuple[DepthRung, ...] = (
    DepthRung(E_V=8, E_B=5),
    DepthRung(E_V=8, E_B=9),
    DepthRung(E_V=8, E_B=12),
)
"""Table 2 reading: ``(E_V; E_B^(1:3)) = (8; 5, 9, 12)`` -- E_V fixed at 8."""

LADDER_TABLE1: tuple[DepthRung, ...] = (
    DepthRung(E_V=8, E_B=5),
    DepthRung(E_V=8, E_B=9),
    DepthRung(E_V=14, E_B=12),
)
"""Table 1 reading: rung 3 uses ``E_V = 14/27``, not 8."""


@dataclass(frozen=True)
class SentryConfig:
    """Frozen system configuration.

    Frozen on purpose.  In particular ``padding`` is "a property of the
    system, not of the call" (SS2.4.1): whichever scheme is chosen "must be
    used identically when generating the training data of Sec. 2.5, since it
    is the training distribution that teaches the shallow mode to ignore the
    tail".  Reading it from here rather than accepting it as a keyword
    argument is what stops deployment and datagen from drifting apart.
    """

    # -- plan geometry (Table 2) -----------------------------------------
    H: int = 50
    """Plan length."""

    m_min: int = 4
    """Unconditional commit after a replan.

    SS2.6: "This is needed for liveness, not for performance."  Those actions
    were produced by the target from the current observation, so verifying
    them against the target's own plan would be vacuous; and without a commit
    the loop could reject at the same ``t`` indefinitely without advancing the
    environment.  Proposition 5 depends on ``m_min >= 1``.
    """

    M: int = 10
    """Euler steps used by plan mode (SS2.1).  Check mode always uses 1."""

    # -- verification (Table 2) -------------------------------------------
    taus: tuple[float, ...] = (0.6, 0.9)
    """``T`` -- verification timesteps, batched.  ``K = |T| = 2`` by default.

    SS2.4.2: the K interpolants "differ only in the leading batch dimension
    and are evaluated in a single batched forward pass, so the cost of T is
    that of one evaluation, not K."
    """

    ladder: tuple[DepthRung, ...] = LADDER_TABLE2
    """Depth cascade ladder.  **Set by Diagnostic D1** (SS2.8), not tuned."""

    mu_esc: float = 0.15
    """Margin below which the cascade escalates to the next rung."""

    alpha: float = 0.05
    """Target false-acceptance level for conformal calibration (eq. 12)."""

    # -- conventions (SS2.9) ----------------------------------------------
    padding: PaddingScheme = "hold"
    """Tail padding for eq. 7.  A system property -- see the class docstring."""

    tau_convention: TauConvention = "one_is_clean"
    """SS2.9 convention (i).  ``one_is_clean``: tau=1 is clean data, as the
    paper takes throughout.  ``zero_is_clean``: the implementation integrates
    in the opposite direction, so eqs. 8 and 9 require ``tau -> 1 - tau``."""

    calibration_mode: CalibrationMode = "per_group"
    """``per_group``: separate alpha-quantile for the pos and rot populations,
    which is what eq. 10's two symbols ``delta_pos, delta_rot`` imply.
    ``joint``: paper-literal.  Eq. 10 evaluated at ``delta_pos = delta_rot = 1``
    collapses to the single scalar ``max(||dpos||, ||drot||)``, so eq. 12
    literally yields one delta shared by both channel groups."""

    # -- paper defects, flag-gated ----------------------------------------
    cascade_guard: CascadeGuard = "prose"
    """Paper defect **D1** -- Algorithm 1's margin test is dead code.

    Line 11 reads ``while N = 0 and j <= J``, while line 13 inside the loop
    reads ``if N > 0 and mu >= mu_esc then break``.  Line 13's guard is exactly
    the negation of SS2.6's stated rule ("a check that returns ``N = 0`` **or**
    ``mu < mu_esc`` is repeated at the next depth"), so the intent is written
    down explicitly -- but line 11 exits the loop the moment ``N > 0``, which
    makes line 13 unreachable in effect.  A check returning ``N > 0`` with
    ``mu < mu_esc`` therefore does not escalate, and the cascade never does the
    one thing SS2.6 introduces it for ("we escalate depth on low confidence").

    The paper fix is one token: line 11 becomes ``while j <= J do``, leaving
    line 13 as the sole exit.  Here, ``prose`` implements that intent;
    ``literal`` reproduces the pseudocode's effective behaviour, for ablation.
    """

    # -- ablation-only, disabled by default (SS2.6) ------------------------
    S_max: Optional[int] = None
    """Forced full-depth refresh cap.

    SS2.6, *No periodic refresh*: "Algorithm 1 contains no timer.  We regard
    this as a falsifiable commitment rather than an omission... If a
    fresh-observation verifier still requires a timer to hold up, the central
    claim of this paper is wrong."  Performance is reported with the timer
    disabled; this exists purely "as an ablation axis".  ``None`` = disabled.
    """

    max_accept: Optional[int] = None
    """Paper defect **D4** -- the acceptance horizon is also the blindness horizon.

    Algorithm 1 line 21 executes the whole accepted prefix open-loop:
    ``execute A_hat[0:N]; k <- k + N``.  Nothing bounds ``N`` except ``H_k``,
    so a single check can license the entire remaining chunk -- after which
    the system is blind for that many steps.  Proposition 2 establishes that a
    cached-context verifier can *never* react to an exogenous event ``Z``;
    SENTRY reacts only at its next check, which may be up to ``N`` steps away.
    The claim that chunk length is "short under perturbation" therefore holds
    only when the perturbation lands near a check boundary.

    ``None`` (default) is paper-faithful.  Setting an integer caps ``N`` and
    makes the reactivity latency an explicit, bounded quantity -- at the cost
    of more checks per accepted step.  Sweeping it traces the
    reactivity-versus-cost trade the paper leaves implicit.
    """

    mu_warn: Optional[float] = None
    """Async prefetch warning level, ``mu_warn > mu_esc`` (SS2.6).

    "Because a low margin precedes rejection, the full-depth replan may be
    launched asynchronously as soon as mu falls below a warning level."  An
    orthogonal systems optimisation, "disabled in the main results so that
    latency numbers reflect the algorithm rather than the schedule."
    """

    # -- training seams (SS2.5); consumed by the deferred training phase ---
    rung_weights: Optional[tuple[float, ...]] = None
    """``w_j`` in eq. 18 -- the multi-exit aggregation ``sum_j w_j c_j(u) L^(j)``.

    ``None`` derives them from the ladder length; see :attr:`w`.  Table 2's
    ``(w_1, w_2, w_3) = (0.6, 0.3, 0.1)`` is what a three-rung ladder gets, but
    the ladder is an *output* of Diagnostic D1 and need not have three rungs, so
    a fixed triple cannot be the field's default.  SS3.5.2: set "proportional
    to the frequency with which rung ``j`` is expected to be evaluated at
    deployment, so that training effort is allocated where inference actually
    spends it.  Rung 1 answers the majority of checks and receives the majority
    of the signal; the deepest rung is consulted only on escalations."

    Deliberately *not* LayerSkip's per-layer scaling: "that schedule exists to
    make a model rely on its early layers, whereas ours exists to match the
    deployment-time load of a fixed ladder, and the two therefore take different
    shapes."
    """

    rung_enable_at: Optional[tuple[float, ...]] = None
    """``u_j`` in the curriculum ``c_j(u) = 1[u >= u_j]``, with ``u`` the
    training fraction.  ``None`` derives them; see :attr:`u_enable`.  Table 2's
    three-rung value is ``(0.3, 0.1, 0.0)``.

    Ordered ``u_1 > u_2 > ... > u_J = 0``, so "training begins with the deepest
    rung alone and admits shallower rungs as the trunk adapter stabilises.
    Without this, the shallow rungs -- whose targets are hardest -- dominate the
    gradient early and destabilise a trunk that is shared with every other rung."
    """

    p_depth: tuple[int, int] = (4, 14)
    """Inclusive support of ``p_depth = U{4, ..., 14}``.

    **Superseded by the multi-exit recipe and kept only for the ablation.**  An
    earlier draft of SS3.5.2 sampled a depth per training example; the current
    text removes that -- "No depth is sampled at this stage.  Depth enters in
    the loss, not in the data" -- because a single pass to ``E_max`` already
    exposes every rung of the ladder at once.  ``StageBTrainer`` follows the
    current text; set ``multi_exit=False`` there to reproduce the old sampler.
    """

    lambda_m: float = 1.0
    lambda_s: float = 0.1
    margin_m: float = 0.2
    """Hinge margin ``m`` in eq. 15."""
    eta: float = 0.05
    """Anti-collapse floor ``eta`` in eq. 16."""

    lora_rank: int = 16
    lora_rank_readout: int = 32
    lora_dropout: float = 0.05

    stage_a_steps: int = 20_000
    stage_b_steps: int = 60_000

    delta_provisional: float = 1.0
    """Paper defect **D3**.

    Eq. 15's ``L_marg`` needs ``d_h``, which needs ``delta_pos/delta_rot``; but
    eq. 12 produces those from calibration *after* training.  Circular.  Stage
    B trains against this provisional value; calibration then runs, with an
    optional re-calibration pass.  Never read as a deployment threshold.
    """

    # -- liveness label (Definition 1) -------------------------------------
    liveness_eps: float = 0.05
    """``epsilon`` in eq. 5.

    Paper defect **D5** makes this threshold meaningful only under a common
    random number -- see :attr:`liveness_m` below.
    """
    liveness_m: int = 1
    """``m`` in eq. 5 -- the horizon over which the fresh plan must agree.

    **Paper defect D5 -- Definition 1 is ill-posed for a stochastic policy.**

    Eq. 5 reads ``max_{0<=h<m} || [pi_theta(o_{t+k})]_h - A_tilde[h] || <= eps``
    as though ``pi_theta(o)`` were a value.  For the flow-matching target it is
    a **sample** from a multi-modal distribution -- precisely the property that
    motivates a diffusion policy over a regressor, and the same property the
    paper invokes to reject a distilled drafter ("mode-averaging failure ...
    cannot represent the multi-modal action distributions that motivate
    diffusion policies in the first place").  Two draws on the *same*
    observation therefore differ by the policy's own sampling spread, and eq. 5
    silently measures that spread instead of staleness.

    This is not a small correction.  Measured on an untrained ``TinyPi0``, two
    ``plan()`` calls on one identical observation disagree by ~4.0 per position
    against ``liveness_eps = 0.05``.  Every sample in a Stage-B batch --
    positives included -- was consequently labelled stale at position 0, which
    reduced eq. 15 to a single term demanding rejection at position 0 on live
    plans, in direct opposition to eq. 14.  Both eq. 12's calibration and eq.
    15's ``h_star`` read this label, so the defect reaches further than D1-D4.

    The resolution is a **common random number**: draw ``A^0`` once per chunk
    under test and re-use it for every fresh plan the label is computed against,
    so the only thing differing between the two plans is the observation.
    :meth:`sentry.core.interfaces.VLABackend.plan` therefore takes a ``noise``
    argument, :func:`sentry.eval.harness.draw_anchor` draws it, and
    :func:`sentry.training.preflight.check_plan_is_deterministic_under_shared_noise`
    asserts backends honour it.

    Unlike D1-D4 this is **not** flag-gated.  The literal reading is not an
    alternative semantics to ablate against -- it is a quantity that is not
    well-defined, so there is nothing to reproduce.  The paper should state the
    shared draw explicitly in Definition 1.
    """

    def __post_init__(self) -> None:
        if self.m_min < 1:
            raise ValueError("m_min must be >= 1; Proposition 5 depends on it")
        if not self.ladder:
            raise ValueError("ladder must contain at least one rung")
        if any(t <= 0.0 or t >= 1.0 for t in self.taus):
            raise ValueError(f"verification timesteps must lie in (0,1), got {self.taus}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must lie in (0,1), got {self.alpha}")
        rungs = [r.E_B for r in self.ladder]
        if rungs != sorted(rungs):
            raise ValueError(
                f"ladder must be ordered E_B^(1) < ... < E_B^(J), got {rungs}"
            )
        if self.mu_warn is not None and self.mu_warn <= self.mu_esc:
            raise ValueError(
                f"mu_warn must exceed mu_esc ({self.mu_warn} <= {self.mu_esc})"
            )
        lo, hi = self.p_depth
        if lo > hi or lo < 1:
            raise ValueError(f"p_depth support must be 1 <= lo <= hi, got {self.p_depth}")
        if self.rung_weights is not None:
            if len(self.rung_weights) != len(self.ladder):
                raise ValueError(
                    f"rung_weights has {len(self.rung_weights)} entries for a ladder "
                    f"of {len(self.ladder)} rungs; eq. 18 needs one w_j per rung"
                )
            if any(w < 0 for w in self.rung_weights):
                raise ValueError(
                    f"rung weights must be non-negative, got {self.rung_weights}"
                )
        if self.rung_enable_at is not None:
            if len(self.rung_enable_at) != len(self.ladder):
                raise ValueError(
                    f"rung_enable_at has {len(self.rung_enable_at)} entries for a "
                    f"ladder of {len(self.ladder)} rungs"
                )
            u = list(self.rung_enable_at)
            if u != sorted(u, reverse=True):
                raise ValueError(
                    f"rung_enable_at must satisfy u_1 >= ... >= u_J, got "
                    f"{self.rung_enable_at}: the curriculum starts with the "
                    "deepest rung alone"
                )
            if u[-1] != 0.0:
                raise ValueError(
                    f"the deepest rung must be enabled from the start (u_J = 0), "
                    f"got {u[-1]}"
                )

    @property
    def w(self) -> tuple[float, ...]:
        """``w_j`` for the current ladder, materialised.

        Table 2's ``(0.6, 0.3, 0.1)`` for a three-rung ladder -- the published
        one.  Otherwise a halving sequence normalised to sum to 1, which encodes
        the same claim the triple does ("rung 1 answers the majority of checks")
        without pretending to a precision the paper does not offer: SS3.5.2 says
        these are initial values, to be "re-estimated once from the observed
        cascade statistics of a first training run".
        """
        if self.rung_weights is not None:
            return self.rung_weights
        J = self.J
        if J == 3:
            return (0.6, 0.3, 0.1)
        raw = [0.5**j for j in range(J)]
        total = sum(raw)
        return tuple(r / total for r in raw)

    @property
    def u_enable(self) -> tuple[float, ...]:
        """``u_j`` for the current ladder, materialised.

        Table 2's ``(0.3, 0.1, 0.0)`` for three rungs; otherwise evenly spaced
        down from ``0.3`` to ``0``, preserving the required ordering
        ``u_1 > ... > u_J = 0`` so the deepest rung trains alone at the start.
        """
        if self.rung_enable_at is not None:
            return self.rung_enable_at
        J = self.J
        if J == 3:
            return (0.3, 0.1, 0.0)
        if J == 1:
            return (0.0,)
        step = 0.3 / (J - 1)
        return tuple(round(0.3 - j * step, 6) for j in range(J))

    @property
    def K(self) -> int:
        """Number of verification timesteps."""
        return len(self.taus)

    @property
    def J(self) -> int:
        """Number of cascade rungs."""
        return len(self.ladder)

    def with_(self, **kw) -> "SentryConfig":
        """Return a copy with fields overridden."""
        return replace(self, **kw)


TABLE2 = SentryConfig()
"""The paper's default configuration (Table 2)."""
