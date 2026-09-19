"""Sample generation following the Stage-B recipe of SS2.5.2.

SS2.5.2 calls data generation "the part of the recipe most easily got wrong",
and gives it explicitly.  We follow it step for step, on the mock environment:

1. Sample a plan anchor ``t``.  Run the **frozen full-depth model in plan
   mode** on ``o_t`` to obtain ``A_t``, and cache it.  "The candidate must come
   from the model's own plan distribution, not from demonstration actions,
   because that is what it will be asked to verify at deployment."
2. Sample an elapsed count ``k ~ U{0, ..., H-1}`` and build ``A_hat`` by
   equation 7 with the **deployed** padding scheme.
3. Choose the check observation ``o*`` according to the sample type.
4. Sample ``tau ~ T`` -- the deployed verification schedule, **not** ``U(0,1)``
   -- and ``eps ~ N(0, I)``.
5. Sample a depth ``E_B ~ p_depth``.

Half of each batch is positive and half negative, with negatives drawn
uniformly from four generators (N1-N4).

This module serves both the deferred Stage-B training and, right now,
conformal calibration (SS2.4.4) and Diagnostic D1 (SS2.8) -- which is the
point of writing it against the recipe rather than ad hoc.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Optional, Sequence

import torch
from torch import Tensor

from sentry.config import SentryConfig
from sentry.core.calibration import liveness
from sentry.core.interfaces import VLABackend
from sentry.core.padding import repad
from sentry.core.types import ChannelSpec, Observation
from sentry.envs.mock_env import MockReachConfig, _rand_xy, render_observation, simulate

__all__ = [
    "SampleKind",
    "Sample",
    "SampleGenConfig",
    "Anchor",
    "NEGATIVES",
    "OBSERVATION_NEGATIVES",
    "generate",
    "phase_of",
    "draw_anchor",
    "build_sample",
    "make_pair",
]

SampleKind = Literal["positive", "N1", "N2", "N3", "N4"]

NEGATIVES: tuple[SampleKind, ...] = ("N1", "N2", "N3", "N4")

OBSERVATION_NEGATIVES: tuple[SampleKind, ...] = ("N1", "N2", "N3")
"""The negatives that make a stale plan by changing ``o*``, not by changing ``A_hat``.

Equation 16 contrasts ``R^shal(o*_+)`` against ``R^shal(o*_-)`` with ``A_hat``,
``tau`` and ``eps`` all held fixed, so its negative partner has to differ in the
**observation**.  N4 corrupts the candidate and leaves the observation identical
to the positive's, so it cannot serve as an eq. 16 partner -- a pair built from
it would have zero separation by construction and would drive ``L_sens`` to its
hinge for reasons that have nothing to do with collapse.  N4 remains a perfectly
good eq. 14 / eq. 15 sample; it is only the *pairing* that excludes it.
"""


@dataclass(frozen=True)
class Sample:
    """One ``(o*, A_hat, y)`` tuple with everything needed to label it."""

    A_hat: Tensor
    """``(H, d_a)`` candidate from equation 7, possibly corrupted (N4)."""
    H_k: int
    obs: Observation
    """The check observation ``o*``."""
    fresh_plan: Optional[Tensor]
    """``(H, d_a)`` full-depth replan from ``o*``, indexed so 0 is "act now".

    Definition 1 is stated against "the plan the target itself would produce
    from the current observation", so this -- not a demonstration action -- is
    the ground truth.  "This is expensive but is done once, offline."

    ``None`` when :func:`build_sample` was told it was not needed: SS2.9 costs
    this at a full ``L_full`` per sample and requires it only "for negatives
    N1-N3".  Anything consuming this for a *label* must handle ``None``;
    anything consuming it for ``h^star`` will never see one, since N1-N3 always
    compute it.
    """
    kind: SampleKind
    live: bool
    h0: Optional[int]
    """Ground-truth first invalid position, for N4 only.

    "N4 is the only generator that supplies a supervised target for the
    accepted *length* rather than for the binary label, and it is what teaches
    the operator to localise the first bad step rather than merely to flag the
    chunk."
    """
    h_star: int
    """``h^star`` for equation 15, resolved by the SS2.5.2 rule.

    "``h^star = H_k``" for positives, "``h^star = h_0``" for N4, and ``h^star``
    from a full-depth evaluation of equation 5 for N1-N3.

    All three branches live in :func:`build_sample` so the rule exists exactly
    once.  Re-deriving it per call site is how positives came to be scored
    against a ``liveness`` label when they are live *by construction*, which
    handed eq. 15 a push-up target on plans that were never stale.
    """
    noise: Tensor
    """The shared ``A^0`` draw behind both the anchor plan and ``fresh_plan``.

    Paper defect D5: without a common random number, eq. 5 compares two
    independent samples of a multi-modal policy and so measures the policy's
    sampling spread rather than staleness.  Carried on the sample so downstream
    code can re-plan *comparably* instead of re-drawing.
    """
    phase: str
    """Manipulation phase, so false acceptance can be broken down per phase."""


@dataclass(frozen=True)
class SampleGenConfig:
    env: MockReachConfig = MockReachConfig()
    delta_min: float = 0.15
    delta_max: float = 0.45
    """``Delta x ~ U(Delta_min, Delta_max)`` for N1 scene perturbation."""
    j_min: int = 5
    """``|j| > j_min`` for N2 plan-observation mismatch."""
    corruption_offset: float = 0.25
    """Magnitude of N4's constant offset."""
    p_sign_flip: float = 0.5
    """N4 applies "a constant offset or a sign flip"; this picks between them."""
    min_H_k: int = 4
    """Skip samples with a live suffix too short to carry a decision."""


def phase_of(ee: Tensor, target: Tensor, cfg: MockReachConfig) -> str:
    """Coarse manipulation phase, for the per-phase breakdown of SS2.4.4/SS2.8.

    The conformal guarantee is *marginal, not conditional*, so it "does not by
    itself bound the false-acceptance rate within a rare but critical phase
    such as final insertion".  Tagging phases is what lets us look.
    """
    d = float(torch.linalg.vector_norm(target - ee))
    if d <= cfg.grasp_radius:
        return "fine_alignment"
    if d <= 3 * cfg.grasp_radius:
        return "approach"
    return "transit"


def generate(
    backend: VLABackend,
    cfg: SentryConfig,
    spec: ChannelSpec,
    n: int,
    gen_cfg: SampleGenConfig = SampleGenConfig(),
    seed: int = 0,
    positive_fraction: float = 0.5,
) -> list[Sample]:
    """Generate ``n`` samples, half positive and half negative.

    Negatives are drawn **uniformly** from the four generators, as SS2.5.2
    specifies.
    """
    rng = random.Random(seed)
    out: list[Sample] = []
    guard = 0
    while len(out) < n and guard < 50 * n:
        guard += 1
        kind: SampleKind = (
            "positive" if rng.random() < positive_fraction else rng.choice(NEGATIVES)
        )
        anchor = draw_anchor(backend, cfg, spec, gen_cfg, rng)
        if anchor is None:
            continue
        s = build_sample(backend, cfg, spec, gen_cfg, rng, anchor, kind)
        if s is not None:
            out.append(s)
    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)}/{n} samples; loosen min_H_k")
    return out


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """Steps 1-2 of the SS2.5.2 recipe, shared by every sample built from it.

    Splitting the anchor out from the sample is what lets one anchor serve a
    positive *and* a matched negative: equation 16 needs both to share
    ``A_hat``, and the amortisation SS2.5.2 recommends -- "cache one plan per
    anchor and reuse it across draws of ``k``, ``tau``, ``eps`` and depth" --
    needs the plan itself to outlive a single sample.
    """

    A: Tensor
    """``(H, d_a)`` the cached full-depth plan."""
    A_hat: Tensor
    """``(H, d_a)`` the eq. 7 candidate, **uncorrupted**.  N4 clones and edits."""
    H_k: int
    k: int
    noise: Tensor
    """``(H, d_a)`` the ``A^0`` draw that produced ``A``.

    Every plan compared against ``A_hat`` must be drawn under this same noise
    (paper defect D5); :func:`build_sample` does so for ``fresh_plan``.
    """
    ee_k: Tensor
    grip_k: float
    target: Tensor
    ee0: Tensor
    phase: str


def draw_anchor(
    backend: VLABackend,
    cfg: SentryConfig,
    spec: ChannelSpec,
    gc: SampleGenConfig,
    rng: random.Random,
    place: Callable[[Observation], Observation] = lambda o: o,
) -> Optional[Anchor]:
    """Steps 1-2: plan anchor, then ``k ~ U{0..H-1}`` and equation 7.

    Args:
        place: hook applied to every rendered observation before the model sees
            it.  The mock environment renders on CPU while the model may live
            on an accelerator, and moving the batch afterwards is too late --
            the ``plan`` call inside generation would already have fed CPU
            images to CUDA weights.

    Returns ``None`` when the draw is unusable (a live suffix too short to carry
    a decision), so callers can simply resample.
    """
    ec = gc.env
    H = cfg.H

    # Step 1: the candidate "must come from the model's own plan distribution,
    # not from demonstration actions, because that is what it will be asked to
    # verify at deployment".
    ee0 = _rand_xy(rng, ec)
    target = _rand_xy(rng, ec, away_from=ee0, min_sep=0.25)
    obs_t = place(render_observation(ee0, target, grip=-1.0, t=0, cfg=ec))

    # The common random number of paper defect D5.  Drawn here, once, and
    # re-used for every plan that will be compared against this one.
    noise = torch.randn(H, backend.d_a)
    A = backend.plan(obs_t, noise)

    # Step 2: elapsed count, then eq. 7 with the DEPLOYED padding scheme.
    k = rng.randrange(0, H)
    if H - k < gc.min_H_k:
        return None
    A_hat, H_k = repad(A[k:H], H, cfg.padding, spec)

    # Scene geometry stays on CPU alongside ``spec``'s statistics; only the
    # tensors the model consumes are placed.
    ee_k, grip_k = simulate(ee0, A.detach().cpu(), k, spec, ec)

    return Anchor(
        A=A,
        A_hat=A_hat,
        H_k=H_k,
        k=k,
        noise=noise,
        ee_k=ee_k,
        grip_k=grip_k,
        target=target,
        ee0=ee0,
        phase=phase_of(ee_k, target, ec),
    )


def build_sample(
    backend: VLABackend,
    cfg: SentryConfig,
    spec: ChannelSpec,
    gc: SampleGenConfig,
    rng: random.Random,
    anchor: Anchor,
    kind: SampleKind,
    place: Callable[[Observation], Observation] = lambda o: o,
    with_fresh_plan: bool = True,
) -> Optional[Sample]:
    """Step 3 onward: choose ``o*`` for ``kind``, label it, resolve ``h^star``.

    ``h^star`` follows SS2.5.2 exactly and in one place -- "``h^star = H_k`` for
    positives; ``h^star = h_0`` for N4; ``h^star`` from a full-depth evaluation
    of equation 5 for N1-N3":

    - **positive** -> ``H_k``.  The scene "has evolved only through the robot's
      own execution of the plan", so the sample is live by construction and
      there is no invalid position to push up.  Re-deriving this from
      :func:`~sentry.core.calibration.liveness` is what previously handed eq. 15
      a push-up target on plans that were never stale.
    - **N4** -> ``h_0``, known by construction.
    - **N1-N3** -> a full-depth evaluation of eq. 5, drawn under the anchor's
      own noise so the comparison isolates the observation (defect D5).

    ``with_fresh_plan`` controls whether the full-depth replan is computed when
    ``h^star`` does not require it.  SS2.9, *Cost of data generation*: "Stage B
    requires a cached plan per anchor and, **for negatives N1-N3**, a full-depth
    liveness evaluation per sample."  Positives and N4 need no such evaluation,
    and at pi_0 scale each one costs a full ``L_full`` -- so training data
    generation passes ``False`` and skips it.  Calibration and Diagnostic D1
    pass ``True``: they need the ground-truth label on *every* sample, including
    the ones whose ``h^star`` is known by construction, because for them the
    label is the thing being measured rather than a training target.

    An N1-N3 sample always computes it regardless, since its ``h^star`` has no
    other source.
    """
    ec = gc.env
    H = cfg.H
    k, H_k = anchor.k, anchor.H_k
    ee_k, grip_k, target = anchor.ee_k, anchor.grip_k, anchor.target

    A_hat = anchor.A_hat
    h0: Optional[int] = None
    live = kind == "positive"

    if kind == "positive":
        obs = render_observation(ee_k, target, grip_k, t=k, cfg=ec)

    elif kind == "N1":
        # Scene perturbation: a task-relevant object displaced by Delta x.
        mag = rng.uniform(gc.delta_min, gc.delta_max)
        ang = rng.uniform(0, 2 * math.pi)
        moved = torch.clamp(
            target + torch.tensor([mag * math.cos(ang), mag * math.sin(ang)]),
            ec.lo,
            ec.hi,
        )
        if float(torch.linalg.vector_norm(moved - target)) < gc.delta_min * 0.5:
            return None  # clamped back onto itself; not a real perturbation
        obs = render_observation(ee_k, moved, grip_k, t=k, cfg=ec)

    elif kind == "N2":
        # Plan-observation mismatch: a different time index of the same episode.
        j = rng.choice([-1, 1]) * rng.randrange(gc.j_min + 1, gc.j_min + 20)
        k2 = k + j
        if not (0 <= k2 < H):
            return None
        ee2, grip2 = simulate(anchor.ee0, anchor.A.detach().cpu(), k2, spec, ec)
        if float(torch.linalg.vector_norm(ee2 - ee_k)) < 1e-3:
            return None  # trajectory already converged; the pair is not stale
        obs = render_observation(ee2, target, grip2, t=k, cfg=ec)

    elif kind == "N3":
        # Cross-task: a different object configuration.
        other = _rand_xy(rng, ec, away_from=target, min_sep=0.3)
        obs = render_observation(ee_k, other, grip_k, t=k, cfg=ec)

    elif kind == "N4":
        # Injected corruption: constant offset or sign flip on a suffix from h0,
        # "which yields a known ground-truth accepted length h_0".
        obs = render_observation(ee_k, target, grip_k, t=k, cfg=ec)
        h0 = rng.randrange(1, H_k) if H_k > 1 else 0
        A_hat = _corrupt(A_hat, h0, H_k, spec, gc, rng)

    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown sample kind {kind!r}")

    obs = place(obs)

    needs_liveness = kind in OBSERVATION_NEGATIVES

    # The expensive ground truth: a full-depth replan from o*, under the
    # anchor's noise.  "This is expensive but is done once, offline."
    fresh_plan = (
        backend.plan(obs, anchor.noise)
        if (with_fresh_plan or needs_liveness)
        else None
    )

    # h^star, by the SS2.5.2 rule -- see the docstring.
    if kind == "positive":
        h_star = H_k
    elif kind == "N4":
        h_star = int(h0)
    else:
        assert fresh_plan is not None  # needs_liveness forced it above
        _, first = liveness(
            fresh_plan, A_hat, H_k, eps=cfg.liveness_eps, m=cfg.liveness_m
        )
        h_star = H_k if first is None else first

    return Sample(
        A_hat=A_hat,
        H_k=H_k,
        obs=obs,
        fresh_plan=fresh_plan,
        kind=kind,
        live=live,
        h0=h0,
        h_star=h_star,
        noise=anchor.noise,
        phase=anchor.phase,
    )


def make_pair(
    backend: VLABackend,
    cfg: SentryConfig,
    spec: ChannelSpec,
    gc: SampleGenConfig,
    rng: random.Random,
    kind: SampleKind,
    place: Callable[[Observation], Observation] = lambda o: o,
) -> Optional[tuple[Sample, Sample]]:
    """One anchor -> ``(sample, eq16_partner)``.

    ``sample`` is of the requested ``kind`` and carries the eq. 14 / eq. 15
    supervision.  ``eq16_partner`` is built from the **same anchor**, so the two
    share ``A_hat`` -- which is what equation 16 requires of ``o*_+`` and
    ``o*_-``.  ``tau`` and ``eps`` are shared by the trainer, which evaluates
    both under one draw.

    The partner's kind is chosen so the pair is always a genuine
    positive-versus-negative contrast, whichever kind ``sample`` is:

    - ``sample`` positive -> partner drawn from :data:`OBSERVATION_NEGATIVES`;
    - ``sample`` negative -> partner is the positive.

    Two negatives contrasted against each other would still separate, but not
    for the reason eq. 16 is measuring: the term exists to punish a shallow mode
    that "ignores ``o*`` and reproduces ``A_hat`` by copying the interpolant",
    and the sharpest evidence against that is a live scene next to a changed
    one.  Restricting the negative side to :data:`OBSERVATION_NEGATIVES` keeps
    the contrast in the observation rather than in the candidate -- see that
    constant for why N4 cannot fill the role.

    Built for **training**, so it pays only what SS2.9's *Cost of data
    generation* says Stage B requires: "a cached plan per anchor and, for
    negatives N1-N3, a full-depth liveness evaluation per sample".  Only the
    ``o*`` of the partner is ever read -- eq. 16 is a distance between two
    reconstructions, and needs no label -- so the partner's replan is skipped
    outright, and the main sample's is skipped whenever its ``h^star`` is known
    by construction.  At pi_0 scale each skipped call is a whole ``L_full``.
    """
    anchor = draw_anchor(backend, cfg, spec, gc, rng, place)
    if anchor is None:
        return None

    sample = build_sample(
        backend, cfg, spec, gc, rng, anchor, kind, place, with_fresh_plan=False
    )
    if sample is None:
        return None

    partner_kind: SampleKind = (
        rng.choice(OBSERVATION_NEGATIVES) if kind == "positive" else "positive"
    )
    partner = build_sample(
        backend, cfg, spec, gc, rng, anchor, partner_kind, place,
        with_fresh_plan=False,
    )
    if partner is None:
        return None

    return sample, partner


def _corrupt(
    A_hat: Tensor,
    h0: int,
    H_k: int,
    spec: ChannelSpec,
    gc: SampleGenConfig,
    rng: random.Random,
) -> Tensor:
    """N4: corrupt the suffix of ``A_hat`` starting at ``h0``."""
    out = A_hat.clone()
    if rng.random() < gc.p_sign_flip:
        # Sign flip -- notably this also flips the gripper channel, which the
        # acceptance rule tests separately (SS2.4.3).
        out[h0:H_k] = -out[h0:H_k]
    else:
        off = gc.corruption_offset
        ang = rng.uniform(0, 2 * math.pi)
        out[h0:H_k, spec.pos[0]] += off * math.cos(ang)
        out[h0:H_k, spec.pos[1]] += off * math.sin(ang)
    return out


