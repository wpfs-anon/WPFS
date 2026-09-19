"""Build SS3.5.2 samples from real LIBERO demonstrations.

This is the SS3.5.2 datagen recipe run against a real corpus instead of the mock
environment, and it follows the same five steps: sample a plan anchor ``t``, run
the frozen full-depth model on ``o_t`` to get ``A_t``, sample ``k ~ U{0..H-1}``,
build ``A_hat`` by eq. 7 with the deployed padding, then choose ``o*`` by sample
type.

**Three of the four negative generators work on demonstrations alone.**  N2
(plan--observation mismatch) pairs the chunk with an observation from a
different index of the same episode; N3 (cross-task) takes one from a different
episode; N4 (injected corruption) edits the candidate and leaves the observation
alone.  Only **N1** -- re-rendering the scene with a task-relevant object
displaced -- needs the simulator, so it is absent here and D1 is reported
without it.  That is a real gap for the headline experiment, where N1 is the
generator closest to the exogenous events Proposition 2 is about; it is not a
gap for the D1 gate, which asks whether truncation error is small next to
staleness signal.

The shared noise draw of paper defect **D5** is threaded through: one ``A^0`` per
anchor, reused for the fresh plan the liveness label is computed against, so
eq. 5 compares two plans that differ only in their observation.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Optional, Sequence

import torch
from torch import Tensor

from sentry.config import SentryConfig
from sentry.core.calibration import (
    calibrate_liveness_eps,
    liveness,
    liveness_distance,
)
from sentry.core.interfaces import VLABackend
from sentry.core.padding import repad
from sentry.core.types import ChannelSpec, Observation
from sentry.envs.libero_data import LiberoCorpus, LiberoEpisode, LiberoPrompt
from sentry.eval.harness import Sample, SampleKind

__all__ = ["LiberoSampleBuilder", "phase_of", "calibrate_eps_and_relabel"]

# Generators that a demonstration corpus can supply unaided.  N1 is deliberately
# absent -- see the module docstring.
DEMO_NEGATIVES: tuple[SampleKind, ...] = ("N2", "N3", "N4")


def phase_of(episode: LiberoEpisode, t: int) -> str:
    """A coarse manipulation phase, read off the demonstration's gripper channel.

    SS3.10 asks for the D1 curve "broken down by manipulation phase", and SS3.10
    also predicts where the method should struggle: "a plan that is stale for a
    reason that is not visible at the truncated depth -- a small occluded
    displacement, a slipping grasp -- will be accepted... fine alignment is where
    that is expected to bite."

    LIBERO encodes the gripper in action channel 6 as ``+-1``, so the transitions
    of that channel segment an episode without any privileged simulator state:
    before the first close the robot is reaching, while closed it is
    transporting, and after it reopens it is retreating.  This is a proxy, not
    ground truth -- it says nothing about *fine alignment* specifically, which
    would need contact information the demonstration does not record.
    """
    grip = episode.actions[:, 6]
    closed = (grip > 0).nonzero(as_tuple=False)
    if closed.numel() == 0:
        return "reach"
    first_close = int(closed[0].item())
    if t < first_close:
        return "reach"
    after = (grip[first_close:] < 0).nonzero(as_tuple=False)
    if after.numel() and t >= first_close + int(after[0].item()):
        return "release"
    return "transport"


class LiberoSampleBuilder:
    """Turns demonstration episodes into :class:`Sample` tuples."""

    def __init__(
        self,
        backend: VLABackend,
        cfg: SentryConfig,
        spec: ChannelSpec,
        corpus: LiberoCorpus,
        prompt: LiberoPrompt,
        state_mean: Tensor,
        state_scale: Tensor,
        j_min: int = 8,
        n4_offset: float = 0.5,
    ) -> None:
        self.backend = backend
        self.cfg = cfg
        self.spec = spec
        self.corpus = corpus
        self.prompt = prompt
        self.state_mean = state_mean
        self.state_scale = state_scale
        self.j_min = j_min
        """``|j| > j_min`` for N2 -- far enough that the scene has genuinely moved."""
        self.n4_offset = n4_offset
        self._tokens: dict[int, Tensor] = {}

    # ------------------------------------------------------------------

    def _obs(self, ep: LiberoEpisode, t: int) -> Observation:
        if ep.index not in self._tokens:
            self._tokens[ep.index] = self.prompt(ep.prompt)
        return ep.observation(
            t, self._tokens[ep.index], self.state_mean, self.state_scale, d_a=self.spec.d_a
        )

    def build(
        self,
        episodes: Sequence[LiberoEpisode],
        n: int,
        rng: random.Random,
        kinds: Sequence[SampleKind] = ("positive", *DEMO_NEGATIVES),
        seed: int = 0,
    ) -> list[Sample]:
        """Draw ``n`` samples, cycling through ``kinds``.

        Half positive and half negative is what SS3.5.2 asks for in a *training*
        batch; here the mix is whatever ``kinds`` says, because D1 wants both
        classes present in usable numbers for the AUC rather than a particular
        ratio.
        """
        H = self.cfg.H
        out: list[Sample] = []

        for i in range(n):
            kind: SampleKind = kinds[i % len(kinds)]
            ep = episodes[rng.randrange(len(episodes))]
            if len(ep) < 4:
                continue

            # Step 1: a plan anchor, and the shared A^0 behind every plan drawn
            # for this sample (defect D5).
            t = rng.randrange(0, max(1, len(ep) - 1))
            noise = torch.randn(
                (H, self.spec.d_a), generator=torch.Generator().manual_seed(seed + i)
            )
            obs_t = self._obs(ep, t)
            A = self.backend.plan(obs_t, noise=noise)

            # Step 2: elapsed count and the eq. 7 candidate.
            k = rng.randrange(0, H)
            A_hat, H_k = repad(
                live_suffix=A[k:H].cpu(),
                H=H,
                scheme=self.cfg.padding,
                spec=self.spec,
                learned_pad=None,
            )

            # Step 3: the check observation, by sample type.
            h0: Optional[int] = None
            if kind == "positive":
                t_star = min(t + k, len(ep) - 1)
                obs_star = self._obs(ep, t_star)
            elif kind == "N2":
                # Same episode, an index far enough away that the scene moved.
                lo, hi = 0, len(ep) - 1
                choices = [
                    u for u in range(lo, hi + 1) if abs(u - (t + k)) > self.j_min
                ]
                if not choices:
                    continue
                obs_star = self._obs(ep, rng.choice(choices))
            elif kind == "N3":
                other = ep
                for _ in range(8):
                    cand = episodes[rng.randrange(len(episodes))]
                    if cand.index != ep.index and cand.prompt != ep.prompt:
                        other = cand
                        break
                if other.index == ep.index:
                    continue
                obs_star = self._obs(other, rng.randrange(len(other)))
            elif kind == "N4":
                # Corruption is applied to the candidate; the observation is the
                # honest one, which is what makes h0 a ground-truth length.
                t_star = min(t + k, len(ep) - 1)
                obs_star = self._obs(ep, t_star)
                if H_k < 2:
                    continue
                h0 = rng.randrange(1, H_k)
                A_hat = A_hat.clone()
                if rng.random() < 0.5:
                    A_hat[h0:H_k, list(self.spec.pos)] += self.n4_offset
                else:
                    A_hat[h0:H_k, self.spec.grip] *= -1.0
            else:
                raise ValueError(f"unsupported sample kind {kind!r}")

            # Step 4: the liveness label, from a fresh full-depth replan under
            # the SAME noise.  Definition 1 is stated against "the plan the
            # target itself would produce from the current observation".
            fresh = self.backend.plan(obs_star, noise=noise).cpu()

            is_live, first = liveness(
                fresh, A_hat, H_k, eps=self.cfg.liveness_eps, m=self.cfg.liveness_m
            )

            # SS3.5.2's h* rule, in one place: H_k for positives, h0 for N4,
            # and a full-depth evaluation of eq. 5 otherwise.
            if kind == "positive":
                h_star = H_k
            elif kind == "N4":
                h_star = h0 if h0 is not None else H_k
            else:
                h_star = first if first is not None else H_k

            out.append(
                Sample(
                    A_hat=A_hat,
                    H_k=H_k,
                    obs=obs_star,
                    fresh_plan=fresh,
                    kind=kind,
                    live=is_live,
                    h0=h0,
                    h_star=h_star,
                    noise=noise,
                    phase=phase_of(ep, min(t + k, len(ep) - 1)),
                )
            )
        return out


def calibrate_eps_and_relabel(
    samples: Sequence[Sample],
    cfg: SentryConfig,
    quantile: float = 0.9,
) -> tuple[float, list[Sample]]:
    """Choose Definition 1's ``epsilon`` from these samples, then relabel them.

    Two passes over the same set, because ``epsilon`` is not knowable in advance:
    the builder labels with whatever ``cfg.liveness_eps`` holds, and those labels
    are then discarded in favour of ones computed against a tolerance measured
    from the positive population (see
    :func:`sentry.core.calibration.calibrate_liveness_eps`).

    ``h_star`` is recomputed with the label, since for N2/N3 it comes from the
    same evaluation of eq. 5; positives keep ``H_k`` and N4 keeps ``h0``, which
    are fixed by construction and do not depend on the tolerance.

    Returns ``(epsilon, relabelled)``.
    """
    pos = [
        liveness_distance(s.fresh_plan, s.A_hat, m=cfg.liveness_m)
        for s in samples
        if s.kind == "positive" and s.fresh_plan is not None
    ]
    if not pos:
        raise ValueError(
            "no positive samples: epsilon is calibrated against chunks that are "
            "live by construction, so at least one is required"
        )
    eps = calibrate_liveness_eps(pos, quantile=quantile)

    out: list[Sample] = []
    for s in samples:
        if s.fresh_plan is None:
            out.append(s)
            continue
        is_live, first = liveness(
            s.fresh_plan, s.A_hat, s.H_k, eps=eps, m=cfg.liveness_m
        )
        if s.kind == "positive":
            h_star = s.H_k
        elif s.kind == "N4":
            h_star = s.h0 if s.h0 is not None else s.H_k
        else:
            h_star = first if first is not None else s.H_k
        out.append(dataclasses.replace(s, live=is_live, h_star=h_star))
    return eps, out
