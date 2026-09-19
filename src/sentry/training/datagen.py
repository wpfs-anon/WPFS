"""Batches for Stage A and Stage B (SS2.5).

The Stage-B recipe is implemented in :mod:`sentry.eval.harness` -- plan
anchors, ``k ~ U{0..H-1}``, positives and N1-N4 -- and reused here rather than
reimplemented, so deployment, calibration and training cannot drift apart.

That reuse used to be a claim rather than a fact: this module reimplemented the
N1 branch by hand and Stage B consequently trained on **N1 alone**.  The cost
was specific.  N4 is, per SS2.5.2, "the only generator that supplies a
supervised target for the accepted *length* rather than for the binary label,
and it is what teaches the operator to localise the first bad step rather than
merely to flag the chunk" -- and it was the only source of ``h^star`` that does
not route through :func:`~sentry.core.calibration.liveness`, hence the only one
immune to the noise-pairing defect below.

What this module adds is the one thing a flat stream of samples does not give:
the **matched positive/negative pairs** equation 16 needs.  SS2.5.2 requires
``o*_+`` and ``o*_-`` to share "the same ``A_hat``, ``tau``, ``eps``".  Draw
them independently and eq. 16 measures sampling noise instead of
observation-sensitivity -- the regulariser silently stops doing its job, and
the failure mode it exists to prevent (a verifier that ignores its visual
input and accepts unconditionally) comes back with no visible symptom.
:func:`sentry.eval.harness.make_pair` builds both halves from one anchor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from sentry.config import SentryConfig
from sentry.core.interfaces import VLABackend
from sentry.core.types import ChannelSpec, Observation
from sentry.envs.mock_env import (
    MockReachConfig,
    _rand_xy,
    raw_plan,
    render_observation,
)
from sentry.eval.harness import NEGATIVES, SampleGenConfig, SampleKind, make_pair

__all__ = ["StageABatch", "StageBBatch", "make_stage_a_batch", "make_stage_b_batch"]


# --------------------------------------------------------------------------
# Stage A
# --------------------------------------------------------------------------


@dataclass
class StageABatch:
    """Demonstration chunks plus their observations.

    Stage A needs nothing exotic: ``A`` is a demonstration chunk,
    ``tau ~ U(0,1)`` and ``eps ~ N(0,I)`` are drawn fresh, and both branches
    share that draw.
    """

    obs: list[Observation]
    A: Tensor
    """``(B, H, d_a)`` demonstration chunks, normalised."""

    def __len__(self) -> int:
        return self.A.shape[0]

    def to(self, device) -> "StageABatch":
        return StageABatch(
            obs=[_obs_to(o, device) for o in self.obs], A=self.A.to(device)
        )


def make_stage_a_batch(
    backend: VLABackend,
    cfg: SentryConfig,
    spec: ChannelSpec,
    batch_size: int,
    env_cfg: MockReachConfig,
    rng: random.Random,
    device: Optional[torch.device | str] = None,
) -> StageABatch:
    """Sample demonstrations from the mock environment.

    ``A`` is the **expert's** chunk, not the model's plan.  Only Stage B is
    required to draw its candidate from the model's own plan distribution --
    "because that is what it will be asked to verify at deployment" (SS2.5.2).
    Stage A has no such requirement: eq. 13 supervises the encoder, and both of
    its branches consume the same ``A^tau``, so what the chunk *is* only decides
    which region of the flow field gets supervised.  Demonstration data is what
    SS2.5.1 trains on, and it is also the better choice here for a reason
    specific to this repository: an untrained ``TinyPi0`` plans essentially
    noise, so training against its own plans would supervise the encoder on a
    region of action space no policy will ever visit.

    The environment renders on CPU while the model may live on an accelerator,
    so each observation is moved to ``device`` *before* the model sees it.
    Moving the batch afterwards is too late -- a forward pass inside generation
    would already have fed CPU images to CUDA weights.
    """
    obs_list, chunks = [], []
    for _ in range(batch_size):
        ee = _rand_xy(rng, env_cfg)
        target = _rand_xy(rng, env_cfg, away_from=ee, min_sep=0.25)
        o = render_observation(ee, target, grip=-1.0, t=0, cfg=env_cfg)
        if device is not None:
            o = _obs_to(o, device)
        obs_list.append(o)
        # Normalised, because every quantity the model consumes is (SS2.9 ii).
        A = spec.normalise(raw_plan(ee, target, env_cfg))
        chunks.append(A.to(o.state.device))
    return StageABatch(obs=obs_list, A=torch.stack(chunks))


# --------------------------------------------------------------------------
# Stage B
# --------------------------------------------------------------------------


@dataclass
class StageBBatch:
    """One Stage-B batch, half positive and half negative.

    ``obs_neg`` holds the matched negative for each element, so equation 16
    can be evaluated against a partner sharing ``A_hat``, ``tau`` and ``eps``.
    """

    A_hat: Tensor
    """``(B, H, d_a)``."""
    H_k: Tensor
    """``(B,)`` int."""
    obs: list[Observation]
    """``o*`` for the eq. 14 / eq. 15 terms.  Any of positive / N1-N4."""
    obs_neg: list[Observation]
    """The eq. 16 partner observation.  Same anchor, same ``k``, same ``A_hat``.

    Named for its role in eq. 16 rather than for its label: the term is the
    symmetric ``|| R^shal(o*_+) - R^shal(o*_-) ||``, and
    :func:`sentry.eval.harness.make_pair` guarantees ``obs`` and ``obs_neg``
    always straddle the live/stale boundary -- so when ``obs`` is itself a
    negative, this holds the *positive*.  Which side is which does not change
    the norm.
    """
    h_star: Tensor
    """``(B,)`` int -- ground-truth first invalid position.

    Resolved by :func:`sentry.eval.harness.build_sample` under the SS2.5.2 rule:
    ``H_k`` for positives, ``h_0`` for N4, a full-depth evaluation of eq. 5 for
    N1-N3.
    """
    is_positive: Tensor
    """``(B,)`` bool, for logging the positive/negative split."""
    kinds: tuple[SampleKind, ...] = ()
    """Which generator produced each element.

    Carried so a preflight check can assert the batch actually contains all four
    negative generators rather than trusting that it does.
    """

    def __len__(self) -> int:
        return self.A_hat.shape[0]

    def to(self, device) -> "StageBBatch":
        return StageBBatch(
            A_hat=self.A_hat.to(device),
            H_k=self.H_k.to(device),
            obs=[_obs_to(o, device) for o in self.obs],
            obs_neg=[_obs_to(o, device) for o in self.obs_neg],
            h_star=self.h_star.to(device),
            is_positive=self.is_positive.to(device),
            kinds=self.kinds,
        )


def _obs_to(o: Observation, device) -> Observation:
    return Observation(
        images=o.images.to(device),
        language=o.language.to(device),
        state=o.state.to(device),
        t=o.t,
    )


def make_stage_b_batch(
    backend: VLABackend,
    cfg: SentryConfig,
    spec: ChannelSpec,
    batch_size: int,
    gen_cfg: SampleGenConfig,
    rng: random.Random,
    device: Optional[torch.device | str] = None,
) -> StageBBatch:
    """Build a batch of matched pairs following SS2.5.2.

    Each element comes from one plan anchor: the eq. 14 / eq. 15 sample and its
    eq. 16 partner share that anchor's ``A_hat`` tensor exactly, and the trainer
    then evaluates both under one draw of ``tau`` and ``eps``.

    "Half of each batch is positive and half negative, with negatives drawn
    uniformly from four generators (N1-N4)" -- both halves are honoured here:
    the positive/negative split alternates deterministically so a small batch
    still carries both, and the negative kind is drawn uniformly from all four.

    ``h_star`` is resolved inside :func:`sentry.eval.harness.build_sample`, in
    one place, by the SS2.5.2 rule -- ``H_k`` for positives, ``h_0`` for N4, a
    full-depth evaluation of eq. 5 for N1-N3.
    """
    def place(o: Observation) -> Observation:
        # Rendered on CPU; the model may not be. Move before any plan() call.
        return _obs_to(o, device) if device is not None else o

    A_hats, H_ks, obs_list, neg_list, h_stars, positives, kinds = (
        [], [], [], [], [], [], []
    )

    guard = 0
    while len(A_hats) < batch_size and guard < 50 * batch_size:
        guard += 1

        # Alternate rather than coin-flip: at the batch sizes preflight uses, a
        # coin flip can produce an all-positive batch, and eq. 15's push-up term
        # would then never fire.
        use_positive = len(A_hats) % 2 == 0
        kind: SampleKind = "positive" if use_positive else rng.choice(NEGATIVES)

        built = make_pair(backend, cfg, spec, gen_cfg, rng, kind, place)
        if built is None:
            continue
        sample, partner = built

        A_hats.append(sample.A_hat)
        H_ks.append(sample.H_k)
        obs_list.append(sample.obs)
        neg_list.append(partner.obs)
        h_stars.append(sample.h_star)
        positives.append(sample.live)
        kinds.append(sample.kind)

    if len(A_hats) < batch_size:
        raise RuntimeError(f"only built {len(A_hats)}/{batch_size} pairs")

    return StageBBatch(
        A_hat=torch.stack(A_hats),
        H_k=torch.tensor(H_ks, dtype=torch.long),
        obs=obs_list,
        obs_neg=neg_list,
        h_star=torch.tensor(h_stars, dtype=torch.long),
        is_positive=torch.tensor(positives, dtype=torch.bool),
        kinds=tuple(kinds),
    )
