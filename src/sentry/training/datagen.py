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


@dataclass
class StageABatch:

    obs: list[Observation]
    A: Tensor

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
    obs_list, chunks = [], []
    for _ in range(batch_size):
        ee = _rand_xy(rng, env_cfg)
        target = _rand_xy(rng, env_cfg, away_from=ee, min_sep=0.25)
        o = render_observation(ee, target, grip=-1.0, t=0, cfg=env_cfg)
        if device is not None:
            o = _obs_to(o, device)
        obs_list.append(o)
        A = spec.normalise(raw_plan(ee, target, env_cfg))
        chunks.append(A.to(o.state.device))
    return StageABatch(obs=obs_list, A=torch.stack(chunks))


@dataclass
class StageBBatch:

    A_hat: Tensor
    H_k: Tensor
    obs: list[Observation]
    obs_neg: list[Observation]
    h_star: Tensor
    is_positive: Tensor
    kinds: tuple[SampleKind, ...] = ()

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
    def place(o: Observation) -> Observation:
        return _obs_to(o, device) if device is not None else o

    A_hats, H_ks, obs_list, neg_list, h_stars, positives, kinds = (
        [], [], [], [], [], [], []
    )

    guard = 0
    while len(A_hats) < batch_size and guard < 50 * batch_size:
        guard += 1

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
