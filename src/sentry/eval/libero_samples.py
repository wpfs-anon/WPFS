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

DEMO_NEGATIVES: tuple[SampleKind, ...] = ("N2", "N3", "N4")


def phase_of(episode: LiberoEpisode, t: int) -> str:
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
        self.n4_offset = n4_offset
        self._tokens: dict[int, Tensor] = {}


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
        H = self.cfg.H
        out: list[Sample] = []

        for i in range(n):
            kind: SampleKind = kinds[i % len(kinds)]
            ep = episodes[rng.randrange(len(episodes))]
            if len(ep) < 4:
                continue

            t = rng.randrange(0, max(1, len(ep) - 1))
            noise = torch.randn(
                (H, self.spec.d_a), generator=torch.Generator().manual_seed(seed + i)
            )
            obs_t = self._obs(ep, t)
            A = self.backend.plan(obs_t, noise=noise)

            k = rng.randrange(0, H)
            A_hat, H_k = repad(
                live_suffix=A[k:H].cpu(),
                H=H,
                scheme=self.cfg.padding,
                spec=self.spec,
                learned_pad=None,
            )

            h0: Optional[int] = None
            if kind == "positive":
                t_star = min(t + k, len(ep) - 1)
                obs_star = self._obs(ep, t_star)
            elif kind == "N2":
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

            fresh = self.backend.plan(obs_star, noise=noise).cpu()

            is_live, first = liveness(
                fresh, A_hat, H_k, eps=self.cfg.liveness_eps, m=self.cfg.liveness_m
            )

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
