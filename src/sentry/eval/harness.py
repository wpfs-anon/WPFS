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


@dataclass(frozen=True)
class Sample:

    A_hat: Tensor
    H_k: int
    obs: Observation
    fresh_plan: Optional[Tensor]
    kind: SampleKind
    live: bool
    h0: Optional[int]
    h_star: int
    noise: Tensor
    phase: str


@dataclass(frozen=True)
class SampleGenConfig:
    env: MockReachConfig = MockReachConfig()
    delta_min: float = 0.15
    delta_max: float = 0.45
    j_min: int = 5
    corruption_offset: float = 0.25
    p_sign_flip: float = 0.5
    min_H_k: int = 4


def phase_of(ee: Tensor, target: Tensor, cfg: MockReachConfig) -> str:
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


@dataclass(frozen=True)
class Anchor:

    A: Tensor
    A_hat: Tensor
    H_k: int
    k: int
    noise: Tensor
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
    ec = gc.env
    H = cfg.H

    ee0 = _rand_xy(rng, ec)
    target = _rand_xy(rng, ec, away_from=ee0, min_sep=0.25)
    obs_t = place(render_observation(ee0, target, grip=-1.0, t=0, cfg=ec))

    noise = torch.randn(H, backend.d_a)
    A = backend.plan(obs_t, noise)

    k = rng.randrange(0, H)
    if H - k < gc.min_H_k:
        return None
    A_hat, H_k = repad(A[k:H], H, cfg.padding, spec)

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
        mag = rng.uniform(gc.delta_min, gc.delta_max)
        ang = rng.uniform(0, 2 * math.pi)
        moved = torch.clamp(
            target + torch.tensor([mag * math.cos(ang), mag * math.sin(ang)]),
            ec.lo,
            ec.hi,
        )
        if float(torch.linalg.vector_norm(moved - target)) < gc.delta_min * 0.5:
            return None
        obs = render_observation(ee_k, moved, grip_k, t=k, cfg=ec)

    elif kind == "N2":
        j = rng.choice([-1, 1]) * rng.randrange(gc.j_min + 1, gc.j_min + 20)
        k2 = k + j
        if not (0 <= k2 < H):
            return None
        ee2, grip2 = simulate(anchor.ee0, anchor.A.detach().cpu(), k2, spec, ec)
        if float(torch.linalg.vector_norm(ee2 - ee_k)) < 1e-3:
            return None
        obs = render_observation(ee2, target, grip2, t=k, cfg=ec)

    elif kind == "N3":
        other = _rand_xy(rng, ec, away_from=target, min_sep=0.3)
        obs = render_observation(ee_k, other, grip_k, t=k, cfg=ec)

    elif kind == "N4":
        obs = render_observation(ee_k, target, grip_k, t=k, cfg=ec)
        h0 = rng.randrange(1, H_k) if H_k > 1 else 0
        A_hat = _corrupt(A_hat, h0, H_k, spec, gc, rng)

    else:
        raise ValueError(f"unknown sample kind {kind!r}")

    obs = place(obs)

    needs_liveness = kind in OBSERVATION_NEGATIVES

    fresh_plan = (
        backend.plan(obs, anchor.noise)
        if (with_fresh_plan or needs_liveness)
        else None
    )

    if kind == "positive":
        h_star = H_k
    elif kind == "N4":
        h_star = int(h0)
    else:
        assert fresh_plan is not None
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
    out = A_hat.clone()
    if rng.random() < gc.p_sign_flip:
        out[h0:H_k] = -out[h0:H_k]
    else:
        off = gc.corruption_offset
        ang = rng.uniform(0, 2 * math.pi)
        out[h0:H_k, spec.pos[0]] += off * math.cos(ang)
        out[h0:H_k, spec.pos[1]] += off * math.sin(ang)
    return out


