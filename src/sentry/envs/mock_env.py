from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from sentry.core.types import ChannelSpec, Observation

__all__ = [
    "ExogenousEvent",
    "MockReachConfig",
    "MockReachEnv",
    "fit_spec",
    "make_planner",
    "render_observation",
    "simulate",
    "raw_plan",
]

_LANGUAGE = (1, 2, 3, 4)


@dataclass(frozen=True)
class ExogenousEvent:

    step: int
    displacement: tuple[float, float] = (0.0, 0.0)
    drift: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class MockReachConfig:
    H: int = 50
    d_a: int = 7
    img_size: int = 32
    d_state: int = 8
    step_size: float = 0.04
    grasp_radius: float = 0.06
    tolerance: float = 0.03
    max_steps: int = 200
    blob_sigma: float = 1.2
    lo: float = 0.05
    hi: float = 0.95
    start: tuple[float, float] = (0.15, 0.15)
    target: tuple[float, float] = (0.80, 0.75)
    event: Optional[ExogenousEvent] = None


def raw_plan(ee: Tensor, target: Tensor, cfg: MockReachConfig) -> Tensor:
    A = torch.zeros(cfg.H, cfg.d_a, dtype=torch.float32)
    p = ee.clone()
    for h in range(cfg.H):
        delta = target - p
        dist = float(torch.linalg.vector_norm(delta))
        move = delta / dist * min(cfg.step_size, dist) if dist > 1e-6 else torch.zeros(2)

        A[h, 0:2] = move
        A[h, 2] = 0.0
        A[h, 3:5] = delta
        A[h, 5] = 0.0
        A[h, 6] = 1.0 if dist <= cfg.grasp_radius else -1.0

        p = torch.clamp(p + move, cfg.lo, cfg.hi)
    return A


def fit_spec(
    cfg: MockReachConfig = MockReachConfig(), n: int = 256, seed: int = 0
) -> ChannelSpec:
    rng = random.Random(seed)
    chunks = []
    for _ in range(n):
        ee = _rand_xy(rng, cfg)
        target = _rand_xy(rng, cfg, away_from=ee, min_sep=0.25)
        chunks.append(raw_plan(ee, target, cfg))
    stacked = torch.cat(chunks, dim=0)

    mean = stacked.mean(dim=0)
    scale = stacked.std(dim=0)
    constant = scale < 1e-6
    mean = torch.where(constant, torch.zeros_like(mean), mean)
    scale = torch.where(constant, torch.ones_like(scale), scale)

    return ChannelSpec(
        d_a=cfg.d_a,
        pos=(0, 1, 2),
        rot=(3, 4, 5),
        grip=6,
        mean=mean,
        scale=scale,
        grip_raw_modes=(-1.0, 1.0),
    )


def make_planner(
    cfg: MockReachConfig, spec: ChannelSpec
) -> Callable[[Observation], Tensor]:

    def planner(obs: Observation) -> Tensor:
        ee = _decode(obs.images[0, 0])
        target = _decode(obs.images[0, 1])
        return spec.normalise(raw_plan(ee, target, cfg))

    return planner


class MockReachEnv:

    def __init__(self, cfg: MockReachConfig, spec: ChannelSpec) -> None:
        self.cfg = cfg
        self.spec = spec
        self.reset()

    def reset(self) -> Observation:
        self._ee = torch.tensor(self.cfg.start, dtype=torch.float32)
        self._target = torch.tensor(self.cfg.target, dtype=torch.float32)
        self._grip = -1.0
        self._t = 0
        self._fired = False
        return self.observe()

    @property
    def t(self) -> int:
        return self._t

    @property
    def terminated(self) -> bool:
        return self._t >= self.cfg.max_steps or self.success

    @property
    def success(self) -> bool:
        return self.distance < self.cfg.tolerance

    @property
    def distance(self) -> float:
        return float(torch.linalg.vector_norm(self._target - self._ee))

    @property
    def target(self) -> Tensor:
        return self._target.clone()

    def step(self, action: Tensor) -> None:
        if action.shape != (self.cfg.d_a,):
            raise ValueError(f"action must be ({self.cfg.d_a},), got {tuple(action.shape)}")
        raw = self.spec.denormalise(action.detach().float())

        self._ee = torch.clamp(self._ee + raw[:2], self.cfg.lo, self.cfg.hi)
        self._grip = 1.0 if float(raw[6]) > 0 else -1.0
        self._t += 1

        ev = self.cfg.event
        if ev is not None:
            if not self._fired and self._t >= ev.step:
                self._target = self._clamp_xy(
                    self._target + torch.tensor(ev.displacement, dtype=torch.float32)
                )
                self._fired = True
            elif self._fired:
                self._target = self._clamp_xy(
                    self._target + torch.tensor(ev.drift, dtype=torch.float32)
                )

    def observe(self) -> Observation:
        return render_observation(self._ee, self._target, self._grip, self._t, self.cfg)

    def _clamp_xy(self, p: Tensor) -> Tensor:
        return torch.clamp(p, self.cfg.lo, self.cfg.hi)


def render_observation(
    ee: Tensor,
    target: Tensor,
    grip: float,
    t: int,
    cfg: MockReachConfig = MockReachConfig(),
) -> Observation:
    S = cfg.img_size
    img = torch.zeros(1, 3, S, S, dtype=torch.float32)
    img[0, 0] = _blob(ee, S, cfg.blob_sigma)
    img[0, 1] = _blob(target, S, cfg.blob_sigma)
    img[0, 2] = grip
    state = torch.zeros(cfg.d_state, dtype=torch.float32)
    state[0:2] = ee
    state[5] = grip
    return Observation(
        images=img,
        language=torch.tensor(_LANGUAGE, dtype=torch.long),
        state=state,
        t=t,
    )


def simulate(
    ee: Tensor, chunk: Tensor, k: int, spec: ChannelSpec, cfg: MockReachConfig
) -> tuple[Tensor, float]:
    p = ee.clone()
    grip = -1.0
    for h in range(min(k, chunk.shape[0])):
        raw = spec.denormalise(chunk[h])
        p = torch.clamp(p + raw[:2], cfg.lo, cfg.hi)
        grip = 1.0 if float(raw[6]) > 0 else -1.0
    return p, grip


def _grid(S: int) -> tuple[Tensor, Tensor]:
    coords = (torch.arange(S, dtype=torch.float32) + 0.5) / S
    return torch.meshgrid(coords, coords, indexing="ij")


def _blob(xy: Tensor, S: int, sigma_px: float) -> Tensor:
    ys, xs = _grid(S)
    sigma = sigma_px / S
    d2 = (xs - xy[0]) ** 2 + (ys - xy[1]) ** 2
    return torch.exp(-d2 / (2 * sigma**2))


def _decode(channel: Tensor) -> Tensor:
    S = channel.shape[-1]
    ys, xs = _grid(S)
    w = channel.clamp_min(0)
    total = w.sum().clamp_min(1e-8)
    return torch.stack([(w * xs).sum() / total, (w * ys).sum() / total])


def _rand_xy(
    rng: random.Random,
    cfg: MockReachConfig,
    away_from: Optional[Tensor] = None,
    min_sep: float = 0.0,
) -> Tensor:
    lo, hi = cfg.lo + 0.05, cfg.hi - 0.05
    for _ in range(100):
        p = torch.tensor([rng.uniform(lo, hi), rng.uniform(lo, hi)], dtype=torch.float32)
        if away_from is None or float(torch.linalg.vector_norm(p - away_from)) >= min_sep:
            return p
    return p
