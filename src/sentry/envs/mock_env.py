"""A scripted reach environment with injectable exogenous events.

Built to satisfy the hypothesis of **Proposition 2** exactly, so that the
proposition can be *demonstrated* rather than merely cited:

- The target's location appears **only in the image**.  Proprioception carries
  the end-effector pose and the gripper, and nothing else.
- An exogenous event ``Z`` displaces the target at a chosen step **without
  altering the robot's own dynamics over that interval** -- the robot keeps
  executing whatever it had committed to, at the same speed, along the same
  path.

Under those two conditions ``s_{t+k}`` is genuinely conditionally independent
of ``Z`` given ``(s_t, a_{t:t+k-1})``, which is what makes
``I(V(A_hat, c_t, s_{t+k}); Z) = 0`` a fact about this environment and not just
an inequality on a page.  A verifier that reads the image reacts; one that
reuses a cached context cannot, "before ``Z`` has changed the robot's dynamics,
i.e. before contact has already occurred."

Action layout (``d_a = 7``): ``pos = (0,1,2)`` translation delta,
``rot = (3,4,5)`` remaining displacement, ``grip = 6`` in ``+-1``.

**Everything crossing the environment boundary is in normalised action space.**
SS2.9 convention (ii) requires all distances in eq. 10 to be computed using the
policy's own statistics "so that per-channel thresholds are commensurable" --
and this environment shows why it is not optional: a per-step translation
delta has a natural scale of ~0.04 while the remaining-displacement channels
range over ~0.5, so an unnormalised ``delta_pos`` would be swamped by
truncation noise while ``delta_rot`` sailed through.  Use :func:`fit_spec` to
obtain the statistics.
"""

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
"""A fixed instruction ("reach the target"); the mock does not vary tasks."""


@dataclass(frozen=True)
class ExogenousEvent:
    """An event ``Z`` occurring in ``(t, t+k]``.

    "an object displaced by an external agent, a distractor introduced, a
    target that begins to move -- which does not alter the robot's own dynamics
    over that interval."
    """

    step: int
    """Environment timestep at which the event fires."""
    displacement: tuple[float, float] = (0.0, 0.0)
    """How far the external agent moves the target."""
    drift: tuple[float, float] = (0.0, 0.0)
    """Per-step motion after the event -- "a target that begins to move"."""


@dataclass(frozen=True)
class MockReachConfig:
    H: int = 50
    d_a: int = 7
    img_size: int = 32
    """32 keeps the soft-centroid decode near-exact (mean error ~2e-5), so that
    positives stay live and any measured staleness is real rather than an
    artefact of the renderer."""
    d_state: int = 8
    step_size: float = 0.04
    """Distance covered per action."""
    grasp_radius: float = 0.06
    """Inside this radius the expert closes the gripper."""
    tolerance: float = 0.03
    max_steps: int = 200
    blob_sigma: float = 1.2
    """Gaussian blob width, in pixels."""
    lo: float = 0.05
    hi: float = 0.95
    """Workspace bounds."""
    start: tuple[float, float] = (0.15, 0.15)
    target: tuple[float, float] = (0.80, 0.75)
    event: Optional[ExogenousEvent] = None


# --------------------------------------------------------------------------
# The expert, in raw action space
# --------------------------------------------------------------------------


def raw_plan(ee: Tensor, target: Tensor, cfg: MockReachConfig) -> Tensor:
    """The expert's chunk in **raw** action units.  ``(H, d_a)``."""
    A = torch.zeros(cfg.H, cfg.d_a, dtype=torch.float32)
    p = ee.clone()
    for h in range(cfg.H):
        delta = target - p
        dist = float(torch.linalg.vector_norm(delta))
        move = delta / dist * min(cfg.step_size, dist) if dist > 1e-6 else torch.zeros(2)

        A[h, 0:2] = move
        A[h, 2] = 0.0
        # The rotation group carries the *remaining displacement*: a smooth
        # function of the scene.  A heading angle would be singular as the
        # end-effector reaches the target -- an artefact of the mock that
        # would surface as spurious staleness.
        A[h, 3:5] = delta
        A[h, 5] = 0.0
        A[h, 6] = 1.0 if dist <= cfg.grasp_radius else -1.0

        # Clamp exactly as MockReachEnv.step does.  If the planner's own
        # rollout diverged from the environment's, a "positive" sample -- a
        # scene that evolved only through the robot's execution of the plan --
        # would drift into staleness for no physical reason.
        p = torch.clamp(p + move, cfg.lo, cfg.hi)
    return A


def fit_spec(
    cfg: MockReachConfig = MockReachConfig(), n: int = 256, seed: int = 0
) -> ChannelSpec:
    """Fit the policy's action statistics, per SS2.9 convention (ii).

    Channels that are identically constant (the unused z and roll slots) get a
    unit scale rather than a degenerate one; they then normalise to zero and
    contribute only whatever truncation noise the backend injects, which is
    the honest behaviour for a padded channel.
    """
    rng = random.Random(seed)
    chunks = []
    for _ in range(n):
        ee = _rand_xy(rng, cfg)
        target = _rand_xy(rng, cfg, away_from=ee, min_sep=0.25)
        chunks.append(raw_plan(ee, target, cfg))
    stacked = torch.cat(chunks, dim=0)                      # (n*H, d_a)

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
    """Build a planner that is a **pure function of the observation**.

    This purity is load-bearing.  :class:`OracleBackend` calls the planner both
    to plan and to define its flow field, and the cached-context baseline works
    by handing it a stale observation.  If the planner could consult the
    environment directly, feeding it a stale observation would change nothing
    and the Proposition 2 demonstration would be vacuous.

    Returns chunks in **normalised** action space.
    """

    def planner(obs: Observation) -> Tensor:
        ee = _decode(obs.images[0, 0])
        target = _decode(obs.images[0, 1])
        return spec.normalise(raw_plan(ee, target, cfg))

    return planner


# --------------------------------------------------------------------------
# The environment
# --------------------------------------------------------------------------


class MockReachEnv:
    """A 2-D reach with a deterministic expert.  Satisfies :class:`Environment`."""

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
        """Execute one **normalised** action.

        Note that ``Z`` fires independently of what the robot does -- that
        independence is the hypothesis of Proposition 2.
        """
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
        """``o_t = [I_t, l_t, s_t]``.

        The target appears **only** in ``I_t``.  ``s_t`` is the end-effector
        pose and gripper -- exactly the information a cached-context verifier
        would refresh, and exactly the information Proposition 2 shows to be
        useless against ``Z``.
        """
        return render_observation(self._ee, self._target, self._grip, self._t, self.cfg)

    def _clamp_xy(self, p: Tensor) -> Tensor:
        return torch.clamp(p, self.cfg.lo, self.cfg.hi)


# --------------------------------------------------------------------------
# Rendering, decoding, simulation
# --------------------------------------------------------------------------


def render_observation(
    ee: Tensor,
    target: Tensor,
    grip: float,
    t: int,
    cfg: MockReachConfig = MockReachConfig(),
) -> Observation:
    """Build an observation for an arbitrary scene configuration.

    Used by :mod:`sentry.eval.harness` to synthesise the Stage-B negatives of
    SS2.5.2 without stepping an environment: *(N1)* re-renders a scene with a
    task-relevant object displaced, *(N2)* renders a different time index of
    the same episode, *(N3)* renders a different object configuration.
    """
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
    """Roll the end-effector forward through ``chunk[:k]`` (normalised actions).

    Mirrors :meth:`MockReachEnv.step` exactly, so a "positive" sample really
    does differ from its anchor only through the robot's own execution of the
    plan (SS2.5.2).
    """
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
    """Recover a position from a blob by soft centroid.

    Any bias here is shared by planning and verification -- both go through the
    same decode -- so it cancels.  What matters is only that the decode
    *responds* when the blob moves.
    """
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
