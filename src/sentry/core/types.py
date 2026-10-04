from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor

__all__ = [
    "DepthRung",
    "Observation",
    "ChannelSpec",
    "CheckResult",
    "Thresholds",
]


@dataclass(frozen=True)
class DepthRung:

    E_V: int
    E_B: int

    def __post_init__(self) -> None:
        if self.E_V < 1:
            raise ValueError(f"E_V must be >= 1, got {self.E_V}")
        if self.E_B < 1:
            raise ValueError(f"E_B must be >= 1, got {self.E_B}")

    def __str__(self) -> str:
        return f"(E_V={self.E_V}, E_B={self.E_B})"

    @property
    def key(self) -> tuple[int, int]:
        return (self.E_V, self.E_B)


def assert_shared_depth(e_perception: int, e_denoising: int) -> None:
    if e_perception != e_denoising:
        raise ValueError(
            "Perception and denoising depth must be identical: attention is "
            "joint at every layer, so skipping layer l skips both the VLM "
            "expert and the action expert.  A single E_B governs both "
            f"(got perception={e_perception}, denoising={e_denoising}).  "
            "See Section 2.9, 'Depth is shared'."
        )


@dataclass(frozen=True)
class Observation:

    images: Tensor

    language: Tensor

    state: Tensor

    t: int

    def with_state(self, state: Tensor, t: int) -> "Observation":
        return replace(self, state=state, t=t)

    def describe(self) -> str:
        return f"Observation(t={self.t}, images={tuple(self.images.shape)})"


@dataclass(frozen=True)
class ChannelSpec:

    d_a: int
    pos: tuple[int, ...]
    rot: tuple[int, ...]
    grip: int
    mean: Tensor
    scale: Tensor
    grip_raw_modes: tuple[float, float] = (-1.0, 1.0)


    def __post_init__(self) -> None:
        idx = list(self.pos) + list(self.rot) + [self.grip]
        if len(set(idx)) != len(idx):
            raise ValueError(f"pos/rot/grip channel indices overlap: {idx}")
        for i in idx:
            if not (0 <= i < self.d_a):
                raise ValueError(f"channel index {i} outside [0, {self.d_a})")
        if self.mean.shape != (self.d_a,):
            raise ValueError(f"mean must be ({self.d_a},), got {tuple(self.mean.shape)}")
        if self.scale.shape != (self.d_a,):
            raise ValueError(f"scale must be ({self.d_a},), got {tuple(self.scale.shape)}")
        if bool((self.scale <= 0).any()):
            raise ValueError("scale must be strictly positive on every channel")
        self._assert_gripper_sign_test_well_posed()

    def _assert_gripper_sign_test_well_posed(self) -> None:
        lo, hi = self.grip_raw_modes
        m = float(self.mean[self.grip])
        s = float(self.scale[self.grip])
        z_lo, z_hi = (lo - m) / s, (hi - m) / s
        if z_lo == 0.0 or z_hi == 0.0 or (z_lo > 0) == (z_hi > 0):
            raise ValueError(
                "Gripper sign test is not well posed: raw modes "
                f"{self.grip_raw_modes} standardise to ({z_lo:.4f}, {z_hi:.4f}), "
                "which are not separated around zero.  See Section 2.9, "
                "convention (iii)."
            )


    def normalise(self, raw: Tensor) -> Tensor:
        return (raw - self.mean) / self.scale

    def denormalise(self, norm: Tensor) -> Tensor:
        return norm * self.scale + self.mean

    def zero_delta_action(self, grip_norm: Tensor) -> Tensor:
        raw = torch.zeros(self.d_a, dtype=self.mean.dtype, device=self.mean.device)
        out = self.normalise(raw).clone().to(grip_norm.device)
        out[self.grip] = grip_norm
        return out


    @property
    def real_channels(self) -> tuple[int, ...]:
        return tuple(sorted({*self.pos, *self.rot, self.grip}))

    @property
    def pos_idx(self) -> Tensor:
        return torch.as_tensor(self.pos, dtype=torch.long, device=self.mean.device)

    @property
    def rot_idx(self) -> Tensor:
        return torch.as_tensor(self.rot, dtype=torch.long, device=self.mean.device)

    def to(self, device: torch.device | str) -> "ChannelSpec":
        return replace(self, mean=self.mean.to(device), scale=self.scale.to(device))

    @staticmethod
    def libero(d_a: int = 32) -> "ChannelSpec":
        return ChannelSpec(
            d_a=d_a,
            pos=(0, 1, 2),
            rot=(3, 4, 5),
            grip=6,
            mean=torch.zeros(d_a),
            scale=torch.ones(d_a),
        )


@dataclass(frozen=True)
class Thresholds:

    pos: float
    rot: float

    def __post_init__(self) -> None:
        if not (self.pos > 0 and self.rot > 0):
            raise ValueError(f"thresholds must be positive, got {self}")

    def __str__(self) -> str:
        return f"Thresholds(pos={self.pos:.6g}, rot={self.rot:.6g})"


@dataclass(frozen=True)
class CheckResult:

    N: int

    margin: Optional[float]

    d: Tensor

    sign_ok: Tensor

    H_k: int

    rung: DepthRung

    @property
    def accepted(self) -> bool:
        return self.N > 0

    @property
    def exhausted(self) -> bool:
        return self.N >= self.H_k

    def is_fragile(self, mu_esc: float) -> bool:
        if self.N == 0:
            return True
        assert self.margin is not None
        return self.margin < mu_esc

    def __str__(self) -> str:
        mu = "None" if self.margin is None else f"{self.margin:+.4f}"
        return f"CheckResult(N={self.N}/{self.H_k}, mu={mu}, rung={self.rung})"
