from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from sentry.core.types import Observation, assert_shared_depth

__all__ = ["OracleConfig", "OracleBackend", "exponential_truncation"]

Planner = Callable[[Observation], Tensor]


def exponential_truncation(sigma_0: float = 0.5, decay: float = 4.0) -> Callable[[float, float], float]:

    def sigma(f_V: float, f_B: float) -> float:
        return sigma_0 * math.exp(-decay * min(f_V, f_B))

    return sigma


@dataclass
class OracleConfig:
    L_V: int = 27
    L_B: int = 18
    H: int = 50
    d_a: int = 7
    M: int = 10
    truncation_sigma: Callable[[float, float], float] = field(
        default_factory=exponential_truncation
    )
    horizon_growth: float = 3.0
    seed: Optional[int] = 0


class OracleBackend:

    def __init__(self, planner: Planner, cfg: OracleConfig = OracleConfig()) -> None:
        self.planner = planner
        self.cfg = cfg
        self.L_V, self.L_B = cfg.L_V, cfg.L_B
        self.H, self.d_a, self.M = cfg.H, cfg.d_a, cfg.M
        self._gen = torch.Generator()
        if cfg.seed is not None:
            self._gen.manual_seed(cfg.seed)

        self.calls: int = 0


    def plan(self, obs: Observation, noise: Optional[Tensor] = None) -> Tensor:
        A = self.planner(obs)
        if A.shape != (self.H, self.d_a):
            raise ValueError(
                f"planner returned {tuple(A.shape)}, expected {(self.H, self.d_a)}"
            )
        return A


    def velocity(
        self,
        A_tau: Tensor,
        tau: Tensor,
        obs: Observation,
        E_V: int,
        E_B: int,
        adapters: bool,
    ) -> Tensor:
        if not (1 <= E_V <= self.L_V):
            raise ValueError(f"E_V={E_V} outside [1, {self.L_V}]")
        if not (1 <= E_B <= self.L_B):
            raise ValueError(f"E_B={E_B} outside [1, {self.L_B}]")
        assert_shared_depth(E_B, E_B)

        self.calls += 1

        A_star = self.planner(obs)
        K = A_tau.shape[0]
        s = tau.view(K, 1, 1).to(A_tau.dtype)

        target = A_star.unsqueeze(0).expand(K, -1, -1)

        sigma = self.cfg.truncation_sigma(E_V / self.L_V, E_B / self.L_B)
        if sigma > 0.0:
            xi = torch.randn(
                A_tau.shape, dtype=A_tau.dtype, device=A_tau.device, generator=self._gen
            )
            H = A_tau.shape[1]
            h = torch.arange(H, dtype=A_tau.dtype, device=A_tau.device)
            growth = (1.0 + self.cfg.horizon_growth * h / max(H - 1, 1)).view(1, H, 1)
            target = target + sigma * growth * xi

        denom = (1.0 - s).clamp_min(1e-6)
        return (target - A_tau) / denom
