from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Sequence

from sentry.core.types import DepthRung

__all__ = [
    "PaddingScheme",
    "TauConvention",
    "CascadeGuard",
    "CalibrationMode",
    "SentryConfig",
    "TABLE2",
    "LADDER_TABLE2",
    "LADDER_TABLE1",
]

PaddingScheme = Literal["hold", "zero_delta", "learned"]
TauConvention = Literal["one_is_clean", "zero_is_clean"]
CascadeGuard = Literal["prose", "literal"]
CalibrationMode = Literal["per_group", "joint"]


LADDER_TABLE2: tuple[DepthRung, ...] = (
    DepthRung(E_V=8, E_B=5),
    DepthRung(E_V=8, E_B=9),
    DepthRung(E_V=8, E_B=12),
)

LADDER_TABLE1: tuple[DepthRung, ...] = (
    DepthRung(E_V=8, E_B=5),
    DepthRung(E_V=8, E_B=9),
    DepthRung(E_V=14, E_B=12),
)


@dataclass(frozen=True)
class SentryConfig:

    H: int = 50

    m_min: int = 4

    M: int = 10

    taus: tuple[float, ...] = (0.6, 0.9)

    ladder: tuple[DepthRung, ...] = LADDER_TABLE2

    mu_esc: float = 0.15

    alpha: float = 0.05

    padding: PaddingScheme = "hold"

    tau_convention: TauConvention = "one_is_clean"

    calibration_mode: CalibrationMode = "per_group"

    cascade_guard: CascadeGuard = "prose"

    S_max: Optional[int] = None

    max_accept: Optional[int] = None

    mu_warn: Optional[float] = None

    rung_weights: Optional[tuple[float, ...]] = None

    rung_enable_at: Optional[tuple[float, ...]] = None

    p_depth: tuple[int, int] = (4, 14)

    lambda_m: float = 1.0
    lambda_s: float = 0.1
    margin_m: float = 0.2
    eta: float = 0.05

    lora_rank: int = 16
    lora_rank_readout: int = 32
    lora_dropout: float = 0.05

    stage_a_steps: int = 20_000
    stage_b_steps: int = 60_000

    delta_provisional: float = 1.0

    liveness_eps: float = 0.05
    liveness_m: int = 1

    def __post_init__(self) -> None:
        if self.m_min < 1:
            raise ValueError("m_min must be >= 1; Proposition 5 depends on it")
        if not self.ladder:
            raise ValueError("ladder must contain at least one rung")
        if any(t <= 0.0 or t >= 1.0 for t in self.taus):
            raise ValueError(f"verification timesteps must lie in (0,1), got {self.taus}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must lie in (0,1), got {self.alpha}")
        rungs = [r.E_B for r in self.ladder]
        if rungs != sorted(rungs):
            raise ValueError(
                f"ladder must be ordered E_B^(1) < ... < E_B^(J), got {rungs}"
            )
        if self.mu_warn is not None and self.mu_warn <= self.mu_esc:
            raise ValueError(
                f"mu_warn must exceed mu_esc ({self.mu_warn} <= {self.mu_esc})"
            )
        lo, hi = self.p_depth
        if lo > hi or lo < 1:
            raise ValueError(f"p_depth support must be 1 <= lo <= hi, got {self.p_depth}")
        if self.rung_weights is not None:
            if len(self.rung_weights) != len(self.ladder):
                raise ValueError(
                    f"rung_weights has {len(self.rung_weights)} entries for a ladder "
                    f"of {len(self.ladder)} rungs; eq. 18 needs one w_j per rung"
                )
            if any(w < 0 for w in self.rung_weights):
                raise ValueError(
                    f"rung weights must be non-negative, got {self.rung_weights}"
                )
        if self.rung_enable_at is not None:
            if len(self.rung_enable_at) != len(self.ladder):
                raise ValueError(
                    f"rung_enable_at has {len(self.rung_enable_at)} entries for a "
                    f"ladder of {len(self.ladder)} rungs"
                )
            u = list(self.rung_enable_at)
            if u != sorted(u, reverse=True):
                raise ValueError(
                    f"rung_enable_at must satisfy u_1 >= ... >= u_J, got "
                    f"{self.rung_enable_at}: the curriculum starts with the "
                    "deepest rung alone"
                )
            if u[-1] != 0.0:
                raise ValueError(
                    f"the deepest rung must be enabled from the start (u_J = 0), "
                    f"got {u[-1]}"
                )

    @property
    def w(self) -> tuple[float, ...]:
        if self.rung_weights is not None:
            return self.rung_weights
        J = self.J
        if J == 3:
            return (0.6, 0.3, 0.1)
        raw = [0.5**j for j in range(J)]
        total = sum(raw)
        return tuple(r / total for r in raw)

    @property
    def u_enable(self) -> tuple[float, ...]:
        if self.rung_enable_at is not None:
            return self.rung_enable_at
        J = self.J
        if J == 3:
            return (0.3, 0.1, 0.0)
        if J == 1:
            return (0.0,)
        step = 0.3 / (J - 1)
        return tuple(round(0.3 - j * step, 6) for j in range(J))

    @property
    def K(self) -> int:
        return len(self.taus)

    @property
    def J(self) -> int:
        return len(self.ladder)

    def with_(self, **kw) -> "SentryConfig":
        return replace(self, **kw)


TABLE2 = SentryConfig()
