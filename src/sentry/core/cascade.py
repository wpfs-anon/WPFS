from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

from sentry.config import SentryConfig
from sentry.core.check import check
from sentry.core.interfaces import VLABackend
from sentry.core.types import ChannelSpec, CheckResult, Observation, Thresholds

__all__ = ["CascadeResult", "cascade", "Verifier", "CascadeVerifier"]


@dataclass(frozen=True)
class CascadeResult:

    verdict: CheckResult

    rungs_used: int

    per_rung: tuple[CheckResult, ...]

    escalated_unresolved: bool

    @property
    def N(self) -> int:
        return self.verdict.N

    @property
    def margin(self) -> Optional[float]:
        return self.verdict.margin


def cascade(
    backend: VLABackend,
    A_hat: Tensor,
    H_k: int,
    obs: Observation,
    cfg: SentryConfig,
    thresholds: Thresholds,
    spec: ChannelSpec,
    eps: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> CascadeResult:
    if not cfg.ladder:
        raise ValueError("cascade requires a non-empty ladder")

    if eps is None:
        gen_device = generator.device if generator is not None else A_hat.device
        eps = torch.randn(
            A_hat.shape, dtype=A_hat.dtype, device=gen_device, generator=generator
        ).to(A_hat.device)

    per_rung: list[CheckResult] = []
    verdict: Optional[CheckResult] = None

    for rung in cfg.ladder:
        verdict = check(
            backend=backend,
            A_hat=A_hat,
            H_k=H_k,
            obs=obs,
            rung=rung,
            thresholds=thresholds,
            taus=cfg.taus,
            spec=spec,
            convention=cfg.tau_convention,
            eps=eps,
        )
        per_rung.append(verdict)

        if not _should_escalate(verdict, cfg):
            break

    assert verdict is not None
    return CascadeResult(
        verdict=verdict,
        rungs_used=len(per_rung),
        per_rung=tuple(per_rung),
        escalated_unresolved=verdict.is_fragile(cfg.mu_esc) and verdict.N > 0,
    )


@runtime_checkable
class Verifier(Protocol):

    def __call__(self, A_hat: Tensor, H_k: int, obs: Observation) -> CascadeResult: ...


class CascadeVerifier:

    def __init__(
        self,
        backend: VLABackend,
        cfg: SentryConfig,
        thresholds: Thresholds,
        spec: ChannelSpec,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.backend = backend
        self.cfg = cfg
        self.thresholds = thresholds
        self.spec = spec
        self.generator = generator

    def __call__(self, A_hat: Tensor, H_k: int, obs: Observation) -> CascadeResult:
        return cascade(
            backend=self.backend,
            A_hat=A_hat,
            H_k=H_k,
            obs=obs,
            cfg=self.cfg,
            thresholds=self.thresholds,
            spec=self.spec,
            generator=self.generator,
        )


def _should_escalate(verdict: CheckResult, cfg: SentryConfig) -> bool:
    if cfg.cascade_guard == "prose":
        return verdict.is_fragile(cfg.mu_esc)
    if cfg.cascade_guard == "literal":
        return verdict.N == 0
    raise ValueError(f"unknown cascade guard {cfg.cascade_guard!r}")
