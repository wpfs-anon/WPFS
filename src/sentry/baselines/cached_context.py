from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from sentry.config import SentryConfig
from sentry.core.cascade import CascadeResult, cascade
from sentry.core.interfaces import VLABackend
from sentry.core.types import ChannelSpec, Observation, Thresholds

__all__ = ["CachedContextVerifier"]


class CachedContextVerifier:

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
        self._cached: Optional[Observation] = None

    def refresh_context(self, obs: Observation) -> None:
        self._cached = obs

    def __call__(self, A_hat: Tensor, H_k: int, obs: Observation) -> CascadeResult:
        if self._cached is None:
            self.refresh_context(obs)

        assert self._cached is not None
        stale = self._cached.with_state(obs.state, obs.t)

        return cascade(
            backend=self.backend,
            A_hat=A_hat,
            H_k=H_k,
            obs=stale,
            cfg=self.cfg,
            thresholds=self.thresholds,
            spec=self.spec,
            generator=self.generator,
        )
