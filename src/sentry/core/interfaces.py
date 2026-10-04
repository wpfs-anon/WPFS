from __future__ import annotations

import inspect
from typing import Optional, Protocol, runtime_checkable

from torch import Tensor

from sentry.core.types import Observation

__all__ = ["VLABackend", "Environment", "assert_no_cache_seam"]


@runtime_checkable
class VLABackend(Protocol):


    L_V: int

    L_B: int

    H: int

    d_a: int

    M: int


    def plan(self, obs: Observation, noise: Optional[Tensor] = None) -> Tensor:
        ...

    def velocity(
        self,
        A_tau: Tensor,
        tau: Tensor,
        obs: Observation,
        E_V: int,
        E_B: int,
        adapters: bool,
    ) -> Tensor:
        ...


@runtime_checkable
class Environment(Protocol):

    def observe(self) -> Observation:
        ...

    def step(self, action: Tensor) -> None:
        ...

    @property
    def terminated(self) -> bool: ...

    @property
    def t(self) -> int:
        ...


def assert_no_cache_seam(backend: object) -> None:
    sig = inspect.signature(backend.velocity)
    banned = {"cache", "kv", "kv_cache", "past_key_values", "context", "ctx"}
    offending = sorted(banned.intersection(sig.parameters))
    if offending:
        raise TypeError(
            f"{type(backend).__name__}.velocity exposes {offending}: plan mode "
            "and check mode differ in depth, adapter state and input image, so "
            "no cache may be shared between them (Section 2.9)."
        )
