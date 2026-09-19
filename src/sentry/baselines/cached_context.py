"""The cached-context verifier of Proposition 2 -- deliberately blind.

SS2.2 describes what existing speculative dVLA pipelines do:

    "Existing speculative dVLA pipelines verify a candidate chunk by reusing
    the visual KV cache ``c_t`` computed at the last full round, feeding the
    candidate to the action expert under that cached context, and comparing the
    reconstructed endpoint to the candidate.  **The only quantity in that test
    that is refreshed is the proprioceptive state ``s_{t+k}``.**  This is not a
    minor economy; it removes exactly the information the test exists to find."

    **Proposition 2.** ... ``I(V(A_hat, c_t, s_{t+k}) ; Z) = 0``.  "No verifier
    of this form can react to ``Z`` before ``Z`` has changed the robot's
    dynamics, i.e. before contact has already occurred."

This module implements exactly that verifier, so the proposition can be
*measured* on the mock environment rather than only proved.  It is otherwise
identical to SENTRY -- same operator, same thresholds, same ladder -- so the
comparison isolates the one variable that matters: whether the conditioning is
fresh.

We reuse the *cached observation* rather than a literal KV cache.  The
information content is identical (the cache is a deterministic function of
``o_t``), and it keeps the comparison exact: the two verifiers differ in their
conditioning and in nothing else.
"""

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
    """``V(A_hat, c_t, s_{t+k})`` -- refreshes proprioception and nothing else.

    Usage mirrors :class:`sentry.core.cascade.CascadeVerifier`, but the caller
    must call :meth:`refresh_context` whenever a full-depth replan happens,
    since that is when a real pipeline would recompute its visual cache.
    """

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
        """Recompute ``c_t``.  A real pipeline does this at each full round."""
        self._cached = obs

    def __call__(self, A_hat: Tensor, H_k: int, obs: Observation) -> CascadeResult:
        """Verify under the cached context, refreshing only ``s_{t+k}``.

        ``obs`` arrives fresh from the loop; we throw away everything in it
        except proprioception.  That discarding *is* the baseline.
        """
        if self._cached is None:
            # No cache yet: the first check after a replan has nothing stale to
            # reuse, so this round is unavoidably fresh.
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
