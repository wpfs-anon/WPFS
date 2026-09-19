"""The depth cascade (SS2.6).

"The margin ``mu`` of Sec. 2.4.3 tells us when the shallow verdict is fragile.
Instead of committing to a single depth, or falling back to the full model on a
hand-specified event such as a gripper switch, we escalate depth on low
confidence."

This is the mechanism that "recovers the behaviour that prior work obtains from
a hand-specified gripper heuristic, but as a consequence of the verifier's own
uncertainty rather than as a rule".  It also answers, structurally, the caveat
that conformal calibration is *marginal rather than conditional*: SS2.4.4 notes
the guarantee "does not by itself bound the false-acceptance rate within a rare
but critical phase such as final insertion", and that "Section 2.6 addresses
the latter structurally, by raising depth rather than by tightening delta."
"""

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
    """Verdict of one cascade, plus the accounting equation 18 needs."""

    verdict: CheckResult
    """The verdict actually acted upon -- the last rung evaluated."""

    rungs_used: int
    """Number of rungs evaluated, ``>= 1``.  Averaged into ``J_bar`` (eq. 18)."""

    per_rung: tuple[CheckResult, ...]
    """Every verdict produced, in ladder order.  Kept for the D1 diagnostic."""

    escalated_unresolved: bool
    """True when the ladder was exhausted while the verdict was still fragile.

    That is: the top rung returned ``N > 0`` but ``mu < mu_esc``.  We accept
    ``N`` anyway (see :func:`cascade`), but the caller may want to count how
    often the ladder ran out of depth -- it is the natural place to look if
    per-phase false-acceptance turns out badly (SS2.8, *Anticipated limitation*).
    """

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
    """Escalate depth until the verdict stops being fragile.

    Given an ordered ladder ``E_B^(1) < E_B^(2) < ... < L_B``, "a check that
    returns ``N = 0`` or ``mu < mu_esc`` is repeated at the next depth.  This
    spends compute where the decision is hard -- which, empirically, is where
    the manipulation is hard."

    **Paper defect D1 -- Algorithm 1 lines 10-17 contain dead code.**  Verbatim::

        10:  N <- 0;  j <- 1
        11:  while N = 0 and j <= J do
        12:      (N, mu) <- CHECK(A_hat, o_t, E_V, E_B^(j))
        13:      if N > 0 and mu >= mu_esc then
        14:          break
        15:      end if
        16:      j <- j + 1
        17:  end while

    Line 13's guard, ``N > 0 and mu >= mu_esc``, is exactly the negation of
    SS2.6's stated escalation rule ("a check that returns ``N = 0`` **or**
    ``mu < mu_esc`` is repeated at the next depth").  So the author's intent is
    not in doubt -- it is written down, in the pseudocode, at line 13.

    But **line 13 can never be the operative exit.**  Line 11 already terminates
    the loop the moment ``N > 0``, so on the only path that reaches line 13 with
    a true guard, line 11 would have ended the loop at the next test regardless;
    and on the path that line 13 exists to catch -- ``N > 0`` with
    ``mu < mu_esc`` -- line 13 declines to break, line 16 increments ``j``, and
    line 11 then exits anyway because ``N != 0``.  The margin test is
    unreachable in effect.  The cascade therefore never escalates on a thin
    margin, which is the one thing SS2.6 introduces it to do.

    The fix is one token: line 11 becomes ``while j <= J do``, leaving line 13
    as the sole exit -- which yields precisely the prose semantics.
    ``cfg.cascade_guard`` selects:

    - ``"prose"``   -- escalate while ``N == 0 or mu < mu_esc`` (the intent);
    - ``"literal"`` -- the pseudocode's *effective* behaviour, for ablation.

    Two deliberate choices the paper leaves implicit:

    - **``eps`` is shared across rungs.**  The cascade re-evaluates *the same
      borderline case* at greater depth, so every rung must answer the same
      query; resampling the noise would conflate "deeper" with "different
      question" and make the escalation a lottery rather than a refinement.
    - **``obs`` is fixed across rungs.**  Algorithm 1 passes the same
      observation to ``CHECK`` at each rung; the environment does not advance
      inside a cascade.

    On exhausting the ladder with ``N > 0`` but ``mu < mu_esc``, we **accept
    ``N`` anyway**.  The top rung is the best verdict available, and forcing a
    replan on low margin at full ladder depth would be a different, unstated
    policy.  ``escalated_unresolved`` records that this happened.
    """
    if not cfg.ladder:
        raise ValueError("cascade requires a non-empty ladder")

    # One shared noise draw for the whole cascade -- see the docstring.
    # Drawn on the generator's device and then moved, so a seeded CPU generator
    # works against a model on an accelerator and the stream depends on the seed
    # alone.  :mod:`sentry.training.rng` does the same for the training path;
    # ``core`` keeps its own copy rather than importing from ``training``.
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

    assert verdict is not None  # ladder is non-empty
    return CascadeResult(
        verdict=verdict,
        rungs_used=len(per_rung),
        per_rung=tuple(per_rung),
        escalated_unresolved=verdict.is_fragile(cfg.mu_esc) and verdict.N > 0,
    )


@runtime_checkable
class Verifier(Protocol):
    """What Algorithm 1's speculative phase needs in order to decide.

    Factoring this out of :func:`sentry.core.loop.run_episode` keeps the loop
    honest in two ways: the Proposition 2 baseline can substitute a
    cached-context verifier without the loop growing a special case, and the
    liveness tests can substitute an adversarial always-reject verifier without
    needing a model at all.
    """

    def __call__(self, A_hat: Tensor, H_k: int, obs: Observation) -> CascadeResult: ...


class CascadeVerifier:
    """The default verifier: SENTRY's depth cascade over a fresh observation."""

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
    """Whether to try the next rung, per the selected guard (paper defect D1)."""
    if cfg.cascade_guard == "prose":
        # N == 0  or  mu < mu_esc
        return verdict.is_fragile(cfg.mu_esc)
    if cfg.cascade_guard == "literal":
        # Algorithm 1 line 11 verbatim: escalate only on outright rejection.
        return verdict.N == 0
    raise ValueError(f"unknown cascade guard {cfg.cascade_guard!r}")
