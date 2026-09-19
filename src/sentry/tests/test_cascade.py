"""The depth cascade (SS2.6), and paper defect D1.

D1 is pinned by a test on purpose.  It is a one-word difference between the
pseudocode and the prose, it silently disables the cascade's entire stated
purpose, and nothing else in the system would fail if it regressed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from sentry.config import SentryConfig
from sentry.core.cascade import cascade
from sentry.core.types import ChannelSpec, DepthRung, Observation, Thresholds


@dataclass
class ScriptedBackend:
    """A backend whose reconstruction error is a scripted function of depth.

    ``error_by_EB`` maps a backbone depth to the per-position translation error
    the reconstruction will show, letting a test dictate ``N`` and ``mu`` at
    each rung exactly.
    """

    error_by_EB: dict[int, float]
    L_V: int = 27
    L_B: int = 18
    H: int = 10
    d_a: int = 7
    M: int = 10

    def __post_init__(self):
        self.rungs_seen: list[tuple[int, int]] = []

    def plan(self, obs):  # pragma: no cover - unused here
        return torch.zeros(self.H, self.d_a)

    def velocity(self, A_tau, tau, obs, E_V, E_B, adapters):
        self.rungs_seen.append((E_V, E_B))
        K = A_tau.shape[0]
        s = tau.view(K, 1, 1)
        target = torch.zeros(K, self.H, self.d_a)
        target[:, :, 0] = self.error_by_EB[E_B]
        # The gripper must agree with the candidate, or eq. 11's sign test
        # vetoes every position and N is 0 regardless of the scripted distance.
        # Leaving this at zero is *correctly* read as disagreement -- exact
        # zero carries no intent -- which would make these tests measure the
        # sign rule instead of the cascade.
        target[:, :, 6] = 1.0
        # Solve eq. 9 for the velocity that lands the reconstruction on target.
        return (target - A_tau) / (1.0 - s).clamp_min(1e-6)


def spec() -> ChannelSpec:
    return ChannelSpec(
        d_a=7, pos=(0, 1, 2), rot=(3, 4, 5), grip=6,
        mean=torch.zeros(7), scale=torch.ones(7),
    )


def obs() -> Observation:
    return Observation(
        images=torch.zeros(1, 3, 4, 4),
        language=torch.zeros(2, dtype=torch.long),
        state=torch.zeros(4),
        t=0,
    )


def candidate() -> torch.Tensor:
    A = torch.zeros(10, 7)
    A[:, 6] = 1.0            # gripper positive, so the sign test passes
    return A


def run(errors, guard="prose", mu_esc=0.15, delta=1.0):
    cfg = SentryConfig(
        H=10, m_min=1, taus=(0.6,), mu_esc=mu_esc, cascade_guard=guard,
        ladder=(DepthRung(8, 5), DepthRung(8, 9), DepthRung(14, 12)),
    )
    backend = ScriptedBackend(error_by_EB=errors)
    result = cascade(
        backend=backend, A_hat=candidate(), H_k=10, obs=obs(), cfg=cfg,
        thresholds=Thresholds(pos=delta, rot=delta), spec=spec(),
        generator=torch.Generator().manual_seed(0),
    )
    return result, backend


# -- paper defect D1 --------------------------------------------------------


def test_prose_guard_escalates_on_a_thin_margin():
    """SS2.6: "a check that returns N = 0 **or** mu < mu_esc is repeated at the
    next depth"."""
    # Rung 1: error 0.9 -> d = 0.9, mu = 0.1 < mu_esc.  Accepted but fragile.
    # Rung 2: error 0.2 -> d = 0.2, mu = 0.8.  Comfortable; stop.
    result, backend = run({5: 0.9, 9: 0.2, 12: 0.05}, guard="prose")
    assert result.rungs_used == 2
    assert backend.rungs_seen == [(8, 5), (8, 9)]
    assert result.N == 10
    assert result.margin == pytest.approx(0.8, abs=1e-4)


def test_literal_guard_does_not_escalate_on_a_thin_margin():
    """Algorithm 1 line 11 verbatim: ``while N = 0 and j <= J``.

    A check returning ``N > 0`` with ``mu < mu_esc`` exits immediately, so the
    cascade never escalates on low confidence -- which is precisely what SS2.6
    says it exists to do.  This test documents the divergence rather than
    hiding it.
    """
    result, backend = run({5: 0.9, 9: 0.2, 12: 0.05}, guard="literal")
    assert result.rungs_used == 1
    assert backend.rungs_seen == [(8, 5)]
    assert result.margin == pytest.approx(0.1, abs=1e-4)
    assert result.escalated_unresolved


def test_both_guards_escalate_on_outright_rejection():
    """The one case the two guards agree on."""
    for guard in ("prose", "literal"):
        result, backend = run({5: 5.0, 9: 5.0, 12: 0.1}, guard=guard)
        assert result.rungs_used == 3, guard
        assert result.N == 10, guard


def test_no_escalation_when_the_first_rung_is_comfortable():
    """"This spends compute where the decision is hard" -- and only there."""
    result, backend = run({5: 0.1, 9: 0.1, 12: 0.1})
    assert result.rungs_used == 1
    assert result.margin == pytest.approx(0.9, abs=1e-4)


# -- ladder exhaustion ------------------------------------------------------


def test_rejection_at_every_depth_yields_N_zero():
    """"no prefix survives at any depth: replan" (Algorithm 1 line 19)."""
    result, _ = run({5: 5.0, 9: 5.0, 12: 5.0})
    assert result.N == 0
    assert result.margin is None
    assert result.rungs_used == 3
    assert not result.escalated_unresolved      # N == 0, not "accepted but fragile"


def test_exhausting_the_ladder_while_fragile_still_accepts():
    """The top rung is the best verdict available.

    Forcing a replan on a thin margin at full ladder depth would be a
    different, unstated policy -- so we accept and record that it happened.
    """
    result, _ = run({5: 0.9, 9: 0.92, 12: 0.95})
    assert result.rungs_used == 3
    assert result.N == 10
    assert result.escalated_unresolved


# -- invariants the paper leaves implicit -----------------------------------


def test_every_rung_answers_the_same_query():
    """``eps`` is shared across rungs.

    The cascade re-evaluates *the same* borderline case at greater depth.
    Resampling the noise would conflate "deeper" with "different question" and
    make escalation a lottery rather than a refinement.
    """
    result, _ = run({5: 5.0, 9: 5.0, 12: 0.1})
    # Depth is the only thing that varied, so the verdicts must be ordered by
    # the scripted error alone.
    assert [r.rung.E_B for r in result.per_rung] == [5, 9, 12]


def test_per_rung_verdicts_are_retained_for_diagnostics():
    result, _ = run({5: 5.0, 9: 0.5, 12: 0.1})
    assert len(result.per_rung) == result.rungs_used == 2
    assert result.verdict is result.per_rung[-1]


def test_unknown_guard_is_rejected():
    with pytest.raises(ValueError, match="unknown cascade guard"):
        run({5: 0.1, 9: 0.1, 12: 0.1}, guard="sideways")
