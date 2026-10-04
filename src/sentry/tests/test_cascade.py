from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from sentry.config import SentryConfig
from sentry.core.cascade import cascade
from sentry.core.types import ChannelSpec, DepthRung, Observation, Thresholds


@dataclass
class ScriptedBackend:

    error_by_EB: dict[int, float]
    L_V: int = 27
    L_B: int = 18
    H: int = 10
    d_a: int = 7
    M: int = 10

    def __post_init__(self):
        self.rungs_seen: list[tuple[int, int]] = []

    def plan(self, obs):
        return torch.zeros(self.H, self.d_a)

    def velocity(self, A_tau, tau, obs, E_V, E_B, adapters):
        self.rungs_seen.append((E_V, E_B))
        K = A_tau.shape[0]
        s = tau.view(K, 1, 1)
        target = torch.zeros(K, self.H, self.d_a)
        target[:, :, 0] = self.error_by_EB[E_B]
        target[:, :, 6] = 1.0
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
    A[:, 6] = 1.0
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


def test_prose_guard_escalates_on_a_thin_margin():
    result, backend = run({5: 0.9, 9: 0.2, 12: 0.05}, guard="prose")
    assert result.rungs_used == 2
    assert backend.rungs_seen == [(8, 5), (8, 9)]
    assert result.N == 10
    assert result.margin == pytest.approx(0.8, abs=1e-4)


def test_literal_guard_does_not_escalate_on_a_thin_margin():
    result, backend = run({5: 0.9, 9: 0.2, 12: 0.05}, guard="literal")
    assert result.rungs_used == 1
    assert backend.rungs_seen == [(8, 5)]
    assert result.margin == pytest.approx(0.1, abs=1e-4)
    assert result.escalated_unresolved


def test_both_guards_escalate_on_outright_rejection():
    for guard in ("prose", "literal"):
        result, backend = run({5: 5.0, 9: 5.0, 12: 0.1}, guard=guard)
        assert result.rungs_used == 3, guard
        assert result.N == 10, guard


def test_no_escalation_when_the_first_rung_is_comfortable():
    result, backend = run({5: 0.1, 9: 0.1, 12: 0.1})
    assert result.rungs_used == 1
    assert result.margin == pytest.approx(0.9, abs=1e-4)


def test_rejection_at_every_depth_yields_N_zero():
    result, _ = run({5: 5.0, 9: 5.0, 12: 5.0})
    assert result.N == 0
    assert result.margin is None
    assert result.rungs_used == 3
    assert not result.escalated_unresolved


def test_exhausting_the_ladder_while_fragile_still_accepts():
    result, _ = run({5: 0.9, 9: 0.92, 12: 0.95})
    assert result.rungs_used == 3
    assert result.N == 10
    assert result.escalated_unresolved


def test_every_rung_answers_the_same_query():
    result, _ = run({5: 5.0, 9: 5.0, 12: 0.1})
    assert [r.rung.E_B for r in result.per_rung] == [5, 9, 12]


def test_per_rung_verdicts_are_retained_for_diagnostics():
    result, _ = run({5: 5.0, 9: 0.5, 12: 0.1})
    assert len(result.per_rung) == result.rungs_used == 2
    assert result.verdict is result.per_rung[-1]


def test_unknown_guard_is_rejected():
    with pytest.raises(ValueError, match="unknown cascade guard"):
        run({5: 0.1, 9: 0.1, 12: 0.1}, guard="sideways")
