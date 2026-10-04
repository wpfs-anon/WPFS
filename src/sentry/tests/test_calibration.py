from __future__ import annotations

import pytest
import torch

from sentry.core.calibration import CalibrationRecord, calibrate, liveness
from sentry.core.types import ChannelSpec


def spec() -> ChannelSpec:
    return ChannelSpec(
        d_a=7, pos=(0, 1, 2), rot=(3, 4, 5), grip=6,
        mean=torch.zeros(7), scale=torch.ones(7),
    )


def test_live_when_the_fresh_plan_agrees():
    A_hat = torch.zeros(10, 7)
    fresh = torch.zeros(10, 7)
    is_live, first = liveness(fresh, A_hat, H_k=10, eps=0.05, m=1)
    assert is_live and first is None


def test_stale_when_the_fresh_plan_diverges_within_the_horizon():
    A_hat = torch.zeros(10, 7)
    fresh = torch.zeros(10, 7)
    fresh[0, 0] = 1.0
    is_live, first = liveness(fresh, A_hat, H_k=10, eps=0.05, m=1)
    assert not is_live and first == 0


def test_horizon_m_bounds_the_binary_label_but_not_the_violation_index():
    A_hat = torch.zeros(10, 7)
    fresh = torch.zeros(10, 7)
    fresh[5, 0] = 1.0

    live_at_1, first = liveness(fresh, A_hat, H_k=10, eps=0.05, m=1)
    assert live_at_1
    assert first == 5

    live_at_8, _ = liveness(fresh, A_hat, H_k=10, eps=0.05, m=8)
    assert not live_at_8


def test_liveness_is_measured_against_the_targets_own_plan():
    A_hat = torch.full((4, 7), 3.0)
    fresh = torch.full((4, 7), 3.0)
    assert liveness(fresh, A_hat, H_k=4, eps=0.05, m=1)[0]


def test_liveness_validation():
    with pytest.raises(ValueError, match="horizon m must be"):
        liveness(torch.zeros(4, 7), torch.zeros(4, 7), 4, eps=0.1, m=0)


def records(stale_pos, stale_rot, live_pos=(), live_rot=(), phase=None):
    out = [
        CalibrationRecord(d_pos=p, d_rot=r, live=False, h_star=0, phase=phase)
        for p, r in zip(stale_pos, stale_rot)
    ]
    out += [
        CalibrationRecord(d_pos=p, d_rot=r, live=True, h_star=None, phase=phase)
        for p, r in zip(live_pos, live_rot)
    ]
    return out


def test_threshold_is_the_lower_tail_quantile_of_the_stale_population():
    d = [float(x) for x in range(1, 101)]
    report = calibrate(records(d, d), alpha=0.1)
    assert report.thresholds.pos == pytest.approx(10.9, abs=0.5)


def test_false_acceptance_tracks_alpha():
    torch.manual_seed(0)
    d = (torch.randn(4000).abs() + 1.0).tolist()
    for alpha in (0.01, 0.05, 0.1, 0.2):
        report = calibrate(records(d, d), alpha=alpha)
        assert report.empirical_false_acceptance == pytest.approx(alpha, abs=0.02)


def test_per_group_gives_independent_thresholds():
    report = calibrate(
        records(stale_pos=[1.0] * 100, stale_rot=[10.0] * 100),
        alpha=0.05, mode="per_group",
    )
    assert report.thresholds.pos == pytest.approx(1.0)
    assert report.thresholds.rot == pytest.approx(10.0)


def test_joint_mode_is_paper_literal():
    report = calibrate(
        records(stale_pos=[1.0] * 100, stale_rot=[10.0] * 100),
        alpha=0.05, mode="joint",
    )
    assert report.thresholds.pos == report.thresholds.rot == pytest.approx(10.0)


def test_live_rejection_is_measured_but_not_targeted():
    report = calibrate(
        records(stale_pos=[5.0] * 100, stale_rot=[5.0] * 100,
                live_pos=[0.1] * 50, live_rot=[0.1] * 50),
        alpha=0.05,
    )
    assert report.live_rejection_rate == pytest.approx(0.0)
    assert report.n_live == 50


def test_sweeping_alpha_trades_false_acceptance_against_efficiency():
    torch.manual_seed(0)
    stale = (torch.randn(2000).abs() + 1.5).tolist()
    live = (torch.randn(2000).abs() * 0.5).tolist()
    rows = records(stale, stale, live, live)

    prev_far, prev_rej = -1.0, 2.0
    for alpha in (0.01, 0.05, 0.1, 0.25):
        r = calibrate(rows, alpha=alpha)
        assert r.empirical_false_acceptance >= prev_far
        assert r.live_rejection_rate <= prev_rej
        prev_far, prev_rej = r.empirical_false_acceptance, r.live_rejection_rate


def test_per_phase_breakdown_exposes_the_conditional_gap():
    rows = (
        records([5.0] * 190, [5.0] * 190, phase="transit")
        + records([0.01] * 10, [0.01] * 10, phase="fine_alignment")
    )
    report = calibrate(rows, alpha=0.05)
    per_phase = report.per_phase_false_acceptance
    assert per_phase["fine_alignment"] > report.empirical_false_acceptance
    assert per_phase["transit"] < per_phase["fine_alignment"]


def test_caveats_are_returned_as_data():
    report = calibrate(records([1.0] * 50, [1.0] * 50), alpha=0.05)
    text = " ".join(report.caveats()).lower()
    assert "exchangeability" in text and "marginal" in text


def test_finite_sample_slack_is_reported():
    report = calibrate(records([1.0] * 200, [1.0] * 200), alpha=0.05)
    assert report.finite_sample_slack == pytest.approx(1 / 200)


def test_only_stale_records_with_a_locatable_violation_are_used():
    rows = [
        CalibrationRecord(d_pos=1.0, d_rot=1.0, live=False, h_star=None),
        CalibrationRecord(d_pos=2.0, d_rot=2.0, live=False, h_star=3),
    ]
    assert calibrate(rows, alpha=0.5).n_stale == 1


def test_empty_stale_population_is_an_error():
    with pytest.raises(ValueError, match="no usable stale records"):
        calibrate(records([], [], live_pos=[1.0], live_rot=[1.0]), alpha=0.05)


def test_degenerate_threshold_points_at_diagnostic_D1():
    with pytest.raises(ValueError, match="Diagnostic D1"):
        calibrate(records([0.0] * 100, [0.0] * 100), alpha=0.05)
