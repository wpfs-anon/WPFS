"""The cost model -- equations 17 and 18, and Table 1 (SS2.7)."""

from __future__ import annotations

import pytest

from sentry.core.cost import (
    RTX4090D,
    StageLatencies,
    l_check,
    l_step,
    speedup_vs_fixed,
    table1,
)
from sentry.core.types import DepthRung


# -- equation 4 -------------------------------------------------------------


def test_L_full_matches_table_1():
    """``L_enc = 11.3``, ``L_pre = 26.7``, ``M*L_den = 20.0`` -> 58.0 ms."""
    assert RTX4090D.L_full == pytest.approx(58.0)
    assert RTX4090D.M * RTX4090D.L_den == pytest.approx(20.0)


def test_denoising_only_acceleration_is_bounded():
    """SS2.1: "any acceleration that touches only ``M`` is bounded by
    ``(L_enc + L_pre)^-1 L_full`` regardless of how aggressively the sampler is
    distilled."  With Table 1's numbers that ceiling is only ~1.53x, which is
    the whole reason the paper attacks depth instead."""
    assert RTX4090D.denoising_bound == pytest.approx(58.0 / 38.0, abs=1e-6)
    assert RTX4090D.denoising_bound < 1.6


# -- equation 17 ------------------------------------------------------------


def test_l_check_scales_linearly_with_each_depth():
    lat = RTX4090D
    a = l_check(DepthRung(8, 5), lat)
    b = l_check(DepthRung(16, 5), lat)
    assert b - a == pytest.approx((8 / 27) * lat.L_enc)


def test_l_check_charges_exactly_one_denoising_step():
    """The last term is a *single* batched velocity evaluation covering all K
    verification timesteps (SS2.4.2), not K of them."""
    lat = RTX4090D
    full_prefix = l_check(DepthRung(lat.L_V, lat.L_B), lat)
    assert full_prefix == pytest.approx(lat.L_enc + lat.L_pre + lat.L_den)
    assert full_prefix < lat.L_full          # cheaper than a plan by (M-1)*L_den


def test_table1_rungs_1_and_2_do_not_reproduce():
    """Table 1's caption says the entries are "projections from equation 17,
    not measurements", so this is pure arithmetic -- and rungs 1 and 2 do not
    come out.  Documented rather than hidden: anyone reimplementing eq. 17 hits
    it, and the paper's own 58.0/12.2 = 4.8x is internally consistent, so the
    discrepancy is in the table rather than in any claim.
    """
    rows = {r["configuration"]: r for r in table1()}

    assert rows["Cascade rung 1"]["computed_ms"] == pytest.approx(12.76, abs=0.01)
    assert rows["Cascade rung 1"]["paper_ms"] == 12.2

    assert rows["Cascade rung 2"]["computed_ms"] == pytest.approx(18.70, abs=0.01)
    assert rows["Cascade rung 2"]["paper_ms"] == 19.0

    # Rung 3 and the full-depth row do reproduce.
    assert rows["Cascade rung 3"]["computed_ms"] == pytest.approx(25.7, abs=0.05)
    assert rows["Full depth (plan mode)"]["delta_ms"] == pytest.approx(0.0)


def test_table1_speedups_are_monotone_in_depth():
    rows = [r for r in table1() if r["configuration"].startswith("Cascade")]
    speedups = [r["computed_speedup"] for r in rows]
    assert speedups == sorted(speedups, reverse=True)


# -- equation 18 ------------------------------------------------------------


def test_l_step_is_a_renewal_reward_ratio():
    """A rejecting round pays a full replan plus its checks and advances by
    ``m_min``; an accepting round pays only checks and advances by ``N_bar``."""
    lat = RTX4090D
    Lc = 12.76
    got = l_step(rho=0.25, N_bar=8.0, J_bar=1.5, m_min=4, lat=lat, L_check_bar=Lc)
    expected = (0.25 * (lat.L_full + 1.5 * Lc) + 0.75 * 1.5 * Lc) / (0.25 * 4 + 0.75 * 8.0)
    assert got == pytest.approx(expected)


def test_never_rejecting_amortises_the_target_away():
    lat = RTX4090D
    cost = l_step(rho=0.0, N_bar=20.0, J_bar=1.0, m_min=4, lat=lat, L_check_bar=12.76)
    assert cost == pytest.approx(12.76 / 20.0)


def test_always_rejecting_degenerates_to_replanning_every_m_min():
    lat = RTX4090D
    cost = l_step(rho=1.0, N_bar=0.0, J_bar=1.0, m_min=4, lat=lat, L_check_bar=12.76)
    assert cost == pytest.approx((lat.L_full + 12.76) / 4)


def test_longer_accepted_prefixes_are_cheaper_per_step():
    lat = RTX4090D
    a = l_step(0.2, 4.0, 1.0, 4, lat, 12.76)
    b = l_step(0.2, 16.0, 1.0, 4, lat, 12.76)
    assert b < a


def test_cascade_escalation_costs_compute():
    """``J_bar`` multiplies the check cost -- escalation is not free."""
    lat = RTX4090D
    shallow = l_step(0.2, 10.0, 1.0, 4, lat, 12.76)
    escalating = l_step(0.2, 10.0, 3.0, 4, lat, 12.76)
    assert escalating > shallow


def test_l_step_validates_its_inputs():
    lat = RTX4090D
    with pytest.raises(ValueError, match="rho must lie"):
        l_step(1.5, 8.0, 1.0, 4, lat, 12.0)
    with pytest.raises(ValueError, match="J_bar must be"):
        l_step(0.2, 8.0, 0.5, 4, lat, 12.0)
    with pytest.raises(ValueError, match="degenerate step rate"):
        l_step(0.0, 0.0, 1.0, 4, lat, 12.0)


# -- the claim SS2.7 stakes -------------------------------------------------


def test_a_shallow_check_is_not_cheaper_than_a_small_drafter():
    """SS2.7's first accounting point, made concrete.

    "a 110M drafter is roughly 25x smaller than the backbone, whereas prefix
    truncation buys a factor of at most ``L_B/E_B``."  So SENTRY is "not a
    claim about ``L_check`` but about equation 18".
    """
    lat = RTX4090D
    best_truncation = lat.L_B / 5           # rung 1's backbone factor
    assert best_truncation < 25.0

    drafter_prefill = lat.L_pre / 25.0
    sentry_check = l_check(DepthRung(8, 5), lat)
    assert sentry_check > drafter_prefill


def test_speedup_against_a_fixed_baseline():
    """"The baseline replans every ``N_exec`` steps at cost ``L_tgt``" (App. A.2)."""
    lat = RTX4090D
    assert speedup_vs_fixed(lat.L_full, N_exec=1, lat=lat) == pytest.approx(1.0)
    assert speedup_vs_fixed(lat.L_full / 8, N_exec=8, lat=lat) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        speedup_vs_fixed(1.0, N_exec=0, lat=lat)


def test_custom_layer_counts():
    """SS2.9 (iv): backbones in this family vary between 18 and 32 layers."""
    lat = StageLatencies(L_enc=11.3, L_pre=26.7, L_den=2.0, M=10, L_V=27, L_B=32)
    assert l_check(DepthRung(8, 16), lat) == pytest.approx(
        (8 / 27) * 11.3 + 0.5 * 26.7 + 2.0
    )
