from __future__ import annotations

import pytest
import torch

from sentry.core.acceptance import (
    accepted_length,
    first_violation,
    margin,
    normalised_distances,
    sign_agreement,
)
from sentry.core.types import ChannelSpec, Thresholds


def spec() -> ChannelSpec:
    return ChannelSpec(
        d_a=7, pos=(0, 1, 2), rot=(3, 4, 5), grip=6,
        mean=torch.zeros(7), scale=torch.ones(7),
    )


def d_table(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def all_signs_ok(d: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(d, dtype=torch.bool)


@pytest.mark.parametrize(
    "rows, expected, why",
    [
        ([[0.1, 0.2, 0.3, 0.4]], 4, "every position passes"),
        ([[2.0, 0.1, 0.1, 0.1]], 0, "position 0 fails -> empty prefix"),
        ([[0.1, 0.1, 2.0, 0.1]], 2, "prefix stops at the first failure"),
        ([[0.1, 2.0, 0.1, 0.1]], 1, "one accepted action"),
        ([[1.0, 1.0, 1.0, 1.0]], 4, "d == 1 is accepted (the condition is <= 1)"),
    ],
)
def test_prefix_length_single_tau(rows, expected, why):
    d = d_table(rows)
    assert accepted_length(d, all_signs_ok(d)) == expected, why


def test_later_pass_cannot_resurrect_an_earlier_failure():
    d = d_table([[0.1, 5.0, 0.1, 0.1, 0.1]])
    assert accepted_length(d, all_signs_ok(d)) == 1


def test_min_over_T_is_conservative():
    d = d_table([
        [0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 9.0, 0.1],
    ])
    assert accepted_length(d, all_signs_ok(d)) == 2


def test_gripper_sign_can_veto_a_passing_distance():
    d = d_table([[0.1, 0.1, 0.1]])
    sign_ok = torch.tensor([[True, False, True]])
    assert accepted_length(d, sign_ok) == 1


def test_empty_suffix_accepts_nothing():
    d = torch.zeros(2, 0)
    assert accepted_length(d, torch.zeros(2, 0, dtype=torch.bool)) == 0


def test_margin_measured_at_last_accepted_position():
    d = d_table([[0.2, 0.7, 5.0], [0.3, 0.4, 5.0]])
    N = accepted_length(d, all_signs_ok(d))
    assert N == 2
    assert margin(d, N) == pytest.approx(1.0 - 0.7)


def test_margin_is_none_when_nothing_accepted():
    d = d_table([[5.0, 0.1]])
    assert margin(d, 0) is None


def test_margin_is_negative_only_if_misused():
    d = d_table([[0.4]])
    assert margin(d, 1) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        margin(d, 5)


def test_thresholds_are_folded_into_the_statistic():
    s = spec()
    A_hat = torch.zeros(4, 7)
    R = torch.zeros(1, 4, 7)
    R[0, :, 0] = 0.5

    tight = normalised_distances(R, A_hat, s, Thresholds(pos=0.25, rot=1.0), H_k=4)
    loose = normalised_distances(R, A_hat, s, Thresholds(pos=1.0, rot=1.0), H_k=4)

    assert torch.all(tight > 1.0)
    assert torch.all(loose <= 1.0)
    torch.testing.assert_close(loose, torch.full((1, 4), 0.5))


def test_max_over_channel_groups():
    s = spec()
    A_hat = torch.zeros(2, 7)
    R = torch.zeros(1, 2, 7)
    R[0, :, 3] = 4.0
    d = normalised_distances(R, A_hat, s, Thresholds(pos=1.0, rot=1.0), H_k=2)
    assert torch.all(d > 1.0)


def test_only_real_entries_are_evaluated():
    s = spec()
    A_hat = torch.zeros(10, 7)
    R = torch.zeros(1, 10, 7)
    R[0, 5:, 0] = 100.0
    d = normalised_distances(R, A_hat, s, Thresholds(pos=1.0, rot=1.0), H_k=5)
    assert d.shape == (1, 5)
    assert torch.all(d <= 1.0)


def test_sign_agreement():
    s = spec()
    A_hat = torch.zeros(3, 7)
    A_hat[:, 6] = torch.tensor([1.0, -1.0, 1.0])
    R = torch.zeros(1, 3, 7)
    R[0, :, 6] = torch.tensor([1.0, -1.0, -1.0])
    ok = sign_agreement(R, A_hat, s, H_k=3)
    assert ok.tolist() == [[True, True, False]]


def test_zero_gripper_is_treated_as_disagreement():
    s = spec()
    A_hat = torch.zeros(2, 7)
    A_hat[:, 6] = torch.tensor([0.0, 1.0])
    R = torch.zeros(1, 2, 7)
    R[0, :, 6] = torch.tensor([1.0, 0.0])
    assert sign_agreement(R, A_hat, s, H_k=2).tolist() == [[False, False]]


def test_first_violation():
    d = d_table([[0.1, 0.1, 3.0, 0.1], [0.1, 0.1, 0.1, 0.1]])
    assert first_violation(d, all_signs_ok(d)) == 2
    clean = d_table([[0.1, 0.1]])
    assert first_violation(clean, all_signs_ok(clean)) is None
