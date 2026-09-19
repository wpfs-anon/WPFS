"""Equations 10 and 11, and the margin (SS2.4.3)."""

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


# -- equation 11 ------------------------------------------------------------


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
    """The cumulative product makes this a *prefix* length, not a count.

    A position that passes after an earlier failure must not contribute --
    otherwise the operator would license executing an action whose predecessor
    it had just rejected.
    """
    d = d_table([[0.1, 5.0, 0.1, 0.1, 0.1]])
    assert accepted_length(d, all_signs_ok(d)) == 1


def test_min_over_T_is_conservative():
    """SS2.4.3 takes the accepted length "conservatively over ``T``".

    One verification timestep objecting is enough to truncate the prefix.
    """
    d = d_table([
        [0.1, 0.1, 0.1, 0.1],   # tau_1 would accept all 4
        [0.1, 0.1, 9.0, 0.1],   # tau_2 stops at 2
    ])
    assert accepted_length(d, all_signs_ok(d)) == 2


def test_gripper_sign_can_veto_a_passing_distance():
    """The gripper is "discrete in intent" and tested separately (eq. 10/11)."""
    d = d_table([[0.1, 0.1, 0.1]])
    sign_ok = torch.tensor([[True, False, True]])
    assert accepted_length(d, sign_ok) == 1


def test_empty_suffix_accepts_nothing():
    d = torch.zeros(2, 0)
    assert accepted_length(d, torch.zeros(2, 0, dtype=torch.bool)) == 0


# -- margin -----------------------------------------------------------------


def test_margin_measured_at_last_accepted_position():
    """``mu = 1 - max_tau d^tau_{N-1}`` -- how close the prefix came to failing."""
    d = d_table([[0.2, 0.7, 5.0], [0.3, 0.4, 5.0]])
    N = accepted_length(d, all_signs_ok(d))
    assert N == 2
    assert margin(d, N) == pytest.approx(1.0 - 0.7)


def test_margin_is_none_when_nothing_accepted():
    """SS2.4.3 defines mu only for ``N >= 1``.

    Returning ``None`` forces the cascade to handle rejection explicitly rather
    than comparing a fabricated ``0.0`` against ``mu_esc``.
    """
    d = d_table([[5.0, 0.1]])
    assert margin(d, 0) is None


def test_margin_is_negative_only_if_misused():
    d = d_table([[0.4]])
    assert margin(d, 1) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        margin(d, 5)


# -- equation 10 ------------------------------------------------------------


def test_thresholds_are_folded_into_the_statistic():
    """``d <= 1`` *is* the acceptance condition, per eq. 10."""
    s = spec()
    A_hat = torch.zeros(4, 7)
    R = torch.zeros(1, 4, 7)
    R[0, :, 0] = 0.5          # pure translation error of 0.5

    tight = normalised_distances(R, A_hat, s, Thresholds(pos=0.25, rot=1.0), H_k=4)
    loose = normalised_distances(R, A_hat, s, Thresholds(pos=1.0, rot=1.0), H_k=4)

    assert torch.all(tight > 1.0)     # rejected
    assert torch.all(loose <= 1.0)    # accepted
    torch.testing.assert_close(loose, torch.full((1, 4), 0.5))


def test_max_over_channel_groups():
    """Eq. 10 takes the max, so either group alone can reject."""
    s = spec()
    A_hat = torch.zeros(2, 7)
    R = torch.zeros(1, 2, 7)
    R[0, :, 3] = 4.0                              # rotation error only
    d = normalised_distances(R, A_hat, s, Thresholds(pos=1.0, rot=1.0), H_k=2)
    assert torch.all(d > 1.0)


def test_only_real_entries_are_evaluated():
    """SS2.4.1: "Agreement is only ever evaluated on real entries, h < H_k"."""
    s = spec()
    A_hat = torch.zeros(10, 7)
    R = torch.zeros(1, 10, 7)
    R[0, 5:, 0] = 100.0        # catastrophic disagreement, but in the padded tail
    d = normalised_distances(R, A_hat, s, Thresholds(pos=1.0, rot=1.0), H_k=5)
    assert d.shape == (1, 5)
    assert torch.all(d <= 1.0)


# -- sign test --------------------------------------------------------------


def test_sign_agreement():
    s = spec()
    A_hat = torch.zeros(3, 7)
    A_hat[:, 6] = torch.tensor([1.0, -1.0, 1.0])
    R = torch.zeros(1, 3, 7)
    R[0, :, 6] = torch.tensor([1.0, -1.0, -1.0])
    ok = sign_agreement(R, A_hat, s, H_k=3)
    assert ok.tolist() == [[True, True, False]]


def test_zero_gripper_is_treated_as_disagreement():
    """A command sitting exactly on the decision boundary carries no intent.

    Accepting it would be the one place the sign test could pass vacuously.
    """
    s = spec()
    A_hat = torch.zeros(2, 7)
    A_hat[:, 6] = torch.tensor([0.0, 1.0])
    R = torch.zeros(1, 2, 7)
    R[0, :, 6] = torch.tensor([1.0, 0.0])
    assert sign_agreement(R, A_hat, s, H_k=2).tolist() == [[False, False]]


# -- first violation --------------------------------------------------------


def test_first_violation():
    d = d_table([[0.1, 0.1, 3.0, 0.1], [0.1, 0.1, 0.1, 0.1]])
    assert first_violation(d, all_signs_ok(d)) == 2
    clean = d_table([[0.1, 0.1]])
    assert first_violation(clean, all_signs_ok(clean)) is None
