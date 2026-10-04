from __future__ import annotations

import pytest
import torch

from sentry.core.padding import PlanExhausted, repad
from sentry.core.types import ChannelSpec


def spec(mean=None, scale=None) -> ChannelSpec:
    return ChannelSpec(
        d_a=7, pos=(0, 1, 2), rot=(3, 4, 5), grip=6,
        mean=torch.zeros(7) if mean is None else mean,
        scale=torch.ones(7) if scale is None else scale,
    )


def suffix(n: int) -> torch.Tensor:
    a = torch.arange(n * 7, dtype=torch.float32).reshape(n, 7)
    a[:, 6] = -1.0
    return a


def test_real_entries_are_copied_with_index_zero_meaning_act_now():
    live = suffix(6)
    A_hat, H_k = repad(live, H=10, scheme="hold", spec=spec())
    assert (A_hat.shape, H_k) == ((10, 7), 6)
    torch.testing.assert_close(A_hat[:6], live)
    torch.testing.assert_close(A_hat[0], live[0])


def test_hold_repeats_the_final_surviving_action():
    live = suffix(3)
    A_hat, _ = repad(live, H=8, scheme="hold", spec=spec())
    for h in range(3, 8):
        torch.testing.assert_close(A_hat[h], live[-1])


def test_zero_delta_is_null_motion_with_the_gripper_held():
    live = suffix(3)
    live[-1, 6] = 1.0
    A_hat, _ = repad(live, H=6, scheme="zero_delta", spec=spec())
    assert torch.all(A_hat[3:, :6] == 0.0)
    assert torch.all(A_hat[3:, 6] == 1.0)


def test_zero_delta_round_trips_through_normalisation():
    mean = torch.full((7,), 0.5)
    mean[6] = 0.0
    s = spec(mean=mean, scale=torch.full((7,), 2.0))
    live = suffix(2)
    live[-1, 6] = 1.0
    A_hat, _ = repad(live, H=4, scheme="zero_delta", spec=s)
    torch.testing.assert_close(A_hat[2, 0], torch.tensor(-0.25))


def test_learned_pad():
    emb = torch.full((7,), 3.0)
    A_hat, _ = repad(suffix(2), H=5, scheme="learned", spec=spec(), learned_pad=emb)
    torch.testing.assert_close(A_hat[2:], emb.expand(3, 7))


def test_learned_pad_is_required():
    with pytest.raises(ValueError, match="requires a learned_pad"):
        repad(suffix(2), H=5, scheme="learned", spec=spec())


def test_no_padding_needed_when_the_suffix_is_full_width():
    live = suffix(5)
    A_hat, H_k = repad(live, H=5, scheme="hold", spec=spec())
    assert H_k == 5
    torch.testing.assert_close(A_hat, live)


def test_exhausted_plan_is_an_error_not_a_silent_empty_check():
    with pytest.raises(PlanExhausted):
        repad(torch.zeros(0, 7), H=5, scheme="hold", spec=spec())


def test_suffix_longer_than_chunk_is_rejected():
    with pytest.raises(ValueError, match="longer than the chunk width"):
        repad(suffix(9), H=5, scheme="hold", spec=spec())


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown padding scheme"):
        repad(suffix(2), H=5, scheme="bogus", spec=spec())
