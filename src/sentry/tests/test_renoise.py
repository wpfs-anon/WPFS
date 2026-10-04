from __future__ import annotations

import pytest
import torch

from sentry.core.renoise import interpolate, reconstruct, to_model_time


@pytest.fixture
def fixture():
    torch.manual_seed(0)
    A_hat = torch.randn(6, 4)
    eps = torch.randn(6, 4)
    s = torch.tensor([0.1, 0.5, 0.6, 0.9, 0.99])
    return A_hat, eps, s


def test_eq9_is_exact_under_constant_velocity(fixture):
    A_hat, eps, s = fixture
    A_s = interpolate(A_hat, s, eps)
    v = (A_hat - eps).unsqueeze(0).expand(len(s), -1, -1)

    R = reconstruct(A_s, s, v, convention="one_is_clean")

    torch.testing.assert_close(R, A_hat.unsqueeze(0).expand_as(R), atol=1e-5, rtol=1e-5)


def test_eq9_exact_under_reversed_convention(fixture):
    A_hat, eps, s = fixture
    A_s = interpolate(A_hat, s, eps)
    v_model = -(A_hat - eps).unsqueeze(0).expand(len(s), -1, -1)

    R = reconstruct(A_s, s, v_model, convention="zero_is_clean")

    torch.testing.assert_close(R, A_hat.unsqueeze(0).expand_as(R), atol=1e-5, rtol=1e-5)


def test_reversed_convention_is_not_a_no_op(fixture):
    A_hat, eps, s = fixture
    A_s = interpolate(A_hat, s, eps)
    v = (A_hat - eps).unsqueeze(0).expand(len(s), -1, -1)

    same = reconstruct(A_s, s, v, "one_is_clean")
    flipped = reconstruct(A_s, s, v, "zero_is_clean")
    assert not torch.allclose(same, flipped)


def test_to_model_time():
    s = torch.tensor([0.6, 0.9])
    torch.testing.assert_close(to_model_time(s, "one_is_clean"), s)
    torch.testing.assert_close(to_model_time(s, "zero_is_clean"), 1.0 - s)


def test_interpolation_endpoints(fixture):
    A_hat, eps, _ = fixture
    near_one = interpolate(A_hat, torch.tensor([0.999]), eps)[0]
    near_zero = interpolate(A_hat, torch.tensor([0.001]), eps)[0]
    torch.testing.assert_close(near_one, A_hat, atol=2e-2, rtol=0)
    torch.testing.assert_close(near_zero, eps, atol=2e-2, rtol=0)


def test_single_shared_noise_draw(fixture):
    A_hat, eps, s = fixture
    A_s = interpolate(A_hat, s, eps)
    for i, si in enumerate(s):
        torch.testing.assert_close(A_s[i], si * A_hat + (1 - si) * eps)


def test_shape_validation(fixture):
    A_hat, _, s = fixture
    with pytest.raises(ValueError, match="eps must match"):
        interpolate(A_hat, s, torch.randn(3, 3))
