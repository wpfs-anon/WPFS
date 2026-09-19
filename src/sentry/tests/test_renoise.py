"""Equations 8 and 9 -- the cheapest tripwire for a tau-direction slip."""

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
    """``R^tau`` recovers ``A_hat`` exactly when ``v = A - eps``.

    With ``A^s = s*A + (1-s)*eps`` and a constant field ``v = A - eps``::

        A^s + (1-s)v = sA + (1-s)eps + (1-s)(A - eps) = A

    This holds for *every* tau, which is why SS2.4.2 can say "any point on that
    path is an admissible query".  A sign or direction error in either equation
    breaks it immediately.
    """
    A_hat, eps, s = fixture
    A_s = interpolate(A_hat, s, eps)
    v = (A_hat - eps).unsqueeze(0).expand(len(s), -1, -1)

    R = reconstruct(A_s, s, v, convention="one_is_clean")

    torch.testing.assert_close(R, A_hat.unsqueeze(0).expand_as(R), atol=1e-5, rtol=1e-5)


def test_eq9_exact_under_reversed_convention(fixture):
    """SS2.9 (i): a backend integrating the other way needs *two* corrections.

    Its time argument becomes ``1 - s``, and since its velocity is
    ``dA/dtau_model`` with ``tau_model = 1 - s``, the chain rule flips the sign
    of ``dA/ds`` as well.  Getting only the time argument right leaves the
    reconstruction wrong by ``2(1-s)v`` -- and still plausible-looking.
    """
    A_hat, eps, s = fixture
    A_s = interpolate(A_hat, s, eps)
    v_model = -(A_hat - eps).unsqueeze(0).expand(len(s), -1, -1)

    R = reconstruct(A_s, s, v_model, convention="zero_is_clean")

    torch.testing.assert_close(R, A_hat.unsqueeze(0).expand_as(R), atol=1e-5, rtol=1e-5)


def test_reversed_convention_is_not_a_no_op(fixture):
    """Guard against the two corrections silently cancelling."""
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
    """``s -> 1`` approaches clean data; ``s -> 0`` approaches pure noise."""
    A_hat, eps, _ = fixture
    near_one = interpolate(A_hat, torch.tensor([0.999]), eps)[0]
    near_zero = interpolate(A_hat, torch.tensor([0.001]), eps)[0]
    torch.testing.assert_close(near_one, A_hat, atol=2e-2, rtol=0)
    torch.testing.assert_close(near_zero, eps, atol=2e-2, rtol=0)


def test_single_shared_noise_draw(fixture):
    """SS2.4.2 draws one ``eps`` for the whole of ``T``.

    Sharing it means the K queries differ only in *where they sit on the same
    noise-to-data line*, which is what makes eq. 11's ``min`` over ``T`` a
    conservative reading of one situation rather than an average over K
    unrelated ones.
    """
    A_hat, eps, s = fixture
    A_s = interpolate(A_hat, s, eps)
    # Every row must be an affine blend of the *same* two endpoints.
    for i, si in enumerate(s):
        torch.testing.assert_close(A_s[i], si * A_hat + (1 - si) * eps)


def test_shape_validation(fixture):
    A_hat, _, s = fixture
    with pytest.raises(ValueError, match="eps must match"):
        interpolate(A_hat, s, torch.randn(3, 3))
