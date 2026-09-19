"""Training objectives and parameter partitioning (SS2.5).

The preflight suite (:mod:`sentry.training.preflight`) exercises the loops
end-to-end; these pin the individual equations, where the mistakes are
one-liners with no visible symptom.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sentry.core.types import ChannelSpec
from sentry.models.lora import set_adapters, wrap_linear
from sentry.training import losses
from sentry.training.losses import (
    behaviour_loss,
    distillation_loss,
    lambda_a,
    margin_loss,
    sensitivity_loss,
    stage_a_loss,
    token_loss,
)
from sentry.training.params import audit, freeze_all_but, split_adapters


# -- equation 13 ------------------------------------------------------------


def test_lambda_anneals_upward():
    """SS2.5.1: the token term is "a fast warm start ... after which the
    behavioural term takes over and **is the one that matters**".

    Annealing downward would train the encoder to match intermediate tokens it
    was never required to match -- "both unnecessary and, at E_V << L_V, likely
    infeasible".
    """
    assert lambda_a(0.0, 2.0) == 0.0
    assert lambda_a(1.0, 2.0) == 2.0
    assert lambda_a(0.5, 2.0) == pytest.approx(1.0)
    assert lambda_a(0.2, 2.0) < lambda_a(0.8, 2.0)


def test_lambda_rejects_out_of_range():
    with pytest.raises(ValueError, match="must lie in"):
        lambda_a(1.5, 1.0)


def test_token_loss_is_zero_for_identical_directions():
    z = torch.randn(4, 8)
    assert float(token_loss(z, z)) == pytest.approx(0.0, abs=1e-6)


def test_token_loss_ignores_scale():
    """Cosine, not L2: what matters is direction in representation space."""
    z = torch.randn(4, 8)
    assert float(token_loss(z, 7.0 * z)) == pytest.approx(0.0, abs=1e-6)


def test_token_loss_is_maximal_for_opposed_directions():
    z = torch.randn(4, 8)
    assert float(token_loss(z, -z)) == pytest.approx(2.0, abs=1e-6)


def test_teacher_is_stop_gradiented():
    """``sg[.]`` in eqs. 13 and 14 -- no gradient may reach the teacher."""
    student = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 4, requires_grad=True)

    behaviour_loss(student, teacher).backward()
    assert student.grad is not None
    assert teacher.grad is None

    student2 = torch.randn(2, 4, requires_grad=True)
    teacher2 = torch.randn(2, 4, requires_grad=True)
    distillation_loss(student2, teacher2).backward()
    assert teacher2.grad is None


def test_stage_a_loss_composition():
    z = torch.randn(3, 6)
    v_s, v_f = torch.randn(2, 4, 7), torch.randn(2, 4, 7)
    _, parts = stage_a_loss(z, z, v_s, v_f, u=0.5, lambda_max=2.0)
    assert parts["lambda_A"] == pytest.approx(1.0)
    assert parts["loss"] == pytest.approx(parts["L_tok"] + 1.0 * parts["L_beh"], abs=1e-5)


# -- equation 15 ------------------------------------------------------------


def test_margin_pushes_valid_positions_below_threshold():
    """Valid positions should sit comfortably under ``1 - m``."""
    comfortable = margin_loss(torch.full((5,), 0.5), h_star=5, H_k=5, m=0.2)
    marginal = margin_loss(torch.full((5,), 0.95), h_star=5, H_k=5, m=0.2)
    assert float(comfortable) == 0.0        # 0.5 < 1 - 0.2
    assert float(marginal) > 0.0            # 0.95 > 0.8


def test_margin_pushes_first_invalid_above_threshold():
    d = torch.tensor([0.1, 0.1, 0.5])       # position 2 is invalid but scores low
    loss = margin_loss(d, h_star=2, H_k=3, m=0.2)
    assert float(loss) == pytest.approx((1.0 + 0.2) - 0.5)


def test_second_term_omitted_when_plan_is_fully_live():
    """SS2.5.2: "The second term is omitted when ``h* = H_k``."

    Keeping it would train the model to reject the padded tail, which eq. 11
    never evaluates -- agreement is "only ever evaluated on real entries".
    """
    d = torch.full((4,), 0.1)
    assert float(margin_loss(d, h_star=4, H_k=4, m=0.2)) == 0.0


def test_h_star_zero_does_not_divide_by_zero():
    """``1/h*`` is undefined when position 0 is already invalid."""
    loss = margin_loss(torch.full((4,), 0.1), h_star=0, H_k=4, m=0.2)
    assert torch.isfinite(loss) and float(loss) > 0.0


def test_margin_validates_bounds():
    with pytest.raises(ValueError, match="h_star"):
        margin_loss(torch.zeros(4), h_star=9, H_k=4, m=0.2)


# -- equation 16 ------------------------------------------------------------


def test_sensitivity_penalises_the_degenerate_optimum():
    """The collapse SS2.5.2 names: a verifier that ignores ``o*``.

    Copying the interpolant scores well on eq. 14 whenever the teacher also
    roughly returns the candidate, so nothing else in ``L_B`` excludes it.
    """
    R = torch.randn(3, 5, 7)
    assert float(sensitivity_loss(R, R.clone(), eta=0.05)) == pytest.approx(0.05)


def test_sensitivity_is_zero_for_an_observation_sensitive_verifier():
    R = torch.randn(3, 5, 7)
    assert float(sensitivity_loss(R, R + 10.0, eta=0.05)) == 0.0


def test_sensitivity_requires_matched_shapes():
    with pytest.raises(ValueError, match="matched pair"):
        sensitivity_loss(torch.randn(2, 3, 4), torch.randn(2, 3, 5), eta=0.05)


def test_sensitivity_gradient_pushes_the_pair_apart():
    R_pos = torch.randn(2, 4, 6, requires_grad=True)
    R_neg = R_pos.detach().clone() + 1e-4
    sensitivity_loss(R_pos, R_neg, eta=1.0).backward()
    assert R_pos.grad is not None and float(R_pos.grad.abs().sum()) > 0


# -- parameter partitioning -------------------------------------------------


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([nn.Linear(4, 4)])
        self.backbone = nn.ModuleList([nn.Linear(4, 4)])
        self.readout = nn.Linear(4, 4)
        wrap_linear(self.encoder, "0", r=2, dropout=0.0)
        wrap_linear(self.backbone, "0", r=2, dropout=0.0)
        wrap_linear(self, "readout", r=2, dropout=0.0)
        # Gates default to off, and a gated-off LoRALinear returns base(x)
        # untouched -- deliberately, since Proposition 4 requires bit-identity.
        # With every gate off no adapter is in the graph and backward has
        # nothing to differentiate, so training code must switch them on.
        set_adapters(self, True)

    def forward(self, x):
        return self.readout(self.backbone[0](self.encoder[0](x)))


def test_split_puts_readout_in_delta_B():
    """SS2.5.2 lists the read-out with the backbone adapters, because feeding
    it features from layer ``E_B`` "is the principal source of mismatch"."""
    g = split_adapters(_Toy())
    assert any("encoder" in n for n, _ in g.delta_V)
    assert any("readout" in n for n, _ in g.delta_B)
    assert not any("readout" in n for n, _ in g.delta_V)


def test_split_is_disjoint_and_complete():
    model = _Toy()
    g = split_adapters(model)
    ids_V = {id(p) for p in g.params_V()}
    ids_B = {id(p) for p in g.params_B()}
    total = sum(1 for n, _ in model.named_parameters() if ".lora_" in n)
    assert not (ids_V & ids_B)
    assert len(ids_V) + len(ids_B) == total


def test_freeze_all_but_is_exact():
    model = _Toy()
    g = split_adapters(model)
    freeze_all_but(model, g.params_V())
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in g.params_V()}


def test_audit_catches_a_leak():
    """The failure that matters: a frozen tensor receiving gradient.

    Stage B re-tuning ``Delta_V`` undoes Stage A's warm-start invisibly.
    """
    model = _Toy()
    g = split_adapters(model)
    freeze_all_but(model, g.params_V())
    for p in g.params_B():
        p.requires_grad_(True)              # simulate a botched freeze

    model(torch.randn(2, 4)).sum().backward()
    rep = audit(model, "Stage A", g.delta_V)
    assert not rep.ok and rep.leaked


def test_audit_tolerates_zero_gradient_on_lora_A():
    """LoRA initialises ``B = 0``, so ``dL/dA`` is exactly zero on the first
    backward.  Membership in the graph is what the audit tests, not magnitude
    -- otherwise every ``lora_A`` reads as disconnected on step 0."""
    model = _Toy()
    g = split_adapters(model)
    freeze_all_but(model, g.params_V())
    model(torch.randn(2, 4)).sum().backward()

    rep = audit(model, "Stage A", g.delta_V)
    a_grads = [p.grad for n, p in g.delta_V if ".lora_A." in n]
    assert all(gr is not None and float(gr.abs().sum()) == 0.0 for gr in a_grads)
    assert rep.ok


# -- eq. 13's two corrections (SS2.5.1) -------------------------------------


def test_real_channels_excludes_the_padding():
    """``d_a`` is padded; only pos/rot/grip carry information."""
    spec = ChannelSpec.libero(d_a=32)
    assert spec.real_channels == (0, 1, 2, 3, 4, 5, 6)
    assert len(spec.real_channels) == 7


def test_behaviour_loss_is_diluted_by_padded_channels():
    """The measured failure mode, reproduced in miniature.

    Both branches emit zero on the padding, so a mean over all ``d_a`` is mostly
    an average of agreement that costs nothing.  On pi0_libero that understated
    the error 4.46x; here the geometry is exact, so the factor is 32/7.
    """
    spec = ChannelSpec.libero(d_a=32)
    v_full = torch.zeros(2, 8, 32)
    v_shal = torch.zeros(2, 8, 32)
    v_shal[..., :7] = 1.0                      # error only on the real channels

    diluted = losses.behaviour_loss(v_shal, v_full)
    honest = losses.behaviour_loss(v_shal, v_full, spec.real_channels)

    assert honest == pytest.approx(1.0)
    assert diluted == pytest.approx(7 / 32)
    assert honest / diluted == pytest.approx(32 / 7)


def test_gripper_sign_loss_fires_only_on_disagreement():
    """SS2.4.3 reads the gripper by sign; eq. 13 as written does not."""
    spec = ChannelSpec.libero(d_a=32)
    g = spec.grip

    R_full = torch.zeros(1, 4, 32)
    R_full[..., g] = torch.tensor([1.0, 1.0, -1.0, -1.0])

    agree = torch.zeros(1, 4, 32)
    agree[..., g] = torch.tensor([0.9, 0.5, -0.9, -0.5])       # same side, past margin
    assert losses.gripper_sign_loss(agree, R_full, g, margin=0.1) == pytest.approx(0.0)

    flipped = torch.zeros(1, 4, 32)
    flipped[..., g] = torch.tensor([-0.9, -0.5, 0.9, 0.5])     # wrong side
    # hinge = relu(m - t*sign(s)) per position: 1.0, 0.6, 1.0, 0.6 -> mean 0.8
    assert losses.gripper_sign_loss(flipped, R_full, g, margin=0.1) == pytest.approx(0.8)

    # Creeping just over zero is still penalised: the margin is the point.
    barely = torch.zeros(1, 4, 32)
    barely[..., g] = torch.tensor([0.01, 0.01, -0.01, -0.01])
    assert losses.gripper_sign_loss(barely, R_full, g, margin=0.1) > 0.0


def test_gripper_sign_loss_ignores_an_intentless_target():
    """A target sitting on the decision boundary carries no sign to preserve."""
    spec = ChannelSpec.libero(d_a=32)
    g = spec.grip
    R_full = torch.zeros(1, 3, 32)             # every target inside the dead zone
    R_shal = torch.zeros(1, 3, 32)
    R_shal[..., g] = torch.tensor([5.0, -5.0, 5.0])
    assert losses.gripper_sign_loss(R_shal, R_full, g) == pytest.approx(0.0)


def test_stage_a_loss_reports_the_flip_rate():
    """The metric that predicts task success has to be visible during training."""
    spec = ChannelSpec.libero(d_a=32)
    z = torch.randn(1, 4, 6)
    v = torch.randn(1, 4, 32)
    R_full = torch.zeros(1, 4, 32)
    R_full[..., spec.grip] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    R_shal = torch.zeros(1, 4, 32)
    R_shal[..., spec.grip] = torch.tensor([1.0, 1.0, -1.0, -1.0])   # half wrong

    _, parts = losses.stage_a_loss(
        z_shal=z, z_full=z, v_shal=v, v_full=v, u=0.5, lambda_max=1.0,
        channels=spec.real_channels, R_shal=R_shal, R_full=R_full,
        grip=spec.grip, lambda_grip=1.0,
    )
    assert parts["grip_flip"] == pytest.approx(0.5)
    assert parts["L_grip"] > 0
