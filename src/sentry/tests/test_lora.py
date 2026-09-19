"""Proposition 4 -- planning is preserved *exactly*, not approximately."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sentry.models.lora import (
    LoRALinear,
    adapters,
    count_lora_parameters,
    lora_parameters,
    set_adapters,
    wrap_linear,
)


@pytest.fixture
def wrapped():
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    reference = nn.Linear(8, 6)
    reference.load_state_dict(base.state_dict())
    return LoRALinear(base, r=4, dropout=0.0), reference


def test_gated_off_is_bit_identical(wrapped):
    """Proposition 4 asserted by **exact** equality, not ``allclose``.

    "the distribution over chunks produced in plan mode equals that of the
    pretrained target, exactly and not approximately, since theta is never
    updated and Delta contributes zero when gated off."

    The gated-off path returns ``base(x)`` untouched rather than
    ``base(x) + 0``: adding a zero tensor is a floating-point operation and can
    perturb the last bit.  "Exactly" has to mean exactly.
    """
    lora, reference = wrapped
    x = torch.randn(3, 8)
    lora.gate = False
    assert torch.equal(lora(x), reference(x))


def test_still_bit_identical_after_the_adapter_is_trained(wrapped):
    """Prop. 4 must hold "for any Delta obtained by the training of Sec. 2.5"."""
    lora, reference = wrapped
    nn.init.normal_(lora.lora_B.weight, std=1.0)   # pretend Stage B ran
    nn.init.normal_(lora.lora_A.weight, std=1.0)
    x = torch.randn(3, 8)

    lora.gate = False
    assert torch.equal(lora(x), reference(x))

    lora.gate = True
    assert not torch.allclose(lora(x), reference(x))


def test_zero_initialised_B_makes_a_fresh_adapter_inert(wrapped):
    """Even gated *on*, a freshly built adapter contributes nothing."""
    lora, reference = wrapped
    lora.gate = True
    x = torch.randn(3, 8)
    torch.testing.assert_close(lora(x), reference(x))


def test_scaling_is_alpha_over_r(wrapped):
    """SS2.9: ``alpha_LoRA = 2r``."""
    lora, _ = wrapped
    assert lora.alpha == 2 * lora.r
    assert lora.scaling == pytest.approx(2.0)


def test_base_weights_are_frozen(wrapped):
    """``theta`` is frozen throughout (SS2.5)."""
    lora, _ = wrapped
    assert not any(p.requires_grad for p in lora.base.parameters())
    assert all(p.requires_grad for p in lora_parameters(lora))


def test_gate_is_not_persisted_state(wrapped):
    """The gate is control flow, not state.

    Plan and check mode alternate within an episode; if the gate travelled in
    ``state_dict`` a checkpoint could be reloaded in check mode and silently
    break Proposition 4.
    """
    lora, _ = wrapped
    lora.gate = True
    assert not any("gate" in k for k in lora.state_dict())


def test_set_adapters_walks_the_tree():
    model = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 4)))
    wrap_linear(model, "0")
    wrap_linear(model[1], "0")
    assert set_adapters(model, True) == 2
    assert all(m.gate for m in model.modules() if isinstance(m, LoRALinear))
    set_adapters(model, False)
    assert not any(m.gate for m in model.modules() if isinstance(m, LoRALinear))


def test_context_manager_restores_state_even_on_exception():
    """An exception must not leave the model stuck in check mode.

    Silently corrupting the *next* plan is exactly the failure Prop. 4 exists
    to rule out.
    """
    model = nn.Sequential(nn.Linear(4, 4))
    lora = wrap_linear(model, "0")
    lora.gate = False

    with pytest.raises(RuntimeError):
        with adapters(model, True):
            assert lora.gate
            raise RuntimeError("boom")

    assert lora.gate is False


def test_wrapping_is_idempotent():
    model = nn.Sequential(nn.Linear(4, 4))
    first = wrap_linear(model, "0")
    second = wrap_linear(model, "0")
    assert first is second
    assert count_lora_parameters(model) == first.r * (4 + 4)


def test_wrapping_a_non_linear_is_rejected():
    model = nn.Sequential(nn.ReLU())
    with pytest.raises(TypeError, match="not nn.Linear"):
        wrap_linear(model, "0")
