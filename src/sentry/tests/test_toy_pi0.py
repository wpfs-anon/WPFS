"""Structural properties of the pi_0 stand-in (SS2.1, SS2.3, SS2.5, SS2.9)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sentry.core.interfaces import VLABackend, assert_no_cache_seam
from sentry.core.types import Observation, assert_shared_depth
from sentry.models.lora import (
    LoRALinear,
    MultiLoRALinear,
    adapter_slot,
    adapters,
    set_adapter_slot,
    set_adapters,
)
from sentry.models.toy_pi0 import TinyPi0, TinyPi0Config


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyPi0(TinyPi0Config()).eval()


def obs(cfg: TinyPi0Config, seed: int = 0) -> Observation:
    g = torch.Generator().manual_seed(seed)
    return Observation(
        images=torch.randn(cfg.n_cam, cfg.in_channels, cfg.img_size, cfg.img_size, generator=g),
        language=torch.randint(0, cfg.vocab_size, (cfg.max_lang,), generator=g),
        state=torch.randn(cfg.d_state, generator=g),
        t=0,
    )


# -- protocol conformance ---------------------------------------------------


def test_satisfies_the_backend_protocol(model):
    assert isinstance(model, VLABackend)
    assert_no_cache_seam(model)


def test_shapes(model):
    cfg = model.cfg
    A_tau = torch.randn(2, cfg.H, cfg.d_a)
    v = model.velocity(A_tau, torch.tensor([0.6, 0.9]), obs(cfg), cfg.L_V, cfg.L_B, False)
    assert v.shape == (2, cfg.H, cfg.d_a)
    assert model.plan(obs(cfg)).shape == (cfg.H, cfg.d_a)


def test_K_interpolants_go_through_one_forward(model):
    """SS2.4.2: the K queries "differ only in the leading batch dimension".

    Equation 17's cost model assumes ``T`` costs one evaluation, not ``K``.
    """
    cfg = model.cfg
    A = torch.randn(1, cfg.H, cfg.d_a)
    batched = model.velocity(
        A.expand(3, -1, -1).contiguous(), torch.tensor([0.3, 0.6, 0.9]),
        obs(cfg), cfg.L_V, cfg.L_B, False,
    )
    assert batched.shape[0] == 3
    # Different taus must give different velocities, or the time conditioning
    # is not wired up and the whole re-noising scheme is vacuous.
    assert not torch.allclose(batched[0], batched[2])


# -- Proposition 4 ----------------------------------------------------------


def test_plan_mode_is_bit_identical_with_adapters_off(model):
    """Prop. 4: "exactly and not approximately"."""
    cfg = model.cfg
    o = obs(cfg)
    A = torch.randn(1, cfg.H, cfg.d_a)
    tau = torch.tensor([0.5])

    baseline = model.velocity(A, tau, o, cfg.L_V, cfg.L_B, adapters=False)

    # Train the adapters into something non-trivial, then gate them off again.
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B.weight, std=0.5)

    after = model.velocity(A, tau, o, cfg.L_V, cfg.L_B, adapters=False)
    assert torch.equal(baseline, after)

    on = model.velocity(A, tau, o, cfg.L_V, cfg.L_B, adapters=True)
    assert not torch.allclose(baseline, on)


def test_gate_is_restored_after_a_velocity_call(model):
    cfg = model.cfg
    set_adapters(model, False)
    model.velocity(torch.randn(1, cfg.H, cfg.d_a), torch.tensor([0.5]), obs(cfg),
                   cfg.L_V, cfg.L_B, adapters=True)
    assert not any(m.gate for m in model.modules() if isinstance(m, LoRALinear))


# -- depth truncation -------------------------------------------------------


def test_depth_changes_the_answer(model):
    """Truncation must actually truncate."""
    cfg = model.cfg
    o = obs(cfg)
    A = torch.randn(1, cfg.H, cfg.d_a)
    tau = torch.tensor([0.6])
    shallow = model.velocity(A, tau, o, 2, 2, adapters=False)
    deep = model.velocity(A, tau, o, cfg.L_V, cfg.L_B, adapters=False)
    assert not torch.allclose(shallow, deep)


def test_truncation_is_a_runtime_prefix_not_a_rebuilt_model(model):
    """Plan and check mode alternate within one episode, so depth is per-call."""
    cfg = model.cfg
    o = obs(cfg)
    A = torch.randn(1, cfg.H, cfg.d_a)
    tau = torch.tensor([0.6])

    first = model.velocity(A, tau, o, 3, 3, adapters=False)
    model.velocity(A, tau, o, cfg.L_V, cfg.L_B, adapters=False)   # interleave
    again = model.velocity(A, tau, o, 3, 3, adapters=False)
    assert torch.equal(first, again)


def test_depth_budgets_are_validated(model):
    cfg = model.cfg
    A = torch.randn(1, cfg.H, cfg.d_a)
    with pytest.raises(ValueError, match="E_V="):
        model.velocity(A, torch.tensor([0.5]), obs(cfg), cfg.L_V + 1, 2, False)
    with pytest.raises(ValueError, match="E_B="):
        model.velocity(A, torch.tensor([0.5]), obs(cfg), 2, cfg.L_B + 1, False)


def test_shallow_backbone_deep_denoiser_is_inexpressible():
    """SS2.9, *Depth is shared*.

    Attention is joint at every layer, so a single ``E_B`` governs both
    pathways.  Configurations that truncate them independently "are not
    expressible without modifying the architecture" -- made a construction-time
    error rather than a silently wrong model.
    """
    assert_shared_depth(5, 5)
    with pytest.raises(ValueError, match="Depth is shared"):
        assert_shared_depth(5, 12)


# -- adapter placement (SS2.9) ----------------------------------------------


def test_adapters_cover_both_experts(model):
    """"Delta_B attaches to the query/key/value/output projections and the MLP
    up/down projections of *both* the VLM expert and the action expert"."""
    blk = model.backbone[0]
    for attr in ("q_vlm", "k_vlm", "v_vlm", "o_vlm",
                 "q_act", "k_act", "v_act", "o_act"):
        assert isinstance(getattr(blk, attr), LoRALinear), attr
    for mlp in (blk.mlp_vlm, blk.mlp_act):
        assert isinstance(mlp.up, LoRALinear)
        assert isinstance(mlp.down, LoRALinear)


def test_adapters_stop_at_the_configured_layer(model):
    """"for layers ``l < E_B`` only"."""
    cfg = model.cfg
    assert isinstance(model.backbone[cfg.lora_layers_B - 1].q_vlm, LoRALinear)
    assert not isinstance(model.backbone[cfg.lora_layers_B].q_vlm, LoRALinear)
    assert isinstance(model.encoder[cfg.lora_layers_V - 1].q, LoRALinear)
    assert not isinstance(model.encoder[cfg.lora_layers_V].q, LoRALinear)


def test_readout_carries_one_higher_rank_adapter_per_rung(model):
    """SS2.5.2: "a per-rung read-out adapter Delta^(j)_out on the action read-out
    projection, one for each E^(j) in the ladder".

    SS2.9 pins down that they share one projection rather than duplicating it:
    they "attach to the **single** action read-out projection and are selected
    by which rung is being evaluated".  So the head is not replicated -- only
    the low-rank update is.
    """
    ro = model.readout
    assert isinstance(ro, MultiLoRALinear)
    assert ro.n_slots == model.cfg.n_rungs
    assert ro.r == model.cfg.lora_rank_readout
    assert ro.r > model.cfg.lora_rank
    # One projection, J updates -- not J projections.
    assert isinstance(ro.base, nn.Linear)
    assert len(ro.lora_A) == len(ro.lora_B) == ro.n_slots


def test_readout_slots_are_independent(model):
    """Selecting a rung must change which update is applied, and nothing else."""
    ro = model.readout
    x = torch.randn(2, 3, ro.in_features)

    with adapters(model, False):
        base = ro(x)

    # Zero-init B means every slot starts inert; give two of them a signature.
    with torch.no_grad():
        nn.init.normal_(ro.lora_B[0].weight, std=0.1)
        nn.init.normal_(ro.lora_B[1].weight, std=0.1)

    model.eval()  # dropout off, so the comparison is deterministic
    with adapters(model, True):
        with adapter_slot(model, 0):
            y0 = ro(x)
        with adapter_slot(model, 1):
            y1 = ro(x)
        with adapter_slot(model, 2):
            y2 = ro(x)

    assert not torch.allclose(y0, base)
    assert not torch.allclose(y1, base)
    assert not torch.allclose(y0, y1)
    # Slot 2 was left at its zero init, so it must still be exactly inert.
    assert torch.equal(y2, base)


def test_adapter_slot_restores_the_previous_rung(model):
    """The cascade escalates mid-check; a stale slot would silently evaluate the
    next rung's read-out against this rung's hidden state."""
    set_adapter_slot(model, 2)
    with adapter_slot(model, 0):
        assert model.readout.slot == 0
    assert model.readout.slot == 2


def test_multi_exit_matches_running_each_depth_separately(model):
    """SS2.5.2's whole claim: one pass to E_max exposes every rung at once,
    because "layer l depends only on layers < l"."""
    torch.manual_seed(0)
    model.eval()
    obs_ = obs(model.cfg)
    A_tau = torch.randn(2, model.cfg.H, model.cfg.d_a)
    tau = torch.tensor([0.6, 0.9])
    exits = [1, 2, 3]

    # Give each slot a distinct update so a slot mix-up cannot pass unnoticed.
    with torch.no_grad():
        for j in range(model.cfg.n_rungs):
            nn.init.normal_(model.readout.lora_B[j].weight, std=0.05 * (j + 1))

    with adapters(model, True):
        joint = model.velocity_multi_exit(A_tau, tau, obs_, E_V=2, exits=exits)
        separate = []
        for j, E_B in enumerate(exits):
            with adapter_slot(model, j):
                separate.append(
                    model.velocity(A_tau, tau, obs_, E_V=2, E_B=E_B, adapters=True)
                )

    assert len(joint) == len(exits)
    for a, b in zip(joint, separate):
        assert torch.allclose(a, b, atol=1e-6), "multi-exit diverged from a per-depth pass"


def test_only_adapters_are_trainable(model):
    """``theta`` is frozen throughout (SS2.5)."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(".lora_A." in n or ".lora_B." in n for n in trainable)


# -- conditioning -----------------------------------------------------------


def test_velocity_depends_on_the_observation(model):
    """The anti-collapse regulariser of eq. 16 exists because a verifier that
    ignores ``o*`` accepts unconditionally.  The architecture must at least
    make observation-sensitivity possible."""
    cfg = model.cfg
    A = torch.randn(1, cfg.H, cfg.d_a)
    tau = torch.tensor([0.6])
    a = model.velocity(A, tau, obs(cfg, seed=1), cfg.L_V, cfg.L_B, False)
    b = model.velocity(A, tau, obs(cfg, seed=2), cfg.L_V, cfg.L_B, False)
    assert not torch.allclose(a, b)


def test_lora_coverage_cannot_exceed_depth():
    with pytest.raises(ValueError, match="cannot exceed model depth"):
        TinyPi0(TinyPi0Config(L_B=4, lora_layers_B=9))
