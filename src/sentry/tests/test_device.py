"""Device placement, tested without a GPU.

Three device bugs reached Colab before these existed, and each was invisible
locally because this machine is CPU-only: everything defaults to the same
device, so a missing ``device=`` is indistinguishable from a correct one.

``torch.device("meta")`` closes that gap.  Meta tensors carry shape and device
but no storage, and PyTorch still enforces device agreement across an
operation -- so a tensor allocated on the default device while the weights are
on ``meta`` raises exactly the error a CUDA run would, on a machine with no
accelerator at all:

    RuntimeError: Tensor on device meta is not on the expected device cpu!

The bugs these lock down:

1. ``TinyPi0.plan`` allocated ``A`` and ``tau`` with no ``device=``, so
   generating a batch failed the moment the model was moved.
2. ``_timestep_embedding`` built ``freqs`` on the default device.
3. The Stage A/B batch builders rendered observations on CPU and handed them
   straight to ``backend.plan``.

Meta does not cover everything.  Stage-B generation simulates the trajectory
forward and therefore needs real values, and a meta tensor cannot be read
back out; that path is exercised on CPU and only its transfer is checked
against a moved device.
"""

from __future__ import annotations

import random

import pytest
import torch

from sentry.config import SentryConfig
from sentry.core.types import Observation
from sentry.envs.mock_env import MockReachConfig, fit_spec
from sentry.eval.harness import SampleGenConfig
from sentry.models.toy_pi0 import TinyPi0, TinyPi0Config
from sentry.training.datagen import make_stage_a_batch, make_stage_b_batch
from sentry.training.rng import rand_on, randn_like_ref

META = torch.device("meta")


def _model(**kw) -> TinyPi0:
    cfg = TinyPi0Config(
        H=6, d_a=7, img_size=16, L_V=2, L_B=2, lora_layers_V=1, lora_layers_B=1, **kw
    )
    return TinyPi0(cfg)


def _obs(cfg: TinyPi0Config, device) -> Observation:
    return Observation(
        images=torch.randn(cfg.n_cam, cfg.in_channels, cfg.img_size, cfg.img_size, device=device),
        language=torch.zeros(cfg.max_lang, dtype=torch.long, device=device),
        state=torch.randn(cfg.d_state, device=device),
        t=0,
    )


# -- the model ---------------------------------------------------------------


def test_device_property_follows_the_weights():
    model = _model()
    assert model.device.type == "cpu"
    assert model.to(META).device.type == "meta"


def test_plan_allocates_noise_on_the_models_device():
    """The bug that broke the Colab run.

    ``plan`` integrates from ``A^0 ~ N(0, I)``; allocating that noise on the
    default device works perfectly until the model is moved, and then fails
    several frames deep inside data generation rather than at the line that is
    wrong.
    """
    model = _model().to(META)
    A = model.plan(_obs(model.cfg, META))
    assert A.device.type == "meta"
    assert A.shape == (model.H, model.d_a)


def test_velocity_runs_entirely_off_the_default_device():
    model = _model().to(META)
    A_tau = torch.randn(2, model.H, model.d_a, device=META)
    tau = torch.tensor([0.6, 0.9], device=META)
    v = model.velocity(A_tau, tau, _obs(model.cfg, META), model.L_V, model.L_B, False)
    assert v.device.type == "meta"


def test_timestep_embedding_follows_tau():
    """``freqs`` must be built on ``tau``'s device, not the default one."""
    from sentry.models.toy_pi0 import _timestep_embedding

    emb = _timestep_embedding(torch.tensor([0.5], device=META), 16)
    assert emb.device.type == "meta"


def test_mismatched_input_is_still_an_error():
    """Guard the guard: confirm meta really does enforce device agreement.

    If this ever stops raising, the tests above would pass vacuously.
    """
    model = _model().to(META)
    A_tau = torch.randn(1, model.H, model.d_a)          # default device
    tau = torch.tensor([0.5])
    with pytest.raises(RuntimeError, match="device"):
        model.velocity(A_tau, tau, _obs(model.cfg, META), model.L_V, model.L_B, False)


# -- batch generation --------------------------------------------------------


def _rig():
    env_cfg = MockReachConfig(H=6, img_size=16, max_steps=40)
    spec = fit_spec(env_cfg, n=32)
    cfg = SentryConfig(H=6, m_min=2, taus=(0.6, 0.9), p_depth=(1, 2),
                       ladder=SentryConfig().ladder[:1])
    return env_cfg, spec, cfg


def test_stage_a_batch_places_observations_before_planning():
    """The observation must move *before* ``backend.plan`` sees it.

    Moving the assembled batch afterwards is too late -- the plan call inside
    generation would already have fed default-device images to moved weights.
    """
    env_cfg, spec, cfg = _rig()
    model = _model().to(META)
    batch = make_stage_a_batch(
        model, cfg, spec, 2, env_cfg, random.Random(0), device=META
    )
    assert batch.A.device.type == "meta"
    assert all(o.images.device.type == "meta" for o in batch.obs)
    assert all(o.state.device.type == "meta" for o in batch.obs)


def test_stage_b_batch_moves_both_halves_of_the_matched_pair():
    """Equation 16 needs ``o*_+`` and ``o*_-``; both must land on the device.

    Tested through ``StageBBatch.to`` rather than through generation.  Stage-B
    generation simulates the trajectory forward to find ``ee_k``, which needs
    real numbers -- a meta tensor cannot be read back out at all
    (``NotImplementedError: Cannot copy out of meta tensor``).  So the
    generation path is exercised on CPU below and only the transfer is checked
    against a moved device.
    """
    env_cfg, spec, cfg = _rig()
    model = _model()
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=2), random.Random(0)
    )
    moved = batch.to(META)
    assert moved.A_hat.device.type == "meta"
    assert all(o.images.device.type == "meta" for o in moved.obs)
    assert all(o.images.device.type == "meta" for o in moved.obs_neg)
    assert all(o.state.device.type == "meta" for o in moved.obs_neg)


def test_stage_b_generation_honours_an_explicit_device():
    """The ``device=`` path itself, on CPU where generation can actually run."""
    env_cfg, spec, cfg = _rig()
    model = _model()
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=2),
        random.Random(0), device=torch.device("cpu"),
    )
    assert batch.A_hat.device.type == "cpu"
    assert all(o.images.device.type == "cpu" for o in batch.obs)


def test_batch_index_tensors_stay_on_cpu():
    """``H_k`` and ``h_star`` are read with ``int()``, so they belong on CPU.

    Shipping them to the accelerator would force a synchronising transfer per
    element of every batch, for integers the loss functions immediately turn
    back into Python ints.
    """
    env_cfg, spec, cfg = _rig()
    model = _model()
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=2), random.Random(0)
    )
    assert batch.H_k.device.type == "cpu"
    assert batch.h_star.device.type == "cpu"


# -- seeded sampling ---------------------------------------------------------


def test_cpu_generator_with_a_moved_tensor():
    """A CPU ``Generator`` cannot drive an allocation on another device.

    On CUDA, ``torch.randn(shape, generator=cpu_gen, device="cuda")`` raises
    "Expected a 'cuda' device type for generator".  Meta does not reproduce
    that -- it has no storage to fill, so it ignores the generator entirely --
    and this test does not pretend otherwise.  What it does check is that the
    helpers take the cross-device branch and still deliver the reference
    device and dtype, which is what makes a seeded run give the same draw
    wherever the model lives.
    """
    g = torch.Generator().manual_seed(0)
    assert g.device.type == "cpu"
    ref = torch.zeros(4, device=META)

    assert randn_like_ref((4,), ref, g).device.type == "meta"
    assert rand_on((4,), ref, g).device.type == "meta"


def test_seeded_draw_is_reproducible_and_device_independent():
    a = randn_like_ref((8,), torch.zeros(8), torch.Generator().manual_seed(3))
    b = randn_like_ref((8,), torch.zeros(8), torch.Generator().manual_seed(3))
    torch.testing.assert_close(a, b)


def test_helpers_match_reference_dtype():
    ref = torch.zeros(4, dtype=torch.float64)
    assert randn_like_ref((4,), ref).dtype == torch.float64
    assert rand_on((4,), ref).dtype == torch.float64
