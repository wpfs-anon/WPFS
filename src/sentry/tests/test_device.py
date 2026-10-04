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


def test_device_property_follows_the_weights():
    model = _model()
    assert model.device.type == "cpu"
    assert model.to(META).device.type == "meta"


def test_plan_allocates_noise_on_the_models_device():
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
    from sentry.models.toy_pi0 import _timestep_embedding

    emb = _timestep_embedding(torch.tensor([0.5], device=META), 16)
    assert emb.device.type == "meta"


def test_mismatched_input_is_still_an_error():
    model = _model().to(META)
    A_tau = torch.randn(1, model.H, model.d_a)
    tau = torch.tensor([0.5])
    with pytest.raises(RuntimeError, match="device"):
        model.velocity(A_tau, tau, _obs(model.cfg, META), model.L_V, model.L_B, False)


def _rig():
    env_cfg = MockReachConfig(H=6, img_size=16, max_steps=40)
    spec = fit_spec(env_cfg, n=32)
    cfg = SentryConfig(H=6, m_min=2, taus=(0.6, 0.9), p_depth=(1, 2),
                       ladder=SentryConfig().ladder[:1])
    return env_cfg, spec, cfg


def test_stage_a_batch_places_observations_before_planning():
    env_cfg, spec, cfg = _rig()
    model = _model().to(META)
    batch = make_stage_a_batch(
        model, cfg, spec, 2, env_cfg, random.Random(0), device=META
    )
    assert batch.A.device.type == "meta"
    assert all(o.images.device.type == "meta" for o in batch.obs)
    assert all(o.state.device.type == "meta" for o in batch.obs)


def test_stage_b_batch_moves_both_halves_of_the_matched_pair():
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
    env_cfg, spec, cfg = _rig()
    model = _model()
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=2),
        random.Random(0), device=torch.device("cpu"),
    )
    assert batch.A_hat.device.type == "cpu"
    assert all(o.images.device.type == "cpu" for o in batch.obs)


def test_batch_index_tensors_stay_on_cpu():
    env_cfg, spec, cfg = _rig()
    model = _model()
    batch = make_stage_b_batch(
        model, cfg, spec, 2, SampleGenConfig(env=env_cfg, min_H_k=2), random.Random(0)
    )
    assert batch.H_k.device.type == "cpu"
    assert batch.h_star.device.type == "cpu"


def test_cpu_generator_with_a_moved_tensor():
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
