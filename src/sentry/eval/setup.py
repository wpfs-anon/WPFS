from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sentry.config import SentryConfig
from sentry.core.types import ChannelSpec, Observation, Thresholds
from sentry.envs.mock_env import (
    MockReachConfig,
    fit_spec,
    make_planner,
)
from sentry.eval.calibrate import SplitCalibration, build_records, calibrate_split
from sentry.eval.harness import SampleGenConfig, generate
from sentry.models.oracle import OracleBackend, OracleConfig, exponential_truncation

__all__ = ["Rig", "build_rig"]


@dataclass
class Rig:

    cfg: SentryConfig
    env_cfg: MockReachConfig
    spec: ChannelSpec
    planner: Callable[[Observation], torch.Tensor]
    thresholds: Thresholds
    calibration: SplitCalibration
    oracle_cfg: OracleConfig

    def backend(self) -> OracleBackend:
        return OracleBackend(self.planner, self.oracle_cfg)


def build_rig(
    cfg: Optional[SentryConfig] = None,
    env_cfg: Optional[MockReachConfig] = None,
    alpha: float = 0.05,
    n_calibration: int = 600,
    sigma_0: float = 0.5,
    seed: int = 0,
) -> Rig:
    cfg = cfg or SentryConfig(H=50)
    env_cfg = env_cfg or MockReachConfig(H=cfg.H, max_steps=200)

    spec = fit_spec(env_cfg)
    planner = make_planner(env_cfg, spec)
    oracle_cfg = OracleConfig(
        H=cfg.H,
        d_a=env_cfg.d_a,
        truncation_sigma=exponential_truncation(sigma_0, 4.0),
        seed=seed,
    )

    def backend() -> OracleBackend:
        return OracleBackend(planner, oracle_cfg)

    samples = generate(
        backend(), cfg, spec, n=n_calibration, gen_cfg=SampleGenConfig(env=env_cfg), seed=seed + 1
    )
    records = build_records(
        backend(), cfg, spec, samples, generator=torch.Generator().manual_seed(seed)
    )
    calibration = calibrate_split(records, alpha=alpha, mode=cfg.calibration_mode, seed=seed)

    return Rig(
        cfg=cfg,
        env_cfg=env_cfg,
        spec=spec,
        planner=planner,
        thresholds=calibration.thresholds,
        calibration=calibration,
        oracle_cfg=oracle_cfg,
    )
