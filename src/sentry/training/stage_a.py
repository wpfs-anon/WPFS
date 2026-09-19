"""Stage A -- shallow perception (SS2.5.1).

Trains ``Delta_V`` only, on encoder layers ``0..E_V-1``.  Nothing else.

The objective (eq. 13) combines a representation term with a behavioural one::

    L_A = (1 - cos(z_shal, z_full)) + lambda_A(u) * ||v(.|z_shal) - sg[v(.|z_full)]||^2

"Matching intermediate visual tokens exactly is both unnecessary and, at
``E_V << L_V``, likely infeasible; what matters is that the truncated encoder
preserves the information the backbone uses."

Three details in SS2.5.1 that this module makes structural rather than
incidental:

1. ``L_beh`` runs the **frozen full-depth backbone** on both branches.  Stage A
   truncates the encoder; the backbone stays at ``L_B``, and its adapters do
   not exist yet.
2. "both branches share the same ``(tau, eps)`` draw" -- otherwise the two
   velocities differ because of noise, not because of the encoder, and the
   term measures nothing.
3. The teacher is under ``sg[.]``, so it runs in ``no_grad`` -- worth roughly
   half the activation memory, which on a 16 GB Colab T4 is the difference
   between fitting and not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from sentry.config import SentryConfig
from sentry.core.renoise import interpolate, reconstruct, to_model_time
from sentry.core.types import ChannelSpec, DepthRung
from sentry.models.lora import adapters as adapters_ctx
from sentry.training.datagen import StageABatch
from sentry.training.losses import stage_a_loss
from sentry.training.params import AdapterGroups, freeze_all_but, split_adapters
from sentry.training.rng import rand_on, randn_like_ref

__all__ = ["StageAConfig", "StageATrainer"]


@dataclass
class StageAConfig:
    E_V: int = 8
    """Encoder truncation depth for the student branch."""
    lambda_max: float = 1.0
    """``lambda_A^max``, the ceiling of the anneal."""
    lr: float = 1e-4
    total_steps: int = 20_000
    """Table 2: Stage A runs 20k steps, LoRA-only.

    Measured, that is roughly ten times too many: two independent runs on
    ``pi0_libero`` reached their best held-out ``L_beh`` at step 1250 and 1750
    and did not improve afterwards.
    """
    grad_clip: Optional[float] = 1.0

    real_channels_only: bool = True
    """Score ``L_beh`` on the channels that carry information.

    Equation 13 averages over all ``d_a``; on ``pi0_libero`` that is 7 real
    channels among 32, and the 25 padded ones are a zero both branches emit for
    free.  Measured dilution: ``4.46x``.  Set ``False`` for the paper-literal
    reading.
    """

    lambda_grip: float = 1.0
    """Weight on the gripper sign term.  ``0.0`` reproduces eq. 13 as written.

    The acceptance rule of SS3.4.3 reads the gripper by **sign**, but eq. 13
    scores it by squared error like any continuous channel.  Trained without
    this term to a 96% reduction in ``L_beh``, the shallow branch still flipped
    the gripper's sign on 11% of chunk positions.
    """
    grip_margin: float = 0.1


class StageATrainer:
    """One optimiser step of equation 13."""

    def __init__(
        self,
        model,
        cfg: SentryConfig,
        a_cfg: StageAConfig,
        groups: Optional[AdapterGroups] = None,
        spec: Optional[ChannelSpec] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.a_cfg = a_cfg
        self.spec = spec
        """Channel semantics.  Without it ``L_beh`` is scored over every padded
        dimension and the gripper term cannot be formed -- i.e. eq. 13 exactly as
        written, which is available but is not the default."""

        # A VLABackend need not be an nn.Module -- a real pi_0 arrives wrapped in
        # an adapter that owns the network rather than being one.
        self.module = model if isinstance(model, torch.nn.Module) else model.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError(
                f"{type(model).__name__} is neither an nn.Module nor a wrapper "
                "exposing one as `.model`; Stage A cannot freeze or optimise it"
            )
        self.groups = groups or split_adapters(self.module)

        # "each of which freezes everything except a small adapter set"
        freeze_all_but(self.module, self.groups.params_V())
        self.opt = torch.optim.AdamW(self.groups.params_V(), lr=a_cfg.lr)
        self.step_idx = 0

    # -- one step ---------------------------------------------------------

    def loss_on(
        self, batch: StageABatch, generator: Optional[torch.Generator] = None
    ) -> tuple[Tensor, dict]:
        """Compute equation 13 without stepping -- used by preflight checks."""
        m = self.model
        u = min(1.0, self.step_idx / max(self.a_cfg.total_steps, 1))

        z_shal_all, z_full_all, v_shal_all, v_full_all = [], [], [], []
        R_shal_all, R_full_all = [], []

        for i, obs in enumerate(batch.obs):
            A = batch.A[i]

            # ONE draw of (tau, eps), shared by both branches (SS2.5.1).
            # tau ~ U(0,1) here: Stage A supervises the whole path, unlike
            # Stage B which draws from the deployed schedule T.
            tau = rand_on((1,), A, generator)
            eps = randn_like_ref(A.shape, A, generator)
            A_tau = interpolate(A, tau, eps)
            tau_model = to_model_time(tau, self.cfg.tau_convention)

            # Teacher: full encoder, full backbone, adapters OFF, no grad.
            with torch.no_grad(), adapters_ctx(self.module, False):
                z_full = m.encode(obs.images, m.L_V)
                v_full = m.velocity_from_tokens(A_tau, tau_model, z_full, obs, m.L_B)

            # Student: truncated encoder WITH Delta_V, then the same frozen
            # full-depth backbone with adapters OFF.  Gradient reaches Delta_V
            # through z_shal alone.
            with adapters_ctx(self.module, True):
                z_shal = m.encode(obs.images, self.a_cfg.E_V)
            with adapters_ctx(self.module, False):
                v_shal = m.velocity_from_tokens(A_tau, tau_model, z_shal, obs, m.L_B)

            z_shal_all.append(z_shal)
            z_full_all.append(z_full)
            v_shal_all.append(v_shal)
            v_full_all.append(v_full)

            # eq. 9's endpoint, which is where the gripper's *sign* lives.  Both
            # branches share A_tau and tau, so this is the velocity difference
            # scaled by (1-tau) -- but the sign test is on the absolute value,
            # not the difference, so it has to be formed explicitly.
            if self.a_cfg.lambda_grip and self.spec is not None:
                R_shal_all.append(
                    reconstruct(A_tau, tau, v_shal, self.cfg.tau_convention)
                )
                R_full_all.append(
                    reconstruct(A_tau, tau, v_full, self.cfg.tau_convention)
                )

        use_spec = self.spec is not None
        return stage_a_loss(
            z_shal=torch.cat(z_shal_all),
            z_full=torch.cat(z_full_all),
            v_shal=torch.cat(v_shal_all),
            v_full=torch.cat(v_full_all),
            u=u,
            lambda_max=self.a_cfg.lambda_max,
            channels=(
                self.spec.real_channels
                if use_spec and self.a_cfg.real_channels_only
                else None
            ),
            R_shal=torch.cat(R_shal_all) if R_shal_all else None,
            R_full=torch.cat(R_full_all) if R_full_all else None,
            grip=self.spec.grip if use_spec else None,
            lambda_grip=self.a_cfg.lambda_grip if use_spec else 0.0,
            grip_margin=self.a_cfg.grip_margin,
        )

    def step(
        self, batch: StageABatch, generator: Optional[torch.Generator] = None
    ) -> dict:
        self.opt.zero_grad(set_to_none=True)
        loss, parts = self.loss_on(batch, generator)
        loss.backward()
        if self.a_cfg.grad_clip is not None:
            parts["grad_norm"] = float(
                torch.nn.utils.clip_grad_norm_(
                    self.groups.params_V(), self.a_cfg.grad_clip
                )
            )
        self.opt.step()
        self.step_idx += 1
        parts["step"] = self.step_idx
        return parts

    # -- checkpointing ----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step_idx": self.step_idx,
            "adapters": {n: p.detach().clone() for n, p in self.groups.delta_V},
            "optimiser": self.opt.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.step_idx = sd["step_idx"]
        by_name = dict(self.groups.delta_V)
        with torch.no_grad():
            for n, v in sd["adapters"].items():
                by_name[n].copy_(v)
        self.opt.load_state_dict(sd["optimiser"])
