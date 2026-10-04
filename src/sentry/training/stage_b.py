from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from sentry.config import SentryConfig
from sentry.core import acceptance
from sentry.core.renoise import interpolate, reconstruct, to_model_time
from sentry.core.types import ChannelSpec, Thresholds
from sentry.models.lora import adapters as adapters_ctx
from sentry.training.datagen import StageBBatch
from sentry.training.losses import multi_exit_loss, stage_b_loss
from sentry.training.params import AdapterGroups, freeze_all_but, split_adapters
from sentry.training.rng import randn_like_ref

__all__ = ["StageBConfig", "StageBTrainer"]


@dataclass
class StageBConfig:
    E_V: int = 8
    lr: float = 1e-4
    total_steps: int = 60_000
    grad_clip: Optional[float] = 1.0
    multi_exit: bool = True
    delta_provisional: float = 1.0


class StageBTrainer:

    def __init__(
        self,
        model,
        cfg: SentryConfig,
        b_cfg: StageBConfig,
        spec: ChannelSpec,
        groups: Optional[AdapterGroups] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.b_cfg = b_cfg
        self.spec = spec

        self.module = model if isinstance(model, torch.nn.Module) else model.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError(
                f"{type(model).__name__} is neither an nn.Module nor a wrapper "
                "exposing one as `.model`; Stage B cannot freeze or optimise it"
            )
        self.groups = groups or split_adapters(self.module)

        freeze_all_but(self.module, self.groups.params_B())
        self.opt = torch.optim.AdamW(self.groups.params_B(), lr=b_cfg.lr)
        self.step_idx = 0
        self.depths_seen: list[int] = []


    def sample_depth(self, rng: random.Random) -> int:
        lo, hi = self.cfg.p_depth
        E_B = rng.randint(lo, min(hi, self.model.L_B))
        self.depths_seen.append(E_B)
        return E_B


    def loss_on(
        self,
        batch: StageBBatch,
        rng: random.Random,
        generator: Optional[torch.Generator] = None,
        E_B: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        m = self.model
        fixed_E_B = E_B
        th = Thresholds(
            pos=self.b_cfg.delta_provisional, rot=self.b_cfg.delta_provisional
        )

        v_shal_all, v_full_all, d_all = [], [], []
        R_pos_all, R_neg_all = [], []
        depths: list[int] = []

        for i, obs in enumerate(batch.obs):
            A_hat = batch.A_hat[i]
            H_k = int(batch.H_k[i])
            E_B = self.sample_depth(rng) if fixed_E_B is None else fixed_E_B
            depths.append(E_B)

            tau_val = self.cfg.taus[rng.randrange(len(self.cfg.taus))]
            tau = torch.tensor([tau_val], device=A_hat.device, dtype=A_hat.dtype)
            eps = randn_like_ref(A_hat.shape, A_hat, generator)

            A_tau = interpolate(A_hat, tau, eps)
            tau_model = to_model_time(tau, self.cfg.tau_convention)

            with torch.no_grad(), adapters_ctx(self.module, False):
                v_full = m.velocity(
                    A_tau, tau_model, obs, m.L_V, m.L_B, adapters=False
                )

            v_shal = m.velocity(
                A_tau, tau_model, obs, self.b_cfg.E_V, E_B, adapters=True
            )
            R_shal = reconstruct(A_tau, tau, v_shal, self.cfg.tau_convention)

            d = acceptance.normalised_distances(R_shal, A_hat, self.spec, th, H_k)
            d_row = d.max(dim=0).values
            d_all.append(torch.nn.functional.pad(d_row, (0, A_hat.shape[0] - H_k)))

            v_shal_all.append(v_shal)
            v_full_all.append(v_full)

            obs_neg = batch.obs_neg[i]
            v_neg = m.velocity(
                A_tau, tau_model, obs_neg, self.b_cfg.E_V, E_B, adapters=True
            )
            R_neg = reconstruct(A_tau, tau, v_neg, self.cfg.tau_convention)
            R_pos_all.append(R_shal)
            R_neg_all.append(R_neg)

        loss, parts = stage_b_loss(
            v_shal=torch.cat(v_shal_all),
            v_full=torch.cat(v_full_all),
            d=torch.stack(d_all),
            h_star=batch.h_star,
            H_k=batch.H_k,
            R_pos=torch.stack(R_pos_all),
            R_neg=torch.stack(R_neg_all),
            lambda_m=self.cfg.lambda_m,
            lambda_s=self.cfg.lambda_s,
            m=self.cfg.margin_m,
            eta=self.cfg.eta,
        )
        parts["E_B"] = sum(depths) / len(depths) if depths else 0.0
        parts["E_B_min"] = min(depths) if depths else 0
        parts["E_B_max"] = max(depths) if depths else 0
        return loss, parts


    def loss_on_multi_exit(
        self,
        batch: StageBBatch,
        rng: random.Random,
        u: float,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, dict]:
        m = self.model
        cfg = self.cfg
        exits = [r.E_B for r in cfg.ladder]
        th = Thresholds(pos=self.b_cfg.delta_provisional, rot=self.b_cfg.delta_provisional)

        J = len(exits)
        v_shal: list[list[Tensor]] = [[] for _ in range(J)]
        v_full_all: list[Tensor] = []
        d_all: list[list[Tensor]] = [[] for _ in range(J)]
        R_pos: list[list[Tensor]] = [[] for _ in range(J)]
        R_neg: list[list[Tensor]] = [[] for _ in range(J)]

        for i, obs in enumerate(batch.obs):
            A_hat = batch.A_hat[i]
            H_k = int(batch.H_k[i])

            tau_val = cfg.taus[rng.randrange(len(cfg.taus))]
            tau = torch.tensor([tau_val], device=A_hat.device, dtype=A_hat.dtype)
            eps = randn_like_ref(A_hat.shape, A_hat, generator)

            A_tau = interpolate(A_hat, tau, eps)
            tau_model = to_model_time(tau, cfg.tau_convention)

            cached = getattr(batch, "v_full", None)
            if cached is not None:
                v_full = cached[i].to(A_hat.device)
            else:
                with torch.no_grad(), adapters_ctx(self.module, False):
                    v_full = m.velocity(A_tau, tau_model, obs, m.L_V, m.L_B, adapters=False)
            v_full_all.append(v_full)

            vs = m.velocity_multi_exit(A_tau, tau_model, obs, self.b_cfg.E_V, exits)
            vs_neg = m.velocity_multi_exit(
                A_tau, tau_model, batch.obs_neg[i], self.b_cfg.E_V, exits
            )

            for j in range(J):
                R = reconstruct(A_tau, tau, vs[j], cfg.tau_convention)
                d = acceptance.normalised_distances(R, A_hat, self.spec, th, H_k)
                d_row = d.max(dim=0).values
                d_all[j].append(
                    torch.nn.functional.pad(d_row, (0, A_hat.shape[0] - H_k))
                )
                v_shal[j].append(vs[j])
                R_pos[j].append(R)
                R_neg[j].append(reconstruct(A_tau, tau, vs_neg[j], cfg.tau_convention))

        per_rung: list[Tensor] = []
        parts: dict = {}
        v_full_cat = torch.cat(v_full_all)
        for j in range(J):
            loss_j, parts_j = stage_b_loss(
                v_shal=torch.cat(v_shal[j]),
                v_full=v_full_cat,
                d=torch.stack(d_all[j]),
                h_star=batch.h_star,
                H_k=batch.H_k,
                R_pos=torch.stack(R_pos[j]),
                R_neg=torch.stack(R_neg[j]),
                lambda_m=cfg.lambda_m,
                lambda_s=cfg.lambda_s,
                m=cfg.margin_m,
                eta=cfg.eta,
            )
            per_rung.append(loss_j)
            for k, v in parts_j.items():
                parts[f"r{j}_{k}"] = v

        total, agg = multi_exit_loss(per_rung, cfg.w, u, cfg.u_enable)
        parts.update(agg)
        parts["u"] = u
        parts["exits"] = tuple(exits)
        return total, parts

    def step(
        self,
        batch: StageBBatch,
        rng: random.Random,
        generator: Optional[torch.Generator] = None,
        u: Optional[float] = None,
    ) -> dict:
        self.opt.zero_grad(set_to_none=True)
        if self.b_cfg.multi_exit:
            if u is None:
                u = min(1.0, self.step_idx / max(self.b_cfg.total_steps, 1))
            loss, parts = self.loss_on_multi_exit(batch, rng, u, generator)
        else:
            loss, parts = self.loss_on(batch, rng, generator)
        loss.backward()
        if self.b_cfg.grad_clip is not None:
            parts["grad_norm"] = float(
                torch.nn.utils.clip_grad_norm_(
                    self.groups.params_B(), self.b_cfg.grad_clip
                )
            )
        self.opt.step()
        self.step_idx += 1
        parts["step"] = self.step_idx
        return parts


    def state_dict(self) -> dict:
        return {
            "step_idx": self.step_idx,
            "adapters": {n: p.detach().clone() for n, p in self.groups.delta_B},
            "optimiser": self.opt.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.step_idx = sd["step_idx"]
        by_name = dict(self.groups.delta_B)
        with torch.no_grad():
            for n, v in sd["adapters"].items():
                by_name[n].copy_(v)
        self.opt.load_state_dict(sd["optimiser"])
