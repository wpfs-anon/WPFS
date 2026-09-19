"""Stage B -- shallow verification (SS2.5.2).

Trains ``Delta_B`` on backbone layers ``0..E_B-1`` -- **both** experts'
attention and MLP projections -- plus a LoRA on the action read-out.
``Delta_V`` from Stage A is frozen.

    ``L_B = L_dist + lambda_m * L_marg + lambda_s * L_sens``

Four things SS2.5.2 is specific about, each of which fails silently if ignored:

1. **``tau ~ T``, the deployed verification schedule, not ``U(0,1)``.**  The
   shallow mode only ever gets queried at the two timesteps in ``T``; spending
   capacity on the rest of the path buys nothing and dilutes what matters.
2. **``E_B ~ p_depth``**, a curriculum over ``U{4..14}``, so "a single
   ``Delta_B`` serves a *range* of depths, which is what makes the depth
   cascade of Sec. 2.6 possible without training one adapter per depth".
3. **Early exit, not interior skipping** -- a contiguous prefix "preserves the
   residual-stream statistics the surviving layers were trained under".
4. **Matched pairs for eq. 16**, sharing ``A_hat``, ``tau`` and ``eps``.

The teacher (``v_full``) is under ``sg[.]`` and therefore ``no_grad``.
"""

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
    """Table 2: Stage B runs 60k steps, LoRA-only."""
    grad_clip: Optional[float] = 1.0
    multi_exit: bool = True
    """Follow the current SS3.5.2 recipe rather than the earlier depth sampler.

    ``True``  -- one pass to ``E_max`` supervises every rung of a **fixed**
    ladder at once, aggregated by eq. 18.  "No depth is sampled at this stage.
    Depth enters in the loss, not in the data."

    ``False`` -- the superseded recipe: draw ``E_B ~ p_depth`` per sample and
    train a single read-out.  Kept as an ablation axis, because it is what an
    earlier draft specified and the comparison is worth reporting.
    """
    delta_provisional: float = 1.0
    """Paper defect **D3**.

    Eq. 15 needs ``d_h``, hence ``delta_pos``/``delta_rot``; eq. 12 produces
    those only *after* training.  Circular.  Train against this provisional
    value, calibrate afterwards, and optionally re-calibrate.  This is never a
    deployment threshold.
    """


class StageBTrainer:
    """One optimiser step of ``L_B``."""

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

        # ``model`` is a VLABackend, which the protocol does not require to be an
        # nn.Module -- a real pi_0 arrives wrapped in an adapter that owns the
        # network rather than being one.  Parameter surgery needs the network.
        self.module = model if isinstance(model, torch.nn.Module) else model.model
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError(
                f"{type(model).__name__} is neither an nn.Module nor a wrapper "
                "exposing one as `.model`; Stage B cannot freeze or optimise it"
            )
        self.groups = groups or split_adapters(self.module)

        # Delta_V frozen; only Delta_B (backbone + read-out) trains.
        freeze_all_but(self.module, self.groups.params_B())
        self.opt = torch.optim.AdamW(self.groups.params_B(), lr=b_cfg.lr)
        self.step_idx = 0
        self.depths_seen: list[int] = []

    # -- depth curriculum -------------------------------------------------

    def sample_depth(self, rng: random.Random) -> int:
        """``E_B ~ p_depth`` over the inclusive support of Table 2's ``U{4..14}``."""
        lo, hi = self.cfg.p_depth
        E_B = rng.randint(lo, min(hi, self.model.L_B))
        self.depths_seen.append(E_B)
        return E_B

    # -- one step ---------------------------------------------------------

    def loss_on(
        self,
        batch: StageBBatch,
        rng: random.Random,
        generator: Optional[torch.Generator] = None,
        E_B: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        """Compute ``L_B`` without stepping -- used by preflight checks.

        ``E_B`` is drawn **per sample**, not per batch.  SS2.5.2 places "sample
        a depth ``E_B ~ p_depth``" as step 5 of the per-sample recipe, and the
        distinction is the whole point of the curriculum: "a single ``Delta_B``
        then serves a *range* of depths, which is what makes the depth cascade
        of Sec. 2.6 possible without training one adapter per depth."  One draw
        per batch makes every gradient step a statement about a single depth,
        and the range is then covered only in expectation across steps rather
        than within each one.  It costs nothing here -- each element is already
        its own forward pass.

        Passing ``E_B`` explicitly pins every element to one depth, which is
        what the preflight ablations need in order to hold depth fixed.
        """
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

            # tau from the DEPLOYED schedule T, not U(0,1)  (SS2.5.2 step 4).
            tau_val = self.cfg.taus[rng.randrange(len(self.cfg.taus))]
            tau = torch.tensor([tau_val], device=A_hat.device, dtype=A_hat.dtype)
            eps = randn_like_ref(A_hat.shape, A_hat, generator)

            A_tau = interpolate(A_hat, tau, eps)
            tau_model = to_model_time(tau, self.cfg.tau_convention)

            # Teacher: full depth, adapters OFF, no grad (it sits under sg[.]).
            with torch.no_grad(), adapters_ctx(self.module, False):
                v_full = m.velocity(
                    A_tau, tau_model, obs, m.L_V, m.L_B, adapters=False
                )

            # Student: truncated, adapters ON.  Delta_V is enabled but frozen.
            v_shal = m.velocity(
                A_tau, tau_model, obs, self.b_cfg.E_V, E_B, adapters=True
            )
            R_shal = reconstruct(A_tau, tau, v_shal, self.cfg.tau_convention)

            d = acceptance.normalised_distances(R_shal, A_hat, self.spec, th, H_k)
            d_row = d.max(dim=0).values                       # conservative over T
            d_all.append(torch.nn.functional.pad(d_row, (0, A_hat.shape[0] - H_k)))

            v_shal_all.append(v_shal)
            v_full_all.append(v_full)

            # Equation 16: the matched partner shares A_hat, tau AND eps.
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

    # -- the current SS3.5.2 recipe: one pass, every rung ------------------

    def loss_on_multi_exit(
        self,
        batch: StageBBatch,
        rng: random.Random,
        u: float,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, dict]:
        r"""Equation 18 over a fixed ladder, from one forward pass per sample.

        The three ways this differs from :meth:`loss_on`, all of them the
        current SS3.5.2 text rather than the earlier draft:

        1. **No depth is sampled.**  The ladder is fixed and every rung is
           supervised on every sample.
        2. **One pass to** ``E_max`` **exposes all of them**, because "layer
           ``l`` depends only on layers ``< l``" -- measured at 2.2x cheaper
           than evaluating the three rungs separately.
        3. **The rungs are aggregated by** ``sum_j w_j c_j(u) L^(j)``, with
           ``w`` matching expected deployment load and ``c_j(u)`` bringing the
           shallow rungs in only once the shared trunk has stabilised.

        ``u`` is the training fraction, and it is a real argument rather than a
        counter: the curriculum is defined on it, and reading it from
        ``step_idx`` would silently change meaning if training is resumed with a
        different ``total_steps``.
        """
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

            # tau from the DEPLOYED schedule T, not U(0,1)  (SS3.5.2 step 4).
            tau_val = cfg.taus[rng.randrange(len(cfg.taus))]
            tau = torch.tensor([tau_val], device=A_hat.device, dtype=A_hat.dtype)
            eps = randn_like_ref(A_hat.shape, A_hat, generator)

            A_tau = interpolate(A_hat, tau, eps)
            tau_model = to_model_time(tau, cfg.tau_convention)

            # Teacher: full depth, adapters OFF, no grad (it sits under sg[.]).
            # SS3.5.2 materialises this offline -- "Stage B then never performs a
            # full-depth forward pass" -- so a cached value is used when the
            # datagen supplied one, and only recomputed here otherwise.
            cached = getattr(batch, "v_full", None)
            if cached is not None:
                v_full = cached[i].to(A_hat.device)
            else:
                with torch.no_grad(), adapters_ctx(self.module, False):
                    v_full = m.velocity(A_tau, tau_model, obs, m.L_V, m.L_B, adapters=False)
            v_full_all.append(v_full)

            # One pass to E_max, every rung read out (SS3.5.2).
            vs = m.velocity_multi_exit(A_tau, tau_model, obs, self.b_cfg.E_V, exits)
            vs_neg = m.velocity_multi_exit(
                A_tau, tau_model, batch.obs_neg[i], self.b_cfg.E_V, exits
            )

            for j in range(J):
                R = reconstruct(A_tau, tau, vs[j], cfg.tau_convention)
                d = acceptance.normalised_distances(R, A_hat, self.spec, th, H_k)
                d_row = d.max(dim=0).values                      # conservative over T
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

    # -- checkpointing ----------------------------------------------------

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
