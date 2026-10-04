from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sentry.core.types import Observation, assert_shared_depth
from sentry.models.lora import (
    LoRALinear,
    MultiLoRALinear,
    adapter_slot,
    wrap_linear,
    wrap_linear_multi,
)
from sentry.models.lora import adapters as adapters_ctx

__all__ = ["TinyPi0Config", "TinyPi0"]


@dataclass(frozen=True)
class TinyPi0Config:

    L_V: int = 6
    d_vision: int = 64
    n_cam: int = 1
    img_size: int = 16
    patch: int = 4
    in_channels: int = 3

    L_B: int = 6
    d_vlm: int = 64
    d_act: int = 32
    n_heads: int = 4
    head_dim: int = 16
    mlp_ratio: int = 2

    vocab_size: int = 64
    max_lang: int = 8
    d_state: int = 8
    H: int = 8
    d_a: int = 7
    M: int = 10

    lora_layers_V: int = 4
    lora_layers_B: int = 5
    lora_rank: int = 16
    lora_rank_readout: int = 32
    lora_dropout: float = 0.05
    n_rungs: int = 3

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.head_dim


class _Mlp(nn.Module):
    def __init__(self, d: int, ratio: int) -> None:
        super().__init__()
        self.up = nn.Linear(d, d * ratio)
        self.down = nn.Linear(d * ratio, d)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))


class _EncoderBlock(nn.Module):

    def __init__(self, d: int, n_heads: int, head_dim: int, ratio: int) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        attn_dim = n_heads * head_dim
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q = nn.Linear(d, attn_dim)
        self.k = nn.Linear(d, attn_dim)
        self.v = nn.Linear(d, attn_dim)
        self.o = nn.Linear(attn_dim, d)
        self.mlp = _Mlp(d, ratio)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        B, L, _ = h.shape
        shape = (B, L, self.n_heads, self.head_dim)
        q = self.q(h).view(shape).transpose(1, 2)
        k = self.k(h).view(shape).transpose(1, 2)
        v = self.v(h).view(shape).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, L, -1)
        x = x + self.o(a)
        return x + self.mlp(self.norm2(x))


class _TwoExpertBlock(nn.Module):

    def __init__(self, cfg: TinyPi0Config) -> None:
        super().__init__()
        self.cfg = cfg
        dv, da, ad = cfg.d_vlm, cfg.d_act, cfg.attn_dim

        self.norm1_vlm, self.norm2_vlm = nn.LayerNorm(dv), nn.LayerNorm(dv)
        self.norm1_act, self.norm2_act = nn.LayerNorm(da), nn.LayerNorm(da)

        self.q_vlm, self.k_vlm, self.v_vlm = (nn.Linear(dv, ad) for _ in range(3))
        self.o_vlm = nn.Linear(ad, dv)
        self.q_act, self.k_act, self.v_act = (nn.Linear(da, ad) for _ in range(3))
        self.o_act = nn.Linear(ad, da)

        self.mlp_vlm = _Mlp(dv, cfg.mlp_ratio)
        self.mlp_act = _Mlp(da, cfg.mlp_ratio)

    def _heads(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)

    def forward(self, vlm: Tensor, act: Tensor) -> tuple[Tensor, Tensor]:
        hv, ha = self.norm1_vlm(vlm), self.norm1_act(act)
        n_vlm = hv.shape[1]

        q = torch.cat([self._heads(self.q_vlm(hv)), self._heads(self.q_act(ha))], dim=2)
        k = torch.cat([self._heads(self.k_vlm(hv)), self._heads(self.k_act(ha))], dim=2)
        v = torch.cat([self._heads(self.v_vlm(hv)), self._heads(self.v_act(ha))], dim=2)

        a = F.scaled_dot_product_attention(q, k, v)
        B, _, L, _ = a.shape
        a = a.transpose(1, 2).reshape(B, L, -1)

        vlm = vlm + self.o_vlm(a[:, :n_vlm])
        act = act + self.o_act(a[:, n_vlm:])

        vlm = vlm + self.mlp_vlm(self.norm2_vlm(vlm))
        act = act + self.mlp_act(self.norm2_act(act))
        return vlm, act


class TinyPi0(nn.Module):

    def __init__(self, cfg: TinyPi0Config = TinyPi0Config()) -> None:
        super().__init__()
        self.cfg = cfg
        self.L_V, self.L_B = cfg.L_V, cfg.L_B
        self.H, self.d_a, self.M = cfg.H, cfg.d_a, cfg.M

        if cfg.lora_layers_V > cfg.L_V or cfg.lora_layers_B > cfg.L_B:
            raise ValueError("LoRA layer coverage cannot exceed model depth")

        n_patch = (cfg.img_size // cfg.patch) ** 2
        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.d_vision, kernel_size=cfg.patch, stride=cfg.patch
        )
        self.vis_pos = nn.Parameter(torch.zeros(1, cfg.n_cam * n_patch, cfg.d_vision))
        nn.init.trunc_normal_(self.vis_pos, std=0.02)
        self.encoder = nn.ModuleList(
            _EncoderBlock(cfg.d_vision, cfg.n_heads, cfg.head_dim, cfg.mlp_ratio)
            for _ in range(cfg.L_V)
        )
        self.vis_proj = nn.Linear(cfg.d_vision, cfg.d_vlm)

        self.lang_embed = nn.Embedding(cfg.vocab_size, cfg.d_vlm)
        self.lang_pos = nn.Parameter(torch.zeros(1, cfg.max_lang, cfg.d_vlm))
        nn.init.trunc_normal_(self.lang_pos, std=0.02)

        self.state_proj = nn.Linear(cfg.d_state, cfg.d_act)
        self.action_proj = nn.Linear(cfg.d_a, cfg.d_act)
        self.action_pos = nn.Parameter(torch.zeros(1, cfg.H, cfg.d_act))
        nn.init.trunc_normal_(self.action_pos, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_act, cfg.d_act), nn.SiLU(), nn.Linear(cfg.d_act, cfg.d_act)
        )

        self.backbone = nn.ModuleList(_TwoExpertBlock(cfg) for _ in range(cfg.L_B))
        self.norm_out = nn.LayerNorm(cfg.d_act)
        self.readout = nn.Linear(cfg.d_act, cfg.d_a)

        self._install_adapters()
        self._freeze_base()


    def _install_adapters(self) -> None:
        cfg = self.cfg
        r, drop = cfg.lora_rank, cfg.lora_dropout

        for blk in self.encoder[: cfg.lora_layers_V]:
            for attr in ("q", "k", "v", "o"):
                wrap_linear(blk, attr, r=r, dropout=drop)
            wrap_linear(blk.mlp, "up", r=r, dropout=drop)
            wrap_linear(blk.mlp, "down", r=r, dropout=drop)

        for blk in self.backbone[: cfg.lora_layers_B]:
            for attr in (
                "q_vlm", "k_vlm", "v_vlm", "o_vlm",
                "q_act", "k_act", "v_act", "o_act",
            ):
                wrap_linear(blk, attr, r=r, dropout=drop)
            for mlp in (blk.mlp_vlm, blk.mlp_act):
                wrap_linear(mlp, "up", r=r, dropout=drop)
                wrap_linear(mlp, "down", r=r, dropout=drop)

        wrap_linear_multi(
            self, "readout", n_slots=cfg.n_rungs, r=cfg.lora_rank_readout, dropout=drop
        )

    def _freeze_base(self) -> None:
        for name, p in self.named_parameters():
            if ".lora_A." not in name and ".lora_B." not in name:
                p.requires_grad_(False)


    def encode(self, images: Tensor, E_V: int) -> Tensor:
        if not (1 <= E_V <= self.L_V):
            raise ValueError(f"E_V={E_V} outside [1, {self.L_V}]")
        if images.ndim != 4:
            raise ValueError(f"images must be (n_cam, C, H, W), got {tuple(images.shape)}")

        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(1, -1, self.cfg.d_vision)
        x = x + self.vis_pos[:, : x.shape[1]]

        for blk in self.encoder[:E_V]:
            x = blk(x)
        return self.vis_proj(x)

    def velocity(
        self,
        A_tau: Tensor,
        tau: Tensor,
        obs: Observation,
        E_V: int,
        E_B: int,
        adapters: bool,
    ) -> Tensor:
        if not (1 <= E_B <= self.L_B):
            raise ValueError(f"E_B={E_B} outside [1, {self.L_B}]")
        if A_tau.ndim != 3:
            raise ValueError(f"A_tau must be (K, H, d_a), got {tuple(A_tau.shape)}")

        with adapters_ctx(self, adapters):
            return self._velocity(A_tau, tau, obs, E_V, E_B)

    def _velocity(
        self, A_tau: Tensor, tau: Tensor, obs: Observation, E_V: int, E_B: int
    ) -> Tensor:
        vis = self.encode(obs.images, E_V)
        return self.velocity_from_tokens(A_tau, tau, vis, obs, E_B)

    def velocity_from_tokens(
        self,
        A_tau: Tensor,
        tau: Tensor,
        vis: Tensor,
        obs: Observation,
        E_B: int,
    ) -> Tensor:
        K = A_tau.shape[0]

        assert_shared_depth(E_B, E_B)

        vis = vis.expand(K, -1, -1) if vis.shape[0] == 1 else vis
        lang = self.lang_embed(obs.language.long()).unsqueeze(0)
        lang = lang + self.lang_pos[:, : lang.shape[1]]
        vlm = torch.cat([vis, lang.expand(K, -1, -1)], dim=1)

        state = self.state_proj(obs.state).view(1, 1, -1).expand(K, -1, -1)
        act = self.action_proj(A_tau) + self.action_pos
        act = act + self.time_mlp(_timestep_embedding(tau, self.cfg.d_act)).unsqueeze(1)
        act = torch.cat([state, act], dim=1)

        for blk in self.backbone[:E_B]:
            vlm, act = blk(vlm, act)

        return self.readout(self.norm_out(act[:, 1:]))

    def velocity_multi_exit(
        self,
        A_tau: Tensor,
        tau: Tensor,
        obs: Observation,
        E_V: int,
        exits: Sequence[int],
    ) -> list[Tensor]:
        exits = list(exits)
        if not exits:
            raise ValueError("need at least one exit depth")
        if exits != sorted(exits) or len(set(exits)) != len(exits):
            raise ValueError(f"exits must be strictly increasing, got {exits}")
        if exits[-1] > self.L_B:
            raise ValueError(f"deepest exit {exits[-1]} exceeds L_B={self.L_B}")
        if len(exits) > self.cfg.n_rungs:
            raise ValueError(
                f"{len(exits)} exits but only {self.cfg.n_rungs} read-out adapters; "
                "rung j selects adapter j, so the ladder cannot be longer"
            )

        K = A_tau.shape[0]
        vis = self.encode(obs.images, E_V)
        vis = vis.expand(K, -1, -1) if vis.shape[0] == 1 else vis
        lang = self.lang_embed(obs.language.long()).unsqueeze(0)
        lang = lang + self.lang_pos[:, : lang.shape[1]]
        vlm = torch.cat([vis, lang.expand(K, -1, -1)], dim=1)

        state = self.state_proj(obs.state).view(1, 1, -1).expand(K, -1, -1)
        act = self.action_proj(A_tau) + self.action_pos
        act = act + self.time_mlp(_timestep_embedding(tau, self.cfg.d_act)).unsqueeze(1)
        act = torch.cat([state, act], dim=1)

        wanted = {depth: j for j, depth in enumerate(exits)}
        out: list[Optional[Tensor]] = [None] * len(exits)

        for ell, blk in enumerate(self.backbone[: exits[-1]]):
            vlm, act = blk(vlm, act)
            depth = ell + 1
            j = wanted.get(depth)
            if j is not None:
                with adapter_slot(self, j):
                    out[j] = self.readout(self.norm_out(act[:, 1:]))

        return [o for o in out if o is not None]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def plan(self, obs: Observation, noise: Optional[Tensor] = None) -> Tensor:
        dev = self.device
        if noise is None:
            A = torch.randn(1, self.H, self.d_a, device=dev)
        else:
            if noise.shape != (self.H, self.d_a):
                raise ValueError(
                    f"noise must be {(self.H, self.d_a)}, got {tuple(noise.shape)}"
                )
            A = noise.to(device=dev).unsqueeze(0).clone()
        dt = 1.0 / self.M
        for i in range(self.M):
            tau = torch.full((1,), i * dt, device=dev)
            v = self.velocity(A, tau, obs, E_V=self.L_V, E_B=self.L_B, adapters=False)
            A = A + dt * v
        return A[0]


def _timestep_embedding(tau: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=torch.float32, device=tau.device)
        / max(half, 1)
    )
    args = tau.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb
