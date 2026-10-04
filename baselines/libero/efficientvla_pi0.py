from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

TOKENS_PER_VIEW = 256
PATCH_GRID, PATCH_SIZE = 16, 14

EXPERT_COHERENCE = [
    0.9994, 0.9692, 0.9542, 0.9404, 0.9929, 0.9449, 0.9371, 0.9665, 0.9597,
    0.9619, 0.9695, 0.9951, 0.9717, 0.9787, 0.9894, 0.9836, 0.9952, 0.9991,
]
BACKBONE_IMPORTANCE = [
    0.7225, 0.0060, 0.0027, 0.0023, 0.0016, 0.0018, 0.0028, 0.0023, 0.0027,
    0.0035, 0.0055, 0.0110, 0.0211, 0.0431, 0.0315, 0.0287, 0.3100, 0.8526,
]


@dataclass
class EfficientVLAConfig:
    prune_layers: bool = True
    n_prune: int = 6

    prune_tokens: bool = True
    k_final: int = 256
    anchor_frac: float = 0.5
    attn_layer: int = 8
    diversity_pool: int = 64

    cache_expert: bool = True
    cache_interval: int = 2
    uniform_cache: bool = False
    coherence_floor: float = 0.96

    eps: float = 1e-8

    def pruned_layer_set(self, n_layers: int) -> set[int]:
        if not self.prune_layers or self.n_prune <= 0:
            return set()
        order = [i for i in sorted(range(n_layers), key=lambda i: BACKBONE_IMPORTANCE[i])
                 if not (self.prune_tokens and i == self.attn_layer)]
        return set(order[: self.n_prune])

    def cached_layer_set(self, n_layers: int) -> set[int]:
        if not self.cache_expert:
            return set()
        if self.uniform_cache:
            return set(range(n_layers))
        return {i for i in range(n_layers)
                if EXPERT_COHERENCE[i] >= self.coherence_floor}


@dataclass
class EfficientVLAState:
    prev_attn_scores: torch.Tensor | None = None
    steps: int = 0
    kept_hist: list[int] = field(default_factory=list)

    def reset(self) -> None:
        self.prev_attn_scores = None
        self.steps = 0


class EfficientVLAPi0:
    def __init__(self, backend, cfg: EfficientVLAConfig | None = None,
                 compile_expert: bool = True):
        self.be = backend
        self.model = backend.model
        self.cfg = cfg or EfficientVLAConfig()
        self.pwe = self.model.paligemma_with_expert
        self.lm = self.pwe.paligemma.language_model
        self.ex = self.pwe.gemma_expert.model
        self.n_layers = self.pwe.paligemma.config.text_config.num_hidden_layers
        self._compiled = None
        if compile_expert:
            try:
                self._compiled = torch.compile(self._denoise_impl,
                                               mode="default", dynamic=False)
            except Exception:
                self._compiled = None

    @torch.no_grad()
    def prefill(self, embs, pad, state, n_views, keep_idx=None):
        from transformers.cache_utils import DynamicCache
        from transformers.models.gemma import modeling_gemma as mg

        cfg = self.cfg
        dev = embs.device
        alive = pad[0].bool().nonzero(as_tuple=True)[0]
        if keep_idx is not None:
            alive = alive[keep_idx]
        n_vis_slots = n_views * TOKENS_PER_VIEW
        pos = (torch.cumsum(pad, dim=1) - 1)[:, alive]
        h = embs[:, alive].contiguous()
        if self.lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            h = h.to(torch.bfloat16)
        n = int(alive.numel())
        cos_, sin_ = self.lm.rotary_emb(h, pos)
        mask = torch.zeros(1, 1, n, n, dtype=torch.float32, device=dev)
        self.lm.config._attn_implementation = "eager"

        skip = cfg.pruned_layer_set(self.n_layers)
        cache = DynamicCache()
        attn_scores = None
        n_vis = None
        for li in range(self.n_layers):
            layer = self.lm.layers[li]
            sa = layer.self_attn
            if li in skip:
                x, _ = layer.input_layernorm(h, None)
                b, s_, _ = x.shape
                k = sa.k_proj(x).view(b, s_, -1, sa.head_dim).transpose(1, 2)
                v = sa.v_proj(x).view(b, s_, -1, sa.head_dim).transpose(1, 2)
                _, k = mg.apply_rotary_pos_emb(k, k, cos_, sin_)
                cache.update(k, v, li, {})
                continue

            want = (li == cfg.attn_layer)
            out = layer(h, attention_mask=mask, position_ids=pos,
                        past_key_value=cache, output_attentions=want,
                        use_cache=True,
                        cache_position=torch.arange(n, device=dev),
                        position_embeddings=(cos_, sin_), adarms_cond=None)
            h = out[0]
            if want and len(out) > 1 and out[1] is not None:
                a = out[1][0].float().mean(0)
                vis_m = alive < n_vis_slots
                txt_m = ~vis_m
                if txt_m.any() and vis_m.any():
                    attn_scores = a[txt_m][:, vis_m].mean(0)
                    n_vis = int(vis_m.sum())

        if attn_scores is not None:
            state.prev_attn_scores = attn_scores
        return cache, n, alive, n_vis

    def _expert_forward(self, suffix_embs, mask4d, position_ids, pk, pv,
                        ca, cm, recompute: bool, adarms=None):
        from transformers.models.gemma import modeling_gemma as mg

        cfg = self.cfg
        cached_layers = cfg.cached_layer_set(self.n_layers)
        h = suffix_embs
        if self.ex.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            h = h.to(torch.bfloat16)
        cos_, sin_ = self.ex.rotary_emb(h, position_ids)
        self.ex.config._attn_implementation = "eager"

        for li, layer in enumerate(self.ex.layers):
            sa = layer.self_attn
            if (li in cached_layers) and (not recompute) and (ca[li] is not None):
                h = h + ca[li]
                h = h + cm[li]
                continue

            residual = h
            x, gate = layer.input_layernorm(h, adarms)
            b, s, _ = x.shape
            q = sa.q_proj(x).view(b, s, -1, sa.head_dim).transpose(1, 2)
            k = sa.k_proj(x).view(b, s, -1, sa.head_dim).transpose(1, 2)
            v = sa.v_proj(x).view(b, s, -1, sa.head_dim).transpose(1, 2)
            q, k = mg.apply_rotary_pos_emb(q, k, cos_, sin_)
            K = torch.cat([pk[li], k], dim=2)
            V = torch.cat([pv[li], v], dim=2)
            att, _ = mg.eager_attention_forward(sa, q, K, V, mask4d,
                                                scaling=sa.scaling)
            a_out = sa.o_proj(att.reshape(b, s, -1))
            if gate is not None:
                a_out = a_out * gate
            h = residual + a_out
            residual = h
            x, gate = layer.post_attention_layernorm(h, adarms)
            m_out = layer.mlp(x)
            if gate is not None:
                m_out = m_out * gate
            h = residual + m_out
            if li in cached_layers:
                ca[li] = a_out
                cm[li] = m_out
        h, _ = self.ex.norm(h, adarms)
        return h

    def _denoise(self, state_vec, prefix_len, pk, pv, x_t, timestep,
                 ca, cm, recompute):
        fn = self._compiled or self._denoise_impl
        return fn(state_vec, prefix_len, pk, pv, x_t, timestep, ca, cm, recompute)

    def _denoise_impl(self, state_vec, prefix_len, pk, pv, x_t, timestep,
                      ca, cm, recompute):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = self.model
        se, sp, sa_, ad = model.embed_suffix(state_vec, x_t, timestep)
        b, s = sp.shape
        dev = sp.device
        full = torch.cat([torch.ones(b, s, prefix_len, dtype=torch.bool, device=dev),
                          make_att_2d_masks(sp, sa_)], dim=2)
        mask4d = model._prepare_attention_masks_4d(full)
        pos = prefix_len + torch.cumsum(sp, dim=1) - 1
        out = self._expert_forward(se, mask4d, pos, pk, pv, ca, cm, recompute, ad)
        y = out[:, -model.config.action_horizon:].to(torch.float32)
        return model.action_out_proj(y)

    def _select_tokens(self, vis_feats, scores, k_final):
        cfg = self.cfg
        n = scores.numel()
        if k_final >= n:
            return torch.arange(n, device=scores.device)
        n_anchor = int(round(cfg.anchor_frac * k_final))
        anchor = torch.topk(scores, n_anchor).indices
        chosen = torch.zeros(n, dtype=torch.bool, device=scores.device)
        chosen[anchor] = True

        f = torch.nn.functional.normalize(vis_feats.float(), dim=-1)
        sim_to_set = (f @ f[anchor].T).max(dim=1).values
        rest = k_final - n_anchor
        if rest > 0:
            rank = torch.argsort(scores, descending=True)
            pool = rank[~chosen[rank]][: max(cfg.diversity_pool, rest) * 4]
            merit = scores[pool] / (scores[pool].max() + cfg.eps) - sim_to_set[pool]
            take = pool[torch.topk(merit, min(rest, pool.numel())).indices]
            chosen[take] = True
        return chosen.nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def plan(self, obs, state: EfficientVLAState, noise=None) -> torch.Tensor:
        be, model, cfg = self.be, self.model, self.cfg
        o1 = be._openpi_obs(obs, batch=1)
        imgs, im, lt, lmk, sv = model._preprocess_observation(o1, train=False)
        live = [i for i, m in enumerate(im) if bool(m.any())]
        embs, pad, _ = model.embed_prefix([imgs[i] for i in live],
                                          [im[i] for i in live], lt, lmk)

        keep_idx = None
        if cfg.prune_tokens and state.prev_attn_scores is not None:
            n_v = int(state.prev_attn_scores.numel())
            k_final = min(cfg.k_final, n_v)
            if k_final < n_v:
                keep_v = self._select_tokens(embs[0, :n_v],
                                             state.prev_attn_scores, k_final)
                total = int(pad[0].bool().sum())
                keep_idx = torch.sort(torch.cat(
                    [keep_v, torch.arange(n_v, total, device=embs.device)])).values

        cache, n, alive, n_vis = self.prefill(embs, pad, state, len(live), keep_idx)
        state.kept_hist.append(int((alive < len(live) * TOKENS_PER_VIEW).sum()))

        x = (noise.to(device=be.device, dtype=torch.float32).unsqueeze(0)
             if noise is not None else model.sample_noise((1, be.H, be.d_a), be.device))
        dt = -1.0 / be.M
        pk = [cache[li][0] for li in range(self.n_layers)]
        pv = [cache[li][1] for li in range(self.n_layers)]
        ca: list = [None] * self.n_layers
        cm: list = [None] * self.n_layers
        for i in range(be.M):
            t = torch.full((1,), 1.0 + i * dt, dtype=torch.float32, device=be.device)
            recompute = (not cfg.cache_expert) or (i % max(cfg.cache_interval, 1) == 0)
            x = x + dt * self._denoise(sv, n, pk, pv, x, t, ca, cm, recompute)

        state.steps += 1
        return x[0].to(device=obs.images.device, dtype=torch.float32)
