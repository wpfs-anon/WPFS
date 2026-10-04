from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

TOKENS_PER_VIEW = 256
PATCH_GRID = 16
PATCH_SIZE = 14


@dataclass
class VLACacheConfig:

    enabled: bool = True
    sim_threshold: float = 0.996
    static_top_k: int = 150
    task_top_k: int = 100
    attn_layer: int = 8
    growth_factor: float = 0.55
    max_reuse_frac: float = 1.0
    schedule_once: bool = True
    max_age: int = 0


@dataclass
class VLACacheState:

    prev_frames: list[np.ndarray] | None = None
    prev_k: list[torch.Tensor] = field(default_factory=list)
    prev_v: list[torch.Tensor] = field(default_factory=list)
    prev_ids: torch.Tensor | None = None
    prev_attn_scores: torch.Tensor | None = None
    schedule: list[float] | None = None
    age: torch.Tensor | None = None
    steps: int = 0
    reuse_hist: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.prev_frames = None
        self.prev_k.clear()
        self.prev_v.clear()
        self.prev_ids = None
        self.prev_attn_scores = None
        self.schedule = None
        self.age = None
        self.steps = 0


def _patch_vectors(frame: np.ndarray) -> np.ndarray:
    h = w = PATCH_GRID
    p = PATCH_SIZE
    f = np.asarray(frame, dtype=np.float32)[: h * p, : w * p]
    return f.reshape(h, p, w, p, -1).transpose(0, 2, 1, 3, 4).reshape(h * w, -1)


def patch_similarity(cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a, b = _patch_vectors(cur), _patch_vectors(ref)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return num / den


class VLACachePi0:

    def __init__(self, backend, cfg: VLACacheConfig | None = None,
                 compiled_denoise=None):
        self.be = backend
        self.model = backend.model
        self.cfg = cfg or VLACacheConfig()
        self.pwe = self.model.paligemma_with_expert
        self.lm = self.pwe.paligemma.language_model
        self.n_layers = self.pwe.paligemma.config.text_config.num_hidden_layers
        self._denoise = compiled_denoise or self._denoise_step

    def _reusable_ids(self, frames, state, n_views, vis_ids, dev):
        cfg = self.cfg
        if state.prev_frames is None:
            return torch.empty(0, dtype=torch.long, device=dev)

        keep, score = [], []
        for v in range(min(n_views, len(frames), len(state.prev_frames))):
            sim = patch_similarity(frames[v], state.prev_frames[v])
            cand = np.nonzero(sim >= cfg.sim_threshold)[0]
            if cand.size == 0:
                continue
            k = min(cfg.static_top_k, cand.size)
            cand = cand[np.argsort(-sim[cand])[:k]]

            if state.prev_attn_scores is not None:
                lo, hi = v * TOKENS_PER_VIEW, (v + 1) * TOKENS_PER_VIEW
                s = state.prev_attn_scores[lo:hi]
                if s.numel() == TOKENS_PER_VIEW:
                    tk = min(cfg.task_top_k, TOKENS_PER_VIEW)
                    hot = set(torch.topk(s, tk).indices.cpu().numpy().tolist())
                    m = np.array([c not in hot for c in cand], dtype=bool)
                    cand = cand[m]
            if cand.size:
                keep.append(cand + v * TOKENS_PER_VIEW)
                score.append(sim[cand])

        if not keep:
            return torch.empty(0, dtype=torch.long, device=dev)
        ids = np.concatenate(keep)
        sc = np.concatenate(score)
        ids = ids[np.argsort(-sc)]
        tok = torch.as_tensor(ids, device=dev, dtype=torch.long)
        return tok[torch.isin(tok, vis_ids)]

    def _schedule(self, entropies) -> list[float]:
        cfg = self.cfg
        e = torch.stack(entropies).float()
        lo, hi = e.min(), e.max()
        reuse = (1.0 - (e - lo) / (hi - lo + 1e-10)).tolist()
        for i in range(1, len(reuse)):
            d = reuse[i] - reuse[i - 1]
            reuse[i] = reuse[i - 1] + d * cfg.growth_factor if d > 0 else reuse[i]
        for i in range(1, len(reuse)):
            reuse[i] = max(reuse[i], reuse[i - 1])
        return [min(max(r, 0.0), cfg.max_reuse_frac) for r in reuse]

    def _layer(self, layer, h, cidx, K_prev, V_prev, cos, sin, want_attn):
        from transformers.models.gemma import modeling_gemma as mg

        sa = layer.self_attn
        hc = h.index_select(1, cidx)
        residual = hc
        x, _ = layer.input_layernorm(hc, None)
        b, c, _ = x.shape
        hd = sa.head_dim
        q = sa.q_proj(x).view(b, c, -1, hd).transpose(1, 2)
        k = sa.k_proj(x).view(b, c, -1, hd).transpose(1, 2)
        v = sa.v_proj(x).view(b, c, -1, hd).transpose(1, 2)
        q, k = mg.apply_rotary_pos_emb(q, k, cos.index_select(1, cidx),
                                       sin.index_select(1, cidx))
        K = K_prev.index_copy(2, cidx, k)
        V = V_prev.index_copy(2, cidx, v)
        att, attn_w = mg.eager_attention_forward(sa, q, K, V, None,
                                                 scaling=sa.scaling)
        hc = residual + sa.o_proj(att.reshape(b, c, -1))
        residual = hc
        x, _ = layer.post_attention_layernorm(hc, None)
        hc = residual + layer.mlp(x)
        return h.index_copy(1, cidx, hc), K, V, (attn_w if want_attn else None)

    @torch.no_grad()
    def prefill(self, embs, pad, frames, state, n_views):
        from transformers.cache_utils import DynamicCache

        cfg = self.cfg
        dev = embs.device
        alive = pad[0].bool().nonzero(as_tuple=True)[0]
        n_vis = n_views * TOKENS_PER_VIEW
        pos = (torch.cumsum(pad, dim=1) - 1)[:, alive]
        h = embs[:, alive].contiguous()
        if self.lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            h = h.to(torch.bfloat16)
        n = int(alive.numel())
        cos, sin = self.lm.rotary_emb(h, pos)

        vis_pos = (alive < n_vis).nonzero(as_tuple=True)[0]
        txt_pos = (alive >= n_vis).nonzero(as_tuple=True)[0]
        vis_ids = alive[vis_pos]

        reusable = torch.empty(0, dtype=torch.long, device=dev)
        if cfg.enabled and state.prev_ids is not None and len(state.prev_k) == self.n_layers \
                and int(state.prev_ids.numel()) == n and bool((state.prev_ids == alive).all()):
            r = self._reusable_ids(frames, state, n_views, vis_ids, dev)
            if r.numel():
                seq = torch.searchsorted(alive, r).clamp(max=n - 1)
                reusable = seq[alive[seq] == r]

        if cfg.max_age > 0 and reusable.numel() and state.age is not None \
                and int(state.age.numel()) == n:
            fresh = state.age[reusable] < cfg.max_age
            reusable = reusable[fresh]

        sched = state.schedule
        need_sched = sched is None or not cfg.schedule_once
        want_all = need_sched

        ks, vs, ents = [], [], []
        attn_scores = None
        sa0 = self.lm.layers[0].self_attn
        zero = torch.zeros(1, self.lm.config.num_key_value_heads, n, sa0.head_dim,
                           dtype=h.dtype, device=dev)
        ok = (len(state.prev_k) == self.n_layers
              and state.prev_k[0].shape[2] == n and reusable.numel() > 0)
        Kp = state.prev_k if ok else [zero] * self.n_layers
        Vp = state.prev_v if ok else [zero] * self.n_layers
        if not ok:
            reusable = torch.empty(0, dtype=torch.long, device=dev)

        all_idx = torch.arange(n, device=dev)
        used = []
        for li in range(self.n_layers):
            frac = 0.0 if sched is None else sched[min(li, len(sched) - 1)]
            n_re = int(round(frac * reusable.numel())) if reusable.numel() else 0
            if n_re > 0:
                re_i = reusable[:n_re]
                mask = torch.ones(n, dtype=torch.bool, device=dev)
                mask[re_i] = False
                cidx = all_idx[mask]
            else:
                cidx = all_idx
            used.append(1.0 - cidx.numel() / n)

            want = want_all or li == cfg.attn_layer
            h, K, V, aw = self._layer(self.lm.layers[li], h, cidx,
                                      Kp[li], Vp[li], cos, sin, want)
            ks.append(K)
            vs.append(V)
            if aw is not None:
                a = aw[0].float().mean(0)
                if want_all:
                    p = a / (a.sum(-1, keepdim=True) + 1e-10)
                    ents.append(-(p * torch.log(p + 1e-10)).sum(-1).mean())
                if li == cfg.attn_layer:
                    rowmap = torch.full((n,), -1, dtype=torch.long, device=dev)
                    rowmap[cidx] = torch.arange(cidx.numel(), device=dev)
                    tr = rowmap[txt_pos]
                    tr = tr[tr >= 0]
                    if tr.numel():
                        full = torch.zeros(n_vis, device=dev)
                        full[vis_ids] = a[tr][:, vis_pos].mean(0)
                        attn_scores = full

        if need_sched and ents:
            state.schedule = self._schedule(ents)

        newage = torch.zeros(n, dtype=torch.long, device=dev)
        if reusable.numel():
            top = reusable[: max(int(round(max(sched or [0.0]) * reusable.numel())), 0)]
            prev = state.age if (state.age is not None and int(state.age.numel()) == n) \
                else torch.zeros(n, dtype=torch.long, device=dev)
            newage[top] = prev[top] + 1
        state.age = newage

        state.prev_k, state.prev_v = ks, vs
        state.prev_ids = alive
        if attn_scores is not None:
            state.prev_attn_scores = attn_scores
        state.reuse_hist.append(float(np.mean(used)))

        cache = DynamicCache()
        for li in range(self.n_layers):
            cache.update(ks[li], vs[li], li, {})
        return cache, n

    def _denoise_step(self, state_vec, prefix_len, cache, x_t, timestep):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = self.model
        se, sp, sa, ad = model.embed_suffix(state_vec, x_t, timestep)
        b, s = sp.shape
        dev = sp.device
        full = torch.cat([torch.ones(b, s, prefix_len, dtype=torch.bool, device=dev),
                          make_att_2d_masks(sp, sa)], dim=2)
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        out, _ = model.paligemma_with_expert.forward(
            attention_mask=model._prepare_attention_masks_4d(full),
            position_ids=prefix_len + torch.cumsum(sp, dim=1) - 1,
            past_key_values=cache, inputs_embeds=[None, se],
            use_cache=False, adarms_cond=[None, ad])
        y = out[1][:, -model.config.action_horizon:].to(torch.float32)
        return model.action_out_proj(y)

    @torch.no_grad()
    def plan(self, obs, state: VLACacheState, noise=None) -> torch.Tensor:
        be, model = self.be, self.model
        o1 = be._openpi_obs(obs, batch=1)
        imgs, im, lt, lmk, sv = model._preprocess_observation(o1, train=False)
        n_views = int(sum(bool(m.any()) for m in im))
        live = [i for i, m in enumerate(im) if bool(m.any())]
        imgs = [imgs[i] for i in live]
        im = [im[i] for i in live]
        embs, pad, _ = model.embed_prefix(imgs, im, lt, lmk)

        frames = list(((obs.images[:n_views].detach().cpu().permute(0, 2, 3, 1).numpy()
                        + 1.0) * 127.5).astype(np.float32))
        cache, plen = self.prefill(embs, pad, frames, state, n_views)

        x = (noise.to(device=be.device, dtype=torch.float32).unsqueeze(0)
             if noise is not None else model.sample_noise((1, be.H, be.d_a), be.device))
        dt = -1.0 / be.M
        for i in range(be.M):
            t = torch.full((1,), 1.0 + i * dt, dtype=torch.float32, device=be.device)
            x = x + dt * self._denoise(sv, plen, cache, x, t)

        state.prev_frames = frames
        state.steps += 1
        return x[0].to(device=obs.images.device, dtype=torch.float32)
