from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

TOKENS_PER_VIEW = 256
PATCH_GRID = 16
PATCH_SIZE = 14


@dataclass
class SpecPruneConfig:

    alpha: float = 0.8
    base_visual_keep: int = 190
    fine_gain: float = 1.25

    k_global: int = 48
    k_local: int = 36
    k_dynamic: int = 24

    sim_threshold: float = 0.95
    frame_lookback_b: float = 2.0
    frame_lookback_k: float = -1.0
    frame_lookback_min: int = 1

    early_layers: tuple[int, ...] = (0, 1)
    goal_layers: tuple[int, ...] = (8, 17)

    dynamic_prune: bool = False
    dynamic_prune_layers: tuple[int, ...] = (6, 9, 12, 15)
    dynamic_update_layers: tuple[int, ...] = (5, 8, 11, 14)
    gamma: float = 0.9
    beta: float = 0.2
    rank_k: float = 1.0
    min_visual_keep: int = 48

    global_from_expert: bool = True
    global_capture_step: int = 5
    global_capture_every: int = 1
    global_first: bool = True

    skip_masked_views: bool = True

    vt_thresh: float = 1.89
    vr_thresh: float = 0.89
    dz_thresh: float = 0.0
    vt_exit: float = 2.38
    vr_exit: float = 1.15

    eps: float = 1e-8

    lang_ref: int = 24
    fixed_prefix_len: bool = True

    def budget(self, precise: bool) -> int:
        n = self.alpha * self.base_visual_keep * (self.fine_gain if precise else 1.0)
        return max(self.min_visual_keep, int(round(n)))

    def visual_budget(self, precise: bool, n_lang: int, n_vis_avail: int) -> int:
        b = self.budget(precise)
        if self.fixed_prefix_len:
            b = b + self.lang_ref - int(n_lang)
        return int(max(self.min_visual_keep, min(b, n_vis_avail)))


@dataclass
class SpecPruneState:

    prev_global: torch.Tensor | None = None
    frames: list[np.ndarray] = field(default_factory=list)
    conf: dict[int, float] = field(default_factory=dict)
    precise: bool = False
    last_action: np.ndarray | None = None
    steps: int = 0
    kept_hist: list[int] = field(default_factory=list)
    precise_hist: list[bool] = field(default_factory=list)

    def reset(self) -> None:
        self.prev_global = None
        self.frames.clear()
        self.conf.clear()
        self.precise = False
        self.last_action = None
        self.steps = 0

    def update_mode(self, action: np.ndarray | None, cfg: SpecPruneConfig) -> None:
        if action is None or len(action) < 6:
            return
        vt = float(np.linalg.norm(action[:3]))
        vr = float(np.linalg.norm(action[3:6]))
        dz = float(action[2])
        if self.precise:
            if vt > cfg.vt_exit or vr > cfg.vr_exit:
                self.precise = False
        elif vt < cfg.vt_thresh and vr < cfg.vr_thresh and dz <= cfg.dz_thresh:
            self.precise = True
        self.last_action = action


def _patch_vectors(frame: np.ndarray) -> np.ndarray:
    h = w = PATCH_GRID
    p = PATCH_SIZE
    f = np.asarray(frame, dtype=np.float32)
    f = f[: h * p, : w * p]
    f = f.reshape(h, p, w, p, -1).transpose(0, 2, 1, 3, 4)
    return f.reshape(h * w, -1)


def dynamic_patch_ids(cur: np.ndarray, ref: np.ndarray, k: int,
                      tau: float) -> np.ndarray:
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    a, b = _patch_vectors(cur), _patch_vectors(ref)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    sim = num / den
    cand = np.nonzero(sim < tau)[0]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, cand.size)
    return cand[np.argpartition(sim[cand], k - 1)[:k]]


class SpecPrunePi0:

    def __init__(self, backend, cfg: SpecPruneConfig | None = None,
                 compiled_denoise=None):
        self.be = backend
        self.model = backend.model
        self.cfg = cfg or SpecPruneConfig()
        self.pwe = self.model.paligemma_with_expert
        self.lm = self.pwe.paligemma.language_model
        self.n_layers = self.pwe.paligemma.config.text_config.num_hidden_layers
        self._denoise = compiled_denoise or self._denoise_step

    def _resolve_layout(self, pad_masks: torch.Tensor, n_views: int):
        total = pad_masks.shape[1]
        n_vis = n_views * TOKENS_PER_VIEW
        valid = pad_masks[0].bool()
        vis_ids = torch.arange(n_vis, device=pad_masks.device)
        vis_live = vis_ids[valid[:n_vis]]
        lang_ids = torch.arange(n_vis, total, device=pad_masks.device)
        lang_live = lang_ids[valid[n_vis:]]
        return dict(total=total, n_vis=n_vis, vis_live=vis_live,
                    lang_live=lang_live)

    @staticmethod
    def _text_to_vision(attn: torch.Tensor, text_pos: torch.Tensor,
                        vis_pos: torch.Tensor) -> torch.Tensor:
        a = attn[0][:, text_pos][:, :, vis_pos]
        return a.float().mean(dim=(0, 1))

    def _layer_entropy(self, attn: torch.Tensor, text_pos: torch.Tensor,
                       vis_pos: torch.Tensor) -> float:
        a = attn[0][:, text_pos][:, :, vis_pos].float().mean(0)
        p = a / (a.sum(-1, keepdim=True) + self.cfg.eps)
        h = -(p * torch.log(p + self.cfg.eps)).sum(-1).mean()
        return float(h / math.log(max(int(vis_pos.numel()), 2)))

    def _rank_weight(self, scores: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(scores, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=scores.device)
        w = torch.sigmoid(-self.cfg.rank_k * rank.float())
        return w / (w.sum() + self.cfg.eps)

    def _select(self, local_scores: torch.Tensor, vis_ids: torch.Tensor,
                dyn_ids: torch.Tensor, state: SpecPruneState,
                budget: int) -> torch.Tensor:
        dev = vis_ids.device
        order = torch.argsort(local_scores, descending=True)
        local_ranked = vis_ids[order]

        cfg = self.cfg
        scale = cfg.alpha * (cfg.fine_gain if state.precise else 1.0)
        n_local = int(round(cfg.k_local * scale)) * 2
        n_global = int(round(cfg.k_global * scale)) * 2

        chosen: list[torch.Tensor] = []
        seen = torch.zeros(int(vis_ids.max().item()) + 1, dtype=torch.bool, device=dev)

        def take(ids: torch.Tensor) -> None:
            if ids.numel() == 0:
                return
            fresh = ids[~seen[ids]]
            if fresh.numel():
                seen[fresh] = True
                chosen.append(fresh)

        def take_global() -> None:
            if state.prev_global is not None and state.prev_global.numel():
                g = state.prev_global.to(dev)
                take(g[g < seen.numel()][:n_global])

        take(dyn_ids)
        if cfg.global_first:
            take_global()
            take(local_ranked[:n_local])
        else:
            take(local_ranked[:n_local])
            take_global()
        take(local_ranked)

        keep = torch.cat(chosen) if chosen else local_ranked[:budget]
        if keep.numel() > budget:
            keep = keep[:budget]
        return torch.sort(keep).values

    @torch.no_grad()
    def prefill(self, embs: torch.Tensor, pad: torch.Tensor, att: torch.Tensor,
                obs_frames: list[np.ndarray], state: SpecPruneState,
                n_views: int):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        cfg = self.cfg
        model = self.model
        lay = self._resolve_layout(pad, n_views)
        dev = embs.device

        n_valid_original = int(pad.sum().item())

        position_ids = torch.cumsum(pad, dim=1) - 1
        hidden = embs
        if self.lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        cos, sin = self.lm.rotary_emb(hidden, position_ids)

        att_2d = make_att_2d_masks(pad, att)
        mask = model._prepare_attention_masks_4d(att_2d)

        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        self.lm.config._attn_implementation = "eager"

        cur_ids = pad[0].bool().nonzero(as_tuple=True)[0]
        hidden = hidden[:, cur_ids].contiguous()
        position_ids = position_ids[:, cur_ids]
        cos, sin = cos[:, cur_ids], sin[:, cur_ids]
        _n = int(cur_ids.numel())
        mask = torch.zeros(1, 1, _n, _n, dtype=mask.dtype, device=dev)
        ids_at_layer: list[torch.Tensor] = []

        def positions_of(ids: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            idx = torch.searchsorted(ids, target)
            idx = idx.clamp(max=ids.numel() - 1)
            return idx[ids[idx] == target]

        text_pos = positions_of(cur_ids, lay["lang_live"])
        vis_pos = positions_of(cur_ids, lay["vis_live"])

        local_sum = torch.zeros(lay["vis_live"].numel(), device=dev, dtype=torch.float32)
        importance: torch.Tensor | None = None
        kept_ids = None
        new_global: list[torch.Tensor] = []

        want_attn = set(cfg.early_layers)
        if not cfg.global_from_expert:
            want_attn |= set(cfg.goal_layers)
        if cfg.dynamic_prune:
            want_attn |= set(cfg.dynamic_update_layers)

        for li in range(self.n_layers):
            ids_at_layer.append(cur_ids)
            layer = self.lm.layers[li]
            need = li in want_attn
            out = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=need,
                use_cache=True,
                cache_position=torch.arange(hidden.shape[1], device=dev),
                position_embeddings=(cos, sin),
                adarms_cond=None,
            )
            hidden = out[0]
            attn = out[1] if need else None

            if li in cfg.early_layers and attn is not None:
                local_sum = local_sum + self._text_to_vision(attn, text_pos, vis_pos)

            if kept_ids is None and li == max(cfg.early_layers):
                dyn_ids = self._dynamic_ids(obs_frames, state, n_views, dev)
                budget = cfg.visual_budget(state.precise,
                                           lay["lang_live"].numel(),
                                           lay["vis_live"].numel())
                kept_vis = self._select(local_sum, lay["vis_live"], dyn_ids,
                                        state, budget)
                kept_ids = torch.sort(torch.cat([kept_vis, lay["lang_live"]])).values
                state.kept_hist.append(int(kept_vis.numel()))

                sel = positions_of(cur_ids, kept_ids)
                hidden = hidden[:, sel].contiguous()
                cur_ids = kept_ids
                position_ids = position_ids[:, sel]
                cos, sin = cos[:, sel], sin[:, sel]
                n = cur_ids.numel()
                mask = torch.zeros(1, 1, n, n, dtype=mask.dtype, device=dev)
                text_pos = positions_of(cur_ids, lay["lang_live"])
                vis_pos = positions_of(cur_ids, kept_vis)
                importance = torch.zeros(n, device=dev, dtype=torch.float32)

            if cfg.dynamic_prune and importance is not None:
                if li in cfg.dynamic_update_layers and attn is not None and vis_pos.numel():
                    scores = self._text_to_vision(attn, text_pos, vis_pos)
                    if li not in state.conf:
                        h = self._layer_entropy(attn, text_pos, vis_pos)
                        state.conf[li] = 1.0 / (h + cfg.eps)
                    s = self._rank_weight(scores) * state.conf[li]
                    importance[vis_pos] = ((1 - cfg.beta) * importance[vis_pos]
                                           + cfg.beta * s)

                if li in cfg.dynamic_prune_layers and vis_pos.numel() > cfg.min_visual_keep:
                    n_keep = max(cfg.min_visual_keep,
                                 int(round(cfg.gamma * vis_pos.numel())))
                    if n_keep < vis_pos.numel():
                        vs = importance[vis_pos]
                        top = torch.topk(vs, n_keep).indices
                        survive = cur_ids[vis_pos[top]]
                        new_ids = torch.sort(
                            torch.cat([survive, lay["lang_live"]])).values
                        sel = positions_of(cur_ids, new_ids)
                        hidden = hidden[:, sel].contiguous()
                        importance = importance[sel]
                        cur_ids = new_ids
                        position_ids = position_ids[:, sel]
                        cos, sin = cos[:, sel], sin[:, sel]
                        n = cur_ids.numel()
                        mask = torch.zeros(1, 1, n, n, dtype=mask.dtype, device=dev)
                        text_pos = positions_of(cur_ids, lay["lang_live"])
                        vis_pos = positions_of(cur_ids, survive)

            if (not cfg.global_from_expert and li in cfg.goal_layers
                    and attn is not None and vis_pos.numel()):
                scores = self._text_to_vision(attn, text_pos, vis_pos)
                k = min(int(round(cfg.k_global * cfg.alpha)) * 2, scores.numel())
                if k > 0:
                    new_global.append(cur_ids[vis_pos[torch.topk(scores, k).indices]])

        if not cfg.global_from_expert:
            state.prev_global = _interleave(new_global) if new_global else None

        cache = self._subset_cache(cache, ids_at_layer, cur_ids)
        return cache, cur_ids, n_valid_original

    def _dynamic_ids(self, frames: list[np.ndarray], state: SpecPruneState,
                     n_views: int, dev) -> torch.Tensor:
        cfg = self.cfg
        if not state.frames:
            return torch.empty(0, dtype=torch.long, device=dev)
        v = 0.0
        if state.last_action is not None:
            v = float(np.linalg.norm(state.last_action[:3]))
        T = int(round(cfg.frame_lookback_b + cfg.frame_lookback_k * v))
        T = max(cfg.frame_lookback_min, T)
        ref = state.frames[-min(T, len(state.frames))]

        scale = cfg.alpha * (cfg.fine_gain if state.precise else 1.0)
        k = int(round(cfg.k_dynamic * scale))
        out = []
        for vi in range(min(n_views, len(frames))):
            if vi >= len(ref):
                break
            ids = dynamic_patch_ids(frames[vi], ref[vi], k, cfg.sim_threshold)
            if ids.size:
                out.append(torch.as_tensor(ids, device=dev, dtype=torch.long)
                           + vi * TOKENS_PER_VIEW)
        return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=dev)

    @staticmethod
    def _subset_cache(cache, ids_at_layer, final_ids):
        from transformers.cache_utils import DynamicCache

        out = DynamicCache()
        for li, ids in enumerate(ids_at_layer):
            k, v = cache[li]
            idx = torch.searchsorted(ids, final_ids)
            k = k[:, :, idx, :].contiguous()
            v = v[:, :, idx, :].contiguous()
            out.update(k, v, li, {})
        return out

    def _denoise_step(self, state_vec, prefix_kv_len: int, prefix_offset: int,
                      cache, x_t, timestep):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = self.model
        suffix_embs, suffix_pad, suffix_att, adarms = model.embed_suffix(
            state_vec, x_t, timestep)
        b, s = suffix_pad.shape
        dev = suffix_pad.device

        prefix_2d = torch.ones(b, s, prefix_kv_len, dtype=torch.bool, device=dev)
        full = torch.cat([prefix_2d, make_att_2d_masks(suffix_pad, suffix_att)], dim=2)
        position_ids = prefix_offset + torch.cumsum(suffix_pad, dim=1) - 1

        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        out, _ = model.paligemma_with_expert.forward(
            attention_mask=model._prepare_attention_masks_4d(full),
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms],
        )
        y = out[1][:, -model.config.action_horizon:].to(torch.float32)
        return model.action_out_proj(y)

    def _denoise_attn(self, state_vec, prefix_kv_len: int, prefix_offset: int,
                      cache, x_t, timestep):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = self.model
        suffix_embs, suffix_pad, suffix_att, adarms = model.embed_suffix(
            state_vec, x_t, timestep)
        b, s = suffix_pad.shape
        dev = suffix_pad.device

        prefix_2d = torch.ones(b, s, prefix_kv_len, dtype=torch.bool, device=dev)
        full = torch.cat([prefix_2d, make_att_2d_masks(suffix_pad, suffix_att)], dim=2)
        position_ids = prefix_offset + torch.cumsum(suffix_pad, dim=1) - 1

        exm = model.paligemma_with_expert.gemma_expert.model
        exm.config._attn_implementation = "eager"
        out = exm.forward(
            inputs_embeds=suffix_embs,
            attention_mask=model._prepare_attention_masks_4d(full),
            position_ids=position_ids, past_key_values=cache,
            use_cache=False, output_attentions=True, adarms_cond=adarms)

        h = out.last_hidden_state[:, -model.config.action_horizon:].to(torch.float32)
        v = model.action_out_proj(h)

        tot = None
        for a in out.attentions:
            m = a[0][:, -model.config.action_horizon:, :prefix_kv_len]
            m = m.float().mean(dim=(0, 1))
            tot = m if tot is None else tot + m
        return v, tot / max(len(out.attentions), 1)

    def _carry_global(self, state: SpecPruneState, kept: torch.Tensor,
                      scores: torch.Tensor, n_views: int) -> None:
        cfg = self.cfg
        n_vis = n_views * TOKENS_PER_VIEW
        vis_mask = kept < n_vis
        vis_ids, vis_sc = kept[vis_mask], scores[vis_mask]
        if vis_ids.numel() == 0:
            state.prev_global = None
            return
        k = min(int(round(cfg.k_global * cfg.alpha)) * 2, int(vis_ids.numel()))
        state.prev_global = vis_ids[torch.topk(vis_sc, k).indices]

    @torch.no_grad()
    def plan(self, obs, state: SpecPruneState, noise=None,
             frames: list[np.ndarray] | None = None) -> torch.Tensor:
        be, model, cfg = self.be, self.model, self.cfg
        o1 = be._openpi_obs(obs, batch=1)
        imgs, img_masks, lang_tokens, lang_masks, state_vec = (
            model._preprocess_observation(o1, train=False))

        n_views = int(sum(bool(m.any()) for m in img_masks))
        if cfg.skip_masked_views:
            live = [i for i, m in enumerate(img_masks) if bool(m.any())]
            imgs = [imgs[i] for i in live]
            img_masks = [img_masks[i] for i in live]
        embs, pad, att = model.embed_prefix(imgs, img_masks, lang_tokens, lang_masks)

        if frames is None:
            frames = _frames_from_obs(obs, n_views)

        cache, kept, n_valid = self.prefill(embs, pad, att, frames, state, n_views)

        if noise is not None:
            x = noise.to(device=be.device, dtype=torch.float32).unsqueeze(0)
        else:
            x = model.sample_noise((1, be.H, be.d_a), be.device)

        dt = -1.0 / be.M
        kv_len = int(kept.numel())
        due = (cfg.global_from_expert
               and (state.prev_global is None
                    or state.steps % max(cfg.global_capture_every, 1) == 0))
        capture = cfg.global_capture_step if due else -1
        for i in range(be.M):
            t = torch.full((1,), 1.0 + i * dt, dtype=torch.float32, device=be.device)
            if i == capture:
                v, expert_scores = self._denoise_attn(
                    state_vec, kv_len, n_valid, cache, x, t)
                self._carry_global(state, kept, expert_scores, n_views)
            else:
                v = self._denoise(state_vec, kv_len, n_valid, cache, x, t)
            x = x + dt * v

        state.frames.append(frames)
        if len(state.frames) > 32:
            state.frames.pop(0)
        state.steps += 1
        state.precise_hist.append(state.precise)
        return x[0].to(device=obs.images.device, dtype=torch.float32)


def _interleave(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    m = min(int(t.numel()) for t in tensors)
    head = torch.stack([t[:m] for t in tensors], dim=1).flatten()
    tail = torch.cat([t[m:] for t in tensors])
    return torch.cat([head, tail])


def _frames_from_obs(obs, n_views: int) -> list[np.ndarray]:
    f = obs.images[:n_views].detach().cpu().permute(0, 2, 3, 1).numpy()
    return list(((f + 1.0) * 127.5).astype(np.float32))
