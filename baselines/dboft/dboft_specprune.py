from dataclasses import dataclass, field

import numpy as np
import torch

from dboft_vlacache import IMAGE_TOKEN_INDEX, N_VIS, VLACacheDB, patch_similarity, patch_vectors


@dataclass
class SPConfig:
    alpha: float = 2.0
    base_visual_keep: int = 214
    fine_gain: float = 1.25
    k_global: int = 108
    k_local: int = 81
    k_dynamic: int = 54
    sim_threshold: float = 0.95
    early_layers: tuple = (0, 1)
    goal_layers: tuple = (13, 27)
    global_capture_step: int = 5
    min_visual_keep: int = 48
    vt_thresh: float = 0.0074
    vr_thresh: float = 0.0233
    dz_thresh: float = 0.0
    vt_exit: float = 0.0107
    vr_exit: float = 0.0317

    def budget(self, precise: bool) -> int:
        n = self.alpha * self.base_visual_keep * (self.fine_gain if precise else 1.0)
        return int(max(self.min_visual_keep, min(round(n), N_VIS)))


@dataclass
class SPState:
    prev_global: torch.Tensor | None = None
    frames: list = field(default_factory=list)
    precise: bool = False
    kept_hist: list = field(default_factory=list)
    precise_hist: list = field(default_factory=list)
    comp_hist: list = field(default_factory=list)


class SpecPruneDB:
    def __init__(self, m, cfg: SPConfig | None = None):
        self.m = m
        self.cfg = cfg or SPConfig()
        self.llm = m.model.llm
        self.L = len(self.llm.layers)
        self.state = SPState()
        self._layer = VLACacheDB(m)._layer
        self.pos_offset = None
        self.n_kept = None
        self._vis_in_cache = None
        self._vis_ids = None

    def reset(self):
        self.state = SPState()

    def update_mode(self, a_env) -> None:
        cfg, st = self.cfg, self.state
        vt, vr, dz = float(np.linalg.norm(a_env[:3])), float(np.linalg.norm(a_env[3:6])), float(a_env[2])
        if st.precise:
            if vt > cfg.vt_exit or vr > cfg.vr_exit:
                st.precise = False
        elif vt < cfg.vt_thresh and vr < cfg.vr_thresh and dz <= cfg.dz_thresh:
            st.precise = True

    def _dynamic(self, cur_vec, k) -> np.ndarray:
        back = 2 if self.state.precise else 1
        if len(self.state.frames) < back or k <= 0:
            return np.zeros(0, dtype=np.int64)
        sim = patch_similarity(cur_vec, self.state.frames[-back])
        cand = np.nonzero(sim < self.cfg.sim_threshold)[0]
        return cand[np.argsort(sim[cand])[:k]]

    def _select(self, local, cur_vec) -> list:
        cfg, st = self.cfg, self.state
        budget = cfg.budget(st.precise)
        scale = cfg.alpha * (cfg.fine_gain if st.precise else 1.0)
        n_dyn, n_glob, n_loc = (int(round(k * scale)) for k in (cfg.k_dynamic, cfg.k_global, cfg.k_local))
        order, seen = [], set()

        def take(idx):
            n0 = len(order)
            for i in idx:
                i = int(i)
                if i not in seen and len(order) < budget:
                    seen.add(i)
                    order.append(i)
            return len(order) - n0

        comp = [take(self._dynamic(cur_vec, n_dyn).tolist())]
        comp.append(take(torch.topk(st.prev_global, min(n_glob, N_VIS)).indices.tolist())
                    if st.prev_global is not None else 0)
        comp.append(take(torch.topk(local, min(n_loc, N_VIS)).indices.tolist()))
        comp.append(take(torch.argsort(local, descending=True).tolist()))
        st.comp_hist.append(tuple(comp))
        return order

    @torch.no_grad()
    def prefix(self, ids, im):
        from transformers.cache_utils import DynamicCache

        cfg, st, m = self.cfg, self.state, self.m
        out = m.model._prepare_inputs_labels_for_multimodal(ids, None, None, None, None, None, im)
        h = out[4]
        n, dev = h.shape[1], h.device
        p0 = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        vis = torch.arange(p0, p0 + N_VIS, device=dev)
        is_vis = torch.zeros(n, dtype=torch.bool, device=dev)
        is_vis[vis] = True
        txt = (~is_vis).nonzero(as_tuple=True)[0]
        txt_after = txt[txt > p0]
        cos, sin = self.llm.rotary_emb(h, torch.arange(n, device=dev)[None])
        cur_vec = patch_vectors(im)

        sa0 = self.llm.layers[0].self_attn
        kvh, hd = sa0.config.num_key_value_heads, sa0.head_dim
        all_idx = torch.arange(n, device=dev)
        zero = torch.zeros(1, kvh, n, hd, dtype=h.dtype, device=dev)
        ks, vs = [], []
        local = torch.zeros(N_VIS, device=dev)
        for li in cfg.early_layers:
            h, K, V, w = self._layer(self.llm.layers[li], h, all_idx, zero, zero, cos, sin, True, n)
            ks.append(K)
            vs.append(V)
            a = w[0].float().mean(0)
            q = txt_after if txt_after.numel() else txt
            local += a[q][:, vis].mean(0)

        kept_vis = torch.tensor(sorted(self._select(local, cur_vec)), device=dev, dtype=torch.long)
        keep = torch.sort(torch.cat([txt, p0 + kept_vis])).values

        hk = h.index_select(1, keep)
        cos_k, sin_k = cos.index_select(1, keep), sin.index_select(1, keep)
        ks = [K.index_select(2, keep) for K in ks]
        vs = [V.index_select(2, keep) for V in vs]
        nk = int(keep.numel())
        idx_k = torch.arange(nk, device=dev)
        zero_k = torch.zeros(1, kvh, nk, hd, dtype=h.dtype, device=dev)
        for li in range(len(cfg.early_layers), self.L):
            hk, K, V, _ = self._layer(self.llm.layers[li], hk, idx_k, zero_k, zero_k, cos_k, sin_k, False, nk)
            ks.append(K)
            vs.append(V)

        cache = DynamicCache()
        for li in range(self.L):
            cache.update(ks[li], vs[li], li, {})
        self.pos_offset, self.n_kept = n, nk
        is_kept_vis = (keep >= p0) & (keep < p0 + N_VIS)
        self._vis_in_cache = is_kept_vis.nonzero(as_tuple=True)[0]
        self._vis_ids = keep[is_kept_vis] - p0
        st.frames = (st.frames + [cur_vec])[-3:]
        st.kept_hist.append(int(kept_vis.numel()))
        st.precise_hist.append(bool(st.precise))
        return cache, nk

    def capture_hooks(self):
        from transformers.models.qwen2 import modeling_qwen2 as mq

        acc = torch.zeros(N_VIS, device=self.llm.layers[0].self_attn.q_proj.weight.device)
        handles = []

        def make(li):
            def hook(module, args, kwargs):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                cos, sin = kwargs["position_embeddings"]
                cache = kwargs["past_key_values"]
                b, s, _ = hs.shape
                q = module.q_proj(hs).view(b, s, -1, module.head_dim).transpose(1, 2)
                q, _ = mq.apply_rotary_pos_emb(q, q, cos, sin)
                lay = cache.layers[li] if hasattr(cache, "layers") else None
                Kp = (lay.keys if lay is not None else cache.key_cache[li])[:, :, : self.n_kept]
                Kp = mq.repeat_kv(Kp, module.num_key_value_groups)
                w = torch.softmax((q @ Kp.transpose(-1, -2)).float() * module.scaling, dim=-1)
                sc = w.mean(dim=(0, 1, 2))
                acc.index_add_(0, self._vis_ids, sc[self._vis_in_cache])
            return hook

        for li in self.cfg.goal_layers:
            handles.append(self.llm.layers[li].self_attn.register_forward_pre_hook(make(li), with_kwargs=True))

        def finish():
            for hd_ in handles:
                hd_.remove()
            self.state.prev_global = acc.detach()
        return finish
