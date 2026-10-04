from dataclasses import dataclass, field

import numpy as np
import torch

from dboft_vlacache import IMAGE_TOKEN_INDEX, N_VIS, VLACacheDB, patch_similarity, patch_vectors

SCALE = N_VIS / 256.0


@dataclass
class OffConfig:
    static_prune_ratio: float = 0.8
    attn_topk_base: int = 30
    primary_topk: int = 236
    primary_topk_precise: int = 240
    sim_threshold: float = 0.986
    goal_layers: tuple = (12, 26)
    global_capture_step: int = -1
    vt_thresh: float = 0.0074
    vr_thresh: float = 0.0233
    dz_thresh: float = 0.0
    vt_exit: float = 0.0107
    vr_exit: float = 0.0317

    def attn_topk(self, precise: bool) -> int:
        k = int(self.attn_topk_base * self.static_prune_ratio)
        if precise:
            r = self.static_prune_ratio
            k = int(k * r ** 2) if r <= 1.0 else int(k * r)
        return int(round(k * SCALE))

    def static_cap(self, precise: bool) -> int:
        return int(round((self.primary_topk_precise if precise else self.primary_topk) * SCALE))


@dataclass
class OffState:
    prev_global: torch.Tensor | None = None
    prev_frame: torch.Tensor | None = None
    precise: bool = False
    kept_hist: list = field(default_factory=list)
    precise_hist: list = field(default_factory=list)
    comp_hist: list = field(default_factory=list)


class SpecPruneOff:
    def __init__(self, m, cfg: OffConfig | None = None):
        self.m = m
        self.cfg = cfg or OffConfig()
        self.llm = m.model.llm
        self.L = len(self.llm.layers)
        self.state = OffState()
        self._layer = VLACacheDB(m)._layer
        self.pos_offset = None
        self.n_kept = None
        self.layer_lengths = None

    def reset(self):
        self.state = OffState()

    def update_mode(self, a_env) -> None:
        cfg, st = self.cfg, self.state
        vt, vr, dz = float(np.linalg.norm(a_env[:3])), float(np.linalg.norm(a_env[3:6])), float(a_env[2])
        if st.precise:
            if vt > cfg.vt_exit or vr > cfg.vr_exit:
                st.precise = False
        elif vt < cfg.vt_thresh and vr < cfg.vr_thresh and dz <= cfg.dz_thresh:
            st.precise = True

    def _static(self, cur_vec, cap) -> torch.Tensor:
        ref = self.state.prev_frame if self.state.prev_frame is not None else cur_vec
        sim = patch_similarity(cur_vec, ref)
        cand = np.nonzero(sim >= self.cfg.sim_threshold)[0]
        if cand.size == 0 or cap <= 0:
            return torch.zeros(0, dtype=torch.long)
        k = min(cap, cand.size)
        top = cand[np.argpartition(sim[cand], -k)[-k:]]
        return torch.as_tensor(np.sort(top), dtype=torch.long)

    @staticmethod
    def _text_to_vis(w, rows, vis_cols):
        a = w[0].float().mean(0)
        return a[rows][:, vis_cols].mean(0)

    @torch.no_grad()
    def prefix(self, ids, im):
        from transformers.cache_utils import DynamicCache

        cfg, st, m = self.cfg, self.state, self.m
        out = m.model._prepare_inputs_labels_for_multimodal(ids, None, None, None, None, None, im)
        h = out[4]
        n, dev = h.shape[1], h.device
        p0 = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        is_vis = torch.zeros(n, dtype=torch.bool, device=dev)
        is_vis[p0:p0 + N_VIS] = True
        txt = (~is_vis).nonzero(as_tuple=True)[0]
        txt_after = txt[txt > p0]
        cos, sin = self.llm.rotary_emb(h, torch.arange(n, device=dev)[None])
        cur_vec = patch_vectors(im)
        precise = st.precise
        k = cfg.attn_topk(precise)
        static = self._static(cur_vec, cfg.static_cap(precise)).to(dev)
        glob = st.prev_global.to(dev) if st.prev_global is not None else torch.zeros(0, dtype=torch.long, device=dev)
        sa0 = self.llm.layers[0].self_attn
        kvh, hd = sa0.config.num_key_value_heads, sa0.head_dim

        def zeros(c):
            z = torch.zeros(1, kvh, c, hd, dtype=h.dtype, device=dev)
            return z, z

        def rows_in(seq, want):
            pos = torch.full((n,), -1, dtype=torch.long, device=dev)
            pos[seq] = torch.arange(seq.numel(), device=dev)
            r = pos[want]
            return r[r >= 0]

        all_idx = torch.arange(n, device=dev)
        h, K0, V0, w0 = self._layer(self.llm.layers[0], h, all_idx, *zeros(n), cos, sin, True, n)
        s0 = self._text_to_vis(w0, txt_after, all_idx[is_vis])
        top2k = torch.topk(s0, min(2 * k, N_VIS)).indices
        topk0 = top2k[:k]
        keep1 = torch.ones(N_VIS, dtype=torch.bool, device=dev)
        keep1[static] = False
        keep1[top2k] = True
        keep1[glob] = True
        S1 = torch.sort(torch.cat([txt, p0 + keep1.nonzero(as_tuple=True)[0]])).values

        h1 = h.index_select(1, S1)
        c1, s1_ = cos.index_select(1, S1), sin.index_select(1, S1)
        n1 = S1.numel()
        h1, K1, V1, w1 = self._layer(self.llm.layers[1], h1, torch.arange(n1, device=dev), *zeros(n1),
                                     c1, s1_, True, n1)
        vis_in_S1 = (S1 >= p0) & (S1 < p0 + N_VIS)
        s1 = self._text_to_vis(w1, rows_in(S1, txt_after), vis_in_S1.nonzero(as_tuple=True)[0])
        vis_ids1 = S1[vis_in_S1] - p0
        cur1 = vis_ids1[torch.topk(s1, min(k, s1.numel())).indices]
        keep2 = keep1.clone()
        keep2[static] = False
        keep2[cur1] = True
        keep2[topk0] = True
        keep2[glob] = True
        keep2 &= keep1
        S2 = torch.sort(torch.cat([txt, p0 + keep2.nonzero(as_tuple=True)[0]])).values

        hk = h1.index_select(1, rows_in(S1, S2))
        c2, s2_ = cos.index_select(1, S2), sin.index_select(1, S2)
        n2 = S2.numel()
        vis_in_S2 = ((S2 >= p0) & (S2 < p0 + N_VIS)).nonzero(as_tuple=True)[0]
        rows2 = rows_in(S2, txt_after)
        ks, vs = [K0, K1], [V0, V1]
        new_glob = []
        for li in range(2, self.L):
            want = li in cfg.goal_layers
            hk, K, V, w = self._layer(self.llm.layers[li], hk, torch.arange(n2, device=dev), *zeros(n2),
                                      c2, s2_, want, n2)
            ks.append(K)
            vs.append(V)
            if want:
                sc = self._text_to_vis(w, rows2, vis_in_S2)
                new_glob.append((S2[vis_in_S2] - p0)[torch.topk(sc, min(k // 2, sc.numel())).indices])

        cache = DynamicCache()
        for li in range(self.L):
            cache.update(ks[li], vs[li], li, {})
        self.pos_offset, self.n_kept = n, n2
        self.layer_lengths = [n, n1] + [n2] * (self.L - 2)
        st.prev_global = torch.unique(torch.cat(new_glob)).cpu() if new_glob else None
        st.prev_frame = cur_vec
        kept = int(keep2.sum())
        st.kept_hist.append(kept)
        st.precise_hist.append(bool(precise))
        st.comp_hist.append((int(static.numel()), int(top2k.numel()), int(cur1.numel()), int(glob.numel()), kept))
        return cache, n2

    def ddim_hooks(self, L):
        masks, handles = {}, []
        dev = self.llm.layers[0].self_attn.q_proj.weight.device
        tri = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
        for P in set(self.layer_lengths):
            masks[P] = torch.cat([torch.ones(L, P, dtype=torch.bool, device=dev), tri], 1)[None, None]

        def make(li):
            P = self.layer_lengths[li]

            def hook(module, args, kwargs):
                kwargs["attention_mask"] = masks[P]
                return args, kwargs
            return hook

        for li, layer in enumerate(self.llm.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(make(li), with_kwargs=True))

        def remove():
            for hd_ in handles:
                hd_.remove()
        return remove

    def crop(self, kv):
        for li, P in enumerate(self.layer_lengths):
            kv.layers[li].crop(P)
