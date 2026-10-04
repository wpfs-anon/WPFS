from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_TOKEN_INDEX = -200
GRID, PATCH = 24, 14
N_VIS = GRID * GRID
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


@dataclass
class VCConfig:
    enabled: bool = True
    sim_threshold: float = 0.996
    static_top_k: int = 338
    task_top_k: int = 225
    attn_layer: int = 13
    growth_factor: float = 0.55
    max_reuse_frac: float = 1.0
    max_age: int = 1


@dataclass
class VCState:
    prev_vec: torch.Tensor | None = None
    prev_k: list = field(default_factory=list)
    prev_v: list = field(default_factory=list)
    prev_n: int | None = None
    prev_p0: int | None = None
    prev_attn: torch.Tensor | None = None
    schedule: list | None = None
    age: torch.Tensor | None = None
    reuse_hist: list = field(default_factory=list)


def patch_vectors(im: torch.Tensor) -> torch.Tensor:
    x = im.detach().float().cpu()
    while x.dim() > 3:
        x = x[0]
    x = (x * CLIP_STD + CLIP_MEAN)[:, : GRID * PATCH, : GRID * PATCH]
    return x.reshape(3, GRID, PATCH, GRID, PATCH).permute(1, 3, 0, 2, 4).reshape(N_VIS, -1)


def patch_similarity(cur: torch.Tensor, ref: torch.Tensor) -> np.ndarray:
    num = (cur * ref).sum(1)
    den = cur.norm(dim=1) * ref.norm(dim=1) + 1e-8
    return (num / den).numpy()


class VLACacheDB:
    def __init__(self, m, cfg: VCConfig | None = None):
        self.m = m
        self.cfg = cfg or VCConfig()
        self.llm = m.model.llm
        self.L = len(self.llm.layers)
        self.state = VCState()

    def reset(self):
        self.state = VCState()

    def _reusable_patches(self, cur_vec) -> np.ndarray:
        st, cfg = self.state, self.cfg
        sim = patch_similarity(cur_vec, st.prev_vec)
        cand = np.nonzero(sim >= cfg.sim_threshold)[0]
        if cand.size == 0:
            return cand
        cand = cand[np.argsort(-sim[cand])[: min(cfg.static_top_k, cand.size)]]
        if st.prev_attn is not None:
            hot = set(torch.topk(st.prev_attn, min(cfg.task_top_k, N_VIS)).indices.tolist())
            cand = cand[np.array([c not in hot for c in cand], dtype=bool)]
        if cfg.max_age > 0 and st.age is not None and cand.size:
            cand = cand[(st.age[cand] < cfg.max_age).numpy()]
        return cand

    def _schedule(self, entropies) -> list:
        e = torch.stack(entropies).float()
        lo, hi = e.min(), e.max()
        reuse = (1.0 - (e - lo) / (hi - lo + 1e-10)).tolist()
        for i in range(1, len(reuse)):
            d = reuse[i] - reuse[i - 1]
            reuse[i] = reuse[i - 1] + d * self.cfg.growth_factor if d > 0 else reuse[i]
        for i in range(1, len(reuse)):
            reuse[i] = max(reuse[i], reuse[i - 1])
        return [min(max(r, 0.0), self.cfg.max_reuse_frac) for r in reuse]

    def _layer(self, layer, h, cidx, K_prev, V_prev, cos, sin, want_attn, n):
        from transformers.models.qwen2 import modeling_qwen2 as mq

        sa = layer.self_attn
        hc = h.index_select(1, cidx)
        residual = hc
        x = layer.input_layernorm(hc)
        b, c, _ = x.shape
        hd = sa.head_dim
        q = sa.q_proj(x).view(b, c, -1, hd).transpose(1, 2)
        k = sa.k_proj(x).view(b, c, -1, hd).transpose(1, 2)
        v = sa.v_proj(x).view(b, c, -1, hd).transpose(1, 2)
        q, k = mq.apply_rotary_pos_emb(q, k, cos.index_select(1, cidx), sin.index_select(1, cidx))
        full = (c == n)
        if full:
            K, V = k, v
        else:
            K = K_prev.index_copy(2, cidx, k)
            V = V_prev.index_copy(2, cidx, v)
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        if full:
            att, _ = sdpa_attention_forward(sa, q, K, V, None, scaling=sa.scaling, is_causal=True)
        else:
            allow = torch.arange(n, device=h.device)[None, :] <= cidx[:, None]
            att, _ = sdpa_attention_forward(sa, q, K, V, allow[None, None], scaling=sa.scaling,
                                            is_causal=False)
        w = None
        if want_attn:
            Kr = mq.repeat_kv(K, sa.num_key_value_groups)
            logits = (q.float() @ Kr.float().transpose(-1, -2)) * sa.scaling
            allow = torch.arange(n, device=h.device)[None, :] <= cidx[:, None]
            w = torch.softmax(logits.masked_fill(~allow[None, None], float("-inf")), dim=-1)
        hc = residual + sa.o_proj(att.reshape(b, c, -1))
        hc = hc + layer.mlp(layer.post_attention_layernorm(hc))
        return h.index_copy(1, cidx, hc), K, V, w

    @torch.no_grad()
    def prefix(self, ids, im, force_full=False):
        from transformers.cache_utils import DynamicCache

        cfg, st, m = self.cfg, self.state, self.m
        out = m.model._prepare_inputs_labels_for_multimodal(ids, None, None, None, None, None, im)
        h = out[4]
        n, dev = h.shape[1], h.device
        p0 = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        vis = torch.arange(p0, p0 + N_VIS, device=dev)
        txt_after = torch.arange(p0 + N_VIS, n, device=dev)
        cos, sin = self.llm.rotary_emb(h, torch.arange(n, device=dev)[None])
        cur_vec = patch_vectors(im)

        ok = (cfg.enabled and not force_full and st.prev_k and st.prev_n == n
              and st.prev_p0 == p0 and st.schedule is not None)
        patches = self._reusable_patches(cur_vec) if ok else np.zeros(0, dtype=np.int64)
        reusable = torch.as_tensor(p0 + patches, device=dev, dtype=torch.long)

        want_all = st.schedule is None
        sa0 = self.llm.layers[0].self_attn
        zero = torch.zeros(1, sa0.config.num_key_value_heads, n, sa0.head_dim, dtype=h.dtype, device=dev)
        Kp = st.prev_k if ok else [zero] * self.L
        Vp = st.prev_v if ok else [zero] * self.L

        all_idx = torch.arange(n, device=dev)
        ks, vs, ents, used = [], [], [], []
        attn_scores = None
        for li, layer in enumerate(self.llm.layers):
            frac = 0.0 if not ok else st.schedule[min(li, len(st.schedule) - 1)]
            n_re = int(round(frac * reusable.numel()))
            if n_re > 0:
                keep = torch.ones(n, dtype=torch.bool, device=dev)
                keep[reusable[:n_re]] = False
                cidx = all_idx[keep]
            else:
                cidx = all_idx
            used.append(1.0 - cidx.numel() / n)
            want = want_all or li == cfg.attn_layer
            h, K, V, w = self._layer(layer, h, cidx, Kp[li], Vp[li], cos, sin, want, n)
            ks.append(K)
            vs.append(V)
            if w is not None:
                a = w[0].float().mean(0)
                if want_all:
                    p = a / (a.sum(-1, keepdim=True) + 1e-10)
                    ents.append(-(p * torch.log(p + 1e-10)).sum(-1).mean())
                if li == cfg.attn_layer and txt_after.numel():
                    rowmap = torch.full((n,), -1, dtype=torch.long, device=dev)
                    rowmap[cidx] = torch.arange(cidx.numel(), device=dev)
                    tr = rowmap[txt_after]
                    tr = tr[tr >= 0]
                    if tr.numel():
                        attn_scores = a[tr][:, vis].mean(0).cpu()

        if want_all and ents:
            st.schedule = self._schedule(ents)
        newage = torch.zeros(N_VIS, dtype=torch.long)
        if ok and patches.size:
            top = patches[: int(round(max(st.schedule) * patches.size))]
            if top.size:
                prev = st.age if st.age is not None else torch.zeros(N_VIS, dtype=torch.long)
                newage[top] = prev[top] + 1
        st.age = newage
        st.prev_k, st.prev_v, st.prev_n, st.prev_p0, st.prev_vec = ks, vs, n, p0, cur_vec
        if attn_scores is not None:
            st.prev_attn = attn_scores
        st.reuse_hist.append(float(np.mean(used)))

        cache = DynamicCache()
        for li in range(self.L):
            cache.update(ks[li], vs[li], li, {})
        return cache, n
