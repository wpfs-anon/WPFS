import os as _os
HOME = _os.environ.get(
    "CORRECTOR_HOME",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import argparse
import json
import os
import sys
import contextlib
import copy
import time
import traceback

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["baseline", "teacher", "aac", "effvla", "vlacache", "specprune"], default="teacher")
ap.add_argument("--k", type=int, default=2, help="draws per correction")
ap.add_argument("--start-step", type=int, default=6,
                help="DDIM step (0..9) a correction starts from; step 8 = abar 0.967, 2 steps left")
ap.add_argument("--no-cache", action="store_true", help="disable the prefix KV cache (control run)")
ap.add_argument("--grip-lock", action="store_true",
                help="a correction never executes a gripper change: the segment is cut just before "
                     "the change and the next call is forced to plan"
                     "")
ap.add_argument("--grip-lock-mode", choices=["both", "reopen"], default="both",
                help="both: block every gripper change inside a correction; reopen: block only "
                     "closed -> open"
                     "")
ap.add_argument("--grip-fresh", type=int, default=0,
                help="DDIM steps of a fresh coarse plan batched with the correction to certify the "
                     "gripper (0 = off)")
ap.add_argument("--c-safe", type=int, default=5, help="actions executed after a plan")
ap.add_argument("--c-max", type=int, default=10, help="cap on actions executed after a correction")
ap.add_argument("--n-min", type=int, default=2, help="agreement below this triggers a replan")
ap.add_argument("--lineage", type=int, default=50, help="maximum steps before a plan is forced")
ap.add_argument("--agree-tau", type=float, default=1.0)
ap.add_argument("--replan", type=int, default=5, help="baseline-style modes: actions per call")
ap.add_argument("--aac-n", type=int, default=5,
                help="AAC: chunks sampled per plan to estimate the entropy (paper: 20)")
ap.add_argument("--aac-move-th", type=float, default=0.15,
                help="AAC: minimum-motion floor (alpha, Eq. 6).  The paper uses 3 in "
                     "LIBERO's normalised [-1, 1] action space; Bridge actions are in metres/radians, "
                     "so the scale must be re-measured (--aac-dump)")
ap.add_argument("--ev-interval", type=int, default=3,
                help="EfficientVLA: recompute each decoder layer every N DDIM steps and reuse its "
                     "stored contribution in between; N=1 disables the cache (identical to baseline)")
ap.add_argument("--vc-max-age", type=int, default=1, help="VLA-Cache: plans a token may carry a reused KV (0 = unlimited, as in the paper)")
ap.add_argument("--vc-sim", type=float, default=0.996, help="VLA-Cache: patch similarity threshold (Eq. 5)")
ap.add_argument("--vc-static-k", type=int, default=338, help="VLA-Cache: top-k static tokens (the paper's 150/256, scaled to 576)")
ap.add_argument("--vc-task-k", type=int, default=225, help="VLA-Cache: task-relevant tokens excluded from reuse (100/256 -> 576)")
ap.add_argument("--vc-attn-layer", type=int, default=13, help="VLA-Cache: layer whose text->vision attention is read (15/32 -> 13/28)")
ap.add_argument("--sp-alpha", type=float, default=2.0, help="SpecPrune: visual-token budget factor (2.0, the pi_0 choice)")
ap.add_argument("--sp-impl", choices=("paper", "official"), default="paper",
                help="SpecPrune: paper = the paper's text (dboft_specprune.py); official = the authors' released rule (dboft_specprune_off.py), where --sp-alpha is STATIC_PRUNE_RATIO")
ap.add_argument("--bench-sp", action="store_true", help="check SpecPrune (keep-all must match the baseline; real alpha: speed and deviation), then exit")
ap.add_argument("--bench-vc", action="store_true", help="check VLA-Cache (agreement with the plain prefill, and speed), then exit")
ap.add_argument("--bench-ev", action="store_true",
                help="measure the deviation and speed of the EfficientVLA cache, then exit")
ap.add_argument("--bench", action="store_true",
                help="time the prefill against the 10 DDIM steps on the first frame, then exit")
ap.add_argument("--aac-dump", default="", help="AAC: write h*, the elbow and the magnitude curve to this file")
ap.add_argument("--cal-calls", type=int, default=12,
                help="first calls of each task used to measure SREF (run as baseline meanwhile)")
ap.add_argument("--cal-m", type=int, default=8, help="independent plans per SREF measurement")
ap.add_argument("--cal-first", type=int, default=12,
                help="only the first N calls of each episode feed SREF (active states only)"
                     "")
ap.add_argument("--sref-file", default="",
                help="per-task SREF (json): loaded if present, otherwise measured with --cal-calls "
                     "and written here"
                     "")
ap.add_argument("--harvest", default="",
                help="directory for student data: per correction the CLIP features, prompt, anchor, "
                     "age and (t, x_t, teacher eps) of --k-harvest draws at every DDIM step")
ap.add_argument("--k-harvest", type=int, default=8,
                help="draws recorded while harvesting; the first --k still feed the certificate")
ap.add_argument("--student", default="",
                help="student checkpoint: corrections run the student (CLIP + small net, 2 DDIM "
                     "steps) instead of the 7B LLM")
ap.add_argument("--log", default=os.environ.get("DBOFT_HOME", f"{HOME}/dboft") + "/results/dboft/calls.jsonl")
a = ap.parse_args()

_H = os.environ.get("DBOFT_HOME", f"{HOME}/dboft")
os.chdir(_H + "/dexbotic")
sys.path.insert(0, _H + "/dexbotic")
sys.path.insert(0, _H + "/dexbotic/playground/benchmarks/simpler")
os.makedirs(os.path.dirname(a.log), exist_ok=True)
LOGF = open(a.log, "a")


def log_call(**kv):
    LOGF.write(json.dumps(kv) + "\n")
    LOGF.flush()


if a.mode == "aac":
    import types as _t
    _added = [_m for _m in ("matplotlib", "matplotlib.pyplot") if _m not in sys.modules]
    for _m in _added:
        sys.modules[_m] = _t.ModuleType(_m)
    sys.path.insert(0, f"{HOME}/third_party/aac_libero")
    try:
        from action_optimization.action_entropy_pi05 import (select_chunk_size, action_magnitude,
                                                             convert_gripper_to_binary)
    finally:
        for _m in _added:
            sys.modules.pop(_m, None)


class LayerCache:
    def __init__(self, llm, interval):
        self.layers = llm.layers
        self.interval = max(1, interval)
        self.delta = [None] * len(self.layers)
        self.recompute = True
        self.astuple = True
        self._orig = None

    def __enter__(self):
        self._orig = [l.forward for l in self.layers]
        for li, layer in enumerate(self.layers):
            layer.forward = self._wrap(li, self._orig[li])
        return self

    def __exit__(self, *exc):
        for layer, f in zip(self.layers, self._orig):
            layer.forward = f
        return False

    def _wrap(self, li, orig):
        def f(hidden_states, *args, **kw):
            if self.recompute or self.delta[li] is None:
                out = orig(hidden_states, *args, **kw)
                h = out[0] if isinstance(out, tuple) else out
                self.delta[li] = (h - hidden_states).detach()
                self.astuple = isinstance(out, tuple)
                return out
            h = hidden_states + self.delta[li]
            return (h,) if self.astuple else h
        return f

    def step(self, i):
        self.recompute = (i % self.interval == 0)


VC = None


SP = None


FORCE_PER_LAYER = False


def ddim_masks(llm, lens, L):
    dev = llm.layers[0].self_attn.q_proj.weight.device
    tri = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    masks = {P: torch.cat([torch.ones(L, P, dtype=torch.bool, device=dev), tri], 1)[None, None]
             for P in set(lens)}
    handles = []
    for li, layer in enumerate(llm.layers):
        def hook(module, args, kwargs, _m=masks[lens[li]]):
            kwargs["attention_mask"] = _m
            return args, kwargs
        handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))

    def remove():
        for h_ in handles:
            h_.remove()
    return remove


def make_sp(m, keep_all=False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if a.sp_impl == "official":
        from dboft_specprune_off import SpecPruneOff, OffConfig
        if keep_all:
            return SpecPruneOff(m, OffConfig(static_prune_ratio=a.sp_alpha, primary_topk=0, primary_topk_precise=0))
        return SpecPruneOff(m, OffConfig(static_prune_ratio=a.sp_alpha))
    from dboft_specprune import SpecPruneDB, SPConfig
    return SpecPruneDB(m, SPConfig(alpha=100.0 if keep_all else a.sp_alpha))


def sp_prefix(m, ids, im):
    global SP
    if SP is None:
        SP = make_sp(m)
    return SP.prefix(ids, im)


def vc_cfg():
    from dboft_vlacache import VCConfig
    return VCConfig(sim_threshold=a.vc_sim, static_top_k=a.vc_static_k, task_top_k=a.vc_task_k,
                    attn_layer=a.vc_attn_layer, max_age=a.vc_max_age)


def vc_prefix(m, ids, im):
    global VC
    if VC is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dboft_vlacache import VLACacheDB
        VC = VLACacheDB(m, vc_cfg())
    return VC.prefix(ids, im)


def aac_dict(m, x, norms):
    A = np.stack([m._denorm(d.float().cpu().numpy(), norms) for d in x])
    return {"normalized_action": x[..., :7].float().cpu().numpy(),
            "action.end_effector_position": A[..., :3],
            "action.end_effector_rotation": A[..., 3:6],
            "action.gripper_close": A[..., 6] - 0.5}


def aac_h(m, x, norms, move_th, C, dump=""):
    d = aac_dict(m, x, norms)
    h, br = select_chunk_size(d, method="gaussian_bernoulli", move_th=move_th)
    h_ent = max(int(np.argmax(np.diff(br["chunk_mean"]))) + 1, 2)
    if dump:
        dd = dict(d)
        dd["action.gripper_close"] = convert_gripper_to_binary(dd["action.gripper_close"])
        with open(dump, "a") as fh:
            fh.write(json.dumps({"h": int(h), "h_ent": int(h_ent),
                                 "mag": [round(float(v), 4) for v in action_magnitude(dd)]}) + "\n")
    return max(2, min(int(h), C)), h_ent


class Teacher:
    def __init__(self):
        self.sref = {}
        self.sref_fixed = (json.load(open(a.sref_file))
                             if a.sref_file and os.path.exists(a.sref_file) else {})
        if self.sref_fixed:
            print("*** fixed SREF loaded from %s: %d tasks ***" % (a.sref_file, len(self.sref_fixed)), flush=True)
        self.cal = {}
        self.A = None
        self.used = 0
        self.last_seg = 0
        self.i0 = None
        self.ep = -1
        self.prompt = None
        self.sch_fresh = None
        self.self_checked = False
        self.g_cur = None
        self.calls_in_ep = 0
        self.hv_buf, self.hv_txt, self.hv_n = [], {}, 0
        self.stu, self.stu_txt = None, {}
        if a.student:
            sys.path.insert(0, f"{HOME}/scripts")
            from dboft_student_net import DboftStudent
            ck = torch.load(a.student, map_location="cpu", weights_only=False)
            self.stu = DboftStudent(**ck["cfg"]).cuda().eval()
            self.stu.load_state_dict(ck["state"])
            self.stu_txt = {k: v.cuda().float() for k, v in ck["txt"].items()}
            print("*** STUDENT %s: %s, val %s ***" % (a.student, ck["cfg"], ck.get("val")), flush=True)
        self.force_plan = False

    @torch.no_grad()
    def prefix(self, m, ids, im):
        out = m.model._prepare_inputs_labels_for_multimodal(ids, None, None, None, None, None, im)
        pos, att, emb = out[1], out[2], out[4]
        kv = m.model.llm(inputs_embeds=emb, attention_mask=att, position_ids=pos,
                         use_cache=True).past_key_values
        return kv, emb.shape[1]

    @torch.no_grad()
    def sample(self, m, ids, im, B, anchor=None, i0=0, pre=None, eps=None):
        sch = m.model.action_head.noise_scheduler
        sch.set_timesteps(10)
        ts = sch.timesteps
        C, D = m.config.chunk_size, m.config.action_dim
        dev = ids.device
        if eps is None:
            eps = torch.randn(B, C, D, device=dev, dtype=im.dtype)
        if anchor is None:
            x, start = eps, 0
        else:
            an = anchor.to(dev, im.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
            x, start = sch.add_noise(an, eps, torch.full((B,), int(ts[i0]), device=dev,
                                                           dtype=torch.long)), i0
        if a.no_cache:
            idsB, imB = ids.repeat(B, 1), im.repeat(B, *([1] * (im.dim() - 1)))
            for t in ts[start:]:
                te = torch.Tensor([t]).to(dev)
                emb = m.model.action_head.time_encoder(te).to(x.dtype).to(dev)
                emb = emb.unsqueeze(1).expand(B, -1, -1)
                nd = {"noise": eps, "noisy_actions": x, "diffusion_timestep_embeddings": emb}
                x = sch.step(m(idsB, images=imB, use_cache=True, noisy_dict=nd).logits,
                             t, x).prev_sample
            return x, len(ts) - start
        kv0, P = pre
        kv = copy.deepcopy(kv0)
        if B > 1:
            kv.batch_repeat_interleave(B)
        ctx = LayerCache(m.model.llm, a.ev_interval) if a.mode == "effvla" else contextlib.nullcontext()
        POSB = SP.pos_offset if (a.mode == "specprune" and SP is not None and SP.pos_offset) else P
        lens = [lay.get_seq_length() for lay in kv.layers] if hasattr(kv, "layers") else []
        per_layer = bool(lens) and (len(set(lens)) > 1 or FORCE_PER_LAYER)
        undo = ddim_masks(m.model.llm, lens, 1 + C * D) if per_layer else None
        try:
            with ctx as lc:
                for _i, t in enumerate(ts[start:]):
                    if lc is not None:
                        lc.step(_i)
                    te = torch.Tensor([t]).to(dev)
                    temb = m.model.action_head.time_encoder(te).to(x.dtype).to(dev)
                    temb = temb.unsqueeze(1).expand(B, -1, -1)
                    a_in = torch.cat([temb, m.model.action_head.noisy_action_projector(
                        x.reshape(B, -1).unsqueeze(-1))], dim=1)
                    L = a_in.shape[1]
                    fin = (SP.capture_hooks() if (a.mode == "specprune" and SP is not None and B == 1
                                                  and hasattr(SP, "capture_hooks")
                                                  and _i == SP.cfg.global_capture_step) else None)
                    o = m.model.llm(inputs_embeds=a_in, past_key_values=kv,
                                    position_ids=torch.arange(POSB, POSB + L, device=dev).unsqueeze(0).expand(B, -1),
                                    cache_position=torch.arange(P, P + L, device=dev),
                                    use_cache=True, output_hidden_states=True)
                    if fin is not None:
                        fin()
                    if per_layer:
                        for li, Pl in enumerate(lens):
                            kv.layers[li].crop(Pl)
                    else:
                        kv.crop(P)
                    pred = m.model.action_head.predict_noise(o.hidden_states[-1][:, 1:, :])
                    x = sch.step(pred.reshape(x.shape), t, x).prev_sample
        finally:
            if undo is not None:
                undo()
        return x, len(ts) - start

    @torch.no_grad()
    def sample_corr_fresh(self, m, ids, im, K, anchor, i0, pre, nf, eps_c=None):
        sch = m.model.action_head.noise_scheduler
        sch.set_timesteps(10)
        tc = list(sch.timesteps[i0:])
        if self.sch_fresh is None or self.sch_fresh.num_inference_steps != nf:
            self.sch_fresh = copy.deepcopy(sch)
            self.sch_fresh.set_timesteps(nf)
        tf = list(self.sch_fresh.timesteps)
        C, D = m.config.chunk_size, m.config.action_dim
        dev, dt = ids.device, im.dtype
        eps = torch.randn(K, C, D, device=dev, dtype=dt) if eps_c is None else eps_c
        an = anchor.to(dev, dt).unsqueeze(0).expand(K, -1, -1).contiguous()
        xc = sch.add_noise(an, eps, torch.full((K,), int(tc[0]), device=dev, dtype=torch.long))
        xf = torch.randn(1, C, D, device=dev, dtype=dt)
        kv0, P = pre
        caches = {}

        def kv_for(B):
            if B not in caches:
                c = copy.deepcopy(kv0)
                if B > 1:
                    c.batch_repeat_interleave(B)
                caches[B] = c
            return caches[B]

        ah = m.model.action_head

        def temb(t, n):
            e = ah.time_encoder(torch.Tensor([t]).to(dev)).to(dt).to(dev)
            return e.unsqueeze(1).expand(n, -1, -1)

        n_it = max(len(tc), len(tf))
        off_f, off_c = n_it - len(tf), n_it - len(tc)
        for j in range(n_it):
            af, ac = j >= off_f, j >= off_c
            xs, ts_ = [], []
            if af:
                xs.append(xf)
                ts_.append(temb(tf[j - off_f], 1))
            if ac:
                xs.append(xc)
                ts_.append(temb(tc[j - off_c], K))
            x = torch.cat(xs, 0)
            B = x.shape[0]
            a_in = torch.cat([torch.cat(ts_, 0),
                              ah.noisy_action_projector(x.reshape(B, -1).unsqueeze(-1))], dim=1)
            L = a_in.shape[1]
            kv = kv_for(B)
            o = m.model.llm(inputs_embeds=a_in, past_key_values=kv,
                            position_ids=torch.arange(P, P + L, device=dev).unsqueeze(0).expand(B, -1),
                            cache_position=torch.arange(P, P + L, device=dev),
                            use_cache=True, output_hidden_states=True)
            kv.crop(P)
            pred = ah.predict_noise(o.hidden_states[-1][:, 1:, :]).reshape(x.shape)
            i = 0
            if af:
                xf = self.sch_fresh.step(pred[:1], tf[j - off_f], xf).prev_sample
                i = 1
            if ac:
                xc = sch.step(pred[i:], tc[j - off_c], xc).prev_sample
        return xc, xf, n_it

    @staticmethod
    def grip_mismatch(fresh, chunk, norms, m, h):
        gf = m._denorm(fresh.float().cpu().numpy(), norms)[:h, 6] > 0.5
        gc = m._denorm(chunk.float().cpu().numpy(), norms)[:h, 6] > 0.5
        d = np.nonzero(gf != gc)[0]
        return int(d[0]) if len(d) else h

    @torch.no_grad()
    def sample_student(self, m, ids, im, K, anchor, i0):
        sch = m.model.action_head.noise_scheduler
        sch.set_timesteps(10)
        ts = sch.timesteps
        dev = ids.device
        clip = m.model.mm_vision_module(im)
        if isinstance(clip, (tuple, list)):
            clip = clip[0]
        p = self.prompt
        if p not in self.stu_txt:
            tid = ids[0][ids[0] >= 0]
            self.stu_txt[p] = m.model.llm.get_input_embeddings()(tid).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem = self.stu.memory(clip[:1].float(), self.stu_txt[p][None], anchor[None].to(dev).float())
        m1 = mem.expand(K, -1, -1)
        H, D = anchor.shape
        eps = torch.randn(K, H, D, device=dev, dtype=torch.float32)
        an = anchor.to(dev).float()[None].expand(K, -1, -1).contiguous()
        x = sch.add_noise(an, eps, torch.full((K,), int(ts[i0]), device=dev, dtype=torch.long))
        age = torch.full((K,), float(self.used), device=dev)
        for t in ts[i0:]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = self.stu(x, torch.full((K,), float(t), device=dev), m1, age)
            x = sch.step(e.float(), t, x).prev_sample
        return x.to(im.dtype), len(ts) - i0

    @torch.no_grad()
    def sample_rec(self, m, ids, im, B, anchor, i0, pre):
        sch = m.model.action_head.noise_scheduler
        sch.set_timesteps(10)
        ts = sch.timesteps
        C, D = m.config.chunk_size, m.config.action_dim
        dev = ids.device
        eps = torch.randn(B, C, D, device=dev, dtype=im.dtype)
        an = anchor.to(dev, im.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
        x = sch.add_noise(an, eps, torch.full((B,), int(ts[i0]), device=dev, dtype=torch.long))
        kv0, P = pre
        kv = copy.deepcopy(kv0)
        if B > 1:
            kv.batch_repeat_interleave(B)
        rec = []
        for t in ts[i0:]:
            te = torch.Tensor([t]).to(dev)
            temb = m.model.action_head.time_encoder(te).to(x.dtype).to(dev)
            temb = temb.unsqueeze(1).expand(B, -1, -1)
            a_in = torch.cat([temb, m.model.action_head.noisy_action_projector(
                x.reshape(B, -1).unsqueeze(-1))], dim=1)
            L = a_in.shape[1]
            o = m.model.llm(inputs_embeds=a_in, past_key_values=kv,
                            position_ids=torch.arange(P, P + L, device=dev).unsqueeze(0).expand(B, -1),
                            cache_position=torch.arange(P, P + L, device=dev),
                            use_cache=True, output_hidden_states=True)
            kv.crop(P)
            pred = m.model.action_head.predict_noise(o.hidden_states[-1][:, 1:, :]).reshape(x.shape)
            rec.append((int(t), x.float().cpu().half(), pred.float().cpu().half()))
            x = sch.step(pred, t, x).prev_sample
        return x, rec

    @torch.no_grad()
    def features(self, m, ids, im):
        clip = m.model.mm_vision_module(im)
        if isinstance(clip, (tuple, list)):
            clip = clip[0]
        p = self.prompt
        if p not in self.hv_txt:
            tid = ids[0][ids[0] >= 0]
            self.hv_txt[p] = m.model.llm.get_input_embeddings()(tid).float().cpu().half()
        return clip[0].float().cpu().half()

    def flush_harvest(self):
        if not a.harvest or not self.hv_buf:
            return
        os.makedirs(a.harvest, exist_ok=True)
        f = os.path.join(a.harvest, "shard_%04d.pt" % self.hv_n)
        torch.save({"rec": self.hv_buf, "txt": dict(self.hv_txt)}, f + ".part")
        os.replace(f + ".part", f)
        print("*** harvest: %d records -> %s ***" % (len(self.hv_buf), f), flush=True)
        self.hv_buf, self.hv_n = [], self.hv_n + 1

    def set_start_step(self, m):
        sch = m.model.action_head.noise_scheduler
        sch.set_timesteps(10)
        ac = sch.alphas_cumprod.cpu()[sch.timesteps.cpu()]
        self.i0 = a.start_step
        print("*** corrections start at DDIM step %d/10 (abar=%.3f) -> %d DDIM steps | prefix cache %s ***"
              % (self.i0, float(ac[self.i0]), 10 - self.i0, "OFF" if a.no_cache else "ON"), flush=True)

    @staticmethod
    def spread(x):
        p = x[:, :, :6].float()
        B = p.shape[0]
        pairs = [torch.linalg.vector_norm(p[i] - p[j], dim=-1)
               for i in range(B) for j in range(i + 1, B)]
        return torch.stack(pairs).median(dim=0).values

    def certificate(self, draws, sref, norms, m):
        h = min(a.c_max, draws.shape[1])
        ok_pose = self.spread(draws)[:h] <= a.agree_tau * sref[:h]
        g = np.stack([m._denorm(d.float().cpu().numpy(), norms)[:h, 6] for d in draws]) > 0.5
        ok_grip = torch.as_tensor(g.all(0) | (~g).all(0), device=ok_pose.device)
        ok = (ok_pose & ok_grip).tolist()
        n = 0
        for v in ok:
            if not v:
                break
            n += 1
        return n

    def infer(self, m, ids, im, args):
        norms = args["action_norms"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if self.i0 is None:
            self.set_start_step(m)
        _pc = {}

        def PRE():
            if "v" not in _pc:
                if a.no_cache:
                    _pc["v"] = None
                elif a.mode == "vlacache":
                    _pc["v"] = vc_prefix(m, ids, im)
                elif a.mode == "specprune":
                    _pc["v"] = sp_prefix(m, ids, im)
                else:
                    _pc["v"] = self.prefix(m, ids, im)
            return _pc["v"]
        self.calls_in_ep += 1
        p = self.prompt
        kind, n, steps, B, ratio, gd, gl, gdir = None, None, 0, 1, None, None, None, None
        hv = None
        g_prev = self.g_cur

        if a.mode != "baseline" and p not in self.sref and p in self.sref_fixed:
            self.sref[p] = torch.tensor(self.sref_fixed[p], device=ids.device)
        if a.mode == "baseline":
            x, steps = self.sample(m, ids, im, 1, pre=PRE())
            chunk, seg, kind = x[0], a.replan, "plan"
        elif a.mode == "specprune":
            x, steps = self.sample(m, ids, im, 1, pre=PRE())
            chunk, seg, kind = x[0], a.replan, "plan"
            SP.update_mode(m._denorm(chunk[:1].float().cpu().numpy(), norms)[0])
        elif a.mode == "vlacache":
            x, steps = self.sample(m, ids, im, 1, pre=PRE())
            chunk, seg, kind = x[0], a.replan, "plan"
        elif a.mode == "effvla":
            x, steps = self.sample(m, ids, im, 1, pre=PRE())
            chunk, seg, kind = x[0], a.replan, "plan"
        elif a.mode == "aac":
            x, steps = self.sample(m, ids, im, a.aac_n, pre=PRE())
            B = a.aac_n
            seg, h_ent_out = aac_h(m, x, norms, a.aac_move_th, m.config.chunk_size, a.aac_dump)
            chunk, kind, n = x[0], "plan", h_ent_out
        elif p not in self.sref:
            x, steps = self.sample(m, ids, im, a.cal_m, pre=PRE())
            B = a.cal_m
            if self.calls_in_ep <= a.cal_first:
                self.cal.setdefault(p, []).append(self.spread(x).cpu())
            if len(self.cal.get(p, [])) >= a.cal_calls:
                self.sref[p] = torch.stack(self.cal[p]).median(dim=0).values.to(x.device)
                s = self.sref[p]
                print("*** SREF task %r: h0 %.4f h4 %.4f h9 %.4f (from %d calls) ***"
                      % (p[-40:], s[0], s[4], s[9], len(self.cal[p])), flush=True)
                if a.sref_file:
                    old = json.load(open(a.sref_file)) if os.path.exists(a.sref_file) else {}
                    old[p] = [float(v) for v in s.cpu()]
                    json.dump(old, open(a.sref_file, "w"), indent=1)
            chunk, seg, kind = x[0], a.replan, "cal"
            self.used = seg
        else:
            need_plan = self.A is None or self.used >= a.lineage or self.force_plan
            if not need_plan:
                rest = self.A[self.last_seg:]
                if rest.shape[0] == 0:
                    rest = self.A[-1:]
                pad = rest[-1:].expand(self.A.shape[0] - rest.shape[0], -1)
                stale = torch.cat([rest, pad], dim=0)[:self.A.shape[0]]
                if a.grip_fresh > 0 and not a.no_cache and not self.self_checked:
                    self.self_checked = True
                    e = torch.randn(a.k, *stale.shape, device=ids.device, dtype=im.dtype)
                    x1, _ = self.sample(m, ids, im, a.k, anchor=stale, i0=self.i0, pre=PRE(), eps=e)
                    x2, _, _ = self.sample_corr_fresh(m, ids, im, a.k, stale, self.i0, PRE(),
                                                      a.grip_fresh, eps_c=e)
                    print("*** mixed-batch self-check: correction differs by max %.4f mean %.5f "
                          "(natural draw spread ~0.2) ***"
                          % (float((x1.float() - x2.float()).abs().max()),
                             float((x1.float() - x2.float()).abs().mean())), flush=True)
                if a.grip_fresh > 0 and not a.no_cache:
                    draws, fresh, steps = self.sample_corr_fresh(m, ids, im, a.k, stale, self.i0,
                                                                PRE(), a.grip_fresh)
                    B = a.k + 1
                elif self.stu is not None and a.harvest:
                    draws, steps = self.sample_student(m, ids, im, a.k, stale, self.i0)
                    B = a.k
                    _, rec = self.sample_rec(m, ids, im, a.k_harvest, stale, self.i0, PRE())
                    hv = {"clip": self.features(m, ids, im), "prompt": p, "ep": self.ep,
                          "call": self.calls_in_ep, "age": int(self.used), "anchor": stale.float().cpu().half(),
                          "steps": rec, "norms": norms, "dagger": True}
                elif self.stu is not None:
                    draws, steps = self.sample_student(m, ids, im, a.k, stale, self.i0)
                    B = a.k
                elif a.harvest:
                    xall, rec = self.sample_rec(m, ids, im, max(a.k_harvest, a.k), stale, self.i0, PRE())
                    draws, steps, B = xall[:a.k], len(rec), max(a.k_harvest, a.k)
                    hv = {"clip": self.features(m, ids, im), "prompt": p, "ep": self.ep,
                          "call": self.calls_in_ep, "age": int(self.used), "anchor": stale.float().cpu().half(),
                          "steps": rec, "norms": norms}
                else:
                    draws, steps = self.sample(m, ids, im, a.k, anchor=stale, i0=self.i0, pre=PRE())
                    B = a.k
                n = self.certificate(draws, self.sref[p], norms, m)
                if a.grip_fresh > 0 and not a.no_cache:
                    gd = self.grip_mismatch(fresh[0], draws[0], norms, m, a.c_max)
                    n = min(n, gd)
                ratio = (self.spread(draws)[:a.c_max] / self.sref[p][:a.c_max]).tolist()
                kind = "try"
                if n >= a.n_min:
                    chunk = draws[0]
                    seg = max(a.n_min, min(n, a.c_max))
                    if a.grip_lock and self.g_cur is not None:
                        g = m._denorm(chunk.float().cpu().numpy(), norms)[:seg, 6] > 0.5
                        if a.grip_lock_mode == "both":
                            d = np.nonzero(g != self.g_cur)[0]
                        else:
                            q = np.concatenate([[self.g_cur], g])
                            d = np.nonzero((~q[:-1]) & q[1:])[0]
                        if len(d):
                            gl = int(d[0])
                            gdir = "open" if g[gl] else "close"
                            seg = gl
                            self.force_plan = True
                    if seg > 0:
                        kind = "corr"
                        self.used += seg
                    else:
                        need_plan = True
                else:
                    need_plan = True
            if need_plan:
                x, b2 = self.sample(m, ids, im, 1, pre=PRE())
                steps += b2
                chunk, seg = x[0], a.c_safe
                kind = "plan" if kind is None else "fallback"
                self.force_plan = False
                self.used = seg
        self.A = chunk
        self.last_seg = seg
        self.g_cur = bool(m._denorm(chunk[seg - 1:seg].float().cpu().numpy(), norms)[0, 6] > 0.5)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if hv is not None:
            hv.update(kind=kind, n=n, seg=int(seg), gl=gl)
            self.hv_buf.append(hv)
            if len(self.hv_buf) >= 100:
                self.flush_harvest()
        extra = {}
        if a.mode == "specprune" and SP is not None and getattr(SP.state, "comp_hist", None):
            extra["sp"] = list(SP.state.comp_hist[-1]) + [int(SP.state.precise_hist[-1])]
        log_call(ep=self.ep, p=p[:30], kind=kind, n=n, seg=seg, B=B, steps=steps, ms=round(ms, 1),
            ratio=[round(v, 2) for v in ratio[:5]] if ratio else None, gd=gd, gl=gl, gdir=gdir, gc=g_prev,
            **extra)
        return m._denorm(chunk[:seg].float().cpu().numpy(), norms).tolist()


T = Teacher()

from simpler_oft import SimplerOFTExp

exp = SimplerOFTExp()
Cls = type(exp.inference_config)


def process_frame(self):
    from flask import request, jsonify
    try:
        self._apply_inference_seed(request.form.get("seed"))
        text = request.form.get("text")
        if str(request.form.get("episode_first_frame")).lower() in ("true", "1"):
            T.A, T.used, T.last_seg, T.g_cur, T.force_plan, T.calls_in_ep = None, 0, 0, None, False, 0
            if a.harvest and len(T.hv_buf) >= 30:
                T.flush_harvest()
            T.ep += 1
            if VC is not None:
                VC.reset()
            if SP is not None:
                SP.reset()
        T.prompt = text
        if a.bench or a.bench_ev or a.bench_vc or a.bench_sp:
            _bench(self.model, text, request.files.getlist("image"), self)
        orig = self.model.inference_action
        self.model.inference_action = lambda ids, im, args: T.infer(self.model, ids, im, args)
        try:
            res = self._get_response(text=text, images=request.files.getlist("image"))
        finally:
            self.model.inference_action = orig
        return jsonify({"response": res})
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        log_call(error=tb[-800:])
        raise


def _bench(m, text, images, cfg):
    import time as _tm
    got = {}

    def capture(ids, im, args):
        got["ids"], got["im"] = ids, im
        raise SystemExit

    orig = m.inference_action
    m.inference_action = capture
    try:
        cfg._get_response(text=text, images=images)
    except SystemExit:
        pass
    except Exception:
        pass
    finally:
        m.inference_action = orig
    if "ids" not in got:
        print("*** BENCH: no observation captured ***", flush=True); os._exit(1)
    ids, im = got["ids"], got["im"]

    def timed(fn, n=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); t0 = _tm.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(); return (_tm.perf_counter() - t0) * 1e3 / n

    with torch.no_grad():
        pre = T.prefix(m, ids, im)
        t_pre = timed(lambda: T.prefix(m, ids, im))
        t_d1 = timed(lambda: T.sample(m, ids, im, 1, pre=pre))
        t_d5 = timed(lambda: T.sample(m, ids, im, 5, pre=pre), n=5)
    if a.bench_sp:
        global SP, FORCE_PER_LAYER
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        C, D = m.config.chunk_size, m.config.action_dim
        g = torch.Generator(device="cuda").manual_seed(7)
        eps = torch.randn(1, C, D, device="cuda", dtype=im.dtype, generator=g)
        with torch.no_grad():
            p_std = T.prefix(m, ids, im)
            ref = T.sample(m, ids, im, 1, pre=p_std, eps=eps)[0]
            g2 = torch.Generator(device="cuda").manual_seed(8)
            eps2 = torch.randn(1, C, D, device="cuda", dtype=im.dtype, generator=g2)
            tn = float((T.sample(m, ids, im, 1, pre=p_std, eps=eps2)[0].float() - ref.float()).abs().mean())
            FORCE_PER_LAYER = True
            d_force = float((T.sample(m, ids, im, 1, pre=p_std, eps=eps)[0].float() - ref.float()).abs().max())
            FORCE_PER_LAYER = False
            orig = a.mode
            a.mode = "specprune"
            SP = make_sp(m, keep_all=True)
            p_all = SP.prefix(ids, im)
            o_all = T.sample(m, ids, im, 1, pre=p_all, eps=eps)[0]
            SP = make_sp(m)
            SP.prefix(ids, im)
            T.sample(m, ids, im, 1, pre=SP.prefix(ids, im), eps=eps)
            p_sp = SP.prefix(ids, im)
            o_sp = T.sample(m, ids, im, 1, pre=p_sp, eps=eps)[0]
            t_plan_std = timed(lambda: T.sample(m, ids, im, 1, pre=T.prefix(m, ids, im), eps=eps), n=6)
            t_plan_sp = timed(lambda: T.sample(m, ids, im, 1, pre=SP.prefix(ids, im), eps=eps), n=6)
            a.mode = orig
        d_all = float((o_all.float() - ref.float()).abs().max())
        d_sp = float((o_sp.float() - ref.float()).abs().mean())
        print("\n*** BENCH SpecPrune on DB-OFT ***", flush=True)
        print("  per-layer mask path on a uniform cache: chunk max diff %.3e (must be ~0)" % d_force, flush=True)
        print("  keep all 576 tokens: chunk max diff %.3e (must be ~ bf16 error)" % d_all, flush=True)
        print("  alpha=%.1f: keeps %d/576 visual tokens, prefix %d/%d | chunk mean diff %.4f = %.0f%% of natural spread (%.4f)"
              % (a.sp_alpha, SP.state.kept_hist[-1], SP.n_kept, SP.pos_offset, d_sp, 100 * d_sp / max(tn, 1e-9), tn), flush=True)
        print("  impl %s | composition of the last plan: %s%s" % (a.sp_impl, SP.state.comp_hist[-1],
              "  (static pruned, text L0, text L1, global, kept)  cache length per layer %s..." % SP.layer_lengths[:3]
              if getattr(SP, "layer_lengths", None) else "  (dynamic, global, local, fill)"), flush=True)
        print("  whole plan: baseline %.1f ms | SpecPrune %.1f ms -> %.2fx" % (t_plan_std, t_plan_sp, t_plan_std / t_plan_sp), flush=True)
        os._exit(0)

    if a.bench_vc:
        from dboft_vlacache import VLACacheDB

        def kv_of(cache, li):
            if hasattr(cache, "layers"):
                return cache.layers[li].keys, cache.layers[li].values
            return cache.key_cache[li], cache.value_cache[li]

        C, D = m.config.chunk_size, m.config.action_dim
        g = torch.Generator(device="cuda").manual_seed(7)
        eps = torch.randn(1, C, D, device="cuda", dtype=im.dtype, generator=g)
        with torch.no_grad():
            p_std = T.prefix(m, ids, im)
            ref = T.sample(m, ids, im, 1, pre=p_std, eps=eps)[0]
            g2 = torch.Generator(device="cuda").manual_seed(8)
            eps2 = torch.randn(1, C, D, device="cuda", dtype=im.dtype, generator=g2)
            tn = float((T.sample(m, ids, im, 1, pre=p_std, eps=eps2)[0].float() - ref.float()).abs().mean())
            vc = VLACacheDB(m, vc_cfg())
            p1 = vc.prefix(ids, im)
            o1 = T.sample(m, ids, im, 1, pre=p1, eps=eps)[0]
            print("\n*** BENCH VLA-Cache on DB-OFT ***", flush=True)
            print("  first plan, no reuse -- K difference per layer against the plain prefill:", flush=True)
            for li in (0, 1, 2, 13, 27):
                kd = float((kv_of(p1[0], li)[0].float() - kv_of(p_std[0], li)[0].float()).abs().max())
                print("    layer %2d: max|dK| = %.3e" % (li, kd), flush=True)
            print("  chunk max diff %.3e  (natural spread mean %.4f)" % (float((o1.float() - ref.float()).abs().max()), tn), flush=True)
            t_std = timed(lambda: T.sample(m, ids, im, 1, pre=T.prefix(m, ids, im), eps=eps), n=6)
            for age in (1, 0):
                cfg = vc_cfg(); cfg.max_age = age
                vc = VLACacheDB(m, cfg)
                vc.prefix(ids, im)
                o2 = T.sample(m, ids, im, 1, pre=vc.prefix(ids, im), eps=eps)[0]
                t_vc = timed(lambda: T.sample(m, ids, im, 1, pre=vc.prefix(ids, im), eps=eps), n=6)
                r = float(np.mean(vc.state.reuse_hist[1:] or [0]))
                d = float((o2.float() - ref.float()).abs().mean())
                print("  max_age=%d: reuses %.0f%% of tokens on average | chunk mean diff %.4f = %.0f%% of natural spread | "
                      "whole plan %.1f vs baseline %.1f ms -> %.2fx"
                      % (age, 100 * r, d, 100 * d / max(tn, 1e-9), t_vc, t_std, t_std / t_vc), flush=True)
        os._exit(0)

    if a.bench_ev:
        C, D = m.config.chunk_size, m.config.action_dim
        g = torch.Generator(device="cuda").manual_seed(7)
        eps = torch.randn(1, C, D, device="cuda", dtype=im.dtype, generator=g)
        orig_mode, orig_iv = a.mode, a.ev_interval
        a.mode = "baseline"
        with torch.no_grad():
            ref = T.sample(m, ids, im, 1, pre=pre, eps=eps)[0]
            t_ref = timed(lambda: T.sample(m, ids, im, 1, pre=pre, eps=eps))
            eps2 = torch.randn(1, C, D, device="cuda", dtype=im.dtype, generator=g)
            natural = float((T.sample(m, ids, im, 1, pre=pre, eps=eps2)[0].float()
                              - ref.float()).abs().mean())
        print("\n*** BENCH EfficientVLA on DB-OFT ***", flush=True)
        print("  no cache: %6.1f ms | natural spread between 2 draws: %.4f" % (t_ref, natural), flush=True)
        print("  %-4s%10s%10s%12s%12s" % ("N", "ms", "speedup", "mean diff", "max diff"), flush=True)
        a.mode = "effvla"
        a.ev_interval = 1
        with torch.no_grad():
            out1 = T.sample(m, ids, im, 1, pre=pre, eps=eps)[0]
            ref2 = None
            a.mode = "baseline"
            ref2 = T.sample(m, ids, im, 1, pre=pre, eps=eps)[0]
            a.mode = "effvla"
        d1 = float((out1.float() - ref.float()).abs().max())
        dd = float((ref2.float() - ref.float()).abs().max())
        print("  CHECK N=1 vs baseline (same eps): max|diff| = %.3e   -> %s"
              % (d1, "IDENTICAL" if d1 == 0.0 else "DIFFERENT -> BUG"), flush=True)
        print("  baseline vs baseline (same eps, rerun): max|diff| = %.3e   (GPU determinism)" % dd, flush=True)
        for iv in (2, 3, 4, 5):
            a.ev_interval = iv
            with torch.no_grad():
                out = T.sample(m, ids, im, 1, pre=pre, eps=eps)[0]
                t_iv = timed(lambda: T.sample(m, ids, im, 1, pre=pre, eps=eps))
            d = (out.float() - ref.float()).abs()
            print("  %-4d%10.1f%9.2fx%12.4f%12.4f   (%.0f%% of natural spread)"
                  % (iv, t_iv, t_ref / t_iv, float(d.mean()), float(d.max()),
                     100 * float(d.mean()) / max(natural, 1e-9)), flush=True)
        a.mode, a.ev_interval = orig_mode, orig_iv
        os._exit(0)

    tot = t_pre + t_d1
    print("\n*** BENCH DB-OFT (one plan) ***", flush=True)
    print("  prefill (once)         %7.1f ms   %3.0f%% of the plan" % (t_pre, 100 * t_pre / tot), flush=True)
    print("  10 DDIM steps (B=1)    %7.1f ms   %3.0f%%" % (t_d1, 100 * t_d1 / tot), flush=True)
    print("  whole plan             %7.1f ms" % tot, flush=True)
    print("  whole plan with B=5    %7.1f ms   (what AAC pays)" % (t_pre + t_d5), flush=True)
    print("  ceiling for SpecPrune/VLA-Cache (prefix only):   %.2fx" % (tot / (tot - t_pre)), flush=True)
    print("  ceiling for EfficientVLA (denoiser only):        %.2fx" % (tot / (tot - t_d1)), flush=True)
    os._exit(0)


Cls.process_frame = process_frame
if a.harvest:
    import signal

    def _flush_and_exit(signum, frame):
        T.flush_harvest()
        os._exit(0)
    signal.signal(signal.SIGTERM, _flush_and_exit)
print("*** DB-OFT %s | K=%d start_step=%d c_safe=%d c_max=%d n_min=%d lineage=%d tau=%.2f grip_fresh=%d grip_lock=%s/%s harvest=%s ***"
      % (a.mode, a.k, a.start_step, a.c_safe, a.c_max, a.n_min, a.lineage, a.agree_tau, a.grip_fresh, a.grip_lock, a.grip_lock_mode, a.harvest or "-"), flush=True)
exp.inference()
