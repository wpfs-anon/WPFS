import os as _os
HOME = _os.environ.get(
    "CORRECTOR_HOME",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os
import sys
import copy
import time
import traceback

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["baseline", "teacher"], default="teacher")
ap.add_argument("--k", type=int, default=2, help="draws per correction")
ap.add_argument("--start-step", type=int, default=6,
                help="DDIM step (0..9) a correction starts from; step 8 = abar 0.967, 2 steps left")
ap.add_argument("--no-cache", action="store_true", help="disable the prefix KV cache (control run)")
ap.add_argument("--grip-lock", action="store_true",
                help="a correction never executes a gripper change: the segment is cut just before "
                     "the change and the next call is forced to plan")
ap.add_argument("--grip-lock-mode", choices=["both", "reopen"], default="both",
                help="both: block every gripper change inside a correction; reopen: block only "
                     "closed -> open")
ap.add_argument("--grip-fresh", type=int, default=0,
                help="DDIM steps of a fresh coarse plan batched with the correction to certify the "
                     "gripper (0 = off)")
ap.add_argument("--c-safe", type=int, default=5, help="actions executed after a plan")
ap.add_argument("--c-max", type=int, default=10, help="cap on actions executed after a correction")
ap.add_argument("--n-min", type=int, default=2, help="agreement below this triggers a replan")
ap.add_argument("--lineage", type=int, default=50, help="maximum steps before a plan is forced")
ap.add_argument("--agree-tau", type=float, default=1.0)
ap.add_argument("--replan", type=int, default=5, help="baseline mode: actions per call")
ap.add_argument("--cal-calls", type=int, default=12,
                help="first calls of each task used to measure SREF (run as baseline meanwhile)")
ap.add_argument("--cal-m", type=int, default=8, help="independent plans per SREF measurement")
ap.add_argument("--cal-first", type=int, default=12,
                help="only the first N calls of each episode feed SREF (active states only)")
ap.add_argument("--sref-file", default="",
                help="per-task SREF (json): loaded if present, otherwise measured with --cal-calls "
                     "and written here")
ap.add_argument("--harvest", default="",
                help="directory for student data: per correction the CLIP features, prompt, anchor, "
                     "age and (t, x_t, teacher eps) of --k-harvest draws at every DDIM step")
ap.add_argument("--k-harvest", type=int, default=8,
                help="draws recorded while harvesting; the first --k still feed the certificate")
ap.add_argument("--student", default="",
                help="student checkpoint: corrections run the student (CLIP + small net, 2 DDIM "
                     "steps) instead of the 7B LLM; the LLM prefix is computed only for plans")
ap.add_argument("--log", default=f"{HOME}/dboft/results/dboft/calls.jsonl")
a = ap.parse_args()

os.chdir(f"{HOME}/dboft/dexbotic")
sys.path.insert(0, f"{HOME}/dboft/dexbotic")
sys.path.insert(0, f"{HOME}/dboft/dexbotic/playground/benchmarks/simpler")
os.makedirs(os.path.dirname(a.log), exist_ok=True)
LOGF = open(a.log, "a")


def log_call(**kv):
    LOGF.write(json.dumps(kv) + "\n")
    LOGF.flush()


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
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        for t in ts[start:]:
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
            pred = m.model.action_head.predict_noise(o.hidden_states[-1][:, 1:, :])
            x = sch.step(pred.reshape(x.shape), t, x).prev_sample
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
                _pc["v"] = None if a.no_cache else self.prefix(m, ids, im)
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
        log_call(ep=self.ep, p=p[:30], kind=kind, n=n, seg=seg, B=B, steps=steps, ms=round(ms, 1),
                 ratio=[round(v, 2) for v in ratio[:5]] if ratio else None, gd=gd, gl=gl, gdir=gdir, gc=g_prev)
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
        T.prompt = text
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
