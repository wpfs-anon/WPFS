import argparse
import glob
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="comma-separated harvest directories")
ap.add_argument("--out", required=True)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--blocks", type=int, default=6)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--rec-per-batch", type=int, default=16, help="corrections per batch (x16 samples)")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--val-frac", type=float, default=0.08, help="fraction of EPISODES held out for validation")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init", default="", help="initialise from a student checkpoint (fine-tuning), e.g. r1")
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--dagger-tasks", default="",
                help="keep DAgger data only for these tasks (e.g. Spoon,Eggplant); teacher data is always kept")
ap.add_argument("--scenes-per-task", type=int, default=24,
                help="episodes per task in one harvest run: 24 on the evaluation scenes, 48 on the training scenes")
ap.add_argument("--scenes", default="", help="keep only scenes lo-hi (scene = ep %% scenes-per-task), e.g. 0-11")
ap.add_argument("--tasks", default="", help="keep data (teacher + DAgger) only for these tasks, e.g. Spoon,Eggplant")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
dev = "cuda"


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dboft_student_net import DboftStudent


R, TXT = [], {}
for d in a.data.split(","):
    for f in sorted(glob.glob(d.strip() + "/shard_*.pt")):
        z = torch.load(f, weights_only=False)
        TXT.update(z["txt"])
        for r in z["rec"]:
            r["group"] = "%s|%s|%d" % (os.path.basename(d.strip()), r["prompt"][-20:], r["ep"])
            R.append(r)
if not R:
    sys.exit("no data")
TASK_KEY = [("stack", "StackCube"), ("carrot", "Carrot"), ("spoon", "Spoon"), ("eggplant", "Eggplant")]


def task_name(prompt):
    q = prompt.lower()
    for k, t in TASK_KEY:
        if k in q:
            return t
    return "?"


import collections
N_SC = a.scenes_per_task
mismatch = sum(1 for r in R if r["ep"] >= 4 * N_SC or TASK_KEY[r["ep"] // N_SC][1] != task_name(r["prompt"]))
print("scene check (scene = ep %% %d, ep // %d must match the task): %d/%d mismatched" % (N_SC, N_SC, mismatch, len(R)), flush=True)
if a.scenes:
    lo, hi = map(int, a.scenes.split("-"))
    R = [r for r in R if lo <= r["ep"] % N_SC <= hi]
if a.tasks:
    keep_t = set(a.tasks.split(","))
    R = [r for r in R if task_name(r["prompt"]) in keep_t]
if a.dagger_tasks:
    keep = set(a.dagger_tasks.split(","))
    R = [r for r in R if not r.get("dagger") or task_name(r["prompt"]) in keep]
count = collections.Counter((task_name(r["prompt"]), "dagger" if r.get("dagger") else "teacher") for r in R)
print("after filtering: %d records | %s" % (len(R), dict(sorted(count.items()))), flush=True)
groups = sorted({r["group"] for r in R}); random.Random(1).shuffle(groups)
nv = max(1, int(len(groups) * a.val_frac)); VAL_G = set(groups[:nv])
TR = [r for r in R if r["group"] not in VAL_G]; VA = [r for r in R if r["group"] in VAL_G]
r0 = R[0]
H, D = r0["anchor"].shape; K = r0["steps"][0][1].shape[0]; T = [s[0] for s in r0["steps"]]
c_img, c_txt = r0["clip"].shape[-1], next(iter(TXT.values())).shape[-1]
Lmax = max(v.shape[0] for v in TXT.values())
print("data: %d corrections (%d train / %d val, split by episode) | K=%d draws x %d DDIM steps %s | "
      "clip %s, txt %d x %d, chunk %dx%d" % (len(R), len(TR), len(VA), K, len(T), T, tuple(r0["clip"].shape),
                                             Lmax, c_txt, H, D), flush=True)
pk = sorted(TXT); tid = {p: i for i, p in enumerate(pk)}
TXT_T = torch.zeros(len(pk), Lmax, c_txt, dtype=torch.float16)
for p, i in tid.items():
    TXT_T[i, :TXT[p].shape[0]] = TXT[p]
TXT_T = TXT_T.to(dev)


def collate(rs):
    clip = torch.stack([r["clip"] for r in rs]).to(dev).float()
    txt = TXT_T[torch.tensor([tid[r["prompt"]] for r in rs], device=dev)].float()
    anc = torch.stack([r["anchor"] for r in rs]).to(dev).float()
    age = torch.tensor([r["age"] for r in rs], device=dev).float()
    xs = torch.stack([torch.stack([s[1] for s in r["steps"]]) for r in rs]).to(dev).float()
    es = torch.stack([torch.stack([s[2] for s in r["steps"]]) for r in rs]).to(dev).float()
    return clip, txt, anc, age, xs, es


from diffusers import DDIMScheduler
SCH = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
SCH.set_timesteps(10)
assert [int(t) for t in SCH.timesteps[-len(T):]] == [int(t) for t in T], (SCH.timesteps, T)


def ddim_step(eps, t, x):
    return SCH.step(eps, int(t), x).prev_sample


def spread(x):
    p = x[:, :, :6]; K_ = p.shape[0]
    c = [torch.linalg.vector_norm(p[i] - p[j], dim=-1) for i in range(K_) for j in range(i + 1, K_)]
    return torch.stack(c).median(0).values.mean()


AC = SCH.alphas_cumprod
WT = torch.tensor([float((1 - AC[t]) / AC[t]) for t in T], device=dev).repeat_interleave(K)
print("loss weight per step %s: %s" % (T, [round(float((1 - AC[t]) / AC[t]), 4) for t in T]), flush=True)
net = DboftStudent(a.d, a.blocks, a.heads, c_img, c_txt, H, D).to(dev)
print("student: d=%d x %d blocks, %.1fM parameters, memory %d tokens" % (
    a.d, a.blocks, sum(p.numel() for p in net.parameters()) / 1e6, r0["clip"].shape[0] + Lmax + H), flush=True)
if a.init:
    ck0 = torch.load(a.init, map_location="cpu", weights_only=False)
    assert ck0["cfg"] == net.cfg, (ck0["cfg"], net.cfg)
    net.load_state_dict(ck0["state"])
    print("initialised from %s (its val: %s)" % (a.init, ck0.get("val")), flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
spe = math.ceil(len(TR) / a.rec_per_batch); total = spe * a.epochs
sch_lr = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, a.warmup)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))


@torch.no_grad()
def validate(rs):
    net.eval()
    rel, x0e, tr = [], [], []
    x0t = collections.defaultdict(list)
    for i in range(0, len(rs), a.rec_per_batch):
        clip, txt, anc, age, xs, es = collate(rs[i:i + a.rec_per_batch])
        mem = net.memory(clip, txt, anc)
        Rn, S = xs.shape[:2]
        for j in range(Rn):
            m1 = mem[j:j + 1].expand(K, -1, -1)
            x = xs[j, 0].clone(); xt = xs[j, 0].clone()
            for s in range(S):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    e_on = net(xs[j, s], torch.full((K,), float(T[s]), device=dev), m1, age[j].expand(K))
                    e_s = net(x, torch.full((K,), float(T[s]), device=dev), m1, age[j].expand(K))
                rel.append((torch.linalg.vector_norm(e_on.float() - es[j, s]) / torch.linalg.vector_norm(es[j, s])).item())
                x = ddim_step(e_s.float(), T[s], x)
                xt = ddim_step(es[j, s], T[s], xt)
            st = spread(xt).item()
            x0e.append(torch.linalg.vector_norm((x - xt)[:, :, :6], dim=-1).mean().item() / max(st, 1e-6))
            x0t[task_name(rs[i + j]["prompt"])].append(x0e[-1])
            tr.append(spread(x).item() / max(st, 1e-6))
    net.train()
    md = lambda v: sorted(v)[len(v) // 2]
    return sum(rel) / len(rel), md(x0e), md(tr), {k: md(v) for k, v in sorted(x0t.items())}


if a.init:
    _r0 = validate(VA)
    print("  INITIAL (before training): VAL x0/spread %.3f | per task: %s"
          % (_r0[1], "  ".join("%s %.3f" % (k, v) for k, v in _r0[3].items())), flush=True)
print("total %d steps (%d/epoch)" % (total, spe), flush=True)
steps, t0, best = 0, time.time(), None
for ep in range(a.epochs):
    random.shuffle(TR)
    for i in range(0, len(TR), a.rec_per_batch):
        clip, txt, anc, age, xs, es = collate(TR[i:i + a.rec_per_batch])
        Rn, S = xs.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem = net.memory(clip, txt, anc)
            memx = mem.repeat_interleave(S * K, 0)
            x = xs.reshape(Rn * S * K, H, D)
            tt = torch.tensor(T, device=dev).float().repeat_interleave(K).repeat(Rn)
            ag = age.repeat_interleave(S * K)
            pred = net(x, tt, memx, ag)
        se = (pred.float() - es.reshape(Rn * S * K, H, D)) ** 2
        loss = (WT.repeat(Rn)[:, None, None] * se).mean() / WT.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch_lr.step(); steps += 1
        if steps % 500 == 0 or steps == total:
            rel, x0e, tr, x0t = validate(VA)
            print("  ep %3d step %5d/%d  mse %.4f | VAL rel_eps %.3f  x0/spread %.3f  spread s/t %.2fx | %.1f min"
                  % (ep + 1, steps, total, loss.item(), rel, x0e, tr, (time.time() - t0) / 60), flush=True)
            print("        per task x0/spread: %s" % "  ".join("%s %.3f" % (k, v) for k, v in x0t.items()), flush=True)
            if best is None or x0e < best[0]:
                best = (x0e, rel, tr, steps)
                torch.save({"state": net.state_dict(), "cfg": net.cfg, "T": T, "val": dict(rel=rel, x0e=x0e, tr=tr),
                            "txt": {p: TXT[p] for p in TXT}}, a.out)
print("best: x0/spread %.3f rel_eps %.3f spread %.2fx at step %d -> %s" % (best[0], best[1], best[2], best[3], a.out))
print("### DB-OFT student training done ###")
