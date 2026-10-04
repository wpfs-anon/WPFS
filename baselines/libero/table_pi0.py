#!/usr/bin/env python3
import os as _os
HOME = _os.environ.get(
    "CORRECTOR_HOME",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import glob, json, math, os, sys

FINAL = f"{HOME}/ckpt/pi0"
STD = f"{HOME}/ckpt/pi0"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
NAMES = {"sp": "SpecPrune-VLA", "vc": "VLA-Cache", "ev": "EfficientVLA", "aac": "AAC",
         "teacher": "teacher (ours)", "student": "student (ours)"}


def mcnemar(b, c):
    t = b + c
    if not t:
        return 1.0
    z = max(abs(b - c) - 1, 0) / math.sqrt(t)
    return min(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))), 1.0)


def load(path):
    with open(path) as f:
        return json.load(f)


def merge(paths):
    eps = succ = steps = plans = 0
    ms_num = l0_num = 0.0
    per_ep = []
    for p in paths:
        d = load(p)
        m = next(k for k in d["results"] if k != "baseline")
        r = d["results"][m]
        eps += r["episodes"]; succ += r["success"]
        steps += r["steps"]; plans += r["plans"]
        ms_num += r["ms_per_step"] * r["steps"]
        l0_num += r["steps"] * d["L_plan"]
        per_ep += r["per_episode"]
    return dict(episodes=eps, success=succ, rate=100.0 * succ / max(eps, 1), steps=steps,
                plans=plans, ms_per_step=ms_num / max(steps, 1), L=ms_num / max(plans, 1),
                L0=l0_num / max(steps, 1), steps_per_plan=steps / max(plans, 1),
                per_episode=per_ep)


rows = {}
for tag in NAMES:
    per_suite = {}
    for suite in SUITES:
        parts = sorted(glob.glob(f"{FINAL}/{suite}/{tag}_s7*.json"))
        b = f"{STD}/{suite}/base_s7.json"
        if not (parts and os.path.exists(b)):
            continue
        r = merge(parts)
        db = load(b)
        rb = db["results"]["baseline"]
        base_ms = rb["plans"] * r["L0"] / max(rb["steps"], 1)
        eb = {(e["task"], e["trial"]): e["ok"] for e in rb["per_episode"]}
        em = {(e["task"], e["trial"]): e["ok"] for e in r["per_episode"]}
        shared = eb.keys() & em.keys()
        only_b = sum(1 for k in shared if eb[k] and not em[k])
        only_m = sum(1 for k in shared if em[k] and not eb[k])
        per_suite[suite] = dict(rate=r["rate"], base=rb["rate"], eps=r["episodes"],
                                speed=base_ms / r["ms_per_step"], p=mcnemar(only_b, only_m),
                                pair=(only_b, only_m), paired=len(shared),
                                base_on_shared=100.0 * sum(eb[k] for k in shared) / max(len(shared), 1),
                                spp=r["steps_per_plan"],
                                L=r["L"], L0=r["L0"], parts=len(parts))
    if per_suite:
        rows[tag] = per_suite

hdr = f"{'method':<16}" + "".join(f"{s.replace('libero_', ''):>16}" for s in SUITES) + f"{'mean':>10}{'speedup':>9}"
print("\n  pi_0 (openpi pi0_libero), 10 tasks x 50 trials x 4 suites = 2,000 scenes, seed 7")
print("  " + hdr)
base_line = f"{'pi_0 (replan 10)':<16}"
got = False
for s in SUITES:
    v = next((r[s]["base"] for r in rows.values() if s in r), None)
    base_line += f"{v:16.1f}" if v is not None else f"{'--':>16}"
    got = got or v is not None
bases = [r[s]["base"] for r in rows.values() for s in r if s in r]
if got:
    vals = [next(r[s]["base"] for r in rows.values() if s in r) for s in SUITES
            if any(s in r for r in rows.values())]
    print("  " + base_line + f"{sum(vals)/len(vals):10.2f}{1.00:8.2f}x")
for tag, per in rows.items():
    line = f"{NAMES[tag]:<16}"
    for s in SUITES:
        if s in per:
            v = per[s]
            mark = "" if v["eps"] == 500 else "*"
            line += f"{v['rate']:10.1f}{mark} ({v['speed']:.2f}x)".rjust(16)
        else:
            line += f"{'--':>16}"
    done = [s for s in SUITES if s in per and per[s]["eps"] == 500]
    if len(done) == len(SUITES):
        vals = [per[s]["rate"] for s in SUITES]
        sp = [per[s]["speed"] for s in SUITES]
        print("  " + line + f"{sum(vals)/len(vals):10.2f}{sum(sp)/len(sp):8.2f}x")
    else:
        print("  " + line + f"{'--':>10}{'--':>9}")
print("  * suite not yet complete; its rate is over the scenes run so far")
print("\n  paired against replan 10 on the same scenes (only-base / only-method, McNemar p):")
for tag, per in rows.items():
    for s in SUITES:
        if s in per:
            v = per[s]
            print(f"    {NAMES[tag]:<14} {s.replace('libero_',''):<9} {v['rate']:5.1f} vs {v['base']:5.1f}"
                  f"   {v['pair'][0]:3d}/{v['pair'][1]:<3d} p {v['p']:.3f}"
                  f"   plan {v['L0']:.1f} -> {v['L']:.1f} ms, {v['spp']:.1f} steps/plan"
                  f"   [{v['eps']} eps, {v['paired']} paired, {v['parts']} part(s)]")
