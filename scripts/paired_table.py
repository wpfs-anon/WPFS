#!/usr/bin/env python
"""Paired comparison of evaluation runs against the baseline, one suite at a time.

    python scripts/paired_table.py results/pi05/libero_10 r5 r10 teacher_L50 student

Each name is a JSON written by eval_corrector.py (`<dir>/<name>_s7.json`).  The first
is the reference (a baseline run).  Every other row shows its success rate, its speedup
(reference ms/step over its ms/step), and the discordant pairs against the reference --
episodes only the reference solves / episodes only this row solves -- with the exact-
continuity McNemar p-value.  All runs must cover the same (task, trial) episodes, which
they do when they share --seed: every episode draws its noise from (seed, task, trial).
"""
import json, math, sys

def load(path):
    r = json.load(open(path))["results"]
    return r.get("corrector") or r.get("baseline")

def mcnemar(b, c):
    if b + c == 0:
        return 1.0
    z = max(abs(b - c) - 1, 0) / math.sqrt(b + c)
    return min(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))), 1.0)

d, names = sys.argv[1], sys.argv[2:]
ref = load(f"{d}/{names[0]}_s7.json")
eb = {(e["task"], e["trial"]): e["ok"] for e in ref["per_episode"]}
print(f"{'run':<24}{'success':>9}{'speedup':>9}{'ms/step':>9}   paired vs {names[0]}")
for n in names:
    r = load(f"{d}/{n}_s7.json")
    er = {(e["task"], e["trial"]): e["ok"] for e in r["per_episode"]}
    b = sum(1 for k in eb if eb[k] and not er[k]); c = sum(1 for k in eb if er[k] and not eb[k])
    extra = "" if n == names[0] else f"   {b} / {c}   p = {mcnemar(b, c):.3f}"
    print(f"{n:<24}{r['rate']:>8.1f}%{ref['ms_per_step'] / r['ms_per_step']:>8.2f}x{r['ms_per_step']:>9.2f}{extra}")
