# The corrector loop

## Time convention

π₀ is a flow-matching policy. Its model time runs **τ = 1 at noise, τ = 0 clean**,
integrated by Euler steps `x ← x − Δτ·v`. The interpolant is `x = τ·ε + (1−τ)·A`
and the velocity target is `v ≈ ε − A`. A plan is ten such steps from τ = 1; a
correction is **one** step from τ₀ = 0.5.

## What a correction is

π₀ produces a 50-step action chunk. Shipped, it executes ten of them and plans
again from noise. The chunk is not worthless after ten steps — it is stale, and
staleness is mostly a change of viewpoint, not a change of intent.

So instead of planning from noise, take the chunk already in hand, re-anchor it
to the current pose, add noise to τ₀ = 0.5, and integrate **one** Euler step
against the current observation:

```
A_stale = hold_pad(A[last_seg:], H)          # shift, pad the tail by holding
A_stale[:, :6] += env.frame_correction()     # re-anchor to where the arm is now
x0 = τ₀·ε + (1−τ₀)·A_stale                   # K independent noise draws
out = integrate(x0, τ₀, steps=1, obs_now)    # one step, K corrected chunks
```

One step from a stale chunk lands inside the teacher's plan distribution. One
step from pure noise returns the conditional mean instead — which scores well on
a nearest-plan metric while being a mode average, and is why the distance to a
reference **centroid** is tracked alongside the distance to the nearest plan.

## What the acceptance test does

The K draws differ only in ε. Where they agree, the correction is determined by
the observation; where they scatter, it is not. Position by position:

```
spread[h] = median over pairs ||out_i[h,:6] − out_j[h,:6]||
ok[h]     = spread[h] ≤ agree_tau · SREF[h]   and   the gripper side is unanimous
N         = length of the longest prefix of ok
```

`N` is how many actions this correction may be trusted for. Below `n_min` the
correction is discarded and π₀ replans; otherwise `N` is clamped to `[n_min,
c_max]` and executed.

**SREF** is the yardstick: the teacher's own natural per-position spread, from 8
honest plans at 4 tasks × 2 time offsets, median over the 28 pairs. It is
computed once per run and is what `agree_tau` multiplies.

## Why the test earns its place

At a matched replan cadence the adaptive test beats a fixed interval by a wide
margin — 39.3 steps/plan and 98.0% against 38.3 steps/plan and 93.3% on the same
episodes. So it is doing more than setting a rhythm.

Its value is concentrated in the tail, not the average. Per-position
disagreement correlates only weakly with per-position error (Spearman 0.15–0.20
for every corrector including the one that reaches parity), and the positions it
rejects are barely worse than average. But the interval distribution has p10 = 1:
about a tenth of the time it stops almost immediately, and those rare early
replans are where the mechanism pays. `scripts/certificate_diagnostic.py`
reproduces both halves of this.

## Distillation

The corrector runs at every correction, so its cost is the thing to attack. Two
students, both trained only against the teacher's velocity field, never against
demonstrations:

**LoRA student.** π₀ truncated to 14 prefix and 12 expert layers with LoRA
adapters, ~7M trained parameters. Truncation alone destroys correction at every
depth; the adapters repair it, relative L2 1.99 → 0.116.

**Network student.** 27.8M parameters. The 50 chunk positions are the queries;
π₀'s frozen SigLIP tokens plus its instruction embedding are the memory. Nothing
of the language model runs.

The second loss term is what makes the network work. The acceptance test reads
per-position draw disagreement and **the velocity loss constrains nothing about
it** — we only measured it and hoped:

```
L = ‖v_s − v_t‖²  +  λ·‖σ_h(student) − σ_h(teacher)‖²
```

where σ_h is the mean pairwise distance between draws at position h, computed on
`x − τ₀·v`. At λ = 1 the disagreement ratio settles at 1.06 instead of 1.10, and
velocity accuracy improves too (0.155 → 0.145) — constraining how the model must
respond to its own input noise appears to regularise what it learns. In the
rollout this is worth 2.7 points: −1.29 at λ = 0 against +1.43 at λ = 1, at the
same speed and the same replan cadence.

## Cost

Measured on an RTX 5090, compiled, batch 4, idle GPU:

| | ms |
|---|---|
| plan (10 steps, full depth, 3 cameras) | 57.81 |
| correction, LoRA student at (14, 12) | 26.52 |
| correction, network student | 7.57 |
| ├ vision encoder, 2 cameras | 6.25 |
| └ the network itself, K=4 draws | 1.27 |

The network's own computation is 17% of its correction; the rest is the price of
looking. Further architectural savings are therefore bounded — with a free
corrector and the current replan rate the ceiling is 4.03×, and the remaining
budget is dominated by teacher plans, which cannot be cheapened without giving
up the anchor every correction is measured against.

## π0.5

openpi's `pi05_libero` plans a **10-action** chunk (π0: 50) directly in LIBERO's
action space, so a stale chunk is reused without re-anchoring. The loop is the
same correction, with three changes that the short chunk forces:

**Lineage.** A plan executes 5 actions (`--c-safe 5`); every later segment is a
correction of the chunk in hand, executed for as long as its K = 4 draws agree
(`--n-min 5`, `--c-max 10`). A plan and its corrections form a lineage that is
cut at 50 steps (`--lineage 50`), after which the next segment is a fresh plan.
Past the chunk's 10 planned actions the remainder is padded by holding the last
action, so most corrections re-plan the tail from a held pose.

**Plan context and age.** The student is told what the plan knew and how old it
is. At every plan the loop keeps the input to the policy's `action_out_proj` —
the action expert's final hidden states for the 10 action tokens, 10 x 1024 —
which the plan computes anyway, and counts the actions executed since. The
student adds the plan tokens to its memory and embeds the age; the teacher's
cost does not change and the student's rises by about 0.1 ms.

**Certificate scale.** The per-position spread the draws are compared against
(`SREF`) is the policy's own plan-to-plan spread, measured at start-up from
seeded draws on the first four tasks' first initial states, so it is the same
for every run of a suite and every subset of its tasks.

Cost on an RTX 5090, compiled, idle GPU: a plan 63.0 ms, a teacher correction
43.3 ms, a student correction 9.6 ms.
