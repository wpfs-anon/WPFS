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

### Measured, not asserted

`scripts/staleness_probe.py` measures that paragraph without scoring a single
episode. On a healthy rollout, with a chunk planned `k` steps ago, it draws 8
fresh plans at the current observation and asks how far each candidate sits from
that distribution: `d_nn` to the nearest of them, `d_cent` to their centroid,
each divided by the reference plans' own — so 1.00 is "as far as two honest
plans already are". 1,950 probes over 12 LIBERO-Spatial episodes, π₀, seed 7,
scored over the 10 actions that would be executed next:

| k | do nothing | 1 step from noise | τ₀ = 0.3 | **τ₀ = 0.5** | τ₀ = 0.7 |
|---|---|---|---|---|---|
| 1 | 0.71 / 1.09 | 0.84 / 0.60 | 0.83 / 0.85 | **0.81 / 0.64** | 0.82 / 0.56 |
| 5 | 1.30 / 1.49 | 0.80 / 0.65 | 1.00 / 1.24 | **0.78 / 0.85** | 0.77 / 0.67 |
| 10 | 1.53 / 1.88 | 0.81 / 0.64 | 1.11 / 1.43 | **0.88 / 1.02** | 0.78 / 0.70 |
| 30 | 1.97 / 2.53 | 0.79 / 0.66 | 1.14 / 1.53 | **0.90 / 1.15** | 0.79 / 0.71 |

`d_nn` / `d_cent`. A candidate is inside the distribution only when `d_nn ≤ 1`
**and** `d_cent ≈ 1`. Three things follow, none of which needed a rollout:

**The chunk does go stale.** Doing nothing leaves the distribution — at the
tight band (the nearest other plan) from k = 3, and at the typical plan-to-plan
spread at k = 10, which is where π₀'s own success-versus-replan curve turns
over. Its `d_cent` grows with it, so the stale chunk drifts *away* from the plan
cloud rather than toward its middle.

**One step from the stale chunk brings it back, at every staleness.** τ₀ = 0.5
holds `d_nn` at 0.78–0.93 from k = 1 to k = 30 while `d_cent` climbs to ≈ 1: not
merely close to some plan, but as far from the centre as a plan should be.

**Both cheap alternatives fail, in opposite directions.** One step from noise —
and τ₀ = 0.7, which is nearly noise — posts the best `d_nn` of the table while
sitting at `d_cent` 0.53–0.71: the conditional mean, a chunk the policy would
never sample. τ₀ = 0.3 does not re-noise enough and stays stale, tracking the
do-nothing row (`d_nn` > 1 from k = 8). So the operator's one hyperparameter is
bracketed by two failure modes that a nearest-plan metric alone cannot tell
apart, which is the reason both statistics are reported.

A fourth observation, smaller: the ratio does not vary systematically along the
chunk (at k = 10 it is 1.71 over positions 0–4, 1.62 over 10–14, 1.81 over the
last five). Staleness is a property of elapsed time, not of position — which is
why one re-noise level serves the whole chunk.

`scripts/staleness_probe.py` writes every value to `ckpt/staleness/`, and
`scripts/staleness_figure.py` draws the two-panel figure from it.

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

The corrector runs at every correction, so its cost is the thing to attack. The
student is trained only against the teacher's velocity field, never against
demonstrations:

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
`x − τ₀·v`. Without the second term nothing keeps the student's draw spread at the
teacher's, and the certificate would read a different quantity than the one it was
calibrated on.

## Cost

Measured on an RTX 5090, compiled, batch 4, idle GPU:

| | ms |
|---|---|
| plan (10 steps, full depth, 3 cameras) | 57.81 |
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
