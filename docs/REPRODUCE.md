# Reproducing every number

All commands assume the repository root as the working directory and
`.venv/bin/python` as the interpreter. Add `JAX_PLATFORMS=cpu` to keep JAX off
the GPU.

## π0.5 on LIBERO (all four suites)

`pipelines/pi05/common.sh` holds every path and default; each stage below is a
thin script over the four programs in `scripts/`. Timings are for one RTX 5090.
Set `CPUS=0-7` (for example) to pin the programs to a few cores.

**Evaluation.** One run is one suite, 10 tasks x 50 initial states:

```bash
PY=.venv/bin/python
L=ckpt/pi05/latency_pi05.json
$PY scripts/latency_probe.py --model pi05 --out $L                     # plan / teacher latency table
EARG="--configs corrector --adaptive --k-draws 4 --tau0 0.5 --grip snap \
      --n-min 5 --c-max 10 --c-safe 5 --agree-tau 1.0 --lineage 50"
S=libero_10
$PY scripts/eval_corrector.py --model pi05 --latency-from $L --suite $S --seed 7 --trials 50 \
    --configs baseline --replan 5 --out ckpt/pi05/$S/r5_s7.json         # 94.0%
$PY scripts/eval_corrector.py --model pi05 --latency-from $L --suite $S --seed 7 --trials 50 \
    $EARG --out ckpt/pi05/$S/teacher_L50_s7.json                        # 94.6% @ 1.69x
$PY scripts/latency_probe.py --model pi05 --net checkpoints/pi05/st_long_pe_g0.pt \
    --out ckpt/pi05/lat_st_long_pe.json                                  # student: ~9.6 ms
$PY scripts/eval_corrector.py --model pi05 --latency-from $L --suite $S --seed 7 --trials 50 \
    $EARG --net checkpoints/pi05/st_long_pe_g0.pt --net-latency 9.59 \
    --out ckpt/pi05/$S/student_s7.json                                   # 94.4% @ 3.43x
$PY scripts/paired_table.py ckpt/pi05/$S r5 teacher_L50 student
```

`--net-latency` is the student's per-correction latency from `latency_probe.py`
on your GPU; speedups are computed from measured per-call latencies and the calls
each run made, not from wall time. Suites take 20 (Goal) to 40 (Long) minutes.

**Training.** Six stages, in order. Each skips finished work.

| stage | what | time | disk |
|---|---|---|---|
| `00_baselines.sh` | replan 5, replan 10, teacher on 4 suites | ~6 h | — |
| `01_rounds_0_2.sh` | r0 (teacher drives), dg (st_r0), dg2 (st_r1), 20 scenes/task/suite; st_r0, st_r1, st_r2 | ~10 h | a 37, 81 and 119 GB cache, one per student |
| `02_plan_context.sh` | pc, st_r2 drives, 60 scenes/task/suite, stores plan context + age; st_r8 (145k steps) | ~5 h | 6 GB shards, 111 GB cache |
| `03_dagger_pd.sh` | pd, st_r8 drives, 60 scenes/task/suite | ~3 h | 6 GB |
| `04_suite_finetune.sh` | st_ft_{spatial,object,goal}: st_r8 on pc+pd of the suite, lr 3e-4, 20 epochs; evaluation | ~2.5 h | 37-50 GB cache each |
| `05_long_dagger.sh` | st_long (lr 1e-4, 30 epochs); pe, st_long drives, 60 scenes/task and 120 on tasks 5, 7; st_long_pe on pc+pd+pe, lr 3e-4, 20 epochs; evaluation | ~5 h | 110 + 174 GB caches |

The memory caches (the policy's prefix embeddings for every harvested
observation, 712 x 2048 fp16 = 2.9 MB each) are what make training fast; they can
be deleted once a student is trained. The training scenes are `env.reset()`
scenes under their own seeds (`--random-scenes --scene-seed`, disjoint blocks per
round: 100000, 200000, 300000, 700000, 800000, 900000), never the evaluation's
fixed initial states.

**Clips.** Any episode, both sides, with the source of every executed action:

```bash
$PY scripts/eval_corrector.py --model pi05 --latency-from $L --suite libero_10 --seed 7 --trials 50 \
    --configs baseline corrector --replan 5 $(echo $EARG | sed 's/--configs corrector//') \
    --net checkpoints/pi05/st_long_pe_g0.pt --net-latency 9.59 \
    --task-ids 9 --trial-ids 6 --video-dir videos/mine --out /tmp/clip.json
```

## π0 on LIBERO-Spatial


All commands assume the repository root as the working directory and
`.venv/bin/python` as the interpreter. Add `JAX_PLATFORMS=cpu` to keep JAX off
the GPU.

### The four evaluation rows

Every run reports **both** the π₀ baseline and the corrector over the same
episodes; the printed comparison at the end is the paired one.

```bash
PY=.venv/bin/python
COMMON="--adaptive --k-draws 4 --tau0 0.5 --grip snap \
        --n-min 5 --c-max 25 --c-safe 10 --agree-tau 1.0"

# teacher, full depth                      98.0% @ 1.63x
$PY scripts/eval_corrector.py $COMMON --seed 7 --out ckpt/teacher_s7.json

# LoRA student                             96.4% @ 2.05x
$PY scripts/eval_corrector.py $COMMON --seed 7 \
    --student checkpoints/final30_14-12.pt --out ckpt/lora_s7.json

# network student, lambda = 1              97.71% @ 3.12x
$PY scripts/eval_corrector.py $COMMON --seed 7 \
    --net checkpoints/spnet_g0.pt --out ckpt/spnet_s7.json
```

The reported table uses seeds `7 17 27 37 47 57 67` for the baseline and network
rows, `7 17 27 37 47` for LoRA, and `7 17 27` for the teacher. Each seed is 100
episodes: 10 tasks × 10 trials of LIBERO-Spatial.

Seeds 7 and up are **evaluation seeds** and were never harvested. The
distillation data comes from seeds 101 and 102.

## Aggregating and testing

`results/` already holds the per-seed JSON, each with per-episode outcomes.
Pooling them and running McNemar over the discordant pairs is what produced
`b = 15, c = 25, p = 0.155`. Do not compare two success rates directly — the
episodes are paired, and the unpaired difference throws that away.

## Rebuilding the students

Harvest first. Roughly 26 minutes per shard on a 5090; the four shards behind
the released students are `libero_spatial` seeds 101 and 102, `libero_goal` 101
and `libero_object` 101, plus `libero_90` seed 101 at 3 trials.

```bash
$PY scripts/harvest_distill.py --suite libero_spatial --seed 101 \
    --tasks 10 --trials 10 --out distill/spatial
```

Then either student. Both write a checkpoint whenever validation improves, so a
crash leaves a usable file.

```bash
# network student, the reported one -- about 2 hours on a 5090
$PY scripts/train_net_student.py \
  --shard distill/train_s101.pt,distill/libero_goal_s101.pt,\
distill/libero_spatial_s102.pt,distill/libero_object_s101.pt,distill/libero_90_s101.pt \
  --gemma-layers 0 --lambda-spread 1.0 \
  --select-shards 0,1,2,3 --val-obs 128 \
  --epochs 120 --val-every 4000 --out distill/spnet

# the lambda = 0 ablation: same command, --lambda-spread 0
```

`--select-shards` decides the checkpoint on the suites that are actually
evaluated. Choosing on the aggregate instead, when one suite dominates the data,
selects for competence on tasks nobody measures — that mistake cost 5 points
once and is why the flag exists. `--val-obs` counts observations **from the
selected shards**, so it does not silently shrink when they are a minority.

## Figures

```bash
# per-stage latency
$PY scripts/stage_latency.py

# does draw disagreement predict error?
$PY scripts/certificate_diagnostic.py

# side-by-side video of episodes where exactly one side succeeds
$PY scripts/render_compare.py --net checkpoints/spnet_g0.pt --seed 17 \
    --render both --max-clips 9 --out-dir clips
```

`render_compare.py` runs a scan pass and then re-runs the divergent episodes
with rendering on. Feed `--scan-json` a previous scan to skip the first pass;
the file is a list of `{task, trial, language, baseline, adaptive}`, so it can
also be synthesised from an evaluation JSON.
