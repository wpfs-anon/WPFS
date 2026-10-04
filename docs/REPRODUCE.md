# Reproducing every number

All commands assume the repository root as the working directory and
`.venv/bin/python` as the interpreter. Add `JAX_PLATFORMS=cpu` to keep JAX off
the GPU.

## π0.5

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

## π0

`pipelines/pi0/common.sh` holds the paths and the corrector flags; the per-call latencies the paper's
π0 rows were priced with are in `data/latency/pi0_policy.json`
(`scripts/latency_probe.py --model pi0` re-measures them on your GPU, `LAT=... ` overrides).

```bash
bash pipelines/pi0/eval.sh                 # all suites; or name some: bash pipelines/pi0/eval.sh libero_10
```

per suite, 10 tasks x 50 initial states, seed 7:

```bash
CORR="--configs corrector --adaptive --k-draws 4 --tau0 0.5 --grip snap --n-min 5 --c-safe 10 --agree-tau 1.0"
$PY scripts/eval_corrector.py --model pi0 --latency-from data/latency/pi0_policy.json --suite $S --seed 7 \
    --trials 50 --configs baseline --replan 10 --out ckpt/pi0/$S/base_s7.json
$PY scripts/eval_corrector.py --model pi0 --latency-from data/latency/pi0_policy.json --suite $S --seed 7 \
    --trials 50 $CORR --c-max 25 --out ckpt/pi0/$S/teacher_s7.json          # --c-max 10 on libero_object
```

WPFS-Small adds `--net <student> --net-latency <ms>`: the student is a 27.8M-parameter network
whose memory is the SigLIP tokens of both cameras plus the instruction embedding (no
language-model layer); `pipelines/pi0/eval.sh` reads it from `checkpoints/pi0/student.pt` (or
`NET=...`), and `scripts/latency_probe.py --model pi0 --net <student>` measures its latency.

## Baselines

`baselines/libero/run.sh <pi0|pi05> [sp vc ev aac]` runs the four accelerators at the operating
points of the tables (selection rule: the highest success rate among configurations faster than
the policy):

| method | π0 | π0.5 |
|---|---|---|
| SpecPrune-VLA | `--alpha 2.0 --global-from expert --capture-every 1` | same |
| VLA-Cache | `--max-age 1` | same |
| EfficientVLA | action-expert cache only: `--n-prune 0 --k-final 512 --cache-interval 3` | same |
| AAC | `--n 5 --move-th 10` (the published N = 20, alpha = 3 costs 0.41x) | `--n 5 --move-th 5.0` |

On π0 each method's plan latency comes from an idle-GPU measurement (`data/latency/pi0_*.json`);
on π0.5 it is measured in the method's own session. `baselines/libero/table_pi0.py` and
`table_pi05.py` print the tables from the runs in `ckpt/pi0` and `ckpt/pi05`. AAC needs the authors' decision code:
`bash setup/09_baselines.sh`. The DB-OFT baselines are in `docs/REPRODUCE_DBOFT.md`.

## Figures

```bash
$PY scripts/staleness_probe.py --model pi0 --suite libero_spatial --tasks 4 --trials 3 \
    --seed 7 --replan 10 --refs 8 --probes 1,2,3,4,5,6,8,10,12,15,20,25,30 \
    --ms 5,10,20 --m-main 10 --taus 0.3,0.5,0.7 --tau-main 0.5
$PY scripts/staleness_figure.py ckpt/staleness/pi0_libero_spatial_s7.json \
    --out ckpt/staleness/staleness_pi0
$PY scripts/stage_latency.py
$PY scripts/certificate_diagnostic.py
```

The probe runs 12 episodes and ~1,950 probes in about 17 minutes on an RTX 5090; nothing in it is
scored by rollout.
