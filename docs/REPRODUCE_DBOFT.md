# DB-OFT on SimplerEnv Bridge

DB-OFT (Dexbotic-OFT, 7B) denoises its 16-action chunk with DDIM **through the whole LLM**, so a
teacher correction costs a pass over the policy, not a small head. Everything below runs on one
RTX 5090; paths hang off `$CORRECTOR_HOME/dboft`.

## Protocol

Four Bridge tasks (Carrot, Spoon, StackCube, Eggplant), the benchmark's 24 fixed scenes each,
`octo-init-rng` labels **0, 2, 6, 8, 12**: 120 episodes per task, 480 per row. The label only
feeds the Octo model and DB-OFT's diffusion noise is unseeded, so the five labels are five
repetitions over the same 24 scenes; paired comparisons therefore pair by scene.

The baseline is DB-OFT replanning every 5 of its 16 actions **with the prefix KV cache on**: the
~620-token image and prompt prefix is computed once per call and reused across the 10 DDIM
steps and the K draws (mean action difference 0.001 against the uncached policy, 2.11x
faster). Every speedup is against that cached baseline.

| method | Carrot | Spoon | StackCube | Eggplant | average | ms per step | speedup |
|---|---|---|---|---|---|---|---|
| DB-OFT, replan 5 (cached) | 60.0 | 85.8 | 31.7 | 94.2 | 67.9 | 56.1 | 1.00x |

A plan costs 280 ms. `scripts/dboft_table.py` prints the table of any set of runs, with a
scene-paired sign-flip test against the first row.

## Setup

```bash
bash setup/05_environment_dboft.sh     # sources, two venvs (server / client), the DB-OFT checkpoint
python setup/06_patch_new_scenes.py    # the client can load training scenes (DBOFT_SCENES=...)
python setup/07_patch_eval_robust.py   # one failing episode no longer ends a run silently
python setup/08_dboft_client.py        # simpler/ link, client configs, variable segment length
```

The server and the SimplerEnv client cannot share one environment (the client pins numpy 1.24.4
and sapien 2.2.2). Three pins the upstream install does not make are in `05`: `pyarrow<15`,
`transformers==4.57.6` and no `kernels`. SAPIEN needs the Vulkan ICD at
`/usr/share/vulkan/icd.d/nvidia_icd.json`, or every episode dies at the first step. Set
`DEXBOTIC_REF`, `BENCHMARK_REF` and `SIMPLER_REF` to pin the upstream commits.

`setup/08_dboft_client.py` matters for the method: the upstream client always executes a fixed
5 of the returned actions; with `DBOFT_VARSEG=1` it executes exactly what the server returns,
which is how the certificate sets the segment length.

## Stages

`pipelines/dboft/common.sh` holds the paths, the seeds and the teacher flags. Evaluation runs
(`eval_run`) use the 24 benchmark scenes; harvest runs (`train_run`) load the 48 training
layouts per task from `data/dboft/train_scenes.json` (`obj-episode-range 0,48`).

| stage | what |
|---|---|
| `scripts/dboft_new_scenes.py` | 48 training layouts per task: the task's own objects moved and rotated inside the original table area, kept only if they settle without drifting, stay visible and lie at least 2.5 cm from every original position (Eggplant, which always rolls down the sink, is judged after settling) |
| `00_calibrate_sref.sh` | the certificate's reference spread, measured once: 8 plans per call over the first calls of 4 episodes per task, active states only (`--cal-first 12`); the released file is `data/dboft/sref_dboft.json` |
| `01_eval.sh [baseline\|teacher\|student] [CKPT]` | evaluation over the seeds, then the table; the student defaults to `checkpoints/dboft/student.pt` |
| `02_harvest.sh` | teacher-driven harvest on the training scenes, 3 runs: per correction the CLIP features before the projector (576 x 1024), prompt embedding, anchor, age, and the teacher's eps for 8 draws at both DDIM steps |
| `03_train_r1.sh` | r1: 60 epochs on the teacher rounds |
| `04_dagger.sh` | student r1 drives, the teacher labels the states it visits, 3 runs |
| `05_train_r8_r11.sh` | r8 = r1 fine-tuned on teacher data plus the Eggplant DAgger data (lr 1e-4, 10 epochs); r11 = r8 fine-tuned with the StackCube DAgger data and a replay of all four tasks (lr 5e-5, 3 epochs) |

**Teacher flags** (WPFS-Large):

```
--k 4 --start-step 8 --n-min 2 --lineage 15 --agree-tau 1.5 \
--grip-lock --grip-lock-mode both --sref-file data/dboft/sref_dboft.json
```

`--start-step 8` re-noises the stale chunk to abar = 0.967 and runs the last 2 DDIM steps.
`--grip-lock --grip-lock-mode both`: **a correction never executes a gripper change**. DB-OFT's
plans change the gripper twice within 16 positions 39% of the time, ending with a spurious
re-open at positions 12-15 that the baseline never reaches because it replans at 5; cutting the
correction before any gripper change and forcing a plan keeps grasp decisions with fresh plans.

**Student.** `scripts/dboft_student_net.py`, 28.4M parameters (d 512 x 6 blocks). Its memory is
the policy's CLIP features plus the prompt embedding, the anchor chunk and the age; it predicts
the teacher's eps at the two DDIM steps the teacher takes, with the loss weighted by
(1 - abar) / abar so the last step does not dominate. DB-OFT's scheduler has 100 training
timesteps (90..0), not 1000.

## Baselines

```bash
bash setup/09_baselines.sh                         # the AAC authors' decision code
bash baselines/dboft/run.sh effvla aac vlacache specprune
```

One instrumented server (`baselines/dboft/dboft_baseline_server.py`) runs all four, with the
same prefix cache and the same measurement as the rows above: EfficientVLA caches each decoder
layer's contribution across DDIM steps (`--ev-interval 2`); AAC samples 5 chunks per plan and
calls the authors' `select_chunk_size` (`--aac-move-th 0.5`, Bridge actions are in metres and
radians); VLA-Cache reuses static visual tokens in the prefill (`--vc-max-age 0`); SpecPrune
prunes visual tokens in the prefill (`--sp-alpha 2.0`; `--sp-impl official` follows the authors'
released rule instead of the paper's text). `--bench`, `--bench-ev`, `--bench-vc` and
`--bench-sp` check each implementation against the plain policy and exit.

## Caveats

* **Selection.** The teacher's lineage and agreement threshold were chosen on these evaluation
  seeds.
* **Run-to-run noise.** Success is a threshold on a 120-step closed loop in bf16: the same
  checkpoint on the same scene flips about a third of its episodes between two machines, and
  per-task totals move by about 5 episodes in 120. Compare rows measured on one machine.
* **Against the published table.** The upstream report gives 76.39 on seeds 0, 2, 4; this
  harness gives 68.1 on those seeds. Dexbotic's own server with the stock client, on the same
  machine, also stays well below the published numbers (Carrot 66.7%, StackCube 30.0% over five
  repetitions), so the gap is in reproducing the benchmark, not in the corrector.
