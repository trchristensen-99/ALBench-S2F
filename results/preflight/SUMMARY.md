# ALBench-S2F pre-flight log

Each task appends a brief summary of what ran, decisions made, anomalies.

## Conventions
- Empirical range CIs (per PI): for n ≥ 4, range = [2nd-lowest, 2nd-highest].
  For n = 3, range = [midpoint(low, mid), midpoint(mid, high)].
- Reported metric: test MSE at epoch of minimum val loss within fixed budget.
- Post-hoc flag: runs where min val loss occurred in the final 10% of epochs.
- All runs: tags `phase=preflight, arch=<>, sweep=<>, seed=<>, d_train=<>`.
- Seeds default `{42, 123, 7}`.

## Task 1 — Infrastructure (in progress)
- Created `configs/preflight/{_base.yaml, arch/{legnet,dream_rnn,dream_attn}.yaml}`
- Created `scripts/preflight/run_single.py` — reusable runner
  taking `(arch, d_train, hp_overrides, seed)`. Trains on K562 + AG-oracle
  pseudolabels, checkpoints at best val, post-hoc-flags runs where min val
  fell in final 10%, logs train/val/test MSE per epoch + param count + GPU-hrs.
  W&B tags follow the schema.
- Created `scripts/preflight/launch.sh` SLURM wrapper. Auto-selects QOS
  (small N → fast, large N → slow_nice), accepts arbitrary `k=v` HP overrides.
- Initialized `results/preflight/pre_flight_decisions.yaml` skeleton.

## Task 2 — D_min provisional — DONE (36/36 results, 2026-05-04)
- All 36 cells (3 archs × 4 D ∈ {500,1000,2000,4000} × 3 seeds) passed
  the val_R² > 0.1 threshold at every D tested.
- **D_min_provisional = 500** (smallest D with all-arch all-seed pass).
- 2 LegNet cells flagged for tight epoch budget (best epoch in final 10%):
  - legnet d=500 seed=123  best_epoch=79/80
  - legnet d=1000 seed=42  best_epoch=78/80
  These surface as a Task 4 input — LegNet may need >80 epoch budget
  for the smaller-D regime.
- Per-(arch,D) min val_R² approximations (typical Var(y_val)=1.5):
  - D=500:  dream_attn +0.419, dream_rnn +0.419, legnet +0.388
  - D=1000: same dream_*; legnet +0.400
  - D=2000: same dream_*; legnet +0.414
  - D=4000: same dream_*; legnet +0.416
- Output: `results/preflight/d_min_provisional.csv` (36 rows).

## Task 3 — Joint LR × BS sweep — DONE (45/45, 2026-05-04→05)

Locked optima from `lock_task3_decisions.py`:

| Arch | LR | BS | val_MSE | flatness 1st-ring | interpretation |
|---|---|---|---|---|---|
| LegNet | 1e-3 | 512 | 0.3126 | 0.060 | moderate ✓ |
| DREAM-RNN | 3e-3 | 256 | 0.1768 | 0.162 | sharp ⚠ |
| DREAM-ATTN | 3e-4 | 128 | 0.2431 | **4.500** | very sharp ⚠⚠ |

**Comparison to published priors:**
- LegNet: locked LR is 5× lower than published 5e-3; BS 2× lower than 1024. DREAM challenge tuned at ~6M yeast seqs; we're at 600k K562 with smoother oracle pseudolabels.
- DREAM-RNN: locked LR is 3× higher than published 1e-3; BS 4× lower than 1024.
- DREAM-ATTN: locked LR matches published 3e-4; BS 4× lower than 512.
- All three arch families found smaller batch sizes than published — consistent with smoother labels + 10× less data.

**HP flatness flags:**
- DREAM-RNN borderline-sharp (1st-ring 0.16 just over 0.15 threshold).
- **DREAM-ATTN very sharp (1st-ring 4.5)** — the optimum's immediate LR-grid neighbors have val_MSE ~4× higher. Attention is brittle to HP choice on this dataset. Per the pre-reg's "what to do if something fails": this is a Task 3 D_min verification flag (in flight on `fast` queue, 16 runs).

## Task 3 verify @ D_min — RUNNING (16 runs on fast queue)
- Tests whether (LR*, BS*) at D=600k still wins at D=500. Required by pre-reg.
- DREAM-ATTN with 1st-ring 4.5 is the highest-risk arch for scale-coupling.
- `analyze_task3_verify.py` will report PASS/FAIL per arch (within 1 grid step) and exit code 2 if scale-coupling detected.

## Task 4 — Epoch budget calibration — RUNNING (3 long runs on slow_nice)
- One run per arch, 240 epochs at D=600k. Each ~12-14h.
- analyze_task4_epoch_budget applies plateau detection + 1.5× rule.

## Task 5 — Augmentation tests (script ready, gated on Task 4 lock)
## Task 6 — Parameterization sensitivity (script ready, gated on Task 4 lock)
## Task 7 — Dropout sensitivity (script ready, gated on Task 4 lock)

## Task 8 — Acquisition method sanity — DONE (2026-05-04)
- 18/18 runs (9 methods × 2 seeds) on cpu_fill, ~30 sec each.
- **All 9 methods PASS**: Jaccard distance vs random in [0.9954, 0.9977]
  — well above the 0.3 sanity threshold.
- Methods covered:
  - Reservoir-based: prm_5, prm_20, motif_grammar, gc_matched, dinuc_shuffle
  - Model-based proxies (k-mer): uncertainty_ensemble, uncertainty_mc_dropout,
    diversity_kmeans, diversity_max_distance
- `acquisition_sanity_flagged` in YAML: empty list.
- Output: `results/preflight/task8_summary.csv` + `figures/task8_jaccard.png`.

## Task 9 — D_min confirmation (script + analyzer ready, gated on Tasks 3+4+5+7)
## Task 10 — Sign-off / pre_flight_decisions.yaml lock (validator + lock helper ready)

## Auto-chain orchestration (complete through Task 9)
- `task3_finalize_and_chain.sh`: polls Task 3 D_max results, runs
  `analyze_hp_flatness` + `lock_task3_decisions`, fires Task 3 verify +
  Task 4 + the Task 4 watcher. Submitted as `2037659` with
  `afterany:pf_task3_launcher`.
- `task4_finalize_and_chain.sh`: polls Task 4, runs
  `analyze_task4_epoch_budget` (locks epoch budget), fires Tasks 5/6/7
  in parallel + the Task 5/6/7 watcher.
- `task5_6_7_finalize_and_chain.sh`: polls Tasks 5+6+7, runs the three
  analyzers (`analyze_task5_augmentations`, `analyze_task6_parameterization`,
  `analyze_task7_dropout`) + diagnostic plots, fires Task 9 + the
  Task 9 watcher.
- `task9_finalize_and_chain.sh`: polls Task 9, runs
  `analyze_task9_d_min_confirm` (writes `d_min.confirmed`), runs
  `task10_finalize.py --dry-run` for the validation report.
- **Final manual step**: `uv run --no-sync python
  scripts/preflight/task10_finalize.py --reviewer NAME` to sign off and
  mark the YAML immutable.

## Post-pre-flight pre-staged
- `launch_main_sweep.sh`: builds the Exp 1.1 main scaling-law sweep
  (3 archs × N_methods × |d_grid| × N_seeds at D_init={0, 600000})
  using locked HPs from the YAML. Refuses without sign-off. Use
  `--execute` to submit (default is plan-only).
- `score_eval_sets.py`: scores a single `best.pt` against the 13-parquet
  expanded eval-set panel. Use after main-sweep ckpts land.

## Aux — Expanded eval-set panel (parallel work, no HP dependence)
- Created `scripts/preflight/generate_eval_sets.py` — sequence-only
  generator for the expanded distribution-shift panel (no retraining
  required; sequences will be scored against main-sweep checkpoints in
  week 6/22).
- Output: `outputs/eval_sets_expanded/` with 13 parquets:
  `prm_{1,5,10,20}pct`, `dinuc_shuffle`, `evoaug_{light,medium,heavy}`,
  `gc_q{1..4}_of_4`, `random_uniform`.
- Local sanity-checked on chr7,13 fallback (seed=42): all PRM rates hit
  target within 1e-4, dinuc shuffle preserves dinuc counts, GC quartiles
  monotonic.
- HPC submission: `bash scripts/preflight/generate_eval_sets.sh` once
  the new ref+alt pool builder writes `pool/test.parquet`.
