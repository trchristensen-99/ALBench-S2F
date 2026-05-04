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

## Task 3 — Joint LR × BS sweep (pending)
## Task 4 — Epoch budget calibration (pending)
## Task 5 — Augmentation tests (pending)
## Task 6 — Parameterization sensitivity (pending)
## Task 7 — Dropout sensitivity (pending)
## Task 8 — Acquisition method sanity (pending)
## Task 9 — D_min confirmation (pending)
## Task 10 — Sign-off / pre_flight_decisions.yaml lock (pending)

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
