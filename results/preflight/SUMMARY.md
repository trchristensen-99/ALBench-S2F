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

## Task 2 — D_min provisional (pending)
## Task 3 — Joint LR × BS sweep (pending)
## Task 4 — Epoch budget calibration (pending)
## Task 5 — Augmentation tests (pending)
## Task 6 — Parameterization sensitivity (pending)
## Task 7 — Dropout sensitivity (pending)
## Task 8 — Acquisition method sanity (pending)
## Task 9 — D_min confirmation (pending)
## Task 10 — Sign-off / pre_flight_decisions.yaml lock (pending)
