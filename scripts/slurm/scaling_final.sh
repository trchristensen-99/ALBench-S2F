#!/bin/bash
# Final scaling law experiments: 6 strategies × 7 sizes × 3-5 replicates.
#
# Each job = one (strategy, size, replicate) — fully independent:
#   1. Draws a unique random subsample from the pool (seed-dependent)
#   2. Runs 20-trial TPE multivariate warm-start HP search on that subsample
#   3. Trains LegNet with the best HP on that same subsample
#   4. Evaluates on the standard chr7/13 test set
#
# Replicates use non-overlapping seed ranges to ensure different subsamples.
# No hardcoded seed=42 — each replicate is fully independent.
#
# Sizes and replicates:
#   1K, 2K:   5 replicates (high variance at small N)
#   5K:       3 replicates
#   10K-100K: 3 replicates each
#
# Total: 6 strats × (2×5 + 5×3) = 6 × 25 = 150 jobs
#
# Array indexing:
#   strat_idx = T / 25
#   Within each strategy: 25 jobs ordered as
#     size_idx 0 (1K): reps 0-4     → jobs 0-4
#     size_idx 1 (2K): reps 0-4     → jobs 5-9
#     size_idx 2 (5K): reps 0-2     → jobs 10-12
#     size_idx 3 (10K): reps 0-2    → jobs 13-15
#     size_idx 4 (20K): reps 0-2    → jobs 16-18
#     size_idx 5 (50K): reps 0-2    → jobs 19-21
#     size_idx 6 (100K): reps 0-2   → jobs 22-24
#
#SBATCH --job-name=scl_fin
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

STRATS=("random" "genomic" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
# Layout: [1K×5, 2K×5, 5K×3, 10K×3, 20K×3, 50K×3, 100K×3] = 25 per strategy
SIZES=(1000 1000 1000 1000 1000 2000 2000 2000 2000 2000 5000 5000 5000 10000 10000 10000 20000 20000 20000 50000 50000 50000 100000 100000 100000)
# Replicate index within each size group
REPS=(0 1 2 3 4 0 1 2 3 4 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)

JOBS_PER_STRAT=25
STRAT_IDX=$((T / JOBS_PER_STRAT))
WITHIN=$((T % JOBS_PER_STRAT))

STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$WITHIN]}
REP=${REPS[$WITHIN]}

# Generate a unique seed per replicate — spread across seed space
# Use large prime offsets to avoid any overlap
SEED=$((1000 * REP + 7919 * STRAT_IDX + 31 * WITHIN + 42))

OUT="outputs/exp1_1_final/k562/legnet_ag_s2"
RESULT="${OUT}/${STRAT}/n${SIZE}/rep${REP}/result.json"

[ -f "${RESULT}" ] && echo "SKIP: ${STRAT} n=${SIZE} rep${REP} already done" && exit 0

echo "=== Scaling Final: ${STRAT} n=${SIZE} rep${REP} seed=${SEED} — $(date) ==="

uv run --no-sync python << PYEOF
import json, os, sys, numpy as np
sys.path.insert(0, ".")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scripts.optuna_legnet_scaling import load_pool_data, train_and_evaluate, get_chr_val, REPO
from pathlib import Path

strategy = "${STRAT}"
n_train = ${SIZE}
seed = ${SEED}

# Load a unique subsample using this replicate's seed
seqs, labels = load_pool_data(strategy, n_train, seed)
_ = get_chr_val()

print(f"Loaded {len(seqs)} sequences (seed={seed})")

def objective(trial):
    lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    return train_and_evaluate(seqs, labels, lr, bs, wd, seed)

# TPE multivariate with warm-start
sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
study = optuna.create_study(direction="maximize", sampler=sampler)

for wc in [
    {"lr": 0.001, "batch_size": 512, "weight_decay": 1e-5},
    {"lr": 0.005, "batch_size": 1024, "weight_decay": 1e-5},
    {"lr": 0.002, "batch_size": 256, "weight_decay": 0.004},
    {"lr": 0.003, "batch_size": 512, "weight_decay": 1e-6},
]:
    study.enqueue_trial(wc)

study.optimize(objective, n_trials=20)
best = study.best_trial
print(f"Best HP: val={best.value:.4f} lr={best.params['lr']:.5f} bs={best.params['batch_size']}")

# Train final model with best HP on the SAME subsample
pool_2m = REPO / "outputs/labeled_pools_2m/k562/ag_s2"
pool_618k = REPO / "outputs/labeled_pools/k562/ag_s2"
pool_dir = str(pool_2m) if (pool_2m / strategy / "pool.npz").exists() else str(pool_618k)

os.system(
    f"uv run --no-sync python experiments/exp1_1_scaling.py "
    f"--task k562 --student legnet --oracle ag_s2 "
    f"--reservoir {strategy} "
    f"--pool-base-dir {pool_dir} "
    f"--n-replicates 1 --seed {seed} "
    f"--output-dir {REPO / 'outputs' / 'exp1_1_final' / 'k562' / 'legnet_ag_s2'} "
    f"--training-sizes {n_train} "
    f"--chr-split --lr {best.params['lr']} --batch-size {best.params['batch_size']} "
    f"--epochs 80 --ensemble-size 1 --early-stop-patience 10"
)

# Move result to rep-indexed path
src = REPO / "outputs" / "exp1_1_final" / "k562" / "legnet_ag_s2" / strategy / f"n{n_train}" / "hp0" / f"seed{seed}" / "result.json"
dst = Path("${RESULT}")
dst.parent.mkdir(parents=True, exist_ok=True)
if src.exists():
    import shutil
    shutil.copy2(src, dst)
    print(f"Result saved to {dst}")
else:
    print(f"WARNING: result not found at {src}")

print("Done!")
PYEOF

echo "=== DONE — $(date) ==="
