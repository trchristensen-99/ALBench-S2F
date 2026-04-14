#!/bin/bash
# Compare HP search strategies across multiple (strategy, size) combos.
#
# Tests: TPE-10, TPE-20, TPE-30, CMA-ES-10, Random-10, TPE-multivariate-10
# On: random n=5K, random n=50K, genomic n=5K, motif_grammar n=10K
# = 6 search methods × 4 test cases = 24 jobs
#
# Array: method_idx * 4 + case_idx
#
#SBATCH --job-name=hp_cmp
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
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

# Test cases: (strategy, size)
CASES_STRAT=("random" "random" "genomic" "motif_grammar")
CASES_SIZE=(5000 50000 5000 10000)

CASE_IDX=$((T % 4))
METHOD_IDX=$((T / 4))

STRAT=${CASES_STRAT[$CASE_IDX]}
SIZE=${CASES_SIZE[$CASE_IDX]}

# Methods: (name, sampler_class, n_trials, extra_args)
METHODS=("TPE_10" "TPE_20" "TPE_30" "CMA_10" "Random_10" "TPE_multi_10")

METHOD=${METHODS[$METHOD_IDX]}

echo "=== HP comparison: ${METHOD} on ${STRAT} n=${SIZE} — $(date) ==="

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
seed = 42
method = "${METHOD}"

seqs, labels = load_pool_data(strategy, n_train, seed)
_ = get_chr_val()

def objective(trial):
    lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    return train_and_evaluate(seqs, labels, lr, bs, wd, seed)

# Configure sampler
if method.startswith("TPE_multi"):
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    n_trials = 10
elif method.startswith("TPE"):
    sampler = optuna.samplers.TPESampler(seed=seed)
    n_trials = int(method.split("_")[1])
elif method.startswith("CMA"):
    sampler = optuna.samplers.CmaEsSampler(seed=seed)
    n_trials = int(method.split("_")[1])
elif method.startswith("Random"):
    sampler = optuna.samplers.RandomSampler(seed=seed)
    n_trials = int(method.split("_")[1])
else:
    raise ValueError(f"Unknown method: {method}")

study = optuna.create_study(direction="maximize", sampler=sampler)

# Warm-start with known-good configs
for wc in [
    {"lr": 0.001, "batch_size": 512, "weight_decay": 1e-5},
    {"lr": 0.005, "batch_size": 1024, "weight_decay": 1e-5},
    {"lr": 0.002, "batch_size": 256, "weight_decay": 0.004},
]:
    study.enqueue_trial(wc)

study.optimize(objective, n_trials=n_trials)

best = study.best_trial
print(f"RESULT: {method} {strategy} n={n_train}: val={best.value:.4f} lr={best.params['lr']:.5f} bs={best.params['batch_size']}")

out_dir = Path("outputs/hp_search_comparison")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / f"{method}_{strategy}_n{n_train}.json", "w") as f:
    json.dump({
        "method": method, "strategy": strategy, "n_train": n_train,
        "best_val": best.value, "best_params": best.params,
        "n_trials": n_trials,
        "all_trials": [{"val": t.value, "params": t.params} for t in study.trials if t.value is not None],
    }, f, indent=2)
PYEOF

echo "=== DONE — $(date) ==="
