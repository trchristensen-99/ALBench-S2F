#!/bin/bash
# Comprehensive HP search comparison at large N (50K, 100K, 200K).
#
# Tests 8 search methods × 3 sizes × 3 strategies = 72 jobs
#
# Methods:
#   0: TPE-10 (baseline, current)
#   1: TPE-20 warm-start
#   2: TPE-30 warm-start
#   3: TPE-50 warm-start (exhaustive)
#   4: CMA-ES-20 warm-start (evolution strategy)
#   5: Random-20 warm-start (baseline comparison)
#   6: TPE-20 multivariate warm-start (models HP correlations)
#   7: TPE-20 no-warm-start (to measure warm-start benefit)
#
# Strategies: random, genomic, motif_grammar
# Sizes: 50000, 100000, 200000
#
# Array: method_idx * 9 + strat_idx * 3 + size_idx
#
#SBATCH --job-name=hp_lg
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

STRATS=("random" "genomic" "motif_grammar")
SIZES=(50000 100000 200000)

METHOD_IDX=$((T / 9))
STRAT_IDX=$(( (T % 9) / 3 ))
SIZE_IDX=$((T % 3))

STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

METHODS=("TPE_10" "TPE_20_ws" "TPE_30_ws" "TPE_50_ws" "CMA_20_ws" "Random_20_ws" "TPE_multi_20_ws" "TPE_20_no_ws")
METHOD=${METHODS[$METHOD_IDX]}

echo "=== HP search: ${METHOD} ${STRAT} n=${SIZE} — $(date) ==="

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

# Parse method config
parts = method.split("_")
use_warm_start = "ws" in method
no_warm_start = "no_ws" in method

if "multi" in method:
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    n_trials = 20
elif "CMA" in method:
    sampler = optuna.samplers.CmaEsSampler(seed=seed)
    n_trials = int(parts[1])
elif "Random" in method:
    sampler = optuna.samplers.RandomSampler(seed=seed)
    n_trials = int(parts[1])
else:  # TPE
    sampler = optuna.samplers.TPESampler(seed=seed)
    n_trials = int(parts[1])

study = optuna.create_study(direction="maximize", sampler=sampler)

# Warm-start with known-good configs (unless testing no-warm-start)
if use_warm_start and not no_warm_start:
    warm_configs = [
        {"lr": 0.001, "batch_size": 512, "weight_decay": 1e-5},
        {"lr": 0.005, "batch_size": 1024, "weight_decay": 1e-5},
        {"lr": 0.002, "batch_size": 256, "weight_decay": 0.004},
        {"lr": 0.003, "batch_size": 512, "weight_decay": 1e-6},
    ]
    # Also load neighbor configs
    for neighbor_n in [n_train // 2, n_train * 2]:
        nf = REPO / "outputs" / "optuna_best" / strategy / f"n{neighbor_n}" / "best_config_seed42.json"
        if nf.exists():
            try:
                nc = json.loads(nf.read_text())["config"]
                warm_configs.append(nc)
            except: pass
    for wc in warm_configs:
        study.enqueue_trial(wc)

study.optimize(objective, n_trials=n_trials)

best = study.best_trial
print(f"RESULT: {method} {strategy} n={n_train}: val={best.value:.4f} lr={best.params['lr']:.5f} bs={best.params['batch_size']}")

# Save
out_dir = Path("outputs/hp_search_large_n")
out_dir.mkdir(parents=True, exist_ok=True)
fname = f"{method}_{strategy}_n{n_train}.json"
with open(out_dir / fname, "w") as f:
    json.dump({
        "method": method, "strategy": strategy, "n_train": n_train,
        "best_val": best.value, "best_params": best.params,
        "n_trials": n_trials, "warm_start": use_warm_start and not no_warm_start,
        "all_trials": [{"val": t.value, "params": t.params, "number": t.number}
                      for t in study.trials if t.value is not None],
    }, f, indent=2)

print(f"Saved: {fname}")
PYEOF

echo "=== DONE — $(date) ==="
