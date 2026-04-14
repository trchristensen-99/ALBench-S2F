#!/bin/bash
# Compare Optuna search algorithms on random n=10K (medium size, good benchmark).
#
# Array:
#   0: TPE (default, current approach)
#   1: CMA-ES (evolution strategy, good for continuous HPs)
#   2: Random (baseline)
#   3: TPE with 30 trials (more exploration)
#   4: TPE with multivariate=True (models HP correlations)
#
#SBATCH --job-name=smp_cmp
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

echo "=== Sampler comparison T=$T — $(date) ==="

python3 << PYEOF
import json, os, sys, numpy as np
sys.path.insert(0, ".")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scripts.optuna_legnet_scaling import load_pool_data, train_and_evaluate, get_chr_val, REPO

strategy = "random"
n_train = 10000
seed = 42

# Pre-load data
seqs, labels = load_pool_data(strategy, n_train, seed)
_ = get_chr_val()  # cache val set

def objective(trial):
    lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    return train_and_evaluate(seqs, labels, lr, bs, wd, seed)

configs = [
    ("TPE_10trials", optuna.samplers.TPESampler(seed=seed), 10),
    ("CMA-ES_10trials", optuna.samplers.CmaEsSampler(seed=seed), 10),
    ("Random_10trials", optuna.samplers.RandomSampler(seed=seed), 10),
    ("TPE_30trials", optuna.samplers.TPESampler(seed=seed), 30),
    ("TPE_multivariate_10", optuna.samplers.TPESampler(seed=seed, multivariate=True), 10),
]

idx = $T
name, sampler, n_trials = configs[idx]

print(f"Running {name}...")
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
print(f"\n=== {name}: val={best.value:.4f} lr={best.params['lr']:.5f} bs={best.params['batch_size']} ===")
print(f"All trials:")
for t in study.trials:
    if t.value is not None:
        print(f"  trial {t.number}: val={t.value:.4f} lr={t.params['lr']:.5f} bs={t.params['batch_size']}")

# Save
out_dir = REPO / "outputs" / "sampler_comparison"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / f"{name}.json", "w") as f:
    json.dump({"name": name, "best_val": best.value, "best_params": best.params,
               "all_trials": [{"val": t.value, "params": t.params} for t in study.trials if t.value]}, f, indent=2)
PYEOF

echo "=== DONE — $(date) ==="
