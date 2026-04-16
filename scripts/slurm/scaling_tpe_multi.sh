#!/bin/bash
# Scaling law experiments with optimized HP search (TPE multivariate warm-start).
#
# Peter's 6 strategies × 7 sizes (1K-200K) = 42 jobs.
# Each job runs:
#   1. TPE multivariate warm-start, 20 trials on chr-split val
#   2. 3 replicates with best HP
#
# Uses biased AG S2 oracle (existing pools).
#
#SBATCH --job-name=scl_tpm
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
SIZES=(1000 2000 5000 10000 20000 50000 100000)

STRAT_IDX=$((T / 7))
SIZE_IDX=$((T % 7))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

OUT="outputs/exp1_1_tpe_multi/k562/legnet_ag_s2"

# Skip if all 3 seeds done
ALL_DONE=true
for SEED in 42 1042 2042; do
    [ ! -f "${OUT}/${STRAT}/n${SIZE}/hp0/seed${SEED}/result.json" ] && ALL_DONE=false
done
$ALL_DONE && echo "SKIP: all seeds done" && exit 0

echo "=== Scaling TPE-Multi: ${STRAT} n=${SIZE} — $(date) ==="

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

seqs, labels = load_pool_data(strategy, n_train, seed)
_ = get_chr_val()

def objective(trial):
    lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    return train_and_evaluate(seqs, labels, lr, bs, wd, seed)

# TPE multivariate with warm-start (the optimized approach)
sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
study = optuna.create_study(direction="maximize", sampler=sampler)

# Warm-start configs
for wc in [
    {"lr": 0.001, "batch_size": 512, "weight_decay": 1e-5},
    {"lr": 0.005, "batch_size": 1024, "weight_decay": 1e-5},
    {"lr": 0.002, "batch_size": 256, "weight_decay": 0.004},
    {"lr": 0.003, "batch_size": 512, "weight_decay": 1e-6},
]:
    study.enqueue_trial(wc)

# Load neighbor configs if available
for neighbor_n in [n_train // 2, n_train * 2]:
    nf = REPO / "outputs" / "optuna_best" / strategy / f"n{neighbor_n}" / "best_config_seed42.json"
    if nf.exists():
        try:
            nc = json.loads(nf.read_text())["config"]
            study.enqueue_trial(nc)
        except: pass

study.optimize(objective, n_trials=20)
best = study.best_trial
print(f"Best HP: val={best.value:.4f} lr={best.params['lr']:.5f} bs={best.params['batch_size']}")

# Save best config
optuna_dir = REPO / "outputs" / "optuna_best_tpe_multi" / strategy / f"n{n_train}"
optuna_dir.mkdir(parents=True, exist_ok=True)
with open(optuna_dir / "best_config.json", "w") as f:
    json.dump({"config": best.params, "val": best.value, "method": "tpe_multi_20_warm"}, f, indent=2)

# Run 3 replicates with best HP
pool_2m = REPO / "outputs/labeled_pools_2m/k562/ag_s2"
pool_618k = REPO / "outputs/labeled_pools/k562/ag_s2"
pool_dir = str(pool_2m) if (pool_2m / strategy / "pool.npz").exists() else str(pool_618k)
out_final = str(REPO / "outputs" / "exp1_1_tpe_multi" / "k562" / "legnet_ag_s2")

for rep_seed in [42, 1042, 2042]:
    result_path = Path(out_final) / strategy / f"n{n_train}" / "hp0" / f"seed{rep_seed}" / "result.json"
    if result_path.exists():
        print(f"  Skip seed={rep_seed}")
        continue
    print(f"  Training seed={rep_seed}...")
    os.system(
        f"uv run --no-sync python experiments/exp1_1_scaling.py "
        f"--task k562 --student legnet --oracle ag_s2 "
        f"--reservoir {strategy} "
        f"--pool-base-dir {pool_dir} "
        f"--n-replicates 1 --seed {rep_seed} "
        f"--output-dir {out_final} "
        f"--training-sizes {n_train} "
        f"--chr-split --lr {best.params['lr']} --batch-size {best.params['batch_size']} "
        f"--epochs 80 --ensemble-size 1 --early-stop-patience 10"
    )

print("Done!")
PYEOF

echo "=== DONE — $(date) ==="
