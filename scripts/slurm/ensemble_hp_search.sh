#!/bin/bash
# Ensembled HP selection: run Optuna 3 times with different seeds,
# pick the HP that works best across ALL 3 runs (consensus HP).
#
# For each (strategy, size): 3 independent Optuna runs (20 trials each),
# then evaluate ALL found HPs on ALL 3 data subsamples, pick the
# HP with highest MEAN val across subsamples.
#
# This is more robust than single-run Optuna because it tests
# HP generalization across data subsamples.
#
# 6 strategies × 6 sizes = 36 jobs
# Each job runs 3 × 20-trial Optuna + cross-evaluation
#
#SBATCH --job-name=ens_hp
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
SIZES=(1000 2000 5000 10000 20000 50000)

STRAT_IDX=$((T / 6))
SIZE_IDX=$((T % 6))
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

echo "=== Ensemble HP: ${STRAT} n=${SIZE} — $(date) ==="

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

_ = get_chr_val()

# Phase 1: Run 3 independent Optuna searches with different seeds
all_best_params = []
for optuna_seed in [42, 1042, 2042]:
    seqs, labels = load_pool_data(strategy, n_train, optuna_seed)

    def objective(trial):
        lr = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        return train_and_evaluate(seqs, labels, lr, bs, wd, optuna_seed)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=optuna_seed))
    # Warm-start
    for wc in [
        {"lr": 0.001, "batch_size": 512, "weight_decay": 1e-5},
        {"lr": 0.005, "batch_size": 1024, "weight_decay": 1e-5},
        {"lr": 0.002, "batch_size": 256, "weight_decay": 0.004},
    ]:
        study.enqueue_trial(wc)

    study.optimize(objective, n_trials=20)

    # Collect top-3 HPs from this run
    sorted_trials = sorted([t for t in study.trials if t.value is not None],
                          key=lambda t: -t.value)
    for t in sorted_trials[:3]:
        all_best_params.append(t.params)

    print(f"  Seed {optuna_seed}: best val={study.best_value:.4f}")

# Phase 2: Cross-evaluate all candidate HPs on all 3 data subsamples
print(f"  Testing {len(all_best_params)} candidate HPs across 3 subsamples...")
hp_scores = []
for hp in all_best_params:
    scores = []
    for eval_seed in [42, 1042, 2042]:
        seqs, labels = load_pool_data(strategy, n_train, eval_seed)
        val = train_and_evaluate(seqs, labels, hp["lr"], hp["batch_size"],
                                hp["weight_decay"], eval_seed)
        scores.append(val)
    hp_scores.append((np.mean(scores), np.std(scores), hp, scores))

# Phase 3: Pick the consensus HP (best mean across all subsamples)
hp_scores.sort(key=lambda x: -x[0])
best_mean, best_std, best_hp, best_scores = hp_scores[0]

print(f"  Consensus HP: lr={best_hp['lr']:.5f} bs={best_hp['batch_size']}")
print(f"  Mean val across subsamples: {best_mean:.4f} ± {best_std:.4f}")
print(f"  Per-subsample: {[f'{s:.4f}' for s in best_scores]}")

# Save
out_dir = Path("outputs/ensemble_hp") / strategy / f"n{n_train}"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "consensus_hp.json", "w") as f:
    json.dump({
        "strategy": strategy, "n_train": n_train,
        "consensus_hp": best_hp,
        "mean_val": best_mean, "std_val": best_std,
        "per_subsample_val": best_scores,
        "all_candidates": [{"mean": m, "std": s, "hp": h, "scores": sc}
                          for m, s, h, sc in hp_scores[:5]],
    }, f, indent=2)

# Phase 4: Run 3 final replicates with consensus HP
print("  Running 3 replicates with consensus HP...")
pool_2m = REPO / "outputs/labeled_pools_2m/k562/ag_s2"
pool_618k = REPO / "outputs/labeled_pools/k562/ag_s2"
pool_dir = str(pool_2m) if (pool_2m / strategy / "pool.npz").exists() else str(pool_618k)
out_final = str(REPO / "outputs" / "exp1_1_ensemble_hp" / "k562" / "legnet_ag_s2")

for rep_seed in [42, 1042, 2042]:
    result_path = Path(out_final) / strategy / f"n{n_train}" / "hp0" / f"seed{rep_seed}" / "result.json"
    if result_path.exists():
        print(f"  Skip seed={rep_seed}")
        continue
    os.system(
        f"uv run --no-sync python experiments/exp1_1_scaling.py "
        f"--task k562 --student legnet --oracle ag_s2 "
        f"--reservoir {strategy} "
        f"--pool-base-dir {pool_dir} "
        f"--n-replicates 1 --seed {rep_seed} "
        f"--output-dir {out_final} "
        f"--training-sizes {n_train} "
        f"--chr-split --lr {best_hp['lr']} --batch-size {best_hp['batch_size']} "
        f"--epochs 80 --ensemble-size 1 --early-stop-patience 10"
    )

print("Done!")
PYEOF

echo "=== DONE — $(date) ==="
