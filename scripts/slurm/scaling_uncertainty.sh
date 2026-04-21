#!/bin/bash
# Scaling laws with uncertainty-based acquisition.
#
# Instead of random subsampling, take the TOP-N most uncertain sequences
# from the uncertainty-ranked pool.
#
# 3 strategies × 10 sizes × 3 reps = 90 jobs
# Array: strat_idx * 30 + size_idx * 3 + rep_idx
#
#SBATCH --job-name=scl_unc
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

STRATS=("evoaug_heavy" "motif_grammar" "prm_1pct")
SIZES=(1000 2000 5000 10000 20000 50000 100000 200000 300000 500000)
SEEDS=(42 1042 2042)

STRAT_IDX=$((T / 30))
SIZE_IDX=$(( (T % 30) / 3 ))
REP_IDX=$((T % 3))

STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$REP_IDX]}

# Size-calibrated HP
if [ $SIZE -le 2000 ]; then
    LR=0.0005
    BS=128
else
    LR=0.003
    BS=256
fi

OUT="outputs/exp1_1_uncertainty/k562/legnet_ag_s2"
RESULT="${OUT}/${STRAT}/n${SIZE}/rep${REP_IDX}/result.json"

[ -f "${RESULT}" ] && echo "SKIP" && exit 0

RANKING_FILE="outputs/uncertainty_ranking/${STRAT}/uncertainty_ranking.npz"
[ ! -f "${RANKING_FILE}" ] && echo "ERROR: ranking not found at ${RANKING_FILE}" && exit 1

echo "=== Uncertainty Scaling: ${STRAT} n=${SIZE} rep${REP_IDX} — $(date) ==="

uv run --no-sync python << PYEOF
import json, os, sys, numpy as np
sys.path.insert(0, ".")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path
REPO = Path(".")

strategy = "${STRAT}"
n_train = ${SIZE}
seed = ${SEED}
rep_idx = ${REP_IDX}

# Load the uncertainty ranking
ranking_data = np.load("${RANKING_FILE}", allow_pickle=True)
ranking = ranking_data["ranking"]  # indices sorted by uncertainty (most uncertain first)
ensemble_mean = ranking_data["ensemble_mean"]  # oracle predictions

# Load the pool
pool_dirs = {
    "5m": REPO / "outputs" / "labeled_pools_5m" / "k562" / "ag_s2" / strategy,
    "2m": REPO / "outputs" / "labeled_pools_2m" / "k562" / "ag_s2" / strategy,
    "618k": REPO / "outputs" / "labeled_pools" / "k562" / "ag_s2" / strategy,
}
for size_key in ["5m", "2m", "618k"]:
    p = pool_dirs[size_key] / "pool.npz"
    if p.exists():
        pool = np.load(p, allow_pickle=True)
        break

all_seqs = pool["sequences"]
all_labels = pool["labels"]

# Select top-N most uncertain sequences
# Add small random jitter per replicate to avoid identical subsets
rng = np.random.default_rng(seed)
n_available = min(len(ranking), len(all_seqs))

if n_train <= n_available:
    # Take top n_train from ranking, with slight randomization per rep
    # Shuffle within the top 2*n_train to get different subsets per rep
    top_pool = ranking[:min(2 * n_train, n_available)]
    selected = rng.choice(top_pool, size=n_train, replace=False)
else:
    selected = ranking[:n_train]

seqs = [str(all_seqs[i]) for i in selected]
labels = all_labels[selected].astype(np.float32)

print(f"Selected {len(seqs)} sequences by uncertainty (from top {min(2*n_train, n_available)})")
print(f"Label mean={labels.mean():.3f} std={labels.std():.3f}")

# Save as oracle_labels for exp1_1_scaling.py compatibility
out_dir = Path("${OUT}") / strategy / f"n{n_train}"
out_dir.mkdir(parents=True, exist_ok=True)
oracle_path = out_dir / f"oracle_labels_uncertainty_rep{rep_idx}.npz"
np.savez_compressed(oracle_path, sequences=np.array(seqs), labels=labels)

# Train LegNet
from models.legnet_student import LegNetStudent, TrainConfig
from scripts.optuna_legnet_scaling import get_chr_val

val_seqs, val_labels = get_chr_val()

config = TrainConfig(
    lr=${LR}, batch_size=${BS}, weight_decay=1e-5,
    epochs=80, early_stopping_patience=10,
)
model = LegNetStudent(ensemble_size=1, train_config=config)
model.fit(sequences=seqs, labels=labels,
          val_sequences=val_seqs, val_labels=val_labels)

# Evaluate on test sets
from data.k562 import K562Dataset
import torch

test_dir = REPO / "data" / "k562" / "test_sets"
results = {"reservoir": strategy, "n_train": n_train, "seed": seed,
           "acquisition": "uncertainty", "hp_config": {"learning_rate": ${LR}, "batch_size": ${BS}}}

test_metrics = {}
for test_name, test_file in [
    ("in_dist", "test_chr7_13_all.tsv"),
    ("ood", "test_ood_designed_k562.tsv"),
    ("snv_pairs", "test_snv_pairs_hashfrag.tsv"),
]:
    f = test_dir / test_file
    if not f.exists():
        continue
    import pandas as pd
    if test_name == "snv_pairs":
        df = pd.read_csv(f, sep="\t")
        ref_seqs = df["sequence_ref"].str[:200].tolist()
        alt_seqs = df["sequence_alt"].str[:200].tolist()
        ref_labels = df["K562_log2FC_ref"].values
        alt_labels = df["K562_log2FC_alt"].values
        ref_preds = model.predict(ref_seqs)
        alt_preds = model.predict(alt_seqs)
        from scipy.stats import pearsonr
        # SNV abs: correlation of all predictions with all labels
        all_preds = np.concatenate([ref_preds, alt_preds])
        all_labels_test = np.concatenate([ref_labels, alt_labels])
        r_abs, _ = pearsonr(all_labels_test, all_preds)
        # SNV delta: correlation of (alt-ref) predicted vs real
        delta_pred = alt_preds - ref_preds
        delta_real = alt_labels - ref_labels
        r_delta, _ = pearsonr(delta_real, delta_pred)
        test_metrics["snv_abs"] = {"pearson_r": float(r_abs), "n": len(all_preds)}
        test_metrics["snv_delta"] = {"pearson_r": float(r_delta), "n": len(delta_pred)}
    else:
        df = pd.read_csv(f, sep="\t")
        test_seqs = df["sequence"].str[:200].tolist()
        test_labels = df["K562_log2FC"].values
        preds = model.predict(test_seqs)
        from scipy.stats import pearsonr
        r, _ = pearsonr(test_labels, preds)
        mse = float(np.mean((preds - test_labels) ** 2))
        test_metrics[test_name] = {"pearson_r": float(r), "mse": mse, "n": len(preds)}

results["test_metrics"] = test_metrics

# Save result
result_path = Path("${RESULT}")
result_path.parent.mkdir(parents=True, exist_ok=True)
with open(result_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved: {result_path}")
for k, v in test_metrics.items():
    print(f"  {k}: pearson_r={v['pearson_r']:.4f}")
PYEOF

echo "=== DONE — $(date) ==="
