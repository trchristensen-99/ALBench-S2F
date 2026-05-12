#!/bin/bash
# Test pool-based validation (10% holdout from pool) vs chr-split validation.
# Uses 3 top HP configs × 9 sizes × 5 strategies × 1 rep each.
# For comparison: the definitive runs already have chr-split validation results.
#
# Array: 0-17 (3 HP × 2 sizes × 3 strategies)
#
#SBATCH --job-name=pool_val
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

# 3 HP configs to test with pool validation
LRS=(0.003 0.0005 0.005)
BSS=(256   128    256)

STRATS=("random" "evoaug_heavy" "genomic")
SIZES=(5000 100000)

HP_IDX=$((T / 6))
SIZE_IDX=$(( (T % 6) / 3 ))
STRAT_IDX=$((T % 3))

LR=${LRS[$HP_IDX]}
BS=${BSS[$HP_IDX]}
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

# BS cap
MAX_BS=$((SIZE / 32))
[ $BS -gt $MAX_BS ] && echo "SKIP: bs=$BS > n/32=$MAX_BS" && exit 0

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/pool_validation_test/k562/legnet_ag_s2"
RESULT="${OUT}/${STRAT}/n${SIZE}/hp${HP_IDX}_lr${LR}_bs${BS}/result.json"
[ -f "${RESULT}" ] && echo "SKIP" && exit 0

sleep $((T % 7))
echo "=== Pool val: ${STRAT} n=${SIZE} lr=${LR} bs=${BS} — $(date) ==="

uv run --no-sync python << PYEOF
import json, os, sys, numpy as np, torch
sys.path.insert(0, ".")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from pathlib import Path
from models.legnet_student import LegNetStudent, TrainConfig
from scripts.optuna_legnet_scaling import get_chr_val
from scipy.stats import pearsonr
import pandas as pd

REPO = Path(".")
strategy = "${STRAT}"
n_train = ${SIZE}

seed = hash((strategy, n_train, "pool_val", ${HP_IDX}, ${T})) % (2**31)
np.random.seed(seed)

# Load pool — sample MORE than n_train so we have 10% for validation
n_total = int(n_train / 0.9) + 1  # 10% held out → need ~11% more
for sk in ["5m", "2m", "618k"]:
    p = REPO / "outputs" / ("labeled_pools_%s" % sk if sk != "618k" else "labeled_pools") / "k562" / "ag_s2" / strategy / "pool.npz"
    if p.exists():
        data = np.load(p, allow_pickle=True)
        if len(data["sequences"]) >= n_total:
            all_seqs = data["sequences"]
            all_labels = data["labels"]
            break

perm = np.random.permutation(len(all_seqs))[:n_total]
total_seqs = [str(all_seqs[i]) for i in perm]
total_labels = all_labels[perm].astype(np.float32)

# Split: 90% train, 10% pool-based validation
n_val = max(50, int(0.1 * len(total_seqs)))
n_actual_train = len(total_seqs) - n_val
train_seqs = total_seqs[:n_actual_train]
train_labels = total_labels[:n_actual_train]
pool_val_seqs = total_seqs[n_actual_train:]
pool_val_labels = total_labels[n_actual_train:]

print(f"Pool validation: {len(train_seqs)} train + {len(pool_val_seqs)} val (from pool)")

config = TrainConfig(lr=${LR}, batch_size=${BS},
                    epochs=80, early_stopping_patience=10)

# Train with pool-based validation
model_pool = LegNetStudent(ensemble_size=1, train_config=config)
model_pool.fit(sequences=train_seqs, labels=train_labels,
              val_sequences=pool_val_seqs, val_labels=pool_val_labels)

# Also train with chr-split validation (same training data) for direct comparison
chr_val_seqs, chr_val_labels = get_chr_val()
np.random.seed(seed)
torch.manual_seed(seed)
model_chr = LegNetStudent(ensemble_size=1, train_config=config)
model_chr.fit(sequences=train_seqs, labels=train_labels,
             val_sequences=chr_val_seqs, val_labels=chr_val_labels)

# Evaluate both on all 3 test sets
test_dir = REPO / "data" / "k562" / "test_sets"

def evaluate_model(model):
    metrics = {}
    for tn, tf in [("in_dist", "test_chr7_13_all.tsv"), ("ood", "test_ood_designed_k562.tsv")]:
        f = test_dir / tf
        if f.exists():
            df = pd.read_csv(f, sep="\t")
            preds = model.predict(df["sequence"].str[:200].tolist())
            r, _ = pearsonr(df["K562_log2FC"].values, preds)
            metrics[tn] = {"pearson_r": float(r), "n": len(preds)}
    snv_f = test_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_f.exists():
        df = pd.read_csv(snv_f, sep="\t")
        rp = model.predict(df["sequence_ref"].str[:200].tolist())
        ap = model.predict(df["sequence_alt"].str[:200].tolist())
        r, _ = pearsonr(df["K562_log2FC_alt"].values - df["K562_log2FC_ref"].values, ap - rp)
        metrics["snv_delta"] = {"pearson_r": float(r), "n": len(rp)}
    return metrics

pool_metrics = evaluate_model(model_pool)
chr_metrics = evaluate_model(model_chr)

result = {"reservoir": strategy, "n_train": len(train_seqs), "n_val_pool": len(pool_val_seqs),
          "seed": seed,
          "pool_val_metrics": pool_metrics,
          "chr_val_metrics": chr_metrics,
          "hp_config": {"learning_rate": ${LR}, "batch_size": ${BS},
                       "weight_decay": 1e-5, "hp_idx": ${HP_IDX}}}
result_path = Path("${RESULT}")
result_path.parent.mkdir(parents=True, exist_ok=True)
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {result_path}")
print("Pool-val test metrics:")
for k, v in pool_metrics.items():
    print(f"  {k}: {v['pearson_r']:.4f}")
print("Chr-val test metrics:")
for k, v in chr_metrics.items():
    print(f"  {k}: {v['pearson_r']:.4f}")
PYEOF

echo "=== DONE — $(date) ==="
