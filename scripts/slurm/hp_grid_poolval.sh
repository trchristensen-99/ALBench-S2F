#!/bin/bash
# HP grid sweep with POOL-BASED validation (10% holdout from pool).
# This matches the definitive script's validation strategy.
# The earlier hp_grid_sweep.sh used chr-split validation which was 0.10 Pearson worse.
#
# 12 HP configs × 2 sizes × 3 strategies = 72 jobs
# Sizes: 5K, 100K. Strategies: random, evoaug_heavy, genomic
#
# Array: 0-71
#
#SBATCH --job-name=hp_pv
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

# 12 HP configs: same as hp_grid_sweep.sh
LRS=(0.0005 0.001  0.001 0.002  0.001 0.003  0.002 0.003 0.005  0.003 0.005 0.008)
BSS=(32     32     64    64     128   128    256   256   256    512   512   512)

STRATS=("random" "evoaug_heavy" "genomic")
SIZES=(5000 100000)

HP_IDX=$((T / 6))
SIZE_IDX=$(( (T % 6) / 3 ))
STRAT_IDX=$((T % 3))

LR=${LRS[$HP_IDX]}
BS=${BSS[$HP_IDX]}
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

# Skip if BS would give fewer than 32 gradient steps per epoch (on 90% train)
TRAIN90=$((SIZE * 9 / 10))
MAX_BS=$((TRAIN90 / 32))
[ $BS -gt $MAX_BS ] && echo "SKIP: bs=$BS > train90/32=$MAX_BS for n=$SIZE" && exit 0

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/hp_grid_poolval/k562/legnet_ag_s2"
RESULT="${OUT}/${STRAT}/n${SIZE}/hp${HP_IDX}_lr${LR}_bs${BS}/result.json"
[ -f "${RESULT}" ] && echo "SKIP" && exit 0

sleep $((T % 7))
echo "=== HP poolval: ${STRAT} n=${SIZE} lr=${LR} bs=${BS} — $(date) ==="

uv run --no-sync python << PYEOF
import json, os, sys, numpy as np, torch
sys.path.insert(0, ".")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from pathlib import Path
from models.legnet_student import LegNetStudent, TrainConfig
from scipy.stats import pearsonr
import pandas as pd

REPO = Path(".")
strategy = "${STRAT}"
n_train = ${SIZE}

seed = hash((strategy, n_train, "hp_poolval", ${HP_IDX}, ${T})) % (2**31)
np.random.seed(seed)

# Load pool
for sk in ["5m", "2m", "618k"]:
    p = REPO / "outputs" / ("labeled_pools_%s" % sk if sk != "618k" else "labeled_pools") / "k562" / "ag_s2" / strategy / "pool.npz"
    if p.exists():
        data = np.load(p, allow_pickle=True)
        if len(data["sequences"]) >= n_train:
            all_seqs = data["sequences"]
            all_labels = data["labels"]
            break

perm = np.random.permutation(len(all_seqs))[:n_train]
all_sampled_seqs = [str(all_seqs[i]) for i in perm]
all_sampled_labels = all_labels[perm].astype(np.float32)

# 10% pool holdout for validation (matching definitive/exp1_1_scaling.py behavior)
n_val = max(100, int(0.1 * n_train))
val_seqs = all_sampled_seqs[:n_val]
val_labels = all_sampled_labels[:n_val]
train_seqs = all_sampled_seqs[n_val:]
train_labels = all_sampled_labels[n_val:]
print(f"Pool validation: {len(train_seqs)} train + {len(val_seqs)} val")

config = TrainConfig(lr=${LR}, batch_size=${BS},
                    epochs=80, early_stopping_patience=10)
model = LegNetStudent(ensemble_size=1, train_config=config)
model.fit(sequences=train_seqs, labels=train_labels,
         val_sequences=val_seqs, val_labels=val_labels)

# Evaluate all 3 test sets
test_metrics = {}
test_dir = REPO / "data" / "k562" / "test_sets"
for tn, tf in [("in_dist", "test_chr7_13_all.tsv"), ("ood", "test_ood_designed_k562.tsv")]:
    f = test_dir / tf
    if f.exists():
        df = pd.read_csv(f, sep="\t")
        preds = model.predict(df["sequence"].str[:200].tolist())
        r, _ = pearsonr(df["K562_log2FC"].values, preds)
        test_metrics[tn] = {"pearson_r": float(r), "n": len(preds)}

snv_f = test_dir / "test_snv_pairs_hashfrag.tsv"
if snv_f.exists():
    df = pd.read_csv(snv_f, sep="\t")
    rp = model.predict(df["sequence_ref"].str[:200].tolist())
    ap = model.predict(df["sequence_alt"].str[:200].tolist())
    r, _ = pearsonr(df["K562_log2FC_alt"].values - df["K562_log2FC_ref"].values, ap - rp)
    test_metrics["snv_delta"] = {"pearson_r": float(r), "n": len(rp)}

result = {"reservoir": strategy, "n_train": n_train, "seed": seed,
          "validation": "pool_10pct", "test_metrics": test_metrics,
          "hp_config": {"learning_rate": ${LR}, "batch_size": ${BS},
                       "weight_decay": 0.01, "hp_idx": ${HP_IDX}}}
result_path = Path("${RESULT}")
result_path.parent.mkdir(parents=True, exist_ok=True)
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {result_path}")
for k, v in test_metrics.items():
    print(f"  {k}: {v['pearson_r']:.4f}")
PYEOF

echo "=== DONE — $(date) ==="
