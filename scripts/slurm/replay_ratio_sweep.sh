#!/bin/bash
# Replay ratio sweep: test different genomic:pool mixing ratios
# for pretrained fine-tuning with episodic replay.
#
# Ratios tested: 10%, 25%, 50% (current), 75%, 100% genomic
# (as fraction of pool size)
#
# 5 ratios × 3 strategies × 3 sizes × 1 rep = 45 jobs
#
#SBATCH --job-name=rp_rat
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

RATIOS=(0.10 0.25 0.50 0.75 1.00)
STRATS=("random" "evoaug_heavy" "genomic")
SIZES=(5000 50000 200000)

RATIO_IDX=$((T / 9))
SIZE_IDX=$(( (T % 9) / 3 ))
STRAT_IDX=$((T % 3))

RATIO=${RATIOS[$RATIO_IDX]}
STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

LR=0.003; BS=256

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/replay_ratio_sweep/k562/legnet_ag_s2"
RESULT="${OUT}/${STRAT}/n${SIZE}/ratio${RATIO}/result.json"
if [ -f "${RESULT}" ]; then echo "SKIP"; exit 0; fi

PRETRAINED="outputs/legnet_uncertainty_models/model_0/model.pt"
[ ! -f "${PRETRAINED}" ] && echo "ERROR: no pretrained model" && exit 1

sleep $((T % 7))
echo "=== Replay ratio=${RATIO}: ${STRAT} n=${SIZE} — $(date) ==="

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
ratio = ${RATIO}

seed = hash((strategy, n_train, ratio, ${T})) % (2**31)
np.random.seed(seed)

# Load pool subsample
for sk in ["5m", "2m", "618k"]:
    p = REPO / "outputs" / ("labeled_pools_%s" % sk if sk != "618k" else "labeled_pools") / "k562" / "ag_s2" / strategy / "pool.npz"
    if p.exists():
        data = np.load(p, allow_pickle=True)
        if len(data["sequences"]) >= n_train:
            all_seqs = data["sequences"]
            all_labels = data["labels"]
            break

perm = np.random.permutation(len(all_seqs))[:n_train]
pool_seqs = [str(all_seqs[i]) for i in perm]
pool_labels = all_labels[perm].astype(np.float32)

# Load genomic training data
from data.k562 import K562Dataset
ds = K562Dataset(data_path="data/k562", split="train")
mapping = {0: "A", 1: "C", 2: "G", 3: "T"}
genomic_seqs = []
genomic_labels = []
for i in range(len(ds)):
    t = ds[i][0]
    seq = ""
    for j in range(t.shape[1]):
        for k in range(4):
            if t[k, j] > 0.5:
                seq += mapping[k]
                break
        else:
            seq += "N"
    genomic_seqs.append(seq)
    genomic_labels.append(float(ds[i][1]))
genomic_labels = np.array(genomic_labels, dtype=np.float32)

# Replay with specified ratio
n_replay = min(int(n_train * ratio), len(genomic_seqs))
replay_idx = np.random.choice(len(genomic_seqs), n_replay, replace=False)
combined_seqs = [genomic_seqs[i] for i in replay_idx] + pool_seqs
combined_labels = np.concatenate([genomic_labels[replay_idx], pool_labels])

# Use default weight_decay (0.01) by NOT specifying it
config = TrainConfig(lr=${LR}, batch_size=${BS}, epochs=80, early_stopping_patience=10)
model = LegNetStudent(ensemble_size=1, train_config=config)
model.models[0].load_state_dict(torch.load("${PRETRAINED}", map_location="cpu"))
print(f"Replay ratio={ratio}: {n_replay} genomic ({ratio*100:.0f}% of pool) + {len(pool_seqs)} pool = {len(combined_seqs)}")

val_seqs, val_labels = get_chr_val()
model.fit(sequences=combined_seqs, labels=combined_labels, val_sequences=val_seqs, val_labels=val_labels)

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
          "replay_ratio": ratio, "n_replay": n_replay, "n_pool": len(pool_seqs),
          "mode": "replay", "test_metrics": test_metrics,
          "hp_config": {"learning_rate": ${LR}, "batch_size": ${BS}, "weight_decay": 0.01}}
result_path = Path("${RESULT}")
result_path.parent.mkdir(parents=True, exist_ok=True)
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {result_path}")
for k, v in test_metrics.items():
    print(f"  {k}: {v['pearson_r']:.4f}")
PYEOF

echo "=== DONE — $(date) ==="
