#!/bin/bash
# Genomic+pool combined training v2: FIXED weight_decay (uses default 0.01).
# Calls exp1_1_scaling.py directly for pipeline consistency.
#
# Two modes:
#   retrain: train from scratch on genomic + pool combined
#   replay:  fine-tune pretrained on pool + genomic replay
#
# For retrain, we use a custom script since exp1_1_scaling.py doesn't
# natively support combined genomic+pool training.
#
# Array: MODE * 75 + STRAT * 15 + SIZE_IDX * 3 + REP
# 2 modes × 5 strats × 5 sizes × 3 reps = 150 jobs
#
#SBATCH --job-name=gp_v2
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

STRATS=("random" "prm_1pct" "prm_20pct" "motif_grammar" "evoaug_heavy")
SIZES=(1000 5000 20000 100000 500000)

MODE=$((T / 75))
WITHIN=$((T % 75))
STRAT_IDX=$((WITHIN / 15))
SIZE_IDX=$(( (WITHIN % 15) / 3 ))
REP_IDX=$((WITHIN % 3))

STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
[ $MODE -eq 0 ] && MODE_NAME="retrain" || MODE_NAME="replay"

LR=0.003; BS=256

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/exp1_1_genomic_plus_pool_v2/k562/legnet_ag_s2"
RESULT="${OUT}/${MODE_NAME}/${STRAT}/n${SIZE}/rep${REP_IDX}/result.json"
if [ -f "${RESULT}" ]; then echo "SKIP"; exit 0; fi

PRETRAINED="outputs/legnet_uncertainty_models/model_${REP_IDX}/model.pt"

sleep $((T % 7))
echo "=== ${MODE_NAME}: ${STRAT} n=${SIZE} rep${REP_IDX} — $(date) ==="

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
rep = ${REP_IDX}
mode = "${MODE_NAME}"

seed = hash((strategy, n_train, rep, mode, ${T})) % (2**31)
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

if mode == "retrain":
    combined_seqs = genomic_seqs + pool_seqs
    combined_labels = np.concatenate([genomic_labels, pool_labels])
    # Use default weight_decay (0.01) by NOT specifying it
    config = TrainConfig(lr=${LR}, batch_size=${BS}, epochs=80, early_stopping_patience=10)
    model = LegNetStudent(ensemble_size=1, train_config=config)
    print(f"Retrain from scratch: {len(genomic_seqs)} genomic + {len(pool_seqs)} pool = {len(combined_seqs)}")
else:
    n_replay = min(len(genomic_seqs), n_train)
    replay_idx = np.random.choice(len(genomic_seqs), n_replay, replace=False)
    combined_seqs = [genomic_seqs[i] for i in replay_idx] + pool_seqs
    combined_labels = np.concatenate([genomic_labels[replay_idx], pool_labels])
    config = TrainConfig(lr=${LR}, batch_size=${BS}, epochs=80, early_stopping_patience=10)
    model = LegNetStudent(ensemble_size=1, train_config=config)
    if os.path.exists("${PRETRAINED}"):
        model.models[0].load_state_dict(torch.load("${PRETRAINED}", map_location="cpu"))
        print(f"Replay: pretrained + {n_replay} genomic + {len(pool_seqs)} pool")

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

result = {"reservoir": strategy, "n_train": n_train, "seed": seed, "rep": rep,
          "mode": mode, "test_metrics": test_metrics,
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
