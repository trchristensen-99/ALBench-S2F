#!/bin/bash
#SBATCH --job-name=scl_pt
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
SIZES=(1000 2000 5000 10000 20000 50000 100000 200000 300000 500000 1000000 2000000 5000000)

STRAT_IDX=$((T / 39))
SIZE_IDX=$(( (T % 39) / 3 ))
REP_IDX=$((T % 3))

STRAT=${STRATS[$STRAT_IDX]}
SIZE=${SIZES[$SIZE_IDX]}

if [ $SIZE -le 2000 ]; then LR=0.0005; BS=128; else LR=0.003; BS=256; fi

POOL_DIR="outputs/labeled_pools_5m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools_2m/k562/ag_s2"
[ ! -f "${POOL_DIR}/${STRAT}/pool.npz" ] && POOL_DIR="outputs/labeled_pools/k562/ag_s2"

OUT="outputs/exp1_1_pretrained/k562/legnet_ag_s2"
RESULT="${OUT}/${STRAT}/n${SIZE}/rep${REP_IDX}/result.json"
[ -f "${RESULT}" ] && echo "SKIP" && exit 0

PRETRAINED="outputs/legnet_uncertainty_models/model_${REP_IDX}/model.pt"
[ ! -f "${PRETRAINED}" ] && echo "ERROR: no pretrained model" && exit 1

sleep $((T % 7))
echo "=== Pretrained: ${STRAT} n=${SIZE} rep${REP_IDX} — $(date) ==="

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

seed = hash((strategy, n_train, rep, ${T})) % (2**31)
np.random.seed(seed)

pool_dirs = {
    "5m": REPO / "outputs" / "labeled_pools_5m" / "k562" / "ag_s2" / strategy,
    "2m": REPO / "outputs" / "labeled_pools_2m" / "k562" / "ag_s2" / strategy,
    "618k": REPO / "outputs" / "labeled_pools" / "k562" / "ag_s2" / strategy,
}
for sk in ["5m", "2m", "618k"]:
    p = pool_dirs[sk] / "pool.npz"
    if p.exists():
        data = np.load(p, allow_pickle=True)
        all_seqs = data["sequences"]
        all_labels = data["labels"]
        if len(all_seqs) >= n_train:
            break

if len(all_seqs) < n_train:
    print(f"ERROR: pool {len(all_seqs)} < {n_train}")
    sys.exit(1)

perm = np.random.permutation(len(all_seqs))[:n_train]
seqs = [str(all_seqs[i]) for i in perm]
labels = all_labels[perm].astype(np.float32)

val_seqs, val_labels = get_chr_val()

config = TrainConfig(lr=${LR}, batch_size=${BS}, weight_decay=1e-5,
                    epochs=80, early_stopping_patience=10)
model = LegNetStudent(ensemble_size=1, train_config=config)
model.models[0].load_state_dict(torch.load("${PRETRAINED}", map_location="cpu"))
print(f"Loaded pretrained from ${PRETRAINED}")

model.fit(sequences=seqs, labels=labels,
         val_sequences=val_seqs, val_labels=val_labels)

test_metrics = {}
test_dir = REPO / "data" / "k562" / "test_sets"
for test_name, test_file in [("in_dist", "test_chr7_13_all.tsv"),
                              ("ood", "test_ood_designed_k562.tsv")]:
    f = test_dir / test_file
    if f.exists():
        df = pd.read_csv(f, sep="\t")
        preds = model.predict(df["sequence"].str[:200].tolist())
        r, _ = pearsonr(df["K562_log2FC"].values, preds)
        test_metrics[test_name] = {"pearson_r": float(r), "n": len(preds)}

snv_f = test_dir / "test_snv_pairs_hashfrag.tsv"
if snv_f.exists():
    df = pd.read_csv(snv_f, sep="\t")
    ref_p = model.predict(df["sequence_ref"].str[:200].tolist())
    alt_p = model.predict(df["sequence_alt"].str[:200].tolist())
    delta_p = alt_p - ref_p
    delta_r = df["K562_log2FC_alt"].values - df["K562_log2FC_ref"].values
    r, _ = pearsonr(delta_r, delta_p)
    test_metrics["snv_delta"] = {"pearson_r": float(r), "n": len(delta_p)}

result = {"reservoir": strategy, "n_train": n_train, "seed": seed, "rep": rep,
          "acquisition": "pretrained", "test_metrics": test_metrics,
          "hp_config": {"learning_rate": ${LR}, "batch_size": ${BS}}}
result_path = Path("${RESULT}")
result_path.parent.mkdir(parents=True, exist_ok=True)
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {result_path}")
for k, v in test_metrics.items():
    print(f"  {k}: {v['pearson_r']:.4f}")
PYEOF

echo "=== DONE — $(date) ==="
