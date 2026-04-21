#!/bin/bash
# Train 3 LegNet models on full genomic train split (real labels)
# for use as uncertainty estimators in acquisition experiments.
#
# Each model is trained with a different seed, producing different
# predictions. The variance across the 3 models = student uncertainty.
#
# Array: 0-2 (one per seed)
#
#SBATCH --job-name=ln_unc
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=02:00:00
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

SEED_IDX=$SLURM_ARRAY_TASK_ID
SEEDS=(42 1042 2042)
SEED=${SEEDS[$SEED_IDX]}

OUT="outputs/legnet_uncertainty_models/model_${SEED_IDX}"

[ -f "${OUT}/model.pt" ] && echo "SKIP" && exit 0

echo "=== Training LegNet model ${SEED_IDX} (seed=${SEED}) on genomic data — $(date) ==="

uv run --no-sync python << PYEOF
import numpy as np, torch, sys, os
sys.path.insert(0, ".")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from models.legnet_student import LegNetStudent, TrainConfig
from data.k562 import K562Dataset
from pathlib import Path

REPO = Path(".")
out_dir = Path("${OUT}")
out_dir.mkdir(parents=True, exist_ok=True)

# Load full genomic training data with real labels
ds = K562Dataset(data_path="data/k562", split="train")
train_seqs = [str(ds[i][0]) for i in range(len(ds))]

# Convert tensor sequences back to strings
mapping = {0: "A", 1: "C", 2: "G", 3: "T"}
seqs = []
for i in range(len(ds)):
    t = ds[i][0]  # (5, 200) tensor
    seq = ""
    for j in range(t.shape[1]):
        for k in range(4):
            if t[k, j] > 0.5:
                seq += mapping[k]
                break
        else:
            seq += "N"
    seqs.append(seq)

labels = np.array([float(ds[i][1]) for i in range(len(ds))])
print(f"Training data: {len(seqs)} sequences, label mean={labels.mean():.3f}")

# Train with fixed good HP
config = TrainConfig(lr=0.003, batch_size=256, weight_decay=1e-5,
                    epochs=80, early_stopping_patience=10)
model = LegNetStudent(ensemble_size=1, train_config=config)

# Use chr19/21/X as val
val_ds = K562Dataset(data_path="data/k562", split="val")
val_seqs = []
for i in range(len(val_ds)):
    t = val_ds[i][0]
    seq = ""
    for j in range(t.shape[1]):
        for k in range(4):
            if t[k, j] > 0.5:
                seq += mapping[k]
                break
        else:
            seq += "N"
    val_seqs.append(seq)
val_labels = np.array([float(val_ds[i][1]) for i in range(len(val_ds))])

np.random.seed(${SEED})
torch.manual_seed(${SEED})

model.fit(sequences=seqs, labels=labels,
         val_sequences=val_seqs, val_labels=val_labels)

# Save model
torch.save(model.models[0].state_dict(), out_dir / "model.pt")
print(f"Saved model to {out_dir / 'model.pt'}")

# Quick test
preds = model.predict(val_seqs[:100])
from scipy.stats import pearsonr
r, _ = pearsonr(val_labels[:100], preds)
print(f"Val Pearson R (first 100): {r:.4f}")
PYEOF

echo "=== DONE — $(date) ==="
