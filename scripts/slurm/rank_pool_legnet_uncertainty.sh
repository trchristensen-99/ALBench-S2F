#!/bin/bash
# Rank pool sequences by LegNet student uncertainty.
#
# Uses 3 pretrained LegNet models (trained on genomic data with real labels).
# For each pool sequence, predict with all 3 models, take variance = uncertainty.
# High variance = the student disagrees with itself = high learning value.
#
# Array: 0=evoaug_heavy, 1=motif_grammar, 2=prm_1pct
#
#SBATCH --job-name=ln_rnk
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

T=$SLURM_ARRAY_TASK_ID

STRATS=("evoaug_heavy" "motif_grammar" "prm_1pct")
POOL_SIZES=("5m" "5m" "618k")

STRAT=${STRATS[$T]}
POOL_SIZE=${POOL_SIZES[$T]}

OUT="outputs/uncertainty_ranking_legnet/${STRAT}"
[ -f "${OUT}/uncertainty_ranking.npz" ] && echo "SKIP" && exit 0

echo "=== LegNet Uncertainty Ranking: ${STRAT} — $(date) ==="

uv run --no-sync python << PYEOF
import numpy as np, torch, sys, os, json, time
sys.path.insert(0, ".")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from models.legnet_student import LegNetStudent, TrainConfig
from pathlib import Path

REPO = Path(".")
out_dir = Path("${OUT}")
out_dir.mkdir(parents=True, exist_ok=True)

# Load pool
pool_dirs = {
    "5m": REPO / "outputs" / "labeled_pools_5m" / "k562" / "ag_s2" / "${STRAT}",
    "2m": REPO / "outputs" / "labeled_pools_2m" / "k562" / "ag_s2" / "${STRAT}",
    "618k": REPO / "outputs" / "labeled_pools" / "k562" / "ag_s2" / "${STRAT}",
}
for size_key in ["${POOL_SIZE}", "5m", "2m", "618k"]:
    p = pool_dirs.get(size_key, pool_dirs["618k"]) / "pool.npz"
    if p.exists():
        data = np.load(p, allow_pickle=True)
        sequences = [str(s)[:200] for s in data["sequences"]]
        labels = data["labels"]
        print(f"Loaded pool: {len(sequences)} sequences from {p}")
        break

# Load 3 pretrained LegNet models and predict
t0 = time.time()
all_preds = []
for model_idx in range(3):
    model_path = REPO / "outputs" / "legnet_uncertainty_models" / f"model_{model_idx}" / "model.pt"
    if not model_path.exists():
        print(f"Model {model_idx} not found at {model_path}")
        continue

    config = TrainConfig(lr=0.003, batch_size=256)
    model = LegNetStudent(ensemble_size=1, train_config=config)
    model.models[0].load_state_dict(torch.load(model_path, map_location="cpu"))
    model.models[0].eval()

    print(f"Predicting with model {model_idx}...")
    preds = model.predict(sequences)
    all_preds.append(preds)
    print(f"  Done: mean={preds.mean():.3f} std={preds.std():.3f}")

elapsed = time.time() - t0
print(f"All models done in {elapsed:.0f}s")

if len(all_preds) < 2:
    print("ERROR: need at least 2 models")
    sys.exit(1)

# Compute variance across models
pred_matrix = np.stack(all_preds, axis=0)
uncertainty = np.var(pred_matrix, axis=0)
ranking = np.argsort(-uncertainty)  # most uncertain first

np.savez_compressed(
    out_dir / "uncertainty_ranking.npz",
    ranking=ranking,
    uncertainty=uncertainty,
    ensemble_mean=np.mean(pred_matrix, axis=0),
    pred_matrix=pred_matrix,
)

summary = {
    "strategy": "${STRAT}",
    "pool_size": len(sequences),
    "n_models": len(all_preds),
    "uncertainty_mean": float(np.mean(uncertainty)),
    "uncertainty_std": float(np.std(uncertainty)),
    "uncertainty_max": float(np.max(uncertainty)),
    "time_sec": elapsed,
}
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved ranking: mean_unc={np.mean(uncertainty):.4f} max={np.max(uncertainty):.4f}")
PYEOF

echo "=== DONE — $(date) ==="
