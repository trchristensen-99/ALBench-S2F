#!/bin/bash
# Train Malinois with exact paper settings (boda2, Gosai et al. 2024).
#
# Key differences from our default:
#   - 3-output multi-task (K562 + HepG2 + SknSh)
#   - L1KLmixed loss (alpha=1, beta=5) across cell-type dim
#   - 1 linear layer, 3 branched layers (140ch), dropout 0.576
#   - Adam(betas=(0.866, 0.879), amsgrad=True)
#   - RC interleave (2x dataset), dup cutoff 0.5
#   - Basset pretrained conv weights (transfer learning)
#   - patience=30
#
# Also downloads pretrained weights if missing and evaluates them directly.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/malinois_paper_mode.sh
#
#SBATCH --job-name=mal_paper
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Malinois paper mode ==="
echo "Node: ${SLURMD_NODENAME} — $(date)"

# Step 1: Download pretrained weights
echo ""
echo "--- Step 1: Download pretrained weights ---"
uv run --no-sync python scripts/download_malinois_weights.py

# Step 2: Evaluate pretrained Malinois (no training)
echo ""
echo "--- Step 2: Evaluate pretrained Malinois ---"
if [ ! -f "outputs/malinois_pretrained_eval/pretrained_eval.json" ]; then
    uv run --no-sync python scripts/eval_pretrained_malinois.py \
        --checkpoint data/pretrained/malinois_trained/torch_checkpoint.pt \
        --output-dir outputs/malinois_pretrained_eval \
        --chr-split --hashfrag
fi

# Step 3: Train paper-mode Malinois (chr-split, 3 seeds)
echo ""
echo "--- Step 3: Train paper-mode Malinois (chr-split) ---"
for SEED in 0 1 2; do
    OUT="outputs/bar_final/k562/malinois_paper/seed_${SEED}"
    if [ -f "${OUT}/seed_${SEED}/result.json" ]; then
        echo "  seed=${SEED}: already done"
        continue
    fi
    echo "  Training seed=${SEED}..."
    uv run --no-sync python experiments/train_malinois_k562.py \
        ++output_dir="outputs/bar_final/k562/malinois_paper" \
        ++seed=${SEED} \
        ++paper_mode=True \
        ++chr_split=True \
        ++pretrained_weights=data/pretrained/basset_pretrained.pkl
    echo "  seed=${SEED} done — $(date)"
done

# Step 4: Train paper-mode without pretrained weights (ablation)
echo ""
echo "--- Step 4: Paper-mode without pretrained weights (ablation) ---"
for SEED in 0 1 2; do
    OUT="outputs/bar_final/k562/malinois_paper_nopretrain/seed_${SEED}"
    if [ -f "${OUT}/seed_${SEED}/result.json" ]; then
        echo "  seed=${SEED}: already done"
        continue
    fi
    echo "  Training seed=${SEED} (no pretrain)..."
    uv run --no-sync python experiments/train_malinois_k562.py \
        ++output_dir="outputs/bar_final/k562/malinois_paper_nopretrain" \
        ++seed=${SEED} \
        ++paper_mode=True \
        ++chr_split=True
    echo "  seed=${SEED} done — $(date)"
done

echo ""
echo "=== All done — $(date) ==="
