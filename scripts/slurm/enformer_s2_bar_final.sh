#!/bin/bash
# Enformer S2 for bar_final (quality-filtered, ref+alt, chr_split).
# Waits for S1 heads to exist, then trains S2 for all 3 cells.
#
# Submit after Enformer S1 cache+heads job (enf_cache) completes:
#   sbatch --qos=default --time=12:00:00 scripts/slurm/enformer_s2_bar_final.sh
#
#SBATCH --job-name=enf_s2_fin
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Enformer S2 bar_final $(date) ==="

for CELL in k562 hepg2 sknsh; do
    CACHE="outputs/bar_final/${CELL}/enformer_cached/embedding_cache"
    S1_DIR="outputs/bar_final/${CELL}/enformer_s1/seed_42"

    # Wait for S1 to exist (in case submitted as dependency)
    if [ ! -f "${S1_DIR}/best_model.pt" ]; then
        echo "WARNING: S1 ${CELL} not found at ${S1_DIR}/best_model.pt, skipping"
        continue
    fi

    echo "=== Enformer S2 ${CELL} ==="
    for SEED in 42 123 456; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_stage2.py \
            ++model_name=enformer \
            ++stage1_result_dir="${S1_DIR}" \
            ++output_dir="outputs/bar_final/${CELL}/enformer_s2/seed_${SEED}" \
            ++data_path="data/${CELL}" \
            ++cell_line="${CELL}" \
            ++chr_split=True \
            ++include_alt_alleles=True \
            ++seed="${SEED}" \
            ++epochs=15 \
            ++batch_size=4 \
            ++grad_accum_steps=2 \
            ++head_lr=0.001 \
            ++encoder_lr=0.0001 \
            ++weight_decay=1e-6 \
            ++hidden_dim=512 \
            ++dropout=0.1 \
            ++early_stop_patience=5 \
            ++max_train_sequences=20000 \
            ++max_val_sequences=2000 \
            ++rc_aug=True \
            ++shift_aug=True \
            ++max_shift=15 \
            ++unfreeze_mode=all \
            ++grad_clip=1.0 \
            ++amp_mode=bfloat16
    done
done

echo "=== Done $(date) ==="
