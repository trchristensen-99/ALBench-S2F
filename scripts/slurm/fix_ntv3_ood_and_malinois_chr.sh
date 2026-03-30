#!/bin/bash
# Fix NTv3 K562 OOD cache misalignment + Malinois chr-split missing SNV/OOD.
#
# Array:
#   0: Rebuild NTv3 K562 OOD embedding cache + retrain 3 seeds
#   1: Malinois K562 chr-split (retrain with SNV+OOD eval)
#   2: Malinois HepG2 chr-split (retrain with SNV+OOD eval)
#   3: Malinois SKNSH chr-split (retrain with SNV+OOD eval)
#
#SBATCH --job-name=fix_misc
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=$SLURM_ARRAY_TASK_ID
echo "=== fix_misc task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

case ${T} in

0)
    echo "NTv3 K562: rebuild OOD cache + retrain 3 seeds"
    # Step 1: Rebuild OOD embeddings only
    uv run --no-sync python scripts/build_ntv3_embedding_cache.py \
        --data-dir data/k562 \
        --output-dir outputs/ntv3_post_k562_cached/embedding_cache \
        --model-variant post \
        --splits test_ood

    # Step 2: Retrain 3 seeds with corrected cache
    for SEED in 42 123 456; do
        echo "--- NTv3 S1 K562 seed ${SEED} ---"
        uv run --no-sync python experiments/train_foundation_cached.py \
            --model ntv3_post \
            --cell-line k562 \
            --output-dir "outputs/ntv3_post_k562_3seeds/seed_${SEED}" \
            --seed ${SEED} \
            --cache-dir outputs/ntv3_post_k562_cached/embedding_cache \
            --lr 0.0005 --weight-decay 1e-6 --dropout 0.1 \
            --epochs 50 --early-stop-patience 7
    done
    ;;

1)
    echo "Malinois K562 chr-split (with SNV+OOD eval)"
    for SEED in 0 1 2; do
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="outputs/chr_split/k562/malinois/seed_${SEED}" \
            ++seed=${SEED} \
            ++cell_line="k562" \
            ++chr_split=True
    done
    ;;

2)
    echo "Malinois HepG2 chr-split (with SNV+OOD eval)"
    for SEED in 0 1 2; do
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="outputs/chr_split/hepg2/malinois/seed_${SEED}" \
            ++seed=${SEED} \
            ++cell_line="hepg2" \
            ++chr_split=True
    done
    ;;

3)
    echo "Malinois SKNSH chr-split (with SNV+OOD eval)"
    for SEED in 0 1 2; do
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="outputs/chr_split/sknsh/malinois/seed_${SEED}" \
            ++seed=${SEED} \
            ++cell_line="sknsh" \
            ++chr_split=True
    done
    ;;

*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
