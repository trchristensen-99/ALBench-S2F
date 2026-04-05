#!/bin/bash
#SBATCH --job-name=mal_p_hep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

uv run --no-sync python scripts/download_malinois_weights.py

for SEED in 0 1 2; do
    for CELL in hepg2 sknsh; do
        OUT="outputs/bar_final/${CELL}/malinois_paper/seed_${SEED}"
        [ -f "${OUT}/result.json" ] && echo "${CELL}/s${SEED}: done" && continue
        echo "Training ${CELL} seed=${SEED} — $(date)"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++output_dir="outputs/bar_final/${CELL}/malinois_paper" \
            ++seed=${SEED} ++paper_mode=True ++chr_split=True \
            ++pretrained_weights=data/pretrained/basset_pretrained.pkl \
            ++cell_line="${CELL}" || true
    done
done
echo "=== Done — $(date) ==="
