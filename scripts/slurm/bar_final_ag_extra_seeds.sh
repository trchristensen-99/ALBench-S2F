#!/bin/bash
#SBATCH --job-name=bar_final_ag_seeds
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# Add seeds 1042 and 2042 for AG S1 and AG S2 across all 3 cell types
# AG S2 uses 20K training, not full dataset

for CELL in k562 hepg2 sknsh; do
    for SEED in 1042 2042; do
        echo "=== AG S1 | cell=$CELL | seed=$SEED ==="
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task "$CELL" --student alphagenome_k562_s1 --reservoir random \
            --chr-split --include-alt-alleles \
            --seed "$SEED"

        echo "=== AG S2 | cell=$CELL | seed=$SEED ==="
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task "$CELL" --student alphagenome_k562_s2 --reservoir random \
            --chr-split --include-alt-alleles \
            --seed "$SEED"
    done
done
