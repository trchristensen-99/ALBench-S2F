#!/bin/bash
# Overnight batch: MPRA-LegNet architecture, LegNet speedups, continual learning
#
# Array tasks:
#   0-6:   MPRA-LegNet architecture scaling (7 training sizes)
#   7-9:   LegNet speedup tests at n=1M (3 BS/LR combos)
#   10-12: Continual learning for neg-aug (3 approaches)
#   13-15: LegNet speedup tests at n=296K for validation (3 combos)
#
#SBATCH --job-name=batch_exp
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

if [ "$T" -le 2 ]; then
    # ══════════════════════════════════════════
    # LegNet speedup tests at n=1M
    # Test larger batch sizes with scaled LR
    # ══════════════════════════════════════════
    IDX=$T
    BS_VALS=(1024 2048 4096)
    LR_VALS=(0.002 0.004 0.008)
    BS=${BS_VALS[$IDX]}
    LR=${LR_VALS[$IDX]}

    OUT="outputs/legnet_speedup/n1000000/bs${BS}_lr${LR}/seed42"
    [ -f "${OUT}/result.json" ] && echo "SKIP" && exit 0

    echo "=== LegNet speedup: bs=${BS} lr=${LR} n=1M — $(date) ==="
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ag_s2 \
        --reservoir random \
        --pool-base-dir outputs/labeled_pools_2m/k562/ag_s2 \
        --n-replicates 1 --seed 42 \
        --output-dir "outputs/legnet_speedup" \
        --training-sizes 1000000 \
        --chr-split --lr "${LR}" --batch-size "${BS}" \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10

elif [ "$T" -le 5 ]; then
    # ══════════════════════════════════════════
    # Continual learning approaches for neg-aug
    # ══════════════════════════════════════════
    IDX=$((T - 3))
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
    export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

    S1_DIR="outputs/oracle_full_856k/s1/oracle_0"

    case $IDX in
        0)  # EWC-like: train with negatives BUT also include OOD positive examples
            LABEL="neg_plus_ood"
            OUT="outputs/oracle_neg_sweep/neg_plus_ood/fold_0"
            # Use the OOD test sequences as positive training examples
            # alongside dinuc-shuffled negatives
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== Neg + OOD positives — $(date) ==="
            # Generate OOD positive training file if not exists
            uv run --no-sync python -c "
import csv, numpy as np
from pathlib import Path
ood_path = Path('data/k562/test_sets/test_ood_designed_k562.tsv')
if not ood_path.exists():
    print('No OOD file'); exit()
out_path = Path('data/synthetic_negatives/ood_positives.tsv')
if out_path.exists():
    print('Already exists'); exit()
import pandas as pd
df = pd.read_csv(ood_path, sep='\t')
# Use a subset of OOD with high activity as positive anchors
high_act = df.nlargest(5000, 'K562_log2FC')
with open(out_path, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['sequence', 'K562_log2FC', 'category'])
    for _, row in high_act.iterrows():
        w.writerow([row['sequence'], row['K562_log2FC'], 'ood_positive'])
print('Saved %d OOD positives' % len(high_act))
" || true
            # Train S2 with both negatives AND OOD positives
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=0 ++n_folds=10 ++stage1_dir="${S1_DIR}" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/synthetic_negatives/dinuc_shuffled_negatives.tsv" \
                ++neg_fraction=0.05 ++wandb_mode=offline
            ;;
        1)  # Lower neg fraction (2%) with lower encoder LR — best Pareto from sweep
            LABEL="frac02_elr1_full"
            OUT="outputs/oracle_neg_sweep/frac02_elr1_fold3/fold_3"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== frac=2% elr=1e-4 fold3 — $(date) ==="
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=3 ++n_folds=10 \
                ++stage1_dir="outputs/oracle_full_856k/s1/oracle_3" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/synthetic_negatives/dinuc_shuffled_negatives.tsv" \
                ++neg_fraction=0.02 ++wandb_mode=offline
            ;;
        2)  # frac=2% on fold 8 for stability check
            LABEL="frac02_elr1_fold8"
            OUT="outputs/oracle_neg_sweep/frac02_elr1_fold8/fold_8"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== frac=2% elr=1e-4 fold8 — $(date) ==="
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=8 ++n_folds=10 \
                ++stage1_dir="outputs/oracle_full_856k/s1/oracle_8" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/synthetic_negatives/dinuc_shuffled_negatives.tsv" \
                ++neg_fraction=0.02 ++wandb_mode=offline
            ;;
    esac

else
    # ══════════════════════════════════════════
    # LegNet speedup validation at n=296K
    # ══════════════════════════════════════════
    IDX=$((T - 6))
    BS_VALS=(1024 2048 4096)
    LR_VALS=(0.002 0.004 0.008)
    BS=${BS_VALS[$IDX]}
    LR=${LR_VALS[$IDX]}

    OUT="outputs/legnet_speedup/n296382/bs${BS}_lr${LR}/seed42"
    [ -f "${OUT}/result.json" ] && echo "SKIP" && exit 0

    echo "=== LegNet speedup validation: bs=${BS} lr=${LR} n=296K — $(date) ==="
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ag_s2 \
        --reservoir genomic \
        --pool-base-dir outputs/labeled_pools/k562/ag_s2 \
        --n-replicates 1 --seed 42 \
        --output-dir "outputs/legnet_speedup" \
        --training-sizes 296382 \
        --chr-split --lr "${LR}" --batch-size "${BS}" \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10
fi

echo "=== Done task ${T} — $(date) ==="
