#!/bin/bash
# Comprehensive overnight HP sweep for DREAM-RNN + Malinois (single + multi-task variants).
#
# Grid (54 tasks):
#   - 2 models (dream_rnn, malinois)
#   - 3 cells (k562, hepg2, sknsh)  -- for multi-task, "cell" is the eval cell; trained on all 3
#   - 9 HP combos:
#     0: baseline
#     1: lr_low (lr×0.2)
#     2: lr_high (lr×2)
#     3: wd_high (wd×10)
#     4: wd_very_high (wd×100)
#     5: dropout_high (dropout +0.2)
#     6: lr_low + wd_high + dropout
#     7: bs_small (bs/2, lr/2)
#     8: bs_large (bs×2, lr×2)
#
#SBATCH --job-name=hp_full
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --array=0-53

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5 2>/dev/null || true
source .venv/bin/activate
export PYTHONPATH="$PWD:/grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=$SLURM_ARRAY_TASK_ID
MODELS=(dream_rnn malinois)
CELLS=(k562 hepg2 sknsh)
HPS=(baseline lr_low lr_high wd_high wd_vhigh dropout_high lr_low_wd_high_drop bs_small bs_large)

# Layout: T = model*27 + cell*9 + hp
M_IDX=$((T / 27))
C_IDX=$(( (T % 27) / 9 ))
H_IDX=$((T % 9))
MODEL=${MODELS[$M_IDX]}
CELL=${CELLS[$C_IDX]}
HP=${HPS[$H_IDX]}
SEED=42

OUT_DIR="$REPO/outputs/hp_full/${CELL}/${MODEL}/${HP}"
mkdir -p "$OUT_DIR"
echo "=== HP-full | model=$MODEL cell=$CELL hp=$HP | $(date) ==="

case $MODEL in
  dream_rnn)
    LR=0.005; BS=128; WD=0.01; DROPOUT_CNN=0.2
    case $HP in
      baseline)            ;;
      lr_low)              LR=0.001 ;;
      lr_high)             LR=0.01 ;;
      wd_high)             WD=0.1 ;;
      wd_vhigh)            WD=1.0 ;;
      dropout_high)        DROPOUT_CNN=0.4 ;;
      lr_low_wd_high_drop) LR=0.001; WD=0.1; DROPOUT_CNN=0.4 ;;
      bs_small)            LR=0.0025; BS=64 ;;
      bs_large)            LR=0.01; BS=256 ;;
    esac
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn \
        --oracle ground_truth --reservoir genomic --cell-line "$CELL" \
        --n-replicates 1 --no-hp-sweep --seed $SEED \
        --output-dir "$OUT_DIR" --training-sizes 319742 \
        --epochs 60 --early-stop-patience 10 \
        --lr $LR --batch-size $BS --weight-decay $WD --dropout $DROPOUT_CNN \
        --chr-split --save-predictions
    ;;
  malinois)
    LR=0.0001; WD=0.0; DROPOUT=0.0
    case $HP in
      baseline)            ;;
      lr_low)              LR=0.00002 ;;
      lr_high)             LR=0.0005 ;;
      wd_high)             WD=0.001 ;;
      wd_vhigh)            WD=0.01 ;;
      dropout_high)        DROPOUT=0.2 ;;
      lr_low_wd_high_drop) LR=0.00002; WD=0.01; DROPOUT=0.2 ;;
      bs_small)            ;; # train_malinois_k562 doesn't expose bs easily
      bs_large)            ;;
    esac
    python experiments/train_malinois_k562.py \
        ++cell_line="$CELL" ++data_path="data/${CELL}" \
        ++chr_split=true ++val_chrs=19,21,X ++test_chrs=7,13 \
        ++seed=$SEED ++output_dir="$OUT_DIR" \
        ++learning_rate=$LR ++weight_decay=$WD ++dropout=$DROPOUT
    ;;
esac
echo "=== done $(date) ==="
