#!/bin/bash
# Fill ALL remaining non-critical gaps.
#
# Array:
#   0-5: Foundation S1 HepG2/SKNSH predictions (6 model×cell combos)
#   6: DREAM-CNN K562 retrain with prediction saving (3 seeds)
#   7: DREAM-RNN K562 retrain with prediction saving (3 seeds, ens=3)
#   8: Malinois K562 retrain with prediction saving (3 seeds)
#   9-11: Yeast AG S2 old-LR largest 3 fractions
#
#SBATCH --job-name=final_gap
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
echo "=== final_gaps task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

case ${T} in

# Foundation S1 HepG2/SKNSH predictions (load head + cached embeddings)
0) echo "Enformer S1 HepG2 predictions"
   uv run --no-sync python scripts/save_foundation_predictions.py --model enformer --cell hepg2 ;;
1) echo "Borzoi S1 HepG2 predictions"
   uv run --no-sync python scripts/save_foundation_predictions.py --model borzoi --cell hepg2 ;;
2) echo "NTv3 S1 HepG2 predictions"
   uv run --no-sync python scripts/save_foundation_predictions.py --model ntv3_post --cell hepg2 ;;
3) echo "Enformer S1 SKNSH predictions"
   uv run --no-sync python scripts/save_foundation_predictions.py --model enformer --cell sknsh ;;
4) echo "Borzoi S1 SKNSH predictions"
   uv run --no-sync python scripts/save_foundation_predictions.py --model borzoi --cell sknsh ;;
5) echo "NTv3 S1 SKNSH predictions"
   uv run --no-sync python scripts/save_foundation_predictions.py --model ntv3_post --cell sknsh ;;

# From-scratch model retraining with prediction saving
6)
    echo "DREAM-CNN K562 retrain with predictions"
    for SEED in 0 1 2; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student dream_cnn \
            --oracle ground_truth --reservoir genomic \
            --n-replicates 1 --seed ${SEED} \
            --output-dir "outputs/dream_cnn_k562_with_preds/seed_${SEED}" \
            --training-sizes 319742 --epochs 80 \
            --ensemble-size 1 --early-stop-patience 10 \
            --save-predictions
    done
    ;;

7)
    echo "DREAM-RNN K562 retrain with predictions (ens=3)"
    for SEED in 42 123 456; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/exp1_1_scaling.py \
            --task k562 --student dream_rnn \
            --oracle ground_truth --reservoir genomic \
            --n-replicates 1 --seed ${SEED} \
            --output-dir "outputs/dream_rnn_k562_with_preds_v2/seed_${SEED}" \
            --training-sizes 319742 --epochs 80 \
            --ensemble-size 3 --early-stop-patience 10 \
            --save-predictions
    done
    ;;

8)
    echo "Malinois K562 retrain with predictions"
    for SEED in 0 1 2; do
        echo "--- seed ${SEED} ---"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++data_path="data/k562" \
            ++output_dir="outputs/malinois_k562_with_preds_v2/seed_${SEED}" \
            ++seed=${SEED} \
            ++cell_line="k562" \
            ++save_predictions=True
    done
    ;;

# Yeast AG S2 old-LR largest 3 fractions
9)
    echo "Yeast AG S2 old-LR n=1213065"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student alphagenome_yeast_s2 \
        --oracle default --reservoir random \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/alphagenome_yeast_s2" \
        --training-sizes 1213065 --epochs 50 \
        --ensemble-size 3 --early-stop-patience 10
    ;;

10)
    echo "Yeast AG S2 old-LR n=3032662"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student alphagenome_yeast_s2 \
        --oracle default --reservoir random \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/alphagenome_yeast_s2" \
        --training-sizes 3032662 --epochs 50 \
        --ensemble-size 3 --early-stop-patience 10
    ;;

11)
    echo "Yeast AG S2 old-LR n=6065324"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student alphagenome_yeast_s2 \
        --oracle default --reservoir random \
        --n-replicates 3 --seed 42 \
        --output-dir "outputs/exp0_oracle_scaling_v4/yeast/alphagenome_yeast_s2" \
        --training-sizes 6065324 --epochs 50 \
        --ensemble-size 3 --early-stop-patience 10
    ;;

*)
    echo "ERROR: unknown task ${T}"
    exit 1
    ;;
esac

echo "=== Done: $(date) ==="
