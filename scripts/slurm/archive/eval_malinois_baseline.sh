#!/bin/bash
#SBATCH --job-name=eval_malinois
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1


source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

MODEL_PATH="${HOME}/my-model.epoch_5-step_19885.pkl"
BODA_DIR="${HOME}/boda2-main"

export PYTHONPATH="$BODA_DIR:$PYTHONPATH"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found"
    exit 1
fi

if [ ! -d "$BODA_DIR" ]; then
    echo "Error: Boda directory $BODA_DIR not found"
    exit 1
fi

mkdir -p outputs/malinois_evaluation
uv pip install lightning hypertune dmslogo
uv run python scripts/analysis/eval_malinois_baseline.py \
    --boda_dir "$BODA_DIR" \
    --model_path "$MODEL_PATH" \
    --data_path data/k562 \
    --test_tsv_dir data/k562/test_sets \
    --output_file outputs/malinois_evaluation/result.json
