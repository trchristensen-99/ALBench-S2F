#!/bin/bash
#SBATCH --job-name=compare_malinois_ag
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

export PYTHONPATH="$BODA_DIR:$PWD:$PYTHONPATH"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Malinois model not found at $MODEL_PATH"
    exit 1
fi
if [ ! -d "$BODA_DIR" ]; then
    echo "Error: boda2 directory not found at $BODA_DIR"
    exit 1
fi

mkdir -p results logs
uv pip install lightning hypertune dmslogo

uv run python scripts/analysis/compare_malinois_ag.py \
    --boda_dir "$BODA_DIR" \
    --model_path "$MODEL_PATH" \
    --test_tsv_dir data/k562/test_sets \
    --ag_outputs_dir outputs \
    --output_file results/k562_comparison.tsv
