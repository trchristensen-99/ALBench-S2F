#!/bin/bash
# Run Malinois evaluation using boda2 tutorial protocol (load_model + FlankBuilder).
# Requires: boda2 repo (BODA_DIR). Malinois artifact is downloaded via HTTP if missing (no VPN/gsutil).
#SBATCH --job-name=malinois_boda2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

set -e
source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

BODA_DIR="${BODA_DIR:-$HOME/boda2-main}"
# Malinois artifact: use existing path or download to repo data/ (persists, no home quota)
MALINOIS_TAR="malinois_artifacts__20211113_021200__287348.tar.gz"
MODEL_PATH="${MALINOIS_ARTIFACT:-}"
if [ -z "$MODEL_PATH" ]; then
  MODEL_PATH="$PWD/data/$MALINOIS_TAR"
fi

if [ ! -d "$BODA_DIR" ]; then
    echo "Error: BODA_DIR not found: $BODA_DIR"
    exit 1
fi

# Download artifact via public HTTP if not present (avoids gsutil/VPN)
if [ ! -f "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
    ARTIFACT_URL="https://storage.googleapis.com/tewhey-public-data/CODA_resources/$MALINOIS_TAR"
    echo "Malinois artifact not found at $MODEL_PATH. Downloading from $ARTIFACT_URL ..."
    mkdir -p "$(dirname "$MODEL_PATH")"
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$MODEL_PATH" "$ARTIFACT_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$MODEL_PATH" "$ARTIFACT_URL"
    else
        echo "Error: need curl or wget to download artifact."
        exit 1
    fi
    echo "Downloaded to $MODEL_PATH"
fi
if [ ! -f "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Malinois artifact missing after download: $MODEL_PATH"
    exit 1
fi

# boda/model/basset.py imports lightning; use a dedicated venv to avoid pydantic conflicts with main venv.
BODA_DIR="${BODA_DIR:-$HOME/boda2-main}"
EVAL_VENV="$PWD/data/boda2_eval_venv_v2"
if [ ! -f "$EVAL_VENV/bin/python" ]; then
  echo "Creating Malinois eval venv at $EVAL_VENV ..."
  uv venv "$EVAL_VENV"
  # lightning>=2.4 removes lightning.app (no lightning_cloud conflict); also has lightning.pytorch
  uv pip install --python "$EVAL_VENV/bin/python" torch numpy pandas scipy imageio "lightning>=2.4,<2.6"
  uv pip install --python "$EVAL_VENV/bin/python" -e . 2>/dev/null || true
fi
PYTHON_BODA="$EVAL_VENV/bin/python"

# Original K562 test set only (chromosome split chr 7, 13 from DATA-Table_S2). No HashFrag TSVs.
mkdir -p outputs/malinois_eval_boda2_tutorial
PYTHONPATH="$PWD:$BODA_DIR" "$PYTHON_BODA" scripts/analysis/eval_malinois_boda2_tutorial.py \
    --boda_dir "$BODA_DIR" \
    --model_path "$MODEL_PATH" \
    --data_path data/k562 \
    --output_file outputs/malinois_eval_boda2_tutorial/result.json
