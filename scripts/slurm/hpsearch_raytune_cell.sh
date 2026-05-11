#!/bin/bash
# Launch one Ray Tune search (strategy × arch × D). Submitted by
# hpsearch_launch_all.sh which iterates over the cells × strategies.
#
# Required env vars (set via sbatch --export or wrapper):
#   STRATEGY  — random | optuna | hyperopt | bohb | pbt
#   ARCH      — legnet | dream_rnn | dream_attn
#   D_TRAIN   — int (5000 or 100000)
#   N_TRIALS  — int (default 50)
#   MAX_EPOCHS — int (default 60)
#   OUT_DIR   — output directory (will be created)
#   GPUS      — number of GPUs in this allocation (default 1)
#   TRIALS_PER_GPU — fractional GPU per trial (default 4)

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
# Activate venv directly so Ray's spawned worker processes can find 'ray'.
source .venv/bin/activate
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

: "${STRATEGY:?STRATEGY required}"
: "${ARCH:?ARCH required}"
: "${D_TRAIN:?D_TRAIN required}"
: "${OUT_DIR:?OUT_DIR required}"
N_TRIALS=${N_TRIALS:-50}
MAX_EPOCHS=${MAX_EPOCHS:-60}
PATIENCE=${PATIENCE:-15}
GPUS=${GPUS:-1}
TRIALS_PER_GPU=${TRIALS_PER_GPU:-4}

mkdir -p "$OUT_DIR"

python -m scripts.preflight.hpsearch.raytune_search \
    --strategy "$STRATEGY" \
    --arch "$ARCH" \
    --d_train "$D_TRAIN" \
    --n_trials "$N_TRIALS" \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --gpus "$GPUS" \
    --trials_per_gpu "$TRIALS_PER_GPU" \
    --output_dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/driver.log"
