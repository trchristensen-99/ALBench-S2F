#!/bin/bash
# Run one consolidated AutoResearch cell: 15 configs (3 roles × 5 each) on a
# single GPU via parallel_gpu_runner. ONE SLURM job per cell instead of 3 —
# matches the shootout pattern.
#
# Env vars (passed via sbatch --export):
#   ARCH       — legnet | dream_attn
#   D_TRAIN    — int
#   K_PARALLEL — int (default 4; tune to GPU memory)
#   ROUND_IDX  — int (default 0)

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

: "${ARCH:?ARCH required}"
: "${D_TRAIN:?D_TRAIN required}"
ROUND_IDX=${ROUND_IDX:-0}
K_PARALLEL=${K_PARALLEL:-4}

CFG_DIR=results/preflight/hpsearch/autoresearch/${ARCH}_d${D_TRAIN}/round_${ROUND_IDX}
CFG=$CFG_DIR/all_configs.json

# Consolidate per-role runner_configs.json into one all_configs.json if needed
if [ ! -f "$CFG" ]; then
    source .venv/bin/activate
    python -m scripts.preflight.hpsearch.autoresearch_consolidate \
        --arch "$ARCH" --d_train "$D_TRAIN" --round_idx "$ROUND_IDX"
fi

uv run --no-sync python scripts/preflight/parallel_gpu_runner.py "$CFG" "$K_PARALLEL"
