#!/bin/bash
# Multi-GPU agent runner: request N GPUs from SLURM, round-robin trials
# across them via parallel_gpu_runner's N_GPUS env var. Each GPU runs
# TRIALS_PER_GPU trials concurrently (default 8), so total concurrency
# is N_GPUS * TRIALS_PER_GPU.
#
# Usage:
#   PROPOSALS=... D_TRAIN=... OUT_DIR=... N_GPUS=4 TRIALS_PER_GPU=8 \
#   QOS=default TIME=10:00:00 bash scripts/slurm/agent_multigpu.sh
#
# Required env:
#   PROPOSALS       path to agent_proposals_*.json
#   D_TRAIN         int (5000 / 20000 / 100000)
#   OUT_DIR         output dir under results/preflight/hpsearch/...
# Optional env:
#   ARCH            default: legnet
#   N_GPUS          default: 4
#   TRIALS_PER_GPU  default: 8
#   QOS             default: default  (slow_nice / default / fast)
#   TIME            default: 10:00:00
#   EPOCHS          default: 60
#   PATIENCE        default: 10
#   JOBNAME         default: agent_${ARCH}_d${D_TRAIN}_mg${N_GPUS}
#   DEP             optional --dependency string (e.g. afterany:2200509)

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

: "${PROPOSALS:?PROPOSALS required}"
: "${D_TRAIN:?D_TRAIN required}"
: "${OUT_DIR:?OUT_DIR required}"
ARCH=${ARCH:-legnet}
N_GPUS=${N_GPUS:-4}
TRIALS_PER_GPU=${TRIALS_PER_GPU:-8}
QOS=${QOS:-default}
TIME=${TIME:-10:00:00}
EPOCHS=${EPOCHS:-60}
PATIENCE=${PATIENCE:-10}
JOBNAME=${JOBNAME:-agent_${ARCH}_d${D_TRAIN}_mg${N_GPUS}}
DEP=${DEP:-}

K_PARALLEL=$(( N_GPUS * TRIALS_PER_GPU ))
mkdir -p "$OUT_DIR"

# Convert proposals → configs.json (skip if already present).
if [ ! -f "$OUT_DIR/configs.json" ]; then
    source .venv/bin/activate
    python -m scripts.preflight.hpsearch._convert_agent_proposals \
        --proposals "$PROPOSALS" \
        --arch "$ARCH" \
        --d_train "$D_TRAIN" \
        --output_dir "$OUT_DIR" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE"
fi

# CPU allocation: 4 cpus per GPU plus 2 base; mem 60G per GPU.
CPUS=$(( N_GPUS * 4 + 2 ))
MEM=$(( N_GPUS * 60 ))G

JOBFILE=$(mktemp)
cat > "$JOBFILE" <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:$N_GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$TIME
#SBATCH --mem=$MEM

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache
export N_GPUS=$N_GPUS
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

echo "[multigpu] N_GPUS=$N_GPUS  trials_per_gpu=$TRIALS_PER_GPU  k_parallel=$K_PARALLEL"
nvidia-smi -L

uv run --no-sync python scripts/preflight/parallel_gpu_runner.py \\
    "$OUT_DIR/configs.json" $K_PARALLEL
EOF

if [ -n "$DEP" ]; then
    JID=$($SBATCH --parsable --dependency="$DEP" "$JOBFILE")
else
    JID=$($SBATCH --parsable "$JOBFILE")
fi
rm -f "$JOBFILE"
echo "$JOBNAME (qos=$QOS, gpus=$N_GPUS, k=$K_PARALLEL) → $JID"
