#!/bin/bash
# dv69: the diversity analysis again, parallel, on V100s.
#
# Two speedups, and the first matters far more than the second:
#
#   The scan re-encoded every sequence once per motif WIDTH. With a vocabulary spanning
#   about fifteen widths, and the one-hot written as a per-character Python loop, encoding
#   dominated the run. It is now vectorised and hoisted so each block is encoded once and
#   reused across widths. The p-value thresholds moved out of the chunk loop too -- they
#   are a property of the motif, not of the data.
#
#   Then the arms are sharded round-robin across four tasks. Round-robin and not blocks,
#   because the pool-size arms get steadily more expensive and a contiguous split would
#   hand one task the 1M pool while another sat idle.
#
# Each shard writes its own JSON; they merge on read, since the arms are disjoint. Shard 0
# carries the 30,000 arm, which is the no-op control -- the selection there IS the whole
# pool, so its three groups must agree.
#
# V100s because they are idle and this is GEMM-bound rather than large; waiting for an
# H100 costs more wall clock than the slower card does.
#SBATCH --job-name=dv69
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/dv69_%A_%a.log
#SBATCH --array=0-3
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" /tmp/$USER/mpl /tmp/$USER/tmp

.venv/bin/python "$KOO/bin/diversity_analysis.py" \
  --out "$KOO/diversity_shard${SLURM_ARRAY_TASK_ID}.json" \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 4
echo "dv69 shard ${SLURM_ARRAY_TASK_ID} done"
