#!/bin/bash
#SBATCH --job-name=screengen
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --qos=slow_nice
#SBATCH --array=1-210%36
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/screengen/%A_%a.out
set -uo pipefail
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd "${ALBENCH_REPO:-/grid/wsbs/home_norepl/christen/ALBench-S2F}"
JOBS=outputs/screen/jobs.sh
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
if [ -z "$LINE" ]; then echo "no job for index ${SLURM_ARRAY_TASK_ID}"; exit 0; fi
# Idempotent: a completed cell is skipped, so resubmitting the array is safe.
OUT=$(echo "$LINE" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
if [ -s "$OUT" ]; then echo "SKIP already present: $OUT"; exit 0; fi
echo "RUN $LINE"
eval "uv run --no-sync python -m albench.cli ${LINE#albench }"
rc=$?
if [ $rc -eq 0 ] && [ -s "$OUT" ]; then echo "DONE $OUT"; else echo "FAIL rc=$rc $OUT"; exit 1; fi
