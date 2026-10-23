#!/bin/bash
#SBATCH --job-name=valvar_cpu
#SBATCH --partition=cpuq --qos=fast
#SBATCH --cpus-per-task=16 --mem=160G --time=03:00:00 --requeue
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/val_variants/logs/cpu_%x_%j.log
# usage: sbatch --export=ALL,STAGE=index|calibrate|gen|assemble|stats  [-J name] cpu.sbatch  [extra args via ARGS]
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments" PYTHONUNBUFFERED=1 TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "stage=$STAGE args=${ARGS:-} on $(hostname) $(date)"
python scripts/val_variants/build_val_variants.py $STAGE ${ARGS:-}
echo "done $(date)"
