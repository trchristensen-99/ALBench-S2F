#!/bin/bash
#SBATCH --job-name=valvar_gpu
#SBATCH --partition=gpuq --qos=fast --gres=gpu:h100:1
#SBATCH --cpus-per-task=8 --mem=90G --time=03:50:00 --requeue
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/val_variants/logs/gpu_%x_%j.log
# Oracle-guided-evolution candidates + oracle_v3 labelling of every variant candidate and of the V0 sequences. Idempotent: finished outputs are skipped.
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments" PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export AG_S2_ORACLE_DIR=$PWD/outputs/oracle_v3/s2
echo "gpu stage on $(hostname) $(date)"
python scripts/val_variants/build_val_variants.py gpu --k 12 --wait_min ${WAIT_MIN:-150}
echo "done $(date)"
