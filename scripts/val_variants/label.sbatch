#!/bin/bash
#SBATCH --job-name=valvar_label
#SBATCH --partition=gpuq --qos=fast --gres=gpu:h100:1
#SBATCH --cpus-per-task=8 --mem=60G --time=01:00:00 --requeue
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/val_variants/logs/label_%j.log
# usage: sbatch --export=ALL,IN=<npz with sequences>,OUT=<labelled npz> label.sbatch   (label_pool.py is atomic + stamps oracle_id)
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments" PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export AG_S2_ORACLE_DIR=$PWD/outputs/oracle_v3/s2
echo "label $IN -> $OUT on $(hostname) $(date)"
uv run --no-sync python scripts/label_pool.py --in "$IN" --out "$OUT"
echo "done $(date)"
