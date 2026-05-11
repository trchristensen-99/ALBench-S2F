#!/bin/bash
#SBATCH --job-name=hps_smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=80G

# Smoke test: verifies the Ray Tune driver + trainable end-to-end.
# 3 random trials of LegNet at D=500, 5 epochs each. Should complete in <5 min.

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
# Activate venv directly so Ray's spawned worker processes can find 'ray'.
# `uv run` only sets the env for the current process, not for Ray's child workers.
source .venv/bin/activate
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

OUT=results/preflight/hpsearch/_smoke_legnet
rm -rf $OUT
mkdir -p $OUT

python -m scripts.preflight.hpsearch.raytune_search \
    --strategy random \
    --arch legnet \
    --d_train 500 \
    --n_trials 3 \
    --max_epochs 5 \
    --patience 5 \
    --gpus 1 \
    --trials_per_gpu 2 \
    --output_dir $OUT 2>&1 | tee $OUT/driver.log
