#!/bin/bash
# Profile every acquisition rule on every reservoir, with two banks each.
#
# The full rule x reservoir cross as TRAINED arms is 8 x 5 = 40 cells at ~3 GPU-h, which is
# 120 GPU-h to answer a question that may well be "nothing changes". The selection itself
# needs no training -- a rule is a function of the bank's predictions over the candidates --
# so this profiles the whole cross by inference and lets the trained follow-up go only where
# the profile says something moved.
#
# Each task does ONE pool with BOTH banks: the reservoir's own 30k bank (the realistic
# setting, and what ab43/ab44 used) and the common 30k GENOMIC bank (holds the ranker fixed,
# so the difference between the columns separates "the candidates changed" from "the model
# that ranks them changed").
#
# ~200k candidates x 5 members x 2 strands per bank; a few minutes each on a V100.
#SBATCH --job-name=acqint
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=default
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/acq50/logs/acqint_%A_%a.log
#SBATCH --array=0-7%4
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq50/logs

# Eight reservoirs spanning the quality range measured by Arm A on the designed endpoint,
# from the two best (motif_shared_core 0.630, motif_ct_enriched 0.540) down through the
# middle (mutagenesis, evoaug, gc_matched) to the weak end (dinuc_shuffle, random) with
# genomic as the real-sequence reference. If the rule ranking is a property of the rule, it
# should be the same in all eight.
RES=(motif_shared_core motif_ct_enriched mutagenesis evoaug_paper2023 gc_matched dinuc_shuffle random genomic)
i=$SLURM_ARRAY_TASK_ID
r=${RES[$i]}

POOL="$PWD/outputs/grid_pools/k562_nested/${r}_d300000.npz"
OWN="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20_${r}"
COMMON="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20"   # the 30k GENOMIC bank

BANKS=("$COMMON")
[ -d "$OWN" ] && [ "$(ls "$OWN"/*.pt 2>/dev/null | wc -l)" -ge 3 ] && BANKS=("$OWN" "$COMMON")

echo "[$(date)] acq50 $r | banks: ${BANKS[*]##*/}"
.venv/bin/python scratchpad/nov25/acq_pool_interaction.py \
  --pool "$POOL" --banks "${BANKS[@]}" \
  --out "$PWD/outputs/acq50/${r}.json"
echo "[$(date)] done $r"
