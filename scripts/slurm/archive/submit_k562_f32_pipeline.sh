#!/bin/bash
# Master script: submit the full K562 float32 cache rebuild pipeline.
#
# This fixes the float16 embedding cache bug where bfloat16→float16 truncation
# zeroed ~60% of embeddings, degrading oracle val Pearson from ~0.90 to ~0.67.
#
# Pipeline:
#   Step 1: Build f32 train/val cache  (1 GPU job, ~60 min)
#   Step 2: Build f32 test caches      (1 GPU job, ~60 min) — parallel with step 1
#   Step 3: Retrain 10 oracle folds    (10-task array, ~1-2h each) — after step 1
#   Step 4: Generate pseudolabels      (1 GPU job, ~20 min) — after steps 2+3
#   Step 5a: AG real-label scaling     (21-task array) — after step 1
#   Step 5b: AG oracle-label scaling   (21-task array) — after step 4
#   Step 5c: DREAM-RNN oracle scaling  (21-task array) — after step 4
#
# Usage (from login node):
#   bash scripts/slurm/submit_k562_f32_pipeline.sh
#
set -euo pipefail

SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

echo "=== K562 Float32 Cache Rebuild Pipeline ==="
echo "Submitting from $(hostname) at $(date)"
echo ""

# Step 1: Build f32 train/val cache
CACHE_JOB=$($SBATCH --parsable scripts/slurm/build_hashfrag_cache_f32.sh)
echo "Step 1: Build f32 train/val cache → job ${CACHE_JOB}"

# Step 2: Build f32 test caches (parallel with step 1)
TEST_CACHE_JOB=$($SBATCH --parsable scripts/slurm/build_hashfrag_test_cache_f32.sh)
echo "Step 2: Build f32 test caches    → job ${TEST_CACHE_JOB}"

# Step 3: Train 10 oracle folds (after train/val cache is ready)
ORACLE_JOB=$($SBATCH --parsable --dependency=afterok:${CACHE_JOB} \
    scripts/slurm/train_oracle_hashfrag_f32_array.sh)
echo "Step 3: Train 10 oracle folds    → job ${ORACLE_JOB} (after ${CACHE_JOB})"

# Step 4: Generate pseudolabels (after oracle training AND test caches)
PL_JOB=$($SBATCH --parsable --dependency=afterok:${ORACLE_JOB},afterok:${TEST_CACHE_JOB} \
    scripts/slurm/generate_pseudolabels_k562_f32.sh)
echo "Step 4: Generate pseudolabels    → job ${PL_JOB} (after ${ORACLE_JOB},${TEST_CACHE_JOB})"

# Step 5a: AG real-label scaling (only needs train/val cache)
REAL_JOB=$($SBATCH --parsable --dependency=afterok:${CACHE_JOB} \
    scripts/slurm/exp0_k562_scaling_ag_real_f32.sh)
echo "Step 5a: AG real-label scaling   → job ${REAL_JOB} (after ${CACHE_JOB})"

# Step 5b: AG oracle-label scaling (needs cache + pseudolabels)
AG_ORACLE_JOB=$($SBATCH --parsable --dependency=afterok:${PL_JOB} \
    scripts/slurm/exp0_k562_scaling_oracle_ag_f32.sh)
echo "Step 5b: AG oracle-label scaling → job ${AG_ORACLE_JOB} (after ${PL_JOB})"

# Step 5c: DREAM-RNN oracle-label scaling (needs pseudolabels only)
DREAM_ORACLE_JOB=$($SBATCH --parsable --dependency=afterok:${PL_JOB} \
    scripts/slurm/exp0_k562_scaling_oracle_dream_f32.sh)
echo "Step 5c: DREAM-RNN oracle scaling → job ${DREAM_ORACLE_JOB} (after ${PL_JOB})"

echo ""
echo "=== Pipeline submitted ==="
echo "Total: 7 job submissions (2 cache + 10 oracle + 1 pseudolabel + 63 scaling tasks)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel ${CACHE_JOB} ${TEST_CACHE_JOB} ${ORACLE_JOB} ${PL_JOB} ${REAL_JOB} ${AG_ORACLE_JOB} ${DREAM_ORACLE_JOB}"
