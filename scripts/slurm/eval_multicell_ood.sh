#!/bin/bash
# Evaluate existing HepG2/SKNSH model checkpoints on cell-line-specific OOD test sets.
#
# OOD files: data/{cell}/test_sets/test_ood_designed_{cell}.tsv
#
# Array tasks:
#   0: Malinois HepG2
#   1: Malinois SKNSH
#   2: Enformer S1 HepG2
#   3: Enformer S1 SKNSH
#   4: Borzoi S1 HepG2
#   5: Borzoi S1 SKNSH
#   6: NTv3 S1 HepG2
#   7: NTv3 S1 SKNSH
#   8: Enformer S2 HepG2
#   9: Enformer S2 SKNSH
#  10: NTv3 S2 HepG2
#  11: NTv3 S2 SKNSH
#  12: AG S1 HepG2
#  13: AG S1 SKNSH
#  14: AG S2 HepG2
#  15: AG S2 SKNSH
#
# Submit: sbatch scripts/slurm/eval_multicell_ood.sh
# Or single task: sbatch --array=0 scripts/slurm/eval_multicell_ood.sh
#
#SBATCH --job-name=ood_eval
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-15

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

mkdir -p logs

echo "=== OOD Evaluation ==="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Ensure data symlinks exist for non-K562 cell lines
for CELL in hepg2 sknsh; do
    mkdir -p "data/${CELL}/test_sets"
    ln -sf "$(pwd)/data/k562/DATA-Table_S2__MPRA_dataset.txt" "data/${CELL}/DATA-Table_S2__MPRA_dataset.txt" 2>/dev/null || true
    ln -sf "$(pwd)/data/k562/hashfrag_splits" "data/${CELL}/hashfrag_splits" 2>/dev/null || true
    # NOTE: test_ood_designed_{cell}.tsv must already exist in data/{cell}/test_sets/
done

SCRIPT="scripts/eval_ood_multicell.py"

case ${SLURM_ARRAY_TASK_ID} in
    0)
        echo "Malinois HepG2"
        uv run --no-sync python ${SCRIPT} \
            --cell-line hepg2 \
            --model-type malinois \
            --result-dirs \
                outputs/malinois_hepg2_3seeds/seed_0/seed_0 \
                outputs/malinois_hepg2_3seeds/seed_1/seed_1 \
                outputs/malinois_hepg2_3seeds/seed_2/seed_2
        ;;
    1)
        echo "Malinois SKNSH"
        uv run --no-sync python ${SCRIPT} \
            --cell-line sknsh \
            --model-type malinois \
            --result-dirs \
                outputs/malinois_sknsh_3seeds/seed_0/seed_0 \
                outputs/malinois_sknsh_3seeds/seed_1/seed_1 \
                outputs/malinois_sknsh_3seeds/seed_2/seed_2
        ;;
    2)
        echo "Enformer S1 HepG2"
        uv run --no-sync python ${SCRIPT} \
            --cell-line hepg2 \
            --model-type foundation_s1 \
            --encoder-name enformer \
            --result-dirs \
                outputs/enformer_hepg2_cached/seed_0/seed_0 \
                outputs/enformer_hepg2_cached/seed_1/seed_1 \
                outputs/enformer_hepg2_cached/seed_2/seed_2
        ;;
    3)
        echo "Enformer S1 SKNSH"
        uv run --no-sync python ${SCRIPT} \
            --cell-line sknsh \
            --model-type foundation_s1 \
            --encoder-name enformer \
            --result-dirs \
                outputs/enformer_sknsh_cached/seed_0/seed_0 \
                outputs/enformer_sknsh_cached/seed_1/seed_1 \
                outputs/enformer_sknsh_cached/seed_2/seed_2
        ;;
    4)
        echo "Borzoi S1 HepG2"
        uv run --no-sync python ${SCRIPT} \
            --cell-line hepg2 \
            --model-type foundation_s1 \
            --encoder-name borzoi \
            --result-dirs \
                outputs/borzoi_hepg2_cached/seed_0/seed_0 \
                outputs/borzoi_hepg2_cached/seed_1/seed_1 \
                outputs/borzoi_hepg2_cached/seed_2/seed_2
        ;;
    5)
        echo "Borzoi S1 SKNSH"
        uv run --no-sync python ${SCRIPT} \
            --cell-line sknsh \
            --model-type foundation_s1 \
            --encoder-name borzoi \
            --result-dirs \
                outputs/borzoi_sknsh_cached/seed_0/seed_0 \
                outputs/borzoi_sknsh_cached/seed_1/seed_1 \
                outputs/borzoi_sknsh_cached/seed_2/seed_2
        ;;
    6)
        echo "NTv3 S1 HepG2"
        uv run --no-sync python ${SCRIPT} \
            --cell-line hepg2 \
            --model-type foundation_s1 \
            --encoder-name ntv3_post \
            --result-dirs \
                outputs/ntv3_post_hepg2_cached/seed_0/seed_0 \
                outputs/ntv3_post_hepg2_cached/seed_1/seed_1 \
                outputs/ntv3_post_hepg2_cached/seed_2/seed_2
        ;;
    7)
        echo "NTv3 S1 SKNSH"
        uv run --no-sync python ${SCRIPT} \
            --cell-line sknsh \
            --model-type foundation_s1 \
            --encoder-name ntv3_post \
            --result-dirs \
                outputs/ntv3_post_sknsh_cached/seed_0/seed_0 \
                outputs/ntv3_post_sknsh_cached/seed_1/seed_1 \
                outputs/ntv3_post_sknsh_cached/seed_2/seed_2
        ;;
    8)
        echo "Enformer S2 HepG2"
        uv run --no-sync python ${SCRIPT} \
            --cell-line hepg2 \
            --model-type enformer_s2 \
            --result-dirs \
                outputs/enformer_hepg2_stage2/seed_0
        ;;
    9)
        echo "Enformer S2 SKNSH"
        uv run --no-sync python ${SCRIPT} \
            --cell-line sknsh \
            --model-type enformer_s2 \
            --result-dirs \
                outputs/enformer_sknsh_stage2/seed_0
        ;;
    10)
        echo "NTv3 S2 HepG2"
        uv run --no-sync python ${SCRIPT} \
            --cell-line hepg2 \
            --model-type ntv3_s2 \
            --result-dirs \
                outputs/ntv3_post_hepg2_stage2/seed_0
        ;;
    11)
        echo "NTv3 S2 SKNSH"
        uv run --no-sync python ${SCRIPT} \
            --cell-line sknsh \
            --model-type ntv3_s2 \
            --result-dirs \
                outputs/ntv3_post_sknsh_stage2/seed_0
        ;;
    12)
        echo "AG S1 HepG2"
        uv run --no-sync python ${SCRIPT} \
            --cell-line hepg2 \
            --model-type ag_s1 \
            --result-dirs \
                outputs/ag_hashfrag_hepg2_cached/seed_0 \
                outputs/ag_hashfrag_hepg2_cached/seed_1 \
                outputs/ag_hashfrag_hepg2_cached/seed_2
        ;;
    13)
        echo "AG S1 SKNSH"
        uv run --no-sync python ${SCRIPT} \
            --cell-line sknsh \
            --model-type ag_s1 \
            --result-dirs \
                outputs/ag_hashfrag_sknsh_cached/seed_0 \
                outputs/ag_hashfrag_sknsh_cached/seed_1 \
                outputs/ag_hashfrag_sknsh_cached/seed_2
        ;;
    14)
        echo "AG S2 HepG2"
        uv run --no-sync python ${SCRIPT} \
            --cell-line hepg2 \
            --model-type ag_s2 \
            --result-dirs \
                outputs/ag_hepg2_stage2/seed_0
        ;;
    15)
        echo "AG S2 SKNSH"
        uv run --no-sync python ${SCRIPT} \
            --cell-line sknsh \
            --model-type ag_s2 \
            --result-dirs \
                outputs/ag_sknsh_stage2/seed_0
        ;;
    *)
        echo "Unknown task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "Done: $(date)"
