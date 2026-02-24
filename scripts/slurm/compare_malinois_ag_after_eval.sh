#!/bin/bash
# Run the Malinois vs AlphaGenome comparison report after the boda2-tutorial eval has finished.
# Optionally run after exporting AG HashFrag: uv run python scripts/analysis/export_ag_hashfrag_results.py
#
# Usage:
#   sbatch scripts/slurm/compare_malinois_ag_after_eval.sh
#   sbatch --dependency=afterok:JOBID scripts/slurm/compare_malinois_ag_after_eval.sh   # after eval job
#SBATCH --job-name=compare_mal_ag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -e
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

MALINOIS_JSON="outputs/malinois_eval_boda2_tutorial/result.json"
if [ ! -f "$MALINOIS_JSON" ]; then
    echo "Error: Malinois results not found at $MALINOIS_JSON. Run eval_malinois_boda2_tutorial.sh first."
    exit 1
fi

# Optional: export AlphaGenome HashFrag if checkpoints exist
if [ -d "outputs/ag_sum/best_model" ] && [ -f "outputs/ag_sum/best_model/checkpoint" ]; then
    uv run python scripts/analysis/export_ag_hashfrag_results.py || true
fi

uv run python scripts/analysis/compare_malinois_alphagenome_results.py \
    --malinois_json "$MALINOIS_JSON" \
    --output outputs/malinois_ag_comparison.md

echo "Comparison report: outputs/malinois_ag_comparison.md"
