#!/bin/bash
# Run from your local machine when connected to VPN/campus so HPC is reachable.
# 1) Sync repo to HPC
# 2) Submit AG chrom-test eval job (writes outputs/ag_chrom_test_results.json)
# 3) Optionally run comparison report after job completes (run manually or via a follow-up job).
set -e
REPO="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
REMOTE="${2:-christen@bamdev4.cshl.edu}"
REMOTE_DIR="/grid/wsbs/home_norepl/christen/ALBench-S2F"
SSH_OPTS="-o ConnectTimeout=30 -o ServerAliveInterval=30"

echo "Syncing $REPO -> $REMOTE:$REMOTE_DIR"
rsync -avz --exclude='.git' --exclude='*.tar.gz' --exclude='__pycache__' \
  -e "ssh $SSH_OPTS" "$REPO/" "$REMOTE:$REMOTE_DIR/"

echo "Submitting AG chrom-test evaluation job..."
ssh $SSH_OPTS "$REMOTE" "cd $REMOTE_DIR && sbatch scripts/slurm/eval_ag_chrom_test.sh"

echo "Done. Check job: squeue -u christen"
echo "After job completes, run on HPC: uv run python scripts/analysis/compare_malinois_alphagenome_results.py --output outputs/malinois_ag_comparison.md"
echo "Or pull outputs/ag_chrom_test_results.json and outputs/malinois_ag_comparison.md and run compare locally."
exit 0
