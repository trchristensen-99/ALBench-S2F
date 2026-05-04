#!/bin/bash
# Generate the expanded eval-set panel on HPC.
#
# This is CPU-only (no GPU) and finishes in <5 min on a single core.
# Submit AFTER the ref+alt pool builder lands so it picks up the
# chromosome-split chr8,9 test parquet from
# outputs/oracle_pseudolabels_k562_ag_s2_refalt/pool/test.parquet.
# If that parquet is missing, the script falls back to the legacy
# chr7,13 TSV — useful for early testing but produces the wrong test
# split for the final pre-flight panel.

set -euo pipefail

cat > /tmp/_pf_eval_sets.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=pf_eval_sets
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpu_fill
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"

uv run --no-sync python scripts/preflight/generate_eval_sets.py --seed 42
EOF

/cm/shared/apps/slurm/current/bin/sbatch /tmp/_pf_eval_sets.sh
rm -f /tmp/_pf_eval_sets.sh
