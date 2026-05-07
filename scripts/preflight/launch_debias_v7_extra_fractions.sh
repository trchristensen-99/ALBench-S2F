#!/bin/bash
# Debias v7: explore ultra-low + larger-mixed neg-aug fractions
# based on v3-v5 findings. c28 (3% dinuc + cpg_inv 0.05) is best so far.
#
# v7 cells:
#   - Ultra-low: 1%, 2% dinuc + cpg_inv 0.05 (does even less neg-aug
#     preserve OOD better while still helping bias?)
#   - Bigger mixed: 5%, 7% dinuc+intergenic + cpg_inv (does mixing help
#     intergenic bias without OOD collapse?)
#   - Stronger λ on c28 recipe: 3% dinuc + cpg_inv at λ=0.10, 0.20
#
# 8 candidates total. parallel_ag_s2_runner with k_parallel=3 on 1 H100.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/outputs/oracle_neg_sweep/debias_sweep_v7
mkdir -p $OUT_BASE
CFG_PATH=$OUT_BASE/configs.json
S1_DIR="$REPO/outputs/oracle_full_856k/s1/oracle_0"
DINUC_TSV="$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv"
MIXED_TSV="$REPO/data/synthetic_negatives/dinuc_plus_intergenic.tsv"

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

configs = [
    # Ultra-low fractions
    ("c70_neg1dinuc_cpginv",  0.01, "$DINUC_TSV", "cpg_invariance", 0.05),
    ("c71_neg2dinuc_cpginv",  0.02, "$DINUC_TSV", "cpg_invariance", 0.05),
    # Mid mixed fractions
    ("c72_neg5mixed_cpginv",  0.05, "$MIXED_TSV", "cpg_invariance", 0.05),
    ("c73_neg7mixed_cpginv",  0.07, "$MIXED_TSV", "cpg_invariance", 0.05),
    # Stronger λ on c28 recipe
    ("c74_neg3dinuc_cpginv_high", 0.03, "$DINUC_TSV", "cpg_invariance", 0.10),
    ("c75_neg3dinuc_cpginv_xhigh", 0.03, "$DINUC_TSV", "cpg_invariance", 0.20),
    # c28 recipe with mixed neg instead of dinuc
    ("c76_neg3mixed_cpginv",  0.03, "$MIXED_TSV", "cpg_invariance", 0.05),
    # 5% mixed + cpg_inv high λ
    ("c77_neg5mixed_cpginv_high", 0.05, "$MIXED_TSV", "cpg_invariance", 0.10),
]

result = []
for label, frac, tsv, mode, lam in configs:
    extra = [f"++negatives_path={tsv}", f"++neg_fraction={frac}"]
    if mode is not None:
        extra.append(f"++debias_mode={mode}")
        extra.append(f"++debias_lambda={lam}")
    result.append({
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1_DIR",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v7/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": extra,
    })

Path("$CFG_PATH").write_text(json.dumps(result, indent=2))
print(f"  wrote {len(result)} v7 configs")
PYEOF

SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v7_fast"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq"
    echo "#SBATCH --qos=fast"
    echo "#SBATCH --gres=gpu:h100:1"
    echo "#SBATCH --cpus-per-task=14"
    echo "#SBATCH --time=03:30:00"
    echo "#SBATCH --mem=200G"
    echo "set -euo pipefail"
    echo "set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5"
    echo "cd $REPO"
    echo 'export PYTHONPATH="$PWD"'
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $CFG_PATH 3"
} > $SCRIPT
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  debias_v7 (fast): $JID — 8 candidates, k_parallel=3"
