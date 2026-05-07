#!/bin/bash
# Debias sweep v6: Sahu-corrected neg-aug labels.
#
# v1-v5 used neg-aug TSVs with labels sampled from the Agarwal lentiviral
# distribution (mean -0.45). The Sahu STARR-seq analysis showed that
# random sequences in episomal K562 MPRA actually have:
#   - mean activity ≈ +0.27 (matches Gosai ctrl_neg, n=503)
#   - small positive tilt with CpG content (~25% activity increase
#     from low to high CpG, mapped to ~+0.06 log2FC linear tilt)
#
# v6 uses regenerated TSV with Sahu-corrected labels (Gosai ctrl_neg
# distribution + CpG tilt). All else same as v3-v5 winning recipes.
#
# Uses parallel_ag_s2_runner with N_GPUS=2 (verified on smoke test:
# 2 AG-S2 fit easily on one H100 with MEM_FRACTION=0.4).
# Total: 8 candidate configs in 1 SLURM job × 2 GPUs.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/outputs/oracle_neg_sweep/debias_sweep_v6
mkdir -p $OUT_BASE
CFG_PATH=$OUT_BASE/configs.json

# (1) Generate Sahu-corrected TSV if not already present
if [ ! -f data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv ]; then
    echo "=== Generating Sahu-corrected neg-aug TSV ==="
    uv run --no-sync python scripts/preflight/_generate_sahu_labeled_negatives.py
fi

# (2) Build v6 configs
SAHU_TSV="$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv"
S1_DIR="$REPO/outputs/oracle_full_856k/s1/oracle_0"

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

# Recipe candidates: vary fraction × loss term
configs = []
for label, frac, debias_mode, lam in [
    ("c60_neg3sahu",        0.03, None, None),
    ("c61_neg5sahu",        0.05, None, None),
    ("c62_neg10sahu",       0.10, None, None),
    ("c63_neg3sahu_cpginv", 0.03, "cpg_invariance", 0.05),
    ("c64_neg5sahu_cpginv", 0.05, "cpg_invariance", 0.05),
    ("c65_neg10sahu_cpginv",0.10, "cpg_invariance", 0.05),
    ("c66_neg10sahu_grad",  0.10, "cpg_gradient_penalty", 0.05),
    ("c67_neg5sahu_cond",   0.05, "conditional_invariance", 0.05),
]:
    extra = [
        f"++negatives_path=$SAHU_TSV",
        f"++neg_fraction={frac}",
    ]
    if debias_mode is not None:
        extra.append(f"++debias_mode={debias_mode}")
        extra.append(f"++debias_lambda={lam}")
    configs.append({
        "label": label,
        "fold_id": 0,
        "n_folds": 10,
        "stage1_dir": "$S1_DIR",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v6/{label}/fold_0",
        "epochs": 80,
        "patience": 15,
        "extra_overrides": extra,
    })

p = Path("$CFG_PATH")
p.write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v6 configs")
PYEOF

# (3) Submit one SLURM job using parallel AG-S2 runner with 2 GPUs × 2 candidates per GPU
SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v6_sahu"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq"
    echo "#SBATCH --qos=slow_nice"
    echo "#SBATCH --gres=gpu:h100:2"
    echo "#SBATCH --cpus-per-task=28"
    echo "#SBATCH --time=06:00:00"
    echo "#SBATCH --mem=400G"
    echo "set -euo pipefail"
    echo "set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5"
    echo "cd $REPO"
    echo 'export PYTHONPATH="$PWD"'
    echo "source scripts/slurm/setup_hpc_deps.sh"
    # 2 GPUs × 2 candidates each = 4 concurrent. With 8 total cells, runs as 2 cycles of 4.
    echo "N_GPUS=2 uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py \\"
    echo "    $CFG_PATH 4"
} > $SCRIPT
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  debias_sweep_v6 (8 candidates, 2 H100s × 2 each): $JID"
