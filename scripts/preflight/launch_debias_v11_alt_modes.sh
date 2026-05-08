#!/bin/bash
# Debias v11: c91 recipe with ALTERNATIVE debias_modes already in train script
# but never tested with c91's freeze pattern + dinuc 3%.
#
# c91 base: blocks 0-2 + dinuc 3% + cpg_invariance λ=0.05.
# v11 swaps cpg_invariance for each of:
#   c109: cpg_gradient_penalty   (penalize ∂pred/∂CpG, not corr(pred, CpG))
#   c110: counterfactual_consistency
#   c111: adaptive_group_dro
#   c112: conditional_invariance (cpg_inv on low-activity batch only)
#   c113: spectral (penalize pred norm on synthetic neg)
#
# 5 cells, 1 H100, k_parallel=3, ~50min × 2 sequential = ~2h.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v11
mkdir -p $OUT
CFG=$OUT/configs.json
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, mode, lam=0.05):
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v11/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path=$DINUC",
            "++neg_fraction=0.03",
            f"++debias_mode={mode}",
            f"++debias_lambda={lam}",
            "++unfreeze_encoder_blocks=[0,1,2]",
        ],
    }

configs = [
    cfg("c109_blk012_dinuc3_cpggrad",     "cpg_gradient_penalty",     0.05),
    cfg("c110_blk012_dinuc3_counterfact", "counterfactual_consistency", 0.05),
    cfg("c111_blk012_dinuc3_adaptdro",    "adaptive_group_dro",       0.05),
    cfg("c112_blk012_dinuc3_condinv",     "conditional_invariance",   0.05),
    cfg("c113_blk012_dinuc3_spectral",    "spectral",                 0.05),
]
Path("$CFG").write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v11 configs (alternative debias modes)")
PYEOF

SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v11"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=fast --gres=gpu:h100:1 --cpus-per-task=14 --time=03:30:00 --mem=200G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo 'export PYTHONPATH="$PWD"'
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $CFG 3"
} > $SCRIPT
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  v11: $JID (5 cells × ~50min × ceil(5/3)=2 sequential = ~2h)"
