#!/bin/bash
# Debias v10: refinements around c91 (the new winner).
# c91 = blocks 0-2 + dinuc 3% + cpg_inv 0.05.
# c91 hits Pareto: test_id=0.954, OOD=0.762, snv_d=0.406, random=0.489.
#
# Variations to test:
#   c100: dinuc 1%      (less neg-aug, OOD-preserving?)
#   c101: dinuc 1.5%
#   c102: dinuc 7%      (more bias reduction?)
#   c103: dinuc 10%     (max push)
#   c104: dinuc 3% + cpg_inv 0.10 (clean λ stack)
#   c105: dinuc 3% + cpg_inv 0.20 (very high λ)
#   c106: dinuc 3% + cpg_inv 0.05 + wd=1e-3 (extra reg)
#   c107: dinuc 3% + cpg_inv 0.05 + ctrl_neg layered 3% (real+synth)
#   c108: c91 + lower encoder_lr (5e-5)

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/outputs/oracle_neg_sweep/debias_sweep_v10
mkdir -p $OUT_BASE
CFG=$OUT_BASE/configs.json
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv
CTRL=$REPO/data/synthetic_negatives_calibration/gosai_ctrl_neg_calibration.tsv

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, frac, lam=0.05, tsv=None, wd=None, enc_lr=None):
    if tsv is None:
        tsv = "$DINUC"
    extras = [
        f"++negatives_path={tsv}",
        f"++neg_fraction={frac}",
        "++debias_mode=cpg_invariance",
        f"++debias_lambda={lam}",
        "++unfreeze_encoder_blocks=[0,1,2]",
    ]
    if wd is not None:
        extras.append(f"++weight_decay={wd}")
    if enc_lr is not None:
        extras.append(f"++encoder_lr={enc_lr}")
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v10/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": extras,
    }

configs = [
    cfg("c100_dinuc1_cpginv",          0.01),
    cfg("c101_dinuc15_cpginv",         0.015),
    cfg("c102_dinuc7_cpginv",          0.07),
    cfg("c103_dinuc10_cpginv",         0.10),
    cfg("c104_dinuc3_cpginv_lam10",    0.03, lam=0.10),
    cfg("c105_dinuc3_cpginv_lam20",    0.03, lam=0.20),
    cfg("c106_dinuc3_cpginv_wd1e3",    0.03, wd=1e-3),
    cfg("c108_dinuc3_cpginv_enclr5e5", 0.03, enc_lr=5e-5),
]
# c107: special — blocks 0-2 + dinuc 3% + cpg_inv + ctrl_neg as second neg-aug
# Workaround: train script supports only ONE TSV. Skip this combo.

Path("$CFG").write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v10 configs")
PYEOF

# Submit as 1 job, k_parallel=3
SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v10"
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
echo "  v10: $JID (8 cells × ~50min × ceil(8/3)=3 sequential = ~2.5h)"
