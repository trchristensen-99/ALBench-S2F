#!/bin/bash
# Train-time × inference-time shift aug sweep.
#
# Premise: Tewhey/MPAC achieves r=0.71 on SNV via 110 models × 18 sliding
# windows. We can't afford 110 models, but we CAN: (a) train with larger
# max_shift to harden the model against position bias, (b) do 18-shift
# TTA at inference. Question: what's the sweet spot?
#
# Training cells (single fold, c86 recipe = blocks 0-2 + Sahu 3% + cpg_inv 0.05):
#   ms15:  max_shift=15  (current default — already in c86_replicate)
#   ms50:  max_shift=50
#   ms100: max_shift=100
#   ms200: max_shift=200 (full adapter range)
#
# 3 NEW trainings + reuse existing c86_replicate seed=42.
# k_parallel=3 on 1 H100 → ~50min/cell × 1 sequential = ~50min total.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/shift_aug_sweep
mkdir -p $OUT
CFG=$OUT/configs.json
SAHU=$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0

uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = []
for ms in (50, 100, 200):
    configs.append({
        "label": f"c86_ms{ms}",
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/shift_aug_sweep/ms{ms}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path=$SAHU",
            "++neg_fraction=0.03",
            "++debias_mode=cpg_invariance",
            "++debias_lambda=0.05",
            "++unfreeze_encoder_blocks=[0,1,2]",
            f"++max_shift={ms}",
        ],
    })
Path("$CFG").write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} max_shift configs")
PYEOF

SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_shift_aug_sweep"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=fast --gres=gpu:h100:1 --cpus-per-task=14 --time=03:00:00 --mem=200G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo 'export PYTHONPATH="$PWD"'
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $CFG 3"
} > $SCRIPT
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  shift_aug_sweep: $JID (3 cells, k_parallel=3)"
