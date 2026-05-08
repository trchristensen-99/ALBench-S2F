#!/bin/bash
# c86 10-fold ensemble: blocks 0-2 unfreeze + Sahu 3% + cpg_inv 0.05.
# Confirmed across 3 seeds (test_id=0.951-0.955, snv_d=0.404-0.408).
# This is the new candidate for the production oracle.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/debias_c86_10fold
mkdir -p $OUT
CFG=$OUT/configs.json
SAHU=$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv

uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = []
for fold in range(10):
    configs.append({
        "label": f"c86_fold_{fold}",
        "fold_id": fold, "n_folds": 10,
        "stage1_dir": f"$REPO/outputs/oracle_full_856k/s1/oracle_{fold}",
        "output_dir": f"outputs/oracle_neg_sweep/debias_c86_10fold/fold_{fold}",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path=$SAHU",
            "++neg_fraction=0.03",
            "++debias_mode=cpg_invariance",
            "++debias_lambda=0.05",
            "++unfreeze_encoder_blocks=[0,1,2]",
        ],
    })
Path("$CFG").write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} c86 fold configs")
PYEOF

# Split into 4 batches
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$CFG").read_text())
batches = [configs[0:3], configs[3:6], configs[6:8], configs[8:10]]
for i, b in enumerate(batches):
    Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(b, indent=2))
print(f"  split into 4 batches: {[len(b) for b in batches]}")
PYEOF

for tag in 0 1 2 3; do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_debias_c86_10fold_b${tag}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=12:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  c86_10fold_b${tag}: $JID"
done
