#!/bin/bash
# Launch 10-fold ensemble of c63 (3% Sahu labels + cpg_invariance λ=0.05)
# alongside the c28 10-fold (job 2115621).
#
# This gives us a head-to-head: Sahu-corrected synthetic labels (c63)
# vs raw dinuc-shuffled (c28) at the same 10-fold ensemble scale.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/outputs/oracle_neg_sweep/debias_c63_10fold
mkdir -p $OUT_BASE
CFG_PATH=$OUT_BASE/configs.json
SAHU_TSV="$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv"

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

configs = []
for fold in range(10):
    configs.append({
        "label": f"c63_fold_{fold}",
        "fold_id": fold, "n_folds": 10,
        "stage1_dir": f"$REPO/outputs/oracle_full_856k/s1/oracle_{fold}",
        "output_dir": f"outputs/oracle_neg_sweep/debias_c63_10fold/fold_{fold}",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path=$SAHU_TSV",
            "++neg_fraction=0.03",
            "++debias_mode=cpg_invariance",
            "++debias_lambda=0.05",
        ],
    })

Path("$CFG_PATH").write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} c63 fold configs")
PYEOF

# Split into 4 jobs of ~3 folds each (k_parallel=3 each, ~1h per fold)
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$CFG_PATH").read_text())
batches = [configs[0:3], configs[3:6], configs[6:8], configs[8:10]]
for i, batch in enumerate(batches):
    Path(f"$OUT_BASE/configs_b{i}.json").write_text(json.dumps(batch, indent=2))
print(f"  split into 4 batches: {[len(b) for b in batches]}")
PYEOF

for tag in 0 1 2 3; do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_debias_c63_10fold_b${tag}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq"
        echo "#SBATCH --qos=slow_nice"
        echo "#SBATCH --gres=gpu:h100:1"
        echo "#SBATCH --cpus-per-task=14"
        echo "#SBATCH --time=12:00:00"
        echo "#SBATCH --mem=200G"
        echo "set -euo pipefail"
        echo "set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5"
        echo "cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py \\"
        echo "    $OUT_BASE/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  c63_10fold_b${tag}: $JID"
done
