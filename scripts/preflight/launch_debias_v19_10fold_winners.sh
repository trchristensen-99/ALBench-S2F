#!/bin/bash
# v19: 10-fold ensembles for the two new Pareto winners from v10-v17:
#   1) c170_03 (blocks {0,5} + Sahu 3% + spectral + λ=0.41) → 10 folds
#   2) c12_grid_f10_lam010 (blocks 0-2 + dinuc 10% + cpg_inv λ=0.10) → 10 folds
# Both have better metric profiles than current c91 10-fold in single-fold;
# need 10-fold to confirm.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv
SAHU=$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv

for ORACLE_NAME in "c170_03_10fold" "c12_grid_f10_lam010_10fold"; do
    OUT=$REPO/outputs/oracle_neg_sweep/debias_${ORACLE_NAME}
    mkdir -p $OUT
    CFG=$OUT/configs.json
    if [ "$ORACLE_NAME" = "c170_03_10fold" ]; then
        TSV=$SAHU; FRAC=0.03; MODE="spectral"; LAM=0.41; BLOCKS="[0,5]"
    else
        TSV=$DINUC; FRAC=0.10; MODE="cpg_invariance"; LAM=0.10; BLOCKS="[0,1,2]"
    fi

    uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = []
for fold in range(10):
    configs.append({
        "label": f"$ORACLE_NAME""_fold_{fold}",
        "fold_id": fold, "n_folds": 10,
        "stage1_dir": f"$REPO/outputs/oracle_full_856k/s1/oracle_{fold}",
        "output_dir": f"outputs/oracle_neg_sweep/debias_$ORACLE_NAME/fold_{fold}",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path=$TSV",
            f"++neg_fraction=$FRAC",
            f"++debias_mode=$MODE",
            f"++debias_lambda=$LAM",
            f"++unfreeze_encoder_blocks=$BLOCKS",
        ],
    })
batches = [configs[0:3], configs[3:6], configs[6:8], configs[8:10]]
for i, b in enumerate(batches):
    Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(b, indent=2))
print(f"  ${ORACLE_NAME}: split into 4 batches: {[len(b) for b in batches]}")
PYEOF

    for tag in 0 1 2 3; do
        SCRIPT=$(mktemp)
        {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_${ORACLE_NAME}_b${tag}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=12:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo "export PYTHONPATH=\"\$PWD\""
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT/configs_b${tag}.json 3"
        } > $SCRIPT
        JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT 2>&1)
        rm -f $SCRIPT
        echo "  ${ORACLE_NAME}_b${tag}: $JID"
    done
done
