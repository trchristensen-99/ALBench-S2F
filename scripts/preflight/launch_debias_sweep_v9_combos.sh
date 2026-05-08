#!/bin/bash
# Debias sweep v9: combinations of effective signals from v8 + earlier work.
#
# Effective signals identified:
#   - blocks 0-2 unfreeze (c86): best test/OOD/SNV, ~5% bias reduction
#   - wd=1e-3 (c85): -14% bias, OOD=0.745
#   - dinuc 3% + cpg_inv 0.05 (c28): -41% bias, OOD=0.74
#   - Sahu 3% + cpg_inv 0.05 (c63): -14% bias, OOD=0.75 (better OOD than c28)
#
# Combinations to test:
#   c91: blocks 0-2 + dinuc 3% + cpg_inv 0.05 (replace Sahu with dinuc in c86)
#   c92: blocks 0-2 + Sahu 3% + cpg_inv 0.05 + wd=1e-3 (add high WD)
#   c93: blocks 0-2 + cpg_inv 0.05 (no neg-aug, freeze + loss debias only)
#   c94: blocks 0-1 unfreeze + Sahu 3% + cpg_inv 0.05 (more aggressive freeze)
#   c95: blocks 0-2 + Sahu 5% + cpg_inv 0.05 (more neg-aug)
#   c96: blocks 0-2 + Sahu 3% + cpg_inv 0.10 (higher λ)
#   c97: blocks 0-2 + dinuc 5% + cpg_inv 0.05 (push bias reduction further)
#   c98: blocks 0,1,2,3 unfreeze + Sahu 3% + cpg_inv 0.05 (one more block, sanity check)
#   c99: blocks 0-2 + dinuc 3% + cpg_inv 0.10 + wd=1e-4 (combo of multiple effective signals)
#
# 9 cells, AG-S2 k_parallel=3 → ~3 sequential runs of ~50 min each = ~2.5h.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/outputs/oracle_neg_sweep/debias_sweep_v9
mkdir -p $OUT_BASE
CFG=$OUT_BASE/configs.json
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv
SAHU=$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, frac, tsv, lam, blocks="[0,1,2]", wd=None, lambda_val=None):
    extras = [
        f"++negatives_path={tsv}",
        f"++neg_fraction={frac}",
        "++debias_mode=cpg_invariance",
        f"++debias_lambda={lam}",
        f"++unfreeze_encoder_blocks={blocks}",
    ]
    if wd is not None:
        extras.append(f"++weight_decay={wd}")
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v9/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": extras,
    }

configs = [
    cfg("c91_blk012_dinuc3_cpginv",       0.03, "$DINUC", 0.05),
    cfg("c92_blk012_sahu3_cpginv_wd1e3",  0.03, "$SAHU",  0.05, wd=1e-3),
    cfg("c94_blk01_sahu3_cpginv",         0.03, "$SAHU",  0.05, blocks="[0,1]"),
    cfg("c95_blk012_sahu5_cpginv",        0.05, "$SAHU",  0.05),
    cfg("c96_blk012_sahu3_cpginv_lam10",  0.03, "$SAHU",  0.10),
    cfg("c97_blk012_dinuc5_cpginv",       0.05, "$DINUC", 0.05),
    cfg("c98_blk0123_sahu3_cpginv",       0.03, "$SAHU",  0.05, blocks="[0,1,2,3]"),
    cfg("c99_blk012_dinuc3_cpginv_lam10_wd1e4", 0.03, "$DINUC", 0.10, wd=1e-4),
]
# c93: blocks 0-2 + cpg_inv 0.05, NO neg-aug
configs.append({
    "label": "c93_blk012_cpginv_only",
    "fold_id": 0, "n_folds": 10,
    "stage1_dir": "$S1",
    "output_dir": "outputs/oracle_neg_sweep/debias_sweep_v9/c93_blk012_cpginv_only/fold_0",
    "epochs": 80, "patience": 15,
    "extra_overrides": [
        "++debias_mode=cpg_invariance",
        "++debias_lambda=0.05",
        "++unfreeze_encoder_blocks=[0,1,2]",
    ],
})

Path("$CFG").write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v9 configs")
PYEOF

# Split into 3 batches (3 cells each) for parallel scheduling
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$CFG").read_text())
batches = [configs[0:3], configs[3:6], configs[6:]]
for i, b in enumerate(batches):
    Path(f"$OUT_BASE/configs_b{i}.json").write_text(json.dumps(b, indent=2))
print(f"  split into 3 batches: {[len(b) for b in batches]}")
PYEOF

for tag in 0 1 2; do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_debias_v9_b${tag}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=06:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT_BASE/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  v9_b${tag}: $JID"
done
