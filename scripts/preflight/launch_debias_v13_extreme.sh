#!/bin/bash
# Debias v13: extreme λ values + stacked-source neg-aug.
#
# c136: c91 + λ=0.50  (push regularization HARD)
# c137: c91 + λ=1.00  (extreme; expect OOD damage but max bias kill)
# c138: c91 with stacked TSV (50% dinuc + 30% Sahu + 20% ctrl_neg) at 3%
# c139: stacked TSV at 5%
# c140: stacked TSV at 7%
# c141: c91 + dropout=0.5 (very high head reg)
# c142: c91 + dinuc 3% + λ=0.05 + cpg_inv WITH 0.001 wd (lighter than c106)

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# Build combined TSV first
if [ ! -f "$REPO/data/synthetic_negatives_combined/dinuc_sahu_ctrl_combined.tsv" ]; then
    echo "=== Building combined negative TSV ==="
    uv run --no-sync python scripts/preflight/_make_combined_negative_tsv.py
fi

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v13
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv
COMBINED=$REPO/data/synthetic_negatives_combined/dinuc_sahu_ctrl_combined.tsv

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, frac, lam, tsv=None, blocks="[0,1,2]", dropout=None, wd=None):
    if tsv is None:
        tsv = "$DINUC"
    extras = [
        f"++negatives_path={tsv}",
        f"++neg_fraction={frac}",
        "++debias_mode=cpg_invariance",
        f"++debias_lambda={lam}",
        f"++unfreeze_encoder_blocks={blocks}",
    ]
    if dropout is not None:
        extras.append(f"++dropout_rate={dropout}")
    if wd is not None:
        extras.append(f"++weight_decay={wd}")
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v13/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": extras,
    }

configs = [
    cfg("c136_dinuc3_lam50",       0.03, 0.50),
    cfg("c137_dinuc3_lam100",      0.03, 1.00),
    cfg("c138_combined3_lam5",     0.03, 0.05, tsv="$COMBINED"),
    cfg("c139_combined5_lam5",     0.05, 0.05, tsv="$COMBINED"),
    cfg("c140_combined7_lam5",     0.07, 0.05, tsv="$COMBINED"),
    cfg("c141_dinuc3_drop50",      0.03, 0.05, dropout=0.5),
    cfg("c142_dinuc3_lam5_wd1e3_l", 0.03, 0.05, wd=1e-3),  # already similar to c106 but explicit
]

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v13 configs")
PYEOF

# Submit single H100 job, k=3, ~50min × 3 sequential = ~2.5h
SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v13"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=fast --gres=gpu:h100:1 --cpus-per-task=14 --time=03:30:00 --mem=200G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo 'export PYTHONPATH="$PWD"'
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT/configs.json 3"
} > $SCRIPT
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  v13: $JID (7 cells, k=3 parallel)"
