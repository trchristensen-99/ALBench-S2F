#!/bin/bash
# Debias sweep v8: novel approaches we haven't tried.
#
# Tier 1 ideas (no code changes — configurable via existing knobs):
#   c80: Real ctrl_neg calibration (use Gosai's 503 ctrl_neg as neg-aug
#        with their REAL measured labels — direct calibration on episomal data)
#   c81: c80 + cpg_invariance λ=0.05
#   c82: Lower encoder LR (5e-5) with c63 recipe (gentler encoder updates)
#   c83: Even lower encoder LR (2e-5) with c63 recipe
#   c84: Higher weight_decay (1e-4) with c63 recipe
#   c85: Highest weight_decay (1e-3) with c63 recipe
#   c86: Different freeze pattern: encoder blocks 0-2 only
#   c87: Different freeze pattern: encoder blocks 3-5 only (vs current 4-5)
#   c88: Longer training: 1% Sahu + cpg_inv 0.05 + epochs=150
#   c89: Multi-source neg-aug: real+ctrl_neg combined at 5%
#   c90: Sahu labels + cpg_invariance very high λ=0.30 (push debias hard)
#
# 11 candidates × ~30-45 min each on AG-S2 with k_parallel=3 across 1-2 GPUs.
# Bundle into 2 SLURM jobs if needed for cluster scheduling.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# Generate the ctrl_neg calibration TSV first
if [ ! -f data/synthetic_negatives_calibration/gosai_ctrl_neg_calibration.tsv ]; then
    echo "=== Generating ctrl_neg calibration TSV ==="
    uv run --no-sync python scripts/preflight/_make_ctrl_neg_calibration_tsv.py
fi

OUT_BASE=$REPO/outputs/oracle_neg_sweep/debias_sweep_v8
mkdir -p $OUT_BASE
CFG_PATH=$OUT_BASE/configs.json

S1_DIR="$REPO/outputs/oracle_full_856k/s1/oracle_0"
DINUC_TSV="$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv"
SAHU_TSV="$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv"
CTRLNEG_TSV="$REPO/data/synthetic_negatives_calibration/gosai_ctrl_neg_calibration.tsv"
MIXED_TSV="$REPO/data/synthetic_negatives/dinuc_plus_intergenic.tsv"

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

S1 = "$S1_DIR"
configs = []

def cfg(label, frac, tsv, mode=None, lam=None, enc_lr=None, head_lr=None, wd=None,
        unfreeze_blocks=None, epochs=80, patience=15):
    extra = [f"++negatives_path={tsv}", f"++neg_fraction={frac}"]
    if mode is not None:
        extra.append(f"++debias_mode={mode}")
        extra.append(f"++debias_lambda={lam}")
    if enc_lr is not None:
        extra.append(f"++encoder_lr={enc_lr}")
    if head_lr is not None:
        extra.append(f"++head_lr={head_lr}")
    if wd is not None:
        extra.append(f"++weight_decay={wd}")
    if unfreeze_blocks is not None:
        extra.append(f"++unfreeze_encoder_blocks={unfreeze_blocks}")
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": S1,
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v8/{label}/fold_0",
        "epochs": epochs, "patience": patience,
        "extra_overrides": extra,
    }

# c80: Real ctrl_neg calibration (10% of training, no extra loss)
configs.append(cfg("c80_ctrlneg_calib", 0.10, "$CTRLNEG_TSV"))
# c81: ctrl_neg calibration + cpg_invariance
configs.append(cfg("c81_ctrlneg_calib_cpginv", 0.10, "$CTRLNEG_TSV", "cpg_invariance", 0.05))
# c82-c83: c63 recipe (3% Sahu + cpg_inv) with lower encoder_lr
configs.append(cfg("c82_sahu3_cpginv_enclr5e5", 0.03, "$SAHU_TSV", "cpg_invariance", 0.05, enc_lr=5e-5))
configs.append(cfg("c83_sahu3_cpginv_enclr2e5", 0.03, "$SAHU_TSV", "cpg_invariance", 0.05, enc_lr=2e-5))
# c84-c85: c63 recipe with higher weight_decay
configs.append(cfg("c84_sahu3_cpginv_wd1e4", 0.03, "$SAHU_TSV", "cpg_invariance", 0.05, wd=1e-4))
configs.append(cfg("c85_sahu3_cpginv_wd1e3", 0.03, "$SAHU_TSV", "cpg_invariance", 0.05, wd=1e-3))
# c86-c87: different freeze patterns
configs.append(cfg("c86_sahu3_cpginv_blocks012", 0.03, "$SAHU_TSV", "cpg_invariance", 0.05,
                   unfreeze_blocks="[0,1,2]"))
configs.append(cfg("c87_sahu3_cpginv_blocks35", 0.03, "$SAHU_TSV", "cpg_invariance", 0.05,
                   unfreeze_blocks="[3,4,5]"))
# c88: longer training, very low neg-aug
configs.append(cfg("c88_sahu1_cpginv_150ep", 0.01, "$SAHU_TSV", "cpg_invariance", 0.05,
                   epochs=150, patience=30))
# c89: c80 (ctrlneg calibration) + cpg_invariance + lower fraction
configs.append(cfg("c89_ctrlneg5_cpginv", 0.05, "$CTRLNEG_TSV", "cpg_invariance", 0.05))
# c90: Sahu + cpg_invariance very high λ
configs.append(cfg("c90_sahu3_cpginv_lambda03", 0.03, "$SAHU_TSV", "cpg_invariance", 0.30))

Path("$CFG_PATH").write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v8 configs")
PYEOF

# Submit one fast-queue job + one slow_nice for redundancy
echo
echo "=== Split into 2 batches (5/6 cells each) for parallel execution ==="
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$CFG_PATH").read_text())
half = len(configs) // 2
Path("$OUT_BASE/configs_a.json").write_text(json.dumps(configs[:half], indent=2))
Path("$OUT_BASE/configs_b.json").write_text(json.dumps(configs[half:], indent=2))
print(f"  Split {len(configs)} into {half} + {len(configs)-half}")
PYEOF

for tag in a b; do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_debias_v8_${tag}"
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
        echo "    $OUT_BASE/configs_${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  v8_${tag}: $JID"
done
