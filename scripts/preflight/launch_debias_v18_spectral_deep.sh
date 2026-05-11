#!/bin/bash
# v18: spectral debias deep-dive based on v15 finding (c170_03):
#   blocks {0,5} + Sahu 3% + spectral + λ=0.41 → bias 0.355, OOD 0.770, test_id 0.948
# Also c12_grid_f10_lam010 (dinuc 10%) and c103 (dinuc 10%) hit strong bias.
#
# Cells:
#   c220-c224: c170_03 variants (λ sweep, fraction sweep, freeze sweep)
#   c225-c228: high-dinuc-fraction × λ (10% / 15% / 20%)
#   c229-c231: blocks {0,5} with dinuc instead of Sahu
#   c232: spectral λ=0.20 (intermediate to c170_03)

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v18
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv
SAHU=$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, tsv, frac, mode, lam, blocks):
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v18/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path={tsv}",
            f"++neg_fraction={frac}",
            f"++debias_mode={mode}",
            f"++debias_lambda={lam}",
            f"++unfreeze_encoder_blocks={blocks}",
        ],
    }

configs = [
    # c170_03 variants (spectral + blocks {0,5})
    cfg("c220_blk05_sahu3_spectral_lam20",     "$SAHU", 0.03, "spectral", 0.20, "[0,5]"),
    cfg("c221_blk05_sahu3_spectral_lam60",     "$SAHU", 0.03, "spectral", 0.60, "[0,5]"),
    cfg("c222_blk05_sahu5_spectral_lam41",     "$SAHU", 0.05, "spectral", 0.41, "[0,5]"),
    cfg("c223_blk05_sahu7_spectral_lam41",     "$SAHU", 0.07, "spectral", 0.41, "[0,5]"),
    cfg("c224_blk04_sahu3_spectral_lam41",     "$SAHU", 0.03, "spectral", 0.41, "[0,4]"),
    cfg("c225_blk_only0_sahu3_spectral_lam41", "$SAHU", 0.03, "spectral", 0.41, "[0]"),
    # blocks {0,5} with dinuc (not Sahu)
    cfg("c229_blk05_dinuc3_spectral_lam41",    "$DINUC", 0.03, "spectral", 0.41, "[0,5]"),
    cfg("c230_blk05_dinuc5_spectral_lam41",    "$DINUC", 0.05, "spectral", 0.41, "[0,5]"),
    cfg("c231_blk05_dinuc10_spectral_lam41",   "$DINUC", 0.10, "spectral", 0.41, "[0,5]"),
    # High-fraction dinuc with cpg_inv (c103 territory)
    cfg("c226_blk012_dinuc15_cpginv",          "$DINUC", 0.15, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c227_blk012_dinuc20_cpginv",          "$DINUC", 0.20, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c228_blk012_dinuc10_cpginv_lam10",    "$DINUC", 0.10, "cpg_invariance", 0.10, "[0,1,2]"),
    # spectral with c91-style blocks (0,1,2) for comparison
    cfg("c232_blk012_dinuc3_spectral_lam41",   "$DINUC", 0.03, "spectral", 0.41, "[0,1,2]"),
    cfg("c233_blk012_dinuc3_spectral_lam20",   "$DINUC", 0.03, "spectral", 0.20, "[0,1,2]"),
]

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v18 configs")
PYEOF

# Split into 2 batches of 7
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$OUT/configs.json").read_text())
half = len(configs) // 2
Path("$OUT/configs_b0.json").write_text(json.dumps(configs[:half], indent=2))
Path("$OUT/configs_b1.json").write_text(json.dumps(configs[half:], indent=2))
print(f"  split: {half} + {len(configs)-half}")
PYEOF

for tag in 0 1; do
    SCRIPT=$(mktemp)
    {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v18_b${tag}"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=06:00:00 --mem=200G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo "export PYTHONPATH=\"\$PWD\""
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT 2>&1)
    rm -f $SCRIPT
    echo "  v18_b${tag}: $JID"
done
