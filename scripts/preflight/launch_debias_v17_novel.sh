#!/bin/bash
# NEW debias approaches (v17):
#   c200: c91 + 3-seed weight averaging (train 3 seeds, average WEIGHTS not predictions)
#         - Different from ensemble averaging; sometimes generalises better
#   c201: c91 with multi-mode debias loss alternation
#         - Different debias_mode per batch (cycle cpg_inv / spectral / conditional)
#         - Hack: epochs 0-25 cpg_inv, 26-50 spectral, 51-80 conditional
#   c202: c91 + EARLY-STOP at half-trained (epochs=40) to preserve OOD better
#   c203: c91 retrained with refined neg-aug labels (pseudo-label round 2):
#         use c91 fold_0 to predict on dinuc neg, replace labels with c91's preds
#   c204: c91 + max_shift=50 (combine with broader training shift)
#   c205: c91 + max_shift=100
#   c206: c91 with bs=128 (current default is bs=64 for AG-S2 — never tested with c91)
#   c207: c91 + extreme high lr (encoder_lr=5e-4 + head_lr=5e-3) — sanity break
#
# 8 cells, single fold, 1 H100, k=3.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v17
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def base(label, seed=1374211415, **extras):
    overrides = [
        f"++negatives_path=$DINUC",
        "++neg_fraction=0.03",
        "++debias_mode=cpg_invariance",
        "++debias_lambda=0.05",
        "++unfreeze_encoder_blocks=[0,1,2]",
    ]
    for k, v in extras.items():
        overrides.append(f"++{k}={v}")
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v17/{label}/fold_0",
        "epochs": extras.get("epochs", 80),
        "patience": 15,
        "seed": seed,
        "extra_overrides": overrides,
    }

configs = [
    # Multi-seed for weight averaging: train 3 with different seeds
    base("c200_seed_42",  seed=42),
    base("c200_seed_123", seed=123),
    base("c200_seed_999", seed=999),
    # Early stop variant
    base("c202_ep40", epochs=40),
    # Larger train shift coupling
    base("c204_ms50",  max_shift=50),
    base("c205_ms100", max_shift=100),
    # Larger BS
    base("c206_bs128", batch_size=128),
    # Extreme HP — sanity break
    base("c207_aggressive", encoder_lr=5e-4, head_lr=5e-3),
]

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v17 configs")
PYEOF

# Single H100 fast job, k=3, ~50min × ceil(8/3)=3 sequential = ~2.5h
SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v17_novel"
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
echo "  v17 novel: $JID (8 cells, k=3 parallel, ~2.5h)"
