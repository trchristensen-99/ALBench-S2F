#!/bin/bash
# Debias v12: 2D λ × fraction grid around c91 + novel freeze variants.
#
# c91 recipe: blocks 0-2 + dinuc N% + cpg_invariance λ=L.
# Already tested in v9/v10: (frac, λ) = (1%, 0.05), (1.5%, 0.05), (3%, 0.05),
# (3%, 0.10), (3%, 0.20), (5%, 0.05), (7%, 0.05), (10%, 0.05).
#
# v12 fills in the 2D grid:
#   λ=0.02 column: (1%, 3%, 5%, 7%) — never tested
#   λ=0.10:        (1%, 2%, 5%, 7%, 10%) — fills middle column
#   λ=0.20:        (1%, 2%, 5%, 7%, 10%) — pushes high λ
#
# Plus novel freeze patterns + dropout variants:
#   c130: block 0 only (most aggressive freeze)
#   c131: block 2 only (skip early)
#   c132: blocks {0,2} (skip middle)
#   c133: blocks {1,2} (skip earliest)
#   c134: c91 + dropout=0.0 (head dropout)
#   c135: c91 + dropout=0.3 (more head reg)
#
# Total: 14 grid + 6 variants = 20 cells.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v12
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, frac, lam, blocks="[0,1,2]", dropout=None):
    extras = [
        f"++negatives_path=$DINUC",
        f"++neg_fraction={frac}",
        "++debias_mode=cpg_invariance",
        f"++debias_lambda={lam}",
        f"++unfreeze_encoder_blocks={blocks}",
    ]
    if dropout is not None:
        extras.append(f"++dropout_rate={dropout}")
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v12/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": extras,
    }

configs = []
# 2D grid: λ × fraction
# λ=0.02 column (never tested)
for frac in (0.01, 0.03, 0.05, 0.07):
    configs.append(cfg(f"c12_grid_f{int(frac*100)}_lam002", frac, 0.02))
# λ=0.10 column (fill middle)
for frac in (0.01, 0.02, 0.05, 0.07, 0.10):
    configs.append(cfg(f"c12_grid_f{int(frac*100)}_lam010", frac, 0.10))
# λ=0.20 column (push high)
for frac in (0.01, 0.02, 0.05, 0.07, 0.10):
    configs.append(cfg(f"c12_grid_f{int(frac*100)}_lam020", frac, 0.20))

# Freeze pattern variants (c91 base recipe: dinuc 3% + cpg_inv λ=0.05)
configs.append(cfg("c130_blk0_only",   0.03, 0.05, blocks="[0]"))
configs.append(cfg("c131_blk2_only",   0.03, 0.05, blocks="[2]"))
configs.append(cfg("c132_blk0and2",    0.03, 0.05, blocks="[0,2]"))
configs.append(cfg("c133_blk1and2",    0.03, 0.05, blocks="[1,2]"))
configs.append(cfg("c134_drop0",       0.03, 0.05, dropout=0.0))
configs.append(cfg("c135_drop3",       0.03, 0.05, dropout=0.3))

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v12 configs (14 grid + 6 freeze/dropout)")
PYEOF

# Split across 3 jobs (~7 cells each)
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$OUT/configs.json").read_text())
batches = [configs[0:7], configs[7:14], configs[14:]]
for i, b in enumerate(batches):
    Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(b, indent=2))
print(f"  split into 3 batches: {[len(b) for b in batches]}")
PYEOF

for tag in 0 1 2; do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_debias_v12_b${tag}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=08:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  v12_b${tag}: $JID"
done
