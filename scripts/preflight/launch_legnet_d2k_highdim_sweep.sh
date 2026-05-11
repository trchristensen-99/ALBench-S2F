#!/bin/bash
# Focused high-dimensional HP sweep at LegNet D=2000.
# This is the highest-coverage low-D LegNet cell (44 existing configs),
# but DROPOUT was never explicitly varied and AUG was always rev_complement.
#
# 6 HP dimensions explored:
#   LR (anchored at best 0.0005)
#   BS (anchored at best 128)
#   WD (anchored at best 0.1)
#   Dropout — NEW dimension {0.0, 0.1, 0.3, 0.5}
#   Aug — NEW dimension {none, rev_complement, rc_shift, rc_shift_evoaug}
#   Capacity (LegNet block_sizes: narrow/default/wide) — NEW dimension
#   Epochs — NEW dimension {30, 60, 90, 150}
#
# Cells:
#   - 2D heatmap: Dropout × Aug (4 × 4 × 2 seeds = 32 cells, main figure)
#   - 1D walk: Capacity (narrow/default/wide × 2 seeds = 6 cells)
#   - 1D walk: Epochs (30/60/90/150 × 1 seed = 4 cells)
#
# Total: 42 cells, fits in fast queue (3.5h × k_parallel=6 V100 = 21 cell-hours).

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/legnet_d2k_highdim
mkdir -p $OUT

uv run --no-sync python <<'PYEOF'
import json
from pathlib import Path

D = 2000
SEEDS = [42, 123]
ANCHOR_LR = 0.0005
ANCHOR_BS = 128
ANCHOR_WD = 0.1

cells = []

# === 2D heatmap: Dropout × Aug ===
DROPOUTS = [0.0, 0.1, 0.3, 0.5]
AUGS = ["none", "rev_complement", "rc_shift", "rc_shift_evoaug"]
for d_val in DROPOUTS:
    for aug in AUGS:
        for seed in SEEDS:
            label = f"d2k_drop{d_val}_aug{aug}_s{seed}"
            cells.append({
                "label": label,
                "arch": "legnet",
                "d_train": D,
                "seed": seed,
                "epochs": 80,
                "patience": 15,
                "aug": aug,
                "output_dir": f"results/preflight/legnet_d2k_highdim/heatmap_drop_aug/{label}",
                "hp_overrides": [
                    f"lr={ANCHOR_LR}",
                    f"batch_size={ANCHOR_BS}",
                    f"weight_decay={ANCHOR_WD}",
                    f"dropout={d_val}",
                ],
            })

# === 1D walk: Capacity (LegNet sizes) ===
# narrow / default / wide (block_sizes from legnet_arch_sweep)
SIZES = {
    "narrow":  "[128,128,64,64,32,32,16,16]",
    "default": "[256,256,128,128,64,64,32,32]",
    "wide":    "[512,512,256,256,128,128,64,64]",
}
for size_name, block_sizes in SIZES.items():
    for seed in SEEDS:
        label = f"d2k_size{size_name}_s{seed}"
        cells.append({
            "label": label,
            "arch": "legnet",
            "d_train": D,
            "seed": seed,
            "epochs": 80,
            "patience": 15,
            "aug": "rev_complement",
            "output_dir": f"results/preflight/legnet_d2k_highdim/walk_capacity/{label}",
            "hp_overrides": [
                f"lr={ANCHOR_LR}",
                f"batch_size={ANCHOR_BS}",
                f"weight_decay={ANCHOR_WD}",
                f"block_sizes={block_sizes}",
            ],
        })

# === 1D walk: Epochs ===
for epochs in [30, 60, 90, 150]:
    label = f"d2k_ep{epochs}_s42"
    cells.append({
        "label": label,
        "arch": "legnet",
        "d_train": D,
        "seed": 42,
        "epochs": epochs,
        "patience": max(15, epochs // 5),
        "aug": "rev_complement",
        "output_dir": f"results/preflight/legnet_d2k_highdim/walk_epochs/{label}",
        "hp_overrides": [
            f"lr={ANCHOR_LR}",
            f"batch_size={ANCHOR_BS}",
            f"weight_decay={ANCHOR_WD}",
        ],
    })

CFG = "results/preflight/legnet_d2k_highdim/configs.json"
Path(CFG).write_text(json.dumps(cells, indent=2))
print(f"  wrote {len(cells)} cells:")
print(f"    Dropout × Aug 2D heatmap: 32")
print(f"    Capacity 1D walk: 6")
print(f"    Epochs 1D walk: 4")
PYEOF

# Split into 2 batches to use both fast slots
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
cells = json.loads(Path("$OUT/configs.json").read_text())
half = len(cells) // 2
Path("$OUT/configs_b0.json").write_text(json.dumps(cells[:half], indent=2))
Path("$OUT/configs_b1.json").write_text(json.dumps(cells[half:], indent=2))
print(f"  split: {half} + {len(cells)-half}")
PYEOF

for tag in 0 1; do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_legnet_d2k_hd_b${tag}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=fast --gres=gpu:v100:1 --cpus-per-task=14 --time=03:30:00 --mem=120G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "export TORCHDYNAMO_DISABLE=1"
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_b${tag}.json 6"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  legnet_d2k_highdim_b${tag}: $JID"
done
