#!/bin/bash
# v23: LegNet D=2000 corner extensions + Shift × EvoAug 2D aug grid.
#
# Corners to fill:
#   - WD high edge: 0.3, 0.5 (current best WD=0.1 was at top of {0.001, 0.01, 0.1})
#   - LR low corner: 5e-5, 1e-4 (best 5e-4, very low untested)
#   - LR high corner: 5e-3, 1e-2 (high LR sparse)
#   - BS high corner: 2048 (highest tested was 1024)
#   - Dropout fills: 0.05, 0.15, 0.4 (between current 0.0/0.1/0.3/0.5)
#
# Aug grid: shift × EvoAug
#   - shift ∈ {0, 15, 50, 100}
#   - EvoAug ∈ {off, on}
#   - All include RC (standard)
#   - 8 combos × 2 seeds = 16 cells
#
# Total cells: ~40 (corners) + 16 (aug grid) = 56 cells

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/legnet_d2k_v23
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

def add(label, hp, aug="rev_complement", max_shift=None):
    for seed in SEEDS:
        full_label = f"{label}_s{seed}"
        overrides = [f"{k}={v}" for k, v in hp.items()]
        cell = {
            "label": full_label,
            "arch": "legnet",
            "d_train": D,
            "seed": seed,
            "epochs": 80,
            "patience": 15,
            "aug": aug,
            "output_dir": f"results/preflight/legnet_d2k_v23/{full_label}",
            "hp_overrides": overrides,
        }
        if max_shift is not None:
            cell["max_shift"] = max_shift
        cells.append(cell)

# === Corner: WD high ===
for wd in (0.3, 0.5):
    add(f"d2k_wd{wd}", {"lr": ANCHOR_LR, "batch_size": ANCHOR_BS, "weight_decay": wd})

# === Corner: LR extremes ===
for lr in (5e-5, 1e-4, 5e-3, 1e-2):
    add(f"d2k_lr{lr}", {"lr": lr, "batch_size": ANCHOR_BS, "weight_decay": ANCHOR_WD})

# === Corner: BS high ===
add(f"d2k_bs2048", {"lr": ANCHOR_LR, "batch_size": 2048, "weight_decay": ANCHOR_WD})

# === Dropout fills ===
for d_val in (0.05, 0.15, 0.4):
    add(f"d2k_drop{d_val}_fill", {"lr": ANCHOR_LR, "batch_size": ANCHOR_BS,
                                    "weight_decay": ANCHOR_WD, "dropout": d_val})

# === Aug grid: shift × EvoAug (8 combos) ===
for shift in [0, 15, 50, 100]:
    for evoaug in [False, True]:
        if shift == 0 and not evoaug:
            aug = "rev_complement"
        elif shift > 0 and not evoaug:
            aug = "rc_shift"
        elif shift == 0 and evoaug:
            aug = "rc_shift_evoaug"  # use evoaug with shift=0
        else:
            aug = "rc_shift_evoaug"
        label = f"d2k_aug_s{shift}_evo{int(evoaug)}"
        for seed in SEEDS:
            full_label = f"{label}_s{seed}"
            cell = {
                "label": full_label,
                "arch": "legnet",
                "d_train": D,
                "seed": seed,
                "epochs": 80,
                "patience": 15,
                "aug": aug,
                "max_shift": shift,  # the runner will use this for shift magnitude
                "output_dir": f"results/preflight/legnet_d2k_v23/{full_label}",
                "hp_overrides": [
                    f"lr={ANCHOR_LR}",
                    f"batch_size={ANCHOR_BS}",
                    f"weight_decay={ANCHOR_WD}",
                    f"max_shift={shift}",
                ],
            }
            cells.append(cell)

CFG = "results/preflight/legnet_d2k_v23/configs.json"
Path(CFG).write_text(json.dumps(cells, indent=2))
print(f"  wrote {len(cells)} v23 cells")
PYEOF

# Split into 4 fast-queue batches
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
cfgs = json.loads(Path("$OUT/configs.json").read_text())
n = len(cfgs); b = (n + 3) // 4
for i in range(4):
    start = i * b
    end = min(start + b, n)
    if start < n:
        Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(cfgs[start:end], indent=2))
        print(f"  batch {i}: {end - start} cells")
PYEOF

for tag in 0 1 2 3; do
    SCRIPT=$(mktemp)
    {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_d2k_v23_b${tag}"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=fast --gres=gpu:v100:1 --cpus-per-task=14 --time=03:30:00 --mem=120G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo "export PYTHONPATH=\"\$PWD\""
    echo "export TORCHDYNAMO_DISABLE=1"
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT 2>&1)
    rm -f $SCRIPT
    echo "  d2k_v23_b${tag}: $JID"
done
