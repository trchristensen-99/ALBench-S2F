#!/bin/bash
# HP local refinement around CURRENT BEST configurations identified by audit.
# Triangulates around best (arch, D) cell using ±1 step in lr/BS/dropout.
#
# Best per (arch, D) (from audit):
#   LegNet:    D=2k:    lr=0.003, BS=512, wd=0.1, aug=rev_c
#              D=30k:   lr=0.001, BS=256
#              D=600k:  lr=0.003, BS=512, aug=none
#   DREAM-RNN: D=2k:    lr=0.003, BS=256, dropout_cnn=0.2, dropout_lstm=0.3
#              D=30k:   lr=0.001, BS=128, dropout_lstm=0.15
#              D=600k:  lr=0.003, BS=256, dropout_lstm=0.0
#
# For each best, sample HPs nearby on a finer grid:
#   lr × {0.7, 1.0, 1.5}
#   BS × {0.5, 1, 2}
#   dropout ∈ {best ± 0.05}

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_local_refine
mkdir -p $OUT

uv run --no-sync python <<PYEOF
import json
import itertools
from pathlib import Path

# (arch, d, best_lr, best_bs, best_aug, [extra_hps])
BEST = [
    ("legnet",    2000,   0.003, 512,  "rev_complement", {"weight_decay": 0.1}),
    ("legnet",    30000,  0.001, 256,  "rev_complement", {"weight_decay": 0.1}),
    ("legnet",    600000, 0.003, 512,  "none",            {"weight_decay": 0.1}),
    ("dream_rnn", 2000,   0.003, 256,  "rev_complement", {"dropout_cnn": 0.2, "dropout_lstm": 0.3}),
    ("dream_rnn", 30000,  0.001, 128,  "rev_complement", {"dropout_cnn": 0.2, "dropout_lstm": 0.15}),
    ("dream_rnn", 600000, 0.003, 256,  "rev_complement", {"dropout_cnn": 0.2, "dropout_lstm": 0.0}),
]

cells = {"legnet": [], "dream_rnn": []}
SEEDS = [42, 123]

for arch, d, best_lr, best_bs, aug, base in BEST:
    # lr × {0.7, 1.0, 1.5}, BS × {0.5, 1, 2}
    lr_grid = [round(best_lr * f, 5) for f in (0.7, 1.0, 1.5)]
    bs_grid = sorted({max(32, int(best_bs * 0.5)), best_bs, min(2048, best_bs * 2)})
    for lr, bs in itertools.product(lr_grid, bs_grid):
        if lr == best_lr and bs == best_bs:
            # Already tested — skip (or keep for variance)
            continue
        for seed in SEEDS:
            hp_str = f"lr{lr}_bs{bs}"
            label = f"{arch}_d{d}_{hp_str}_s{seed}"
            overrides = [f"lr={lr}", f"batch_size={bs}"]
            for k, v in base.items():
                overrides.append(f"{k}={v}")
            cells[arch].append({
                "label": label,
                "arch": arch,
                "d_train": d,
                "seed": seed,
                "epochs": 80 if d >= 30000 else 60,
                "patience": 15,
                "aug": aug,
                "output_dir": f"results/preflight/hp_local_refine/{label}",
                "hp_overrides": overrides,
            })

for arch, c in cells.items():
    Path(f"$OUT/configs_{arch}.json").write_text(json.dumps(c, indent=2))
    print(f"  {arch}: {len(c)} cells")
PYEOF

declare -A KP=([legnet]=6 [dream_rnn]=4)

for arch in legnet dream_rnn; do
    K=${KP[$arch]}
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_hp_local_${arch}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=12:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "export TORCHDYNAMO_DISABLE=1"
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_${arch}.json $K"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  hp_local_${arch}: $JID (k_parallel=$K)"
done
