#!/bin/bash
# HP edge v4: newly-emergent edges from latest 685+ runs.
#
# LegNet D=4k:    BS=64 LOW edge → test BS=32
# LegNet D=300k:  lr=0.001 LOW edge → lr {0.0003, 0.0005}; BS=256 LOW edge → BS=128
# LegNet D=600k:  wd=0.1 HIGH edge → test wd {0.3, 0.5}
# DREAM-RNN D=30k: wd=0.001 LOW edge → wd {0.0, 0.0001}; BS=128 LOW edge → BS=64
# DREAM-RNN D=60k: BS=128 LOW edge → BS=64

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_edge_v4
mkdir -p $OUT

uv run --no-sync python <<'PYEOF'
import json
from pathlib import Path

SEEDS = [42, 123]
cells = {"legnet": [], "dream_rnn": []}

def add(arch, d, hp, aug="rev_complement"):
    for seed in SEEDS:
        hp_str = "_".join(f"{k}{v}" for k, v in hp.items())
        label = f"{arch}_d{d}_{hp_str}_s{seed}"
        cells[arch].append({
            "label": label,
            "arch": arch,
            "d_train": d,
            "seed": seed,
            "epochs": 80 if d >= 30000 else 60,
            "patience": 15,
            "aug": aug,
            "output_dir": f"results/preflight/hp_edge_v4/{label}",
            "hp_overrides": [f"{k}={v}" for k, v in hp.items()],
        })

# LegNet D=4k: BS=32
add("legnet", 4000, {"lr": 0.0005, "batch_size": 32, "weight_decay": 0.1})
add("legnet", 4000, {"lr": 0.001,  "batch_size": 32, "weight_decay": 0.1})

# LegNet D=300k: lower lr + lower BS
for lr in (0.0003, 0.0005):
    for bs in (128, 256):
        add("legnet", 300000, {"lr": lr, "batch_size": bs, "weight_decay": 0.1})

# LegNet D=600k: higher wd
for wd in (0.3, 0.5):
    add("legnet", 600000, {"lr": 0.003, "batch_size": 512, "weight_decay": wd})

# DREAM-RNN D=30k: lower wd + BS=64
for wd in (0.0, 0.0001):
    for bs in (64, 128):
        add("dream_rnn", 30000, {"lr": 0.001, "batch_size": bs,
                                  "dropout_cnn": 0.2, "dropout_lstm": 0.3, "weight_decay": wd})

# DREAM-RNN D=60k: BS=64
add("dream_rnn", 60000, {"lr": 0.001, "batch_size": 64,
                          "dropout_cnn": 0.2, "dropout_lstm": 0.3, "weight_decay": 0.01})

for arch, c in cells.items():
    Path(f"results/preflight/hp_edge_v4/configs_{arch}.json").write_text(json.dumps(c, indent=2))
    print(f"  {arch}: {len(c)} cells")
PYEOF

declare -A KP=([legnet]=4 [dream_rnn]=2)
declare -A GPU=([legnet]="v100" [dream_rnn]="v100")

for arch in legnet dream_rnn; do
    K=${KP[$arch]}
    G=${GPU[$arch]}
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_hp_edge_v4_${arch}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:${G}:1 --cpus-per-task=14 --time=12:00:00 --mem=120G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "export TORCHDYNAMO_DISABLE=1"
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_${arch}.json $K"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT 2>&1)
    rm -f $SCRIPT
    echo "  hp_edge_v4_$arch: $JID"
done
