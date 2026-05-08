#!/bin/bash
# HP edge v3: address remaining edges identified by latest audit (666 results).
#
# Remaining edges:
#   DREAM-ATTN: BS=64 still LOW edge across nearly all D → test BS=32
#   DREAM-RNN  D=30k: dropout_cnn=0.2 LOW edge → test {0.3, 0.4, 0.5}
#                     weight_decay=0.01 LOW edge → test {0.001, 0.005}
#   DREAM-RNN  D=60k: BS=128 LOW edge → test BS=64
#   LegNet     D=100k: lr=0.003 HIGH edge → test {0.005, 0.01}
#   LegNet     D=10k:  BS=512 HIGH edge → test BS=1024
#   LegNet     D=4k:   lr=0.0005 LOW edge → test {0.0001, 0.0003}
#   LegNet     D=60k:  BS=256 LOW edge → test {64, 128}
#
# ~46 cells split by arch.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_edge_v3
mkdir -p $OUT

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

SEEDS = [42, 123]

cells = {"legnet": [], "dream_rnn": [], "dream_attn": []}

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
            "output_dir": f"results/preflight/hp_edge_v3/{label}",
            "hp_overrides": [f"{k}={v}" for k, v in hp.items()],
        })

# DREAM-ATTN BS=32 across all D values
for d in (500, 1000, 2000, 4000, 5000, 6000, 30000, 60000, 100000, 600000):
    add("dream_attn", d, {"lr": 0.0003, "batch_size": 32, "weight_decay": 0.01})

# DREAM-RNN D=30k: dropout_cnn × wd
for dc in (0.3, 0.4, 0.5):
    for wd in (0.001, 0.005, 0.01):
        add("dream_rnn", 30000, {"lr": 0.001, "batch_size": 128,
                                 "dropout_cnn": dc, "dropout_lstm": 0.15,
                                 "weight_decay": wd})

# DREAM-RNN D=60k: BS=64
add("dream_rnn", 60000, {"lr": 0.001, "batch_size": 64,
                          "dropout_cnn": 0.2, "dropout_lstm": 0.3,
                          "weight_decay": 0.01})

# LegNet D=100k: high lr
for lr in (0.005, 0.01):
    add("legnet", 100000, {"lr": lr, "batch_size": 512, "weight_decay": 0.1})

# LegNet D=10k: BS=1024
add("legnet", 10000, {"lr": 0.001, "batch_size": 1024, "weight_decay": 0.1})

# LegNet D=4k: lower lr
for lr in (0.0001, 0.0003):
    add("legnet", 4000, {"lr": lr, "batch_size": 128, "weight_decay": 0.1})

# LegNet D=60k: smaller BS
for bs in (64, 128):
    add("legnet", 60000, {"lr": 0.003, "batch_size": bs, "weight_decay": 0.1})

for arch, c in cells.items():
    Path(f"$OUT/configs_{arch}.json").write_text(json.dumps(c, indent=2))
    print(f"  {arch}: {len(c)} cells")
PYEOF

# LegNet, DREAM-RNN on V100 (free), DREAM-ATTN on H100
declare -A KP=([legnet]=4 [dream_rnn]=2 [dream_attn]=4)
declare -A GPU=([legnet]="v100" [dream_rnn]="v100" [dream_attn]="h100")
declare -A QOS=([legnet]="default" [dream_rnn]="default" [dream_attn]="default")

for arch in legnet dream_rnn dream_attn; do
    K=${KP[$arch]}
    G=${GPU[$arch]}
    Q=${QOS[$arch]}
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_hp_edge_v3_${arch}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=$Q --gres=gpu:${G}:1 --cpus-per-task=14 --time=12:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "export TORCHDYNAMO_DISABLE=1"
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_${arch}.json $K"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  hp_edge_v3_$arch: $JID (gpu:$G, k=$K, qos=$Q)"
done
