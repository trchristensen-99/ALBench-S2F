#!/bin/bash
# Fill scaling-curve gap at D=300k.
# Currently: D coverage = {500, 1k, 2k, 4k, 6k, 10k, 30k, 60k, 100k, 600k}.
# D=300k is missing (between 100k and 600k — important for scaling law fit).
#
# Per arch: 6 HP samples × 2 seeds = 12 cells. 3 archs × 12 = 36 cells total.
# Use V100 nodes for LegNet/DREAM-RNN (idle, free), H100 for DREAM-ATTN.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_d300k
mkdir -p $OUT

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

# Use known good HPs from audit + a couple of variations
LEGNET_CFGS = [
    {"lr": 0.003, "batch_size": 512, "weight_decay": 0.1},
    {"lr": 0.001, "batch_size": 256, "weight_decay": 0.1},
    {"lr": 0.003, "batch_size": 256, "weight_decay": 0.1},
    {"lr": 0.005, "batch_size": 512, "weight_decay": 0.1},
    {"lr": 0.001, "batch_size": 512, "weight_decay": 0.1},
    {"lr": 0.003, "batch_size": 1024, "weight_decay": 0.1},
]
DRNN_CFGS = [
    {"lr": 0.003, "batch_size": 256, "dropout_cnn": 0.2, "dropout_lstm": 0.0},
    {"lr": 0.003, "batch_size": 256, "dropout_cnn": 0.2, "dropout_lstm": 0.15},
    {"lr": 0.001, "batch_size": 256, "dropout_cnn": 0.2, "dropout_lstm": 0.0},
    {"lr": 0.003, "batch_size": 128, "dropout_cnn": 0.2, "dropout_lstm": 0.0},
    {"lr": 0.005, "batch_size": 256, "dropout_cnn": 0.2, "dropout_lstm": 0.0},
    {"lr": 0.003, "batch_size": 512, "dropout_cnn": 0.2, "dropout_lstm": 0.0},
]
DATTN_CFGS = [
    {"lr": 0.0003, "batch_size": 64,  "weight_decay": 0.01},
    {"lr": 0.0003, "batch_size": 128, "weight_decay": 0.01},
    {"lr": 0.001,  "batch_size": 64,  "weight_decay": 0.01},
    {"lr": 0.0001, "batch_size": 64,  "weight_decay": 0.01},
    {"lr": 0.0003, "batch_size": 128, "weight_decay": 0.001},
    {"lr": 0.0003, "batch_size": 64,  "weight_decay": 0.1},
]

D = 300000
SEEDS = [42, 123]

cells = {"legnet": [], "dream_rnn": [], "dream_attn": []}
for arch, cfgs, aug in [
    ("legnet", LEGNET_CFGS, "rev_complement"),
    ("dream_rnn", DRNN_CFGS, "rev_complement"),
    ("dream_attn", DATTN_CFGS, "rev_complement"),
]:
    for hp in cfgs:
        for seed in SEEDS:
            hp_str = "_".join(f"{k}{v}" for k, v in hp.items())
            label = f"{arch}_d{D}_{hp_str}_s{seed}"
            cells[arch].append({
                "label": label,
                "arch": arch,
                "d_train": D,
                "seed": seed,
                "epochs": 80,
                "patience": 15,
                "aug": aug,
                "output_dir": f"results/preflight/hp_d300k/{label}",
                "hp_overrides": [f"{k}={v}" for k, v in hp.items()],
            })

for arch, c in cells.items():
    Path(f"$OUT/configs_{arch}.json").write_text(json.dumps(c, indent=2))
    print(f"  {arch}: {len(c)} cells")
PYEOF

# LegNet/DREAM-RNN on V100 (idle), DREAM-ATTN on H100 (faster for attention)
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
        echo "#SBATCH --job-name=pf_hp_d300k_${arch}"
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
    echo "  hp_d300k_$arch: $JID (gpu:$G, k_parallel=$K, qos=$Q)"
done
