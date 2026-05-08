#!/bin/bash
# HP global random search — explore broader HP regions to find potential
# global optima the grid sweeps may have missed.
#
# Strategy: at each scaling-relevant D (2k, 30k, 600k), sample 6 random
# (lr, BS, dropout, wd) tuples per arch covering wider ranges than tested before:
#   LegNet:  lr ∈ [1e-4, 0.02], BS ∈ {64, 128, 256, 512, 1024},
#            wd ∈ {0.001, 0.01, 0.1, 0.3}
#   DREAM-RNN: lr ∈ [1e-4, 0.02], BS ∈ {64, 128, 256, 512},
#              dropout_cnn ∈ [0.0, 0.5], dropout_lstm ∈ [0.0, 0.5]
#   DREAM-ATTN: lr ∈ [1e-4, 0.005], BS ∈ {32, 64, 128, 256},
#               wd ∈ {0.001, 0.01, 0.1}
#
# 3 D × 3 arch × 6 samples × 2 seeds = 108 cells (large but parallel-friendly).
# Run on parallel_gpu_runner with k_parallel=6/4/4.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_global_random
mkdir -p $OUT

uv run --no-sync python <<PYEOF
import json
import random
from pathlib import Path

random.seed(20260508)

LEGNET_LR_LOG  = (-4.0, -1.7)   # 1e-4 to ~0.02
DRNN_LR_LOG    = (-4.0, -1.7)
DATTN_LR_LOG   = (-4.0, -2.3)   # 1e-4 to ~0.005
LEGNET_BS  = [64, 128, 256, 512, 1024]
DRNN_BS    = [64, 128, 256, 512]
DATTN_BS   = [32, 64, 128, 256]
LEGNET_WD  = [0.001, 0.01, 0.1, 0.3]
DRNN_WD    = [0.001, 0.01, 0.1]
DATTN_WD   = [0.001, 0.01, 0.1]

D_LIST = [2000, 30000, 600000]
N_SAMPLES = 6
SEEDS = [42, 123]

def sample_legnet():
    lr = round(10 ** random.uniform(*LEGNET_LR_LOG), 5)
    return {"lr": lr,
            "batch_size": random.choice(LEGNET_BS),
            "weight_decay": random.choice(LEGNET_WD)}

def sample_drnn():
    lr = round(10 ** random.uniform(*DRNN_LR_LOG), 5)
    return {"lr": lr,
            "batch_size": random.choice(DRNN_BS),
            "weight_decay": random.choice(DRNN_WD),
            "dropout_cnn": round(random.uniform(0.0, 0.5), 2),
            "dropout_lstm": round(random.uniform(0.0, 0.5), 2)}

def sample_dattn():
    lr = round(10 ** random.uniform(*DATTN_LR_LOG), 5)
    return {"lr": lr,
            "batch_size": random.choice(DATTN_BS),
            "weight_decay": random.choice(DATTN_WD)}

cells = {"legnet": [], "dream_rnn": [], "dream_attn": []}
for arch, sampler in [("legnet", sample_legnet),
                      ("dream_rnn", sample_drnn),
                      ("dream_attn", sample_dattn)]:
    for d in D_LIST:
        for i in range(N_SAMPLES):
            hp = sampler()
            for seed in SEEDS:
                hp_str = "_".join(f"{k}{v}" for k, v in hp.items())
                label = f"{arch}_d{d}_{hp_str}_s{seed}"
                hp_overrides = [f"{k}={v}" for k, v in hp.items()]
                cells[arch].append({
                    "label": label,
                    "arch": arch,
                    "d_train": d,
                    "seed": seed,
                    "epochs": 80 if d >= 30000 else 60,
                    "patience": 15,
                    "aug": "rev_complement",
                    "output_dir": f"results/preflight/hp_global_random/{label}",
                    "hp_overrides": hp_overrides,
                })

for arch, c in cells.items():
    Path(f"$OUT/configs_{arch}.json").write_text(json.dumps(c, indent=2))
    print(f"  {arch}: {len(c)} cells")
PYEOF

declare -A KP=([legnet]=6 [dream_rnn]=4 [dream_attn]=4)

for arch in legnet dream_rnn dream_attn; do
    K=${KP[$arch]}
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_hp_global_${arch}"
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
    echo "  hp_global_${arch}: $JID (k_parallel=$K)"
done
