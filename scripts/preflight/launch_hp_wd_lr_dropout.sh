#!/bin/bash
# NEW: WD × LR coupling at d=600k for LegNet, and Dropout × LR for DREAM-RNN.
# WD has always been 0.1 for LegNet — never co-varied with LR. Same for DREAM-RNN dropout.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_wd_lr_dropout
mkdir -p $OUT

uv run --no-sync python <<'PYEOF'
import json
from pathlib import Path

cells = {"legnet": [], "dream_rnn": []}
SEEDS = [42, 123]

# LegNet WD × LR at d=600k
for lr in (0.001, 0.003, 0.005):
    for wd in (0.001, 0.01, 0.1, 0.3):
        for seed in SEEDS:
            cells["legnet"].append({
                "label": f"legnet_d600k_lr{lr}_wd{wd}_s{seed}",
                "arch": "legnet",
                "d_train": 600000,
                "seed": seed,
                "epochs": 80,
                "patience": 15,
                "aug": "none",
                "output_dir": f"results/preflight/hp_wd_lr_dropout/legnet_d600k_lr{lr}_wd{wd}_s{seed}",
                "hp_overrides": [f"lr={lr}", "batch_size=512", f"weight_decay={wd}"],
            })

# DREAM-RNN dropout_lstm × LR at d=600k
for lr in (0.001, 0.003, 0.005):
    for dl in (0.0, 0.1, 0.2, 0.3):
        for seed in SEEDS:
            cells["dream_rnn"].append({
                "label": f"drnn_d600k_lr{lr}_dl{dl}_s{seed}",
                "arch": "dream_rnn",
                "d_train": 600000,
                "seed": seed,
                "epochs": 80,
                "patience": 15,
                "aug": "rev_complement",
                "output_dir": f"results/preflight/hp_wd_lr_dropout/drnn_d600k_lr{lr}_dl{dl}_s{seed}",
                "hp_overrides": [f"lr={lr}", "batch_size=256",
                                 "dropout_cnn=0.2", f"dropout_lstm={dl}",
                                 "weight_decay=0.01"],
            })

for arch, c in cells.items():
    Path(f"results/preflight/hp_wd_lr_dropout/configs_{arch}.json").write_text(json.dumps(c, indent=2))
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
        echo "#SBATCH --job-name=pf_hp_wdlr_${arch}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=default --gres=gpu:${G}:1 --cpus-per-task=14 --time=12:00:00 --mem=120G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "export TORCHDYNAMO_DISABLE=1"
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_${arch}.json $K"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  hp_wdlr_$arch: $JID (gpu:$G k=$K)"
done
