#!/bin/bash
# NEW: Architecture size × D coupling.
# task6_parameterization tested arch sizes only at d=600k. Question:
# does the best arch size depend on N? Test S/M/L at d=2k, 30k, 100k.
#
# For LegNet, "size" varies in block_sizes/ks/hidden_dim (see legnet_arch_sweep).
# For DREAM-RNN, size = hidden_dim ∈ {160, 320, 640} (default 320).
# For DREAM-ATTN, size = core_layers ∈ {4, 8, 12} (default 8).
#
# 3 archs × 3 sizes × 3 D × 2 seeds = 54 cells.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_arch_size_x_d
mkdir -p $OUT

uv run --no-sync python <<'PYEOF'
import json
from pathlib import Path

cells = {"legnet": [], "dream_rnn": [], "dream_attn": []}
SEEDS = [42, 123]
DS = [2000, 30000, 100000]

# LegNet: hidden_dim ∈ {64, 256, 1024} (default 256)
for d in DS:
    for hd in (64, 256, 1024):
        for seed in SEEDS:
            cells["legnet"].append({
                "label": f"legnet_d{d}_hd{hd}_s{seed}",
                "arch": "legnet",
                "d_train": d,
                "seed": seed,
                "epochs": 80 if d >= 30000 else 60,
                "patience": 15,
                "aug": "rev_complement",
                "output_dir": f"results/preflight/hp_arch_size_x_d/legnet_d{d}_hd{hd}_s{seed}",
                "hp_overrides": [
                    f"hidden_dim={hd}",
                    f"lr={0.003 if d >= 30000 else 0.001}",
                    f"batch_size={256 if d >= 30000 else 128}",
                    "weight_decay=0.1",
                ],
            })

# DREAM-RNN: hidden_dim ∈ {160, 320, 640}
for d in DS:
    for hd in (160, 320, 640):
        for seed in SEEDS:
            cells["dream_rnn"].append({
                "label": f"drnn_d{d}_hd{hd}_s{seed}",
                "arch": "dream_rnn",
                "d_train": d,
                "seed": seed,
                "epochs": 80 if d >= 30000 else 60,
                "patience": 15,
                "aug": "rev_complement",
                "output_dir": f"results/preflight/hp_arch_size_x_d/drnn_d{d}_hd{hd}_s{seed}",
                "hp_overrides": [
                    f"hidden_dim={hd}",
                    f"cnn_filters={hd//2}",
                    "lr=0.003",
                    "batch_size=256",
                    "dropout_cnn=0.2",
                    "dropout_lstm=0.3",
                    "weight_decay=0.01",
                ],
            })

# DREAM-ATTN: core_layers ∈ {4, 8, 12}
for d in DS:
    for cl in (4, 8, 12):
        for seed in SEEDS:
            cells["dream_attn"].append({
                "label": f"dattn_d{d}_cl{cl}_s{seed}",
                "arch": "dream_attn",
                "d_train": d,
                "seed": seed,
                "epochs": 80 if d >= 30000 else 60,
                "patience": 15,
                "aug": "rc_shift" if d >= 30000 else "rev_complement",
                "output_dir": f"results/preflight/hp_arch_size_x_d/dattn_d{d}_cl{cl}_s{seed}",
                "hp_overrides": [
                    f"core_layers={cl}",
                    "lr=0.0003",
                    f"batch_size={128 if d >= 30000 else 64}",
                    "weight_decay=0.01",
                ],
            })

for arch, c in cells.items():
    Path(f"$OUT/configs_{arch}.json".replace("$OUT", str(Path("results/preflight/hp_arch_size_x_d")))).write_text(json.dumps(c, indent=2))
    print(f"  {arch}: {len(c)} cells")
PYEOF

declare -A KP=([legnet]=4 [dream_rnn]=2 [dream_attn]=4)
declare -A GPU=([legnet]="v100" [dream_rnn]="v100" [dream_attn]="h100")

for arch in legnet dream_rnn dream_attn; do
    K=${KP[$arch]}
    G=${GPU[$arch]}
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_hp_size_x_d_${arch}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=default --gres=gpu:${G}:1 --cpus-per-task=14 --time=12:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "export TORCHDYNAMO_DISABLE=1"
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_${arch}.json $K"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  hp_size_x_d_$arch: $JID (gpu:$G k=$K)"
done
