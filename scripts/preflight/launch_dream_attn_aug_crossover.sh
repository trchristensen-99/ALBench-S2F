#!/bin/bash
# DREAM-ATTN aug × N crossover sweep (analog to LegNet's).
#
# Findings so far for DREAM-ATTN:
#   d=30k:  rev_complement = 0.45  vs rc_shift = 0.94 (rev_c better)
#   d=600k: rev_complement = 0.18  vs rc_shift = 0.16 (rc_shift better)
# Crossover unknown — sweep d ∈ {500, 5k, 30k, 100k, 600k} for both augs.
#
# 5 D × 2 augs × 2 seeds = 20 cells, k_parallel=4 on 1 H100.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/results/preflight/dream_attn_aug_crossover
mkdir -p $OUT_BASE
CFG_PATH=$OUT_BASE/configs.json

uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = []
for aug in ("rev_complement", "rc_shift"):
    for d in (500, 5000, 30000, 100000, 600000):
        for seed in (42, 123):
            label = f"dattn_{aug}_d{d}_s{seed}"
            configs.append({
                "label": label,
                "arch": "dream_attn",
                "d_train": d,
                "seed": seed,
                "epochs": 108,
                "patience": 15,
                "aug": aug,
                "output_dir": f"results/preflight/dream_attn_aug_crossover/{aug}/d{d}/seed{seed}",
                "hp_overrides": ["lr=0.0003", "batch_size=64"],
            })
Path("$CFG_PATH").write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} configs")
PYEOF

SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_dattn_aug_crossover_fast"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq"
    echo "#SBATCH --qos=fast"
    echo "#SBATCH --gres=gpu:h100:1"
    echo "#SBATCH --cpus-per-task=14"
    echo "#SBATCH --time=03:30:00"
    echo "#SBATCH --mem=200G"
    echo "set -euo pipefail"
    echo "set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5"
    echo "cd $REPO"
    echo 'export PYTHONPATH="$PWD"'
    echo "export TORCHDYNAMO_DISABLE=1"
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $CFG_PATH 4"
} > $SCRIPT
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  dream_attn_aug_crossover (fast): $JID — 20 cells, k_parallel=4"
