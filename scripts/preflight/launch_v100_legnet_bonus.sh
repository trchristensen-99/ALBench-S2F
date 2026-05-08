#!/bin/bash
# Bonus throughput: route extra LegNet HP work to idle V100 nodes.
# V100 nodes (bamgpu01-14) are mostly idle. LegNet runs fine on V100.
#
# Sample 24 random HPs covering wider ranges than tested, at D ∈ {6k, 100k}
# (under-explored at these D values).

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_v100_bonus
mkdir -p $OUT

uv run --no-sync python <<PYEOF
import json
import random
from pathlib import Path
random.seed(20260508 + 1)

cells = []
for d in (6000, 100000):
    for _ in range(6):
        lr = round(10 ** random.uniform(-4.0, -1.7), 5)
        bs = random.choice([64, 128, 256, 512, 1024])
        wd = random.choice([0.001, 0.01, 0.1, 0.3])
        for seed in (42, 123):
            label = f"legnet_d{d}_lr{lr}_bs{bs}_wd{wd}_s{seed}"
            cells.append({
                "label": label,
                "arch": "legnet",
                "d_train": d,
                "seed": seed,
                "epochs": 80 if d >= 30000 else 60,
                "patience": 15,
                "aug": "rev_complement",
                "output_dir": f"results/preflight/hp_v100_bonus/{label}",
                "hp_overrides": [f"lr={lr}", f"batch_size={bs}", f"weight_decay={wd}"],
            })

Path("$OUT/configs.json").write_text(json.dumps(cells, indent=2))
print(f"  wrote {len(cells)} V100 LegNet cells")
PYEOF

# V100 has 16GB so k_parallel=3 max for LegNet
SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_v100_legnet_bonus"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=default --gres=gpu:v100:1 --cpus-per-task=14 --time=12:00:00 --mem=120G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo 'export PYTHONPATH="$PWD"'
    echo "export TORCHDYNAMO_DISABLE=1"
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs.json 3"
} > $SCRIPT
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  v100_legnet_bonus: $JID (k_parallel=3 on V100)"
