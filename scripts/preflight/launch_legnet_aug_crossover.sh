#!/bin/bash
# Map the LegNet aug × N crossover precisely.
#
# Findings so far:
#   d=500:   aug=none gives 1.42, rev_complement gives 0.92 (rev_c better)
#   d=600k:  aug=none gives 0.16, rev_complement gives 0.29 (none better)
#
# Crossover is somewhere in 4k-30k. This sweep maps it at d ∈
# {10k, 30k, 100k} for both aug regimes × 2 seeds = 12 LegNet cells.
#
# Uses the parallel multi-model GPU runner to bundle all 12 cells into
# ONE H100 SLURM allocation. LegNet is small (~2M params) so 6 can
# train concurrently. Total wall ~30-45 min vs ~2 hr serial.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

CFG_PATH=$REPO/results/preflight/legnet_aug_crossover/configs.json
mkdir -p $(dirname $CFG_PATH)

# Generate the 12-cell config
uv run --no-sync python <<PYEOF
import json
from pathlib import Path

REPO = Path("$REPO")
configs = []
# aug=none winner HPs (from task3 retry): lr=3e-3, bs=512
# aug=rev_complement winner HPs (from task3 orig): lr=5e-4, bs=512
SETUPS = [
    ("none", "0.003", "512"),
    ("rev_complement", "0.0005", "512"),
]
for aug, lr, bs in SETUPS:
    for d in (10000, 30000, 100000):
        for seed in (42, 123):
            label = f"legnet_aug{aug}_d{d}_s{seed}"
            out = f"results/preflight/legnet_aug_crossover/{aug}/d{d}/seed{seed}"
            configs.append({
                "label": label,
                "arch": "legnet",
                "d_train": d,
                "seed": seed,
                "epochs": 35,  # locked epoch_budget for legnet
                "patience": 15,
                "aug": aug,
                "output_dir": out,
                "hp_overrides": [f"lr={lr}", f"batch_size={bs}"],
            })

cfg_path = Path("$CFG_PATH")
cfg_path.write_text(json.dumps(configs, indent=2))
print(f"  Wrote {len(configs)} configs to {cfg_path}")
PYEOF

echo
echo "=== Submit ONE H100 job that runs all 12 cells in parallel ==="
SCRIPT=$(mktemp)
cat > $SCRIPT <<'EOFSH'
#!/bin/bash
#SBATCH --job-name=pf_legnet_aug_crossover
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --mem=200G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python scripts/preflight/parallel_gpu_runner.py \
    /grid/wsbs/home_norepl/christen/ALBench-S2F/results/preflight/legnet_aug_crossover/configs.json \
    6
EOFSH
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  legnet_aug_crossover (12 cells, 6 parallel): $JID"
echo
echo "Output: results/preflight/legnet_aug_crossover/{aug}/d{N}/seed{S}/"
