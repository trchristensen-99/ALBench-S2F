#!/bin/bash
# Test the LegNet RC-channel hypothesis: does adding in_channels=5 (RC
# indicator) make rev_complement augmentation actually help?
#
# Hypothesis: LegNet underperforms with RC aug because it doesn't get
# an explicit "this is RC view" channel like DREAM-RNN/ATTN do. Adding
# that channel should fix the gap.
#
# Test cells (10 total):
#   - d=600k, in_channels=5, aug=rev_complement, lr ∈ {5e-4, 3e-3} × 2 seeds = 4
#   - d=600k, in_channels=5, aug=rc_shift, lr=5e-4, 2 seeds = 2
#   - d=600k, in_channels=4, aug=rev_complement (control replicate), 2 seeds = 2
#   - d=30k, in_channels=5, aug=rev_complement, 2 seeds = 2
#
# Bundled into ONE H100 with parallel_gpu_runner k_parallel=6.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/results/preflight/legnet_rc_channel_test
mkdir -p $OUT_BASE
CFG_PATH=$OUT_BASE/configs.json

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

configs = []
# d=600k, in_channels=5, aug=rev_complement
for lr in (5e-4, 3e-3):
    for seed in (42, 123):
        label = f"legnet_ic5_revc_lr{lr}_s{seed}"
        configs.append({
            "label": label,
            "arch": "legnet",
            "d_train": 600000,
            "seed": seed,
            "epochs": 35,
            "patience": 15,
            "aug": "rev_complement",
            "output_dir": f"results/preflight/legnet_rc_channel_test/d600k_ic5_revc/lr{lr}/seed{seed}",
            "hp_overrides": [f"lr={lr}", "batch_size=512", "in_channels=5"],
        })

# d=600k, in_channels=5, aug=rc_shift
for seed in (42, 123):
    label = f"legnet_ic5_rcshift_s{seed}"
    configs.append({
        "label": label,
        "arch": "legnet",
        "d_train": 600000,
        "seed": seed,
        "epochs": 35,
        "patience": 15,
        "aug": "rc_shift",
        "output_dir": f"results/preflight/legnet_rc_channel_test/d600k_ic5_rcshift/seed{seed}",
        "hp_overrides": ["lr=5e-4", "batch_size=512", "in_channels=5"],
    })

# d=600k, in_channels=4, aug=rev_complement (control — replicate the puzzle)
for seed in (42, 123):
    label = f"legnet_ic4_revc_ctrl_s{seed}"
    configs.append({
        "label": label,
        "arch": "legnet",
        "d_train": 600000,
        "seed": seed,
        "epochs": 35,
        "patience": 15,
        "aug": "rev_complement",
        "output_dir": f"results/preflight/legnet_rc_channel_test/d600k_ic4_revc_ctrl/seed{seed}",
        "hp_overrides": ["lr=5e-4", "batch_size=512", "in_channels=4"],
    })

# d=30k, in_channels=5, aug=rev_complement
for seed in (42, 123):
    label = f"legnet_ic5_revc_d30k_s{seed}"
    configs.append({
        "label": label,
        "arch": "legnet",
        "d_train": 30000,
        "seed": seed,
        "epochs": 35,
        "patience": 15,
        "aug": "rev_complement",
        "output_dir": f"results/preflight/legnet_rc_channel_test/d30k_ic5_revc/seed{seed}",
        "hp_overrides": ["lr=5e-4", "batch_size=512", "in_channels=5"],
    })

p = Path("$CFG_PATH")
p.write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} configs to {p}")
PYEOF

SCRIPT=$(mktemp)
cat > $SCRIPT <<'EOFSH'
#!/bin/bash
#SBATCH --job-name=pf_legnet_rc_channel
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=03:00:00
#SBATCH --mem=200G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python scripts/preflight/parallel_gpu_runner.py \
    /grid/wsbs/home_norepl/christen/ALBench-S2F/results/preflight/legnet_rc_channel_test/configs.json \
    6
EOFSH
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  legnet_rc_channel_test: $JID (10 cells, k_parallel=6)"
