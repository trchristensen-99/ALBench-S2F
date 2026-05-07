#!/bin/bash
# HP refinement sweep at d=30k (the missing middle for the scaling-law sweep).
#
# Combines:
#   1. NARROW local grid around current best HPs at d=30k for all 3 archs
#      - 3 LR × 3 BS = 9 cells per arch (logspaced around current lock)
#   2. WIDER exploration: LegNet aug crossover + alt regimes
#      - aug=none vs rev_complement at d=30k for LegNet (9 cells extra)
#      - explore wider LR/BS extremes (10× away from current best)
#
# All cells use 1 seed (42) to keep budget tight; multi-seed verification
# saved for after we identify new best HPs per arch.
#
# Total: ~36 cells. Bundled into ONE SLURM job using:
#   - 4 H100s on slow_nice
#   - parallel_gpu_runner.py with N_GPUS=4, k_parallel=24
#   - Round-robin process placement (6 cells per GPU)
# ETA: ~30-60 min wall (limited by slowest cell = d=30k epoch_budget runs)

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/results/preflight/hp_refinement_d30k
mkdir -p $OUT_BASE
CFG_PATH=$OUT_BASE/configs.json

# Generate the configs
uv run --no-sync python <<PYEOF
import json
from pathlib import Path

configs = []

# ─────────────────────────────────────────────────────────────────
# (1) NARROW local refinement at d=30k for each arch
# ─────────────────────────────────────────────────────────────────
# LegNet — current best aug=none at d=600k (lr=3e-3 bs=512). Test both
# augs at d=30k since aug-vs-N coupling is real.
for aug in ("rev_complement", "none"):
    lr_grid = (1e-4, 1e-3, 1e-2) if aug == "rev_complement" else (1e-3, 3e-3, 1e-2)
    for lr in lr_grid:
        for bs in (256, 512, 1024):
            label = f"legnet_d30k_lr{lr}_bs{bs}_{aug}"
            configs.append({
                "label": label,
                "arch": "legnet",
                "d_train": 30000,
                "seed": 42,
                "epochs": 35,
                "patience": 15,
                "aug": aug,
                "output_dir": f"results/preflight/hp_refinement_d30k/legnet/{aug}/lr{lr}_bs{bs}/seed42",
                "hp_overrides": [f"lr={lr}", f"batch_size={bs}"],
            })

# DREAM-RNN — current best aug=rev_complement (lr=3e-3 bs=256)
for lr in (1e-3, 3e-3, 1e-2):
    for bs in (128, 256, 512):
        label = f"drnn_d30k_lr{lr}_bs{bs}"
        configs.append({
            "label": label,
            "arch": "dream_rnn",
            "d_train": 30000,
            "seed": 42,
            "epochs": 66,
            "patience": 15,
            "aug": "rev_complement",
            "output_dir": f"results/preflight/hp_refinement_d30k/dream_rnn/rev_complement/lr{lr}_bs{bs}/seed42",
            "hp_overrides": [f"lr={lr}", f"batch_size={bs}", "dropout_lstm=0.15"],
        })

# DREAM-ATTN — current best aug=rc_shift (lr=3e-4 bs=64 from retry)
for lr in (1e-4, 3e-4, 1e-3):
    for bs in (32, 64, 128):
        label = f"dattn_d30k_lr{lr}_bs{bs}"
        configs.append({
            "label": label,
            "arch": "dream_attn",
            "d_train": 30000,
            "seed": 42,
            "epochs": 108,
            "patience": 15,
            "aug": "rc_shift",
            "output_dir": f"results/preflight/hp_refinement_d30k/dream_attn/rc_shift/lr{lr}_bs{bs}/seed42",
            "hp_overrides": [f"lr={lr}", f"batch_size={bs}"],
        })

# ─────────────────────────────────────────────────────────────────
# (2) WIDER exploration: extreme HPs for LegNet at d=600k (revisit)
# ─────────────────────────────────────────────────────────────────
# Test if very small LR (1e-4) or very large BS (2048) helps anywhere
for arch, aug, ep, dr_args in [
    ("legnet", "none", 35, []),
    ("dream_rnn", "rev_complement", 66, ["dropout_lstm=0.15"]),
    ("dream_attn", "rc_shift", 108, []),
]:
    # Wide LR (1 cell at extreme low) + wide BS (1 cell at extreme high)
    base_lr = {"legnet": 3e-3, "dream_rnn": 3e-3, "dream_attn": 3e-4}[arch]
    base_bs = {"legnet": 512, "dream_rnn": 256, "dream_attn": 64}[arch]
    extras = [
        ("xlow_lr", 1e-5 if arch == "dream_attn" else 1e-4, base_bs),
        ("xlarge_bs", base_lr, 2048 if arch != "dream_attn" else 512),
    ]
    for tag, lr, bs in extras:
        label = f"{arch}_d600k_{tag}_lr{lr}_bs{bs}"
        configs.append({
            "label": label,
            "arch": arch,
            "d_train": 600000,
            "seed": 42,
            "epochs": ep,
            "patience": 15,
            "aug": aug,
            "output_dir": f"results/preflight/hp_refinement_d30k/extras/{arch}/{tag}/seed42",
            "hp_overrides": [f"lr={lr}", f"batch_size={bs}"] + dr_args,
        })

p = Path("$CFG_PATH")
p.write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} configs to {p}")
PYEOF

echo
echo "=== Submit ONE SLURM job (4 H100s, 24 parallel) ==="
SCRIPT=$(mktemp)
cat > $SCRIPT <<'EOFSH'
#!/bin/bash
#SBATCH --job-name=pf_hp_refinement_d30k
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=28
#SBATCH --time=06:00:00
#SBATCH --mem=400G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

# 4 GPUs × 6 cells/GPU = 24 concurrent
N_GPUS=4 uv run --no-sync python scripts/preflight/parallel_gpu_runner.py \
    /grid/wsbs/home_norepl/christen/ALBench-S2F/results/preflight/hp_refinement_d30k/configs.json \
    24
EOFSH
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  hp_refinement_d30k: $JID (4 H100s, 24 cells parallel)"
