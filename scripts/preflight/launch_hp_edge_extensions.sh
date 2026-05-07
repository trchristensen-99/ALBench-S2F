#!/bin/bash
# HP edge-extension sweep based on _analyze_hp_high_dim.py findings.
#
# Targets 6 cases where the best cell sat at a grid edge:
#   1. dream_rnn aug=rev_complement: dropout_lstm BELOW 0.05 (extend to 0.0, 0.025)
#   2. dream_attn aug=rc_shift: BS BELOW 64 (extend to 32)
#   3. dream_attn aug=rev_complement: BS BELOW 128 (extend to 64)
#   4. legnet aug=rev_complement: LR BELOW 5e-4 (extend to 1e-4, 3e-4)
#   5. legnet aug=rc_shift: LR BELOW 1e-3 (extend to 1e-4, 3e-4, 5e-4)
#   6. (legnet aug=rev_complement dropout below 0.0 — already at floor)
#
# Plus 4 "wider" exploration cells testing weight_decay variations for
# legnet (currently fixed at 0.1; try 0.01 and 0.5 — true global probe).
#
# Total: 22 cells. Bundled with parallel_gpu_runner k_parallel=12 on a
# single H100 (LegNet/DREAM-RNN are tiny, fit easily). ~30-45 min wall.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_BASE=$REPO/results/preflight/hp_edge_extensions
mkdir -p $OUT_BASE
CFG_PATH=$OUT_BASE/configs.json

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

configs = []

# 1. dream_rnn aug=rev_complement, dropout_lstm ∈ {0.0, 0.025}
for dr in (0.0, 0.025):
    for seed in (42, 123):
        configs.append({
            "label": f"drnn_revc_drlstm{dr}_s{seed}",
            "arch": "dream_rnn", "d_train": 600000, "seed": seed,
            "epochs": 66, "patience": 15, "aug": "rev_complement",
            "output_dir": f"results/preflight/hp_edge_extensions/dream_rnn/revc_drlstm{dr}/seed{seed}",
            "hp_overrides": ["lr=0.003", "batch_size=256", f"dropout_lstm={dr}"],
        })

# 2. dream_attn aug=rc_shift, BS=32
for seed in (42, 123):
    configs.append({
        "label": f"dattn_rcshift_bs32_s{seed}",
        "arch": "dream_attn", "d_train": 600000, "seed": seed,
        "epochs": 108, "patience": 15, "aug": "rc_shift",
        "output_dir": f"results/preflight/hp_edge_extensions/dream_attn/rcshift_bs32/seed{seed}",
        "hp_overrides": ["lr=0.0003", "batch_size=32"],
    })

# 3. dream_attn aug=rev_complement, BS=64
for seed in (42, 123):
    configs.append({
        "label": f"dattn_revc_bs64_s{seed}",
        "arch": "dream_attn", "d_train": 600000, "seed": seed,
        "epochs": 108, "patience": 15, "aug": "rev_complement",
        "output_dir": f"results/preflight/hp_edge_extensions/dream_attn/revc_bs64/seed{seed}",
        "hp_overrides": ["lr=0.0003", "batch_size=64"],
    })

# 4. legnet aug=rev_complement, LR ∈ {1e-4, 3e-4}
for lr in (1e-4, 3e-4):
    for seed in (42, 123):
        configs.append({
            "label": f"legnet_revc_lr{lr}_s{seed}",
            "arch": "legnet", "d_train": 600000, "seed": seed,
            "epochs": 35, "patience": 15, "aug": "rev_complement",
            "output_dir": f"results/preflight/hp_edge_extensions/legnet/revc_lr{lr}/seed{seed}",
            "hp_overrides": [f"lr={lr}", "batch_size=512"],
        })

# 5. legnet aug=rc_shift, LR ∈ {1e-4, 3e-4, 5e-4}
for lr in (1e-4, 3e-4, 5e-4):
    for seed in (42, 123):
        configs.append({
            "label": f"legnet_rcshift_lr{lr}_s{seed}",
            "arch": "legnet", "d_train": 600000, "seed": seed,
            "epochs": 35, "patience": 15, "aug": "rc_shift",
            "output_dir": f"results/preflight/hp_edge_extensions/legnet/rcshift_lr{lr}/seed{seed}",
            "hp_overrides": [f"lr={lr}", "batch_size=512"],
        })

# Global wider exploration: legnet weight_decay sweep at locked HPs (aug=none)
for wd in (0.01, 0.5):
    for seed in (42, 123):
        configs.append({
            "label": f"legnet_none_wd{wd}_s{seed}",
            "arch": "legnet", "d_train": 600000, "seed": seed,
            "epochs": 35, "patience": 15, "aug": "none",
            "output_dir": f"results/preflight/hp_edge_extensions/legnet/none_wd{wd}/seed{seed}",
            "hp_overrides": ["lr=0.003", "batch_size=512", f"weight_decay={wd}"],
        })

p = Path("$CFG_PATH")
p.write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} edge-extension configs")
PYEOF

# Submit with parallel_gpu_runner — 1 H100, k_parallel=12 (LegNet+DREAM are small)
SCRIPT=$(mktemp)
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_hp_edge_extensions"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq"
    echo "#SBATCH --qos=slow_nice"
    echo "#SBATCH --gres=gpu:h100:1"
    echo "#SBATCH --cpus-per-task=14"
    echo "#SBATCH --time=04:00:00"
    echo "#SBATCH --mem=200G"
    echo "set -euo pipefail"
    echo "set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5"
    echo "cd $REPO"
    echo 'export PYTHONPATH="$PWD"'
    echo "export TORCHDYNAMO_DISABLE=1"
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $CFG_PATH 12"
} > $SCRIPT
JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
rm -f $SCRIPT
echo "  hp_edge_extensions: $JID (22 cells, k_parallel=12)"
