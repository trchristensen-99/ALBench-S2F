#!/bin/bash
#SBATCH --job-name=shootout_d20k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=03:30:00
#SBATCH --mem=120G

# Train 4 LegNet variants at D=20k in parallel on a single V100, pick the winner.
# Each config is a candidate "default" for the Peter Colab.

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1

OUT=results/preflight/shootout_d20k_legnet
rm -rf "$OUT"
mkdir -p "$OUT"

# Configs to compare. Each runs run_single.py at D=20k.
cat > "$OUT/configs.json" <<'EOF'
[
  {
    "label": "current_colab_default",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/current_colab_default",
    "hp_overrides": [
      "lr=0.003", "batch_size=512", "weight_decay=0.05", "dropout=0.1",
      "block_sizes=[256,256,128,128,64,64,32,32]", "ks=5"
    ]
  },
  {
    "label": "legnet_published_default",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/legnet_published_default",
    "hp_overrides": [
      "lr=0.005", "batch_size=1024", "weight_decay=0.1", "dropout=0.0",
      "block_sizes=[256,256,128,128,64,64,32,32]", "ks=5"
    ]
  },
  {
    "label": "wider_arch",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/wider_arch",
    "hp_overrides": [
      "lr=0.003", "batch_size=512", "weight_decay=0.05", "dropout=0.1",
      "block_sizes=[512,512,256,256,128,128,64,64]", "ks=5"
    ]
  },
  {
    "label": "with_shift_aug",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rc_shift",
    "output_dir": "results/preflight/shootout_d20k_legnet/with_shift_aug",
    "hp_overrides": [
      "lr=0.003", "batch_size=512", "weight_decay=0.05", "dropout=0.1",
      "block_sizes=[256,256,128,128,64,64,32,32]", "ks=5"
    ]
  }
]
EOF

uv run --no-sync python scripts/preflight/parallel_gpu_runner.py "$OUT/configs.json" 4 2>&1 | tee "$OUT/driver.log"

echo ""
echo "=== Summary ==="
for d in "$OUT"/*/; do
    label=$(basename "$d")
    if [ -f "$d/result.json" ]; then
        uv run --no-sync python -c "
import json, sys
r = json.load(open('$d/result.json'))
print(f'  {sys.argv[1]:>30s}  val_mse={r[\"best_val_mse\"]:.4f}  test_mse={r[\"test_mse_at_best_val\"]:.4f}  best_ep={r[\"best_epoch\"]}')
" "$label"
    fi
done
