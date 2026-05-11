#!/bin/bash
#SBATCH --job-name=shootout_d20k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=160G

# Train 8 LegNet variants at D=20k in parallel on a single V100, pick the winner.
# Each config is a candidate "best D=20k LegNet" for the Peter Colab.
# LegNet is ~2-7M params; we can fit 8 trials simultaneously on a V100's 16GB.

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1

OUT=results/preflight/shootout_d20k_legnet
rm -rf "$OUT"
mkdir -p "$OUT"

cat > "$OUT/configs.json" <<'EOF'
[
  {
    "label": "colab_default",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/colab_default",
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
    "label": "low_lr_bs256",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/low_lr_bs256",
    "hp_overrides": [
      "lr=0.001", "batch_size=256", "weight_decay=0.1", "dropout=0.0",
      "block_sizes=[256,256,128,128,64,64,32,32]", "ks=5"
    ]
  },
  {
    "label": "high_lr_bs1024",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/high_lr_bs1024",
    "hp_overrides": [
      "lr=0.01", "batch_size=1024", "weight_decay=0.1", "dropout=0.0",
      "block_sizes=[256,256,128,128,64,64,32,32]", "ks=5"
    ]
  },
  {
    "label": "wider_arch",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/wider_arch",
    "hp_overrides": [
      "lr=0.003", "batch_size=512", "weight_decay=0.05", "dropout=0.0",
      "block_sizes=[512,512,256,256,128,128,64,64]", "ks=5"
    ]
  },
  {
    "label": "deeper_arch",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/deeper_arch",
    "hp_overrides": [
      "lr=0.003", "batch_size=512", "weight_decay=0.1", "dropout=0.0",
      "block_sizes=[256,256,128,128,128,64,64,64,32]", "ks=5"
    ]
  },
  {
    "label": "with_shift_aug",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rc_shift",
    "output_dir": "results/preflight/shootout_d20k_legnet/with_shift_aug",
    "hp_overrides": [
      "lr=0.005", "batch_size=1024", "weight_decay=0.1", "dropout=0.0",
      "block_sizes=[256,256,128,128,64,64,32,32]", "ks=5"
    ]
  },
  {
    "label": "low_dropout_aggressive",
    "arch": "legnet", "d_train": 20000, "seed": 42,
    "epochs": 80, "patience": 15, "aug": "rev_complement",
    "output_dir": "results/preflight/shootout_d20k_legnet/low_dropout_aggressive",
    "hp_overrides": [
      "lr=0.005", "batch_size=512", "weight_decay=0.05", "dropout=0.05",
      "block_sizes=[256,256,128,128,64,64,32,32]", "ks=5"
    ]
  }
]
EOF

uv run --no-sync python scripts/preflight/parallel_gpu_runner.py "$OUT/configs.json" 4 2>&1 | tee "$OUT/driver.log"

echo ""
echo "=== Summary ==="
uv run --no-sync python - <<PYEOF
import json
from pathlib import Path
out = Path("$OUT")
results = []
for d in sorted(out.iterdir()):
    if d.is_dir() and (d / 'result.json').exists():
        r = json.loads((d / 'result.json').read_text())
        results.append((d.name, r['best_val_mse'], r['test_mse_at_best_val'], r['best_epoch']))
results.sort(key=lambda x: x[1])
print(f"{'label':30s} {'val_mse':>9s} {'test_mse':>9s} {'best_ep':>8s}")
for n, v, t, e in results:
    print(f"{n:30s} {v:>9.4f} {t:>9.4f} {e:>8d}")
PYEOF
