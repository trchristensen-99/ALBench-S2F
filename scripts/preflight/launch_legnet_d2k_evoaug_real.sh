#!/bin/bash
# REAL training-time EvoAug 2D sweep at LegNet D=2000.
#
# Now that EvoAug is actually applied at training time (via EvoAugTransform):
#   - intensity ∈ {light, medium, heavy}  — operation magnitude
#   - apply_prob ∈ {0.0, 0.25, 0.5, 0.75, 1.0} — per-sample probability
#
# 3 intensities × 5 probabilities × 2 seeds = 30 cells
# Anchors: lr=0.0005, BS=128, WD=0.1, aug=rev_complement (RC baseline)
# EvoAug applied AFTER RC, ON TOP OF the standard RC augmentation.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/legnet_d2k_evoaug
mkdir -p $OUT

uv run --no-sync python <<'PYEOF'
import json
from pathlib import Path

SEEDS = [42, 123]
INTENSITIES = ["light", "medium", "heavy"]
PROBS = [0.0, 0.25, 0.5, 0.75, 1.0]

cells = []
for intensity in INTENSITIES:
    for prob in PROBS:
        for seed in SEEDS:
            label = f"d2k_evoaug_{intensity}_p{int(prob*100):03d}_s{seed}"
            cells.append({
                "label": label,
                "arch": "legnet",
                "d_train": 2000,
                "seed": seed,
                "epochs": 80,
                "patience": 15,
                "aug": "rev_complement",  # RC standard baseline
                "evoaug_intensity": intensity if prob > 0 else None,
                "evoaug_prob": prob,
                "output_dir": f"results/preflight/legnet_d2k_evoaug/{label}",
                "hp_overrides": ["lr=0.0005", "batch_size=128", "weight_decay=0.1"],
            })

CFG = "results/preflight/legnet_d2k_evoaug/configs.json"
Path(CFG).write_text(json.dumps(cells, indent=2))
print(f"  wrote {len(cells)} cells (3 intensity × 5 prob × 2 seeds)")
PYEOF

# Split into 3 fast jobs
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
cfgs = json.loads(Path("$OUT/configs.json").read_text())
n = len(cfgs); b = (n + 2) // 3
for i in range(3):
    Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(cfgs[i*b:(i+1)*b], indent=2))
print(f"  split: {[len(Path(f'$OUT/configs_b{i}.json').read_text().count('label')) for i in range(3)]}")
PYEOF

for tag in 0 1 2; do
    SCRIPT=$(mktemp)
    {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_d2k_evoaug_${tag}"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=fast --gres=gpu:v100:1 --cpus-per-task=14 --time=03:30:00 --mem=120G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo "export PYTHONPATH=\"\$PWD\""
    echo "export TORCHDYNAMO_DISABLE=1"
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT 2>&1)
    rm -f $SCRIPT
    echo "  pf_d2k_evoaug_${tag}: $JID"
done
