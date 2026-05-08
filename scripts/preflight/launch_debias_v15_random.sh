#!/bin/bash
# Debias v15: RANDOM SEARCH across debias HP space.
# Goal: find unexpected far-flung combinations the targeted sweeps may miss.
#
# 16 random samples spanning:
#   blocks: random subset {0..5} of size 1-3
#   neg-aug source: random choice (dinuc, Sahu, ctrl_neg, combined, intergenic)
#   frac: log-uniform [0.005, 0.15]
#   debias_mode: random choice from 5 modes
#   lam: log-uniform [0.01, 0.50]
#   wd: log-uniform [1e-6, 1e-2]
#   dropout: U(0.0, 0.4)

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v15
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0

uv run --no-sync python <<PYEOF
import json
import math
import random
from pathlib import Path

random.seed(20260508)

NEG_TSVS = [
    "$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv",
    "$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv",
    "$REPO/data/synthetic_negatives_combined/dinuc_sahu_ctrl_combined.tsv",
]
MODES = ["cpg_invariance", "cpg_gradient_penalty", "counterfactual_consistency",
         "conditional_invariance", "spectral"]

def loguniform(lo, hi):
    return 10 ** random.uniform(math.log10(lo), math.log10(hi))

configs = []
for i in range(16):
    n_blocks = random.randint(1, 3)
    blocks = sorted(random.sample(list(range(6)), n_blocks))
    blocks_str = "[" + ",".join(str(b) for b in blocks) + "]"
    tsv = random.choice(NEG_TSVS)
    frac = round(loguniform(0.005, 0.15), 4)
    mode = random.choice(MODES)
    lam = round(loguniform(0.01, 0.50), 3)
    wd = round(loguniform(1e-6, 1e-2), 6)
    dropout = round(random.uniform(0.0, 0.4), 2)

    short_mode = mode.replace("_", "")[:8]
    label = f"c170_{i:02d}_blk{''.join(map(str, blocks))}_f{int(frac*1000):03d}_{short_mode}_lam{lam:.2f}".replace(".","p")
    extras = [
        f"++negatives_path={tsv}",
        f"++neg_fraction={frac}",
        f"++debias_mode={mode}",
        f"++debias_lambda={lam}",
        f"++unfreeze_encoder_blocks={blocks_str}",
        f"++weight_decay={wd}",
        f"++dropout_rate={dropout}",
    ]
    configs.append({
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v15/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": extras,
    })
    print(f"  {label}")
    print(f"    blocks={blocks}, tsv={Path(tsv).name}, frac={frac:.3f}, mode={mode}, "
          f"lam={lam:.3f}, wd={wd:.1e}, drop={dropout}")

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"\n  wrote {len(configs)} v15 random configs")
PYEOF

# 16 cells / k=3 = ~6 sequential @ ~50min = ~5h.  Split into 3 batches.
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$OUT/configs.json").read_text())
n = len(configs)
b = (n + 2) // 3
batches = [configs[i:i+b] for i in range(0, n, b)]
for i, batch in enumerate(batches):
    Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(batch, indent=2))
print(f"  split: {[len(b) for b in batches]}")
PYEOF

NUM_BATCHES=$(ls "$OUT"/configs_b*.json | wc -l)
for tag in $(seq 0 $((NUM_BATCHES-1))); do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_debias_v15_b${tag}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=08:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  v15_b${tag}: $JID"
done
