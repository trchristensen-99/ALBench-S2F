#!/bin/bash
# Debias v14: fill remaining axes of debias HP space.
#   - encoder_lr × head_lr (rarely varied)
#   - freeze extremes (full encoder unfrozen / head only / late blocks)
#   - neg-aug source ratios (different dinuc/Sahu mixes)
#   - epoch budget (50 vs 120)
#   - alternative neg-aug types (intergenic, GC-matched if available)

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v14
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
DINUC=$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv
SAHU=$REPO/data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv
INTERG=$REPO/data/synthetic_negatives/intergenic_negatives.tsv
GCMATCH=$REPO/data/synthetic_negatives/gc_matched_negatives.tsv
COMBINED=$REPO/data/synthetic_negatives_combined/dinuc_sahu_ctrl_combined.tsv

# Build extra mixed TSVs if needed
if [ ! -f "$REPO/data/synthetic_negatives_combined/dinuc_sahu_70_30.tsv" ]; then
    uv run --no-sync python <<PYEOF
import pandas as pd
from pathlib import Path

d = pd.read_csv("$DINUC", sep="\t")[["sequence", "K562_log2FC"]]
s = pd.read_csv("$SAHU", sep="\t")[["sequence", "K562_log2FC"]]
out_dir = Path("$REPO/data/synthetic_negatives_combined")
out_dir.mkdir(parents=True, exist_ok=True)

for tag, p_d, p_s in [("70_30", 0.7, 0.3), ("30_70", 0.3, 0.7), ("50_50", 0.5, 0.5)]:
    n_total = 50000
    n_d = int(n_total * p_d)
    n_s = int(n_total * p_s)
    n_d = min(n_d, len(d)); n_s = min(n_s, len(s))
    df = pd.concat([d.sample(n_d, random_state=42), s.sample(n_s, random_state=42)])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    out = out_dir / f"dinuc_sahu_{tag}.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"  {out}: {len(df)} ({p_d:.0%} dinuc, {p_s:.0%} Sahu)")
PYEOF
fi

uv run --no-sync python <<PYEOF
import json
from pathlib import Path
import os

def cfg(label, **kwargs):
    extras = []
    extras.append(f"++negatives_path={kwargs.get('tsv', '$DINUC')}")
    extras.append(f"++neg_fraction={kwargs.get('frac', 0.03)}")
    extras.append(f"++debias_mode={kwargs.get('mode', 'cpg_invariance')}")
    extras.append(f"++debias_lambda={kwargs.get('lam', 0.05)}")
    extras.append(f"++unfreeze_encoder_blocks={kwargs.get('blocks', '[0,1,2]')}")
    if "enc_lr" in kwargs:
        extras.append(f"++encoder_lr={kwargs['enc_lr']}")
    if "head_lr" in kwargs:
        extras.append(f"++head_lr={kwargs['head_lr']}")
    if "wd" in kwargs:
        extras.append(f"++weight_decay={kwargs['wd']}")
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v14/{label}/fold_0",
        "epochs": kwargs.get("epochs", 80),
        "patience": kwargs.get("patience", 15),
        "extra_overrides": extras,
    }

configs = []

# Encoder LR axis
configs.append(cfg("c150_enclr1e5",  enc_lr=1e-5))
configs.append(cfg("c151_enclr5e4",  enc_lr=5e-4))

# Head LR axis (default 1e-3)
configs.append(cfg("c152_headlr1e4",  head_lr=1e-4))
configs.append(cfg("c153_headlr5e4",  head_lr=5e-4))
configs.append(cfg("c154_headlr5e3",  head_lr=5e-3))

# Freeze extremes
configs.append(cfg("c155_full_encoder", blocks="[0,1,2,3,4,5]"))
configs.append(cfg("c156_head_only",    blocks="[]"))
configs.append(cfg("c157_blk345_late",  blocks="[3,4,5]"))

# Source ratios (dinuc/Sahu mixes)
combined_dir = "$REPO/data/synthetic_negatives_combined"
configs.append(cfg("c158_mix7030", tsv=f"{combined_dir}/dinuc_sahu_70_30.tsv", frac=0.03))
configs.append(cfg("c159_mix3070", tsv=f"{combined_dir}/dinuc_sahu_30_70.tsv", frac=0.03))
configs.append(cfg("c160_mix5050", tsv=f"{combined_dir}/dinuc_sahu_50_50.tsv", frac=0.03))

# Alt neg-aug TYPES (with c91 recipe)
import os
if os.path.exists("$INTERG"):
    configs.append(cfg("c161_intergenic", tsv="$INTERG", frac=0.03))
if os.path.exists("$GCMATCH"):
    configs.append(cfg("c162_gcmatch", tsv="$GCMATCH", frac=0.03))

# Epoch budget
configs.append(cfg("c163_50ep",  epochs=50, patience=10))
configs.append(cfg("c164_120ep", epochs=120, patience=30))

# Extreme wd
configs.append(cfg("c165_wd0",     wd=0.0))
configs.append(cfg("c166_wd1e2",   wd=1e-2))

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v14 configs")
PYEOF

# Split into 3 batches of ~7 cells each
uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$OUT/configs.json").read_text())
n = len(configs)
b = (n + 2) // 3
batches = [configs[i:i+b] for i in range(0, n, b)]
for i, batch in enumerate(batches):
    Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(batch, indent=2))
print(f"  split into {len(batches)} batches: {[len(b) for b in batches]}")
PYEOF

NUM_BATCHES=$(ls "$OUT"/configs_b*.json | wc -l)
for tag in $(seq 0 $((NUM_BATCHES-1))); do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_debias_v14_b${tag}"
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
    echo "  v14_b${tag}: $JID"
done
