#!/bin/bash
# HP edge extensions + coverage gaps (v2) — driven by hp_grid_audit.py findings.
#
# 1. LegNet D=2k/4k: lr=0.003 was at LOW edge → test lr={0.001, 0.0005} × BS={128, 64, 256}
# 2. DREAM-RNN D=2k/4k: lr=0.003 was at HIGH edge → test lr={0.005, 0.01} × BS={128, 256}
# 3. DREAM-RNN D=30k: lr=0.001 was at LOW edge → test lr={0.0003, 0.0005}
# 4. DREAM-ATTN coverage gaps: D ∈ {2000, 4000, 5000, 100000} (3-4 configs each, need ≥6)
# 5. LegNet D=10k: lr=0.0005 at LOW edge → test lr=0.0003

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT=$REPO/results/preflight/hp_edge_v2
mkdir -p $OUT
CFG=$OUT/configs.json

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

cells = []

def add(arch, d, lr, bs, aug, seed, label_extras="", **hp):
    label = f"{arch}_d{d}_lr{lr}_bs{bs}{label_extras}_s{seed}"
    overrides = [f"lr={lr}", f"batch_size={bs}"]
    for k, v in hp.items():
        overrides.append(f"{k}={v}")
    cells.append({
        "label": label,
        "arch": arch,
        "d_train": d,
        "seed": seed,
        "epochs": 80 if d >= 30000 else 60,
        "patience": 15,
        "aug": aug,
        "output_dir": f"results/preflight/hp_edge_v2/{label}",
        "hp_overrides": overrides,
    })

# ── LegNet D=2k, 4k extensions: lr was at LOW edge ──
for d in (2000, 4000):
    for lr in (0.001, 0.0005):
        for bs in (256, 128, 64):
            for seed in (42, 123):
                add("legnet", d, lr, bs, "rev_complement", seed)

# ── LegNet D=10k: lr=0.0005 at LOW edge ──
for lr in (0.0003, 0.0001):
    for bs in (256, 512):
        for seed in (42, 123):
            add("legnet", 10000, lr, bs, "rev_complement", seed)

# ── DREAM-RNN D=2k, 4k: lr=0.003 was at HIGH edge ──
for d in (2000, 4000):
    for lr in (0.005, 0.01):
        for bs in (128, 256):
            for seed in (42, 123):
                add("dream_rnn", d, lr, bs, "rev_complement", seed)

# ── DREAM-RNN D=30k: lr=0.001 at LOW edge ──
for lr in (0.0003, 0.0005):
    for seed in (42, 123):
        add("dream_rnn", 30000, lr, 128, "rev_complement", seed)

# ── DREAM-ATTN coverage gaps: D=2k, 4k, 5k, 100k ──
for d in (2000, 4000, 5000, 100000):
    for aug in ("rev_complement", "rc_shift"):
        for bs in (64, 128):
            for seed in (42, 123):
                add("dream_attn", d, 0.0003, bs, aug, seed)

# Split by arch (different k_parallel)
legnet_cells = [c for c in cells if c["arch"] == "legnet"]
drnn_cells   = [c for c in cells if c["arch"] == "dream_rnn"]
dattn_cells  = [c for c in cells if c["arch"] == "dream_attn"]

Path("$OUT/configs_legnet.json").write_text(json.dumps(legnet_cells, indent=2))
Path("$OUT/configs_dream_rnn.json").write_text(json.dumps(drnn_cells, indent=2))
Path("$OUT/configs_dream_attn.json").write_text(json.dumps(dattn_cells, indent=2))

print(f"  LegNet cells: {len(legnet_cells)}")
print(f"  DREAM-RNN cells: {len(drnn_cells)}")
print(f"  DREAM-ATTN cells: {len(dattn_cells)}")
print(f"  Total: {len(cells)}")
PYEOF

# Submit one job per arch with appropriate k_parallel
declare -A KPARALLEL=(
    [legnet]=6
    [dream_rnn]=4
    [dream_attn]=4
)

for arch in legnet dream_rnn dream_attn; do
    K=${KPARALLEL[$arch]}
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=pf_hp_edge_v2_$arch"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=12:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "export TORCHDYNAMO_DISABLE=1"
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo "uv run --no-sync python scripts/preflight/parallel_gpu_runner.py $OUT/configs_${arch}.json $K"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
    rm -f $SCRIPT
    echo "  hp_edge_v2_$arch: $JID (k_parallel=$K)"
done
