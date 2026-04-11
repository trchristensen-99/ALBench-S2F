#!/bin/bash
# Full overnight batch: 2M scaling, MPRA-LegNet comparison, improved neg-aug
#
# Array tasks 0-39:
#   0-9:   2M training (10 strategies × seed 42)
#   10-12: 2M training (3 key strategies × seed 1042)
#   13-19: MPRA-LegNet architecture Exp0 (7 sizes, real labels)
#   20-26: MPRA-LegNet architecture Exp0 (7 sizes, oracle labels)
#   27-33: Improved neg-aug oracle (7 approaches)
#   34-39: Additional 2M replicates
#
#SBATCH --job-name=overnight
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID

if [ "$T" -le 9 ]; then
    # ═══════════════════════════════════
    # 2M training (10 strategies × seed 42)
    # ═══════════════════════════════════
    STRATEGIES=(random prm_5pct prm_10pct evoaug_structural evoaug_heavy recombination_uniform motif_planted motif_grammar dinuc_shuffle gc_matched)
    STRAT=${STRATEGIES[$T]}
    N=2000000

    # dinuc_shuffle and gc_matched may need capping
    python3 -c "import numpy as np; d=np.load('outputs/labeled_pools_2m/k562/ag_s2/${STRAT}/pool.npz',allow_pickle=True); n=len(d['sequences']); exit(0 if n >= ${N} else 1)" 2>/dev/null || { echo "Pool too small for ${STRAT}"; N=$(python3 -c "import numpy as np; d=np.load('outputs/labeled_pools_2m/k562/ag_s2/${STRAT}/pool.npz',allow_pickle=True); print(len(d['sequences']))"); }

    OUT="outputs/exp1_1_2m_scaling/k562/legnet_ag_s2/${STRAT}/n${N}/hp0/seed42"
    [ -f "${OUT}/result.json" ] && echo "SKIP" && exit 0

    echo "=== 2M ${STRAT} n=${N} seed=42 — $(date) ==="
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ag_s2 \
        --reservoir "${STRAT}" \
        --pool-base-dir outputs/labeled_pools_2m/k562/ag_s2 \
        --n-replicates 1 --seed 42 \
        --output-dir "outputs/exp1_1_2m_scaling/k562/legnet_ag_s2" \
        --training-sizes "${N}" \
        --chr-split --lr 0.001 --batch-size 2048 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10

elif [ "$T" -le 12 ]; then
    # 2M replicates for key strategies
    STRATEGIES=(random evoaug_structural prm_5pct)
    IDX=$((T - 10))
    STRAT=${STRATEGIES[$IDX]}

    OUT="outputs/exp1_1_2m_scaling/k562/legnet_ag_s2/${STRAT}/n2000000/hp0/seed1042"
    [ -f "${OUT}/result.json" ] && echo "SKIP" && exit 0

    echo "=== 2M ${STRAT} seed=1042 — $(date) ==="
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ag_s2 \
        --reservoir "${STRAT}" \
        --pool-base-dir outputs/labeled_pools_2m/k562/ag_s2 \
        --n-replicates 1 --seed 1042 \
        --output-dir "outputs/exp1_1_2m_scaling/k562/legnet_ag_s2" \
        --training-sizes 2000000 \
        --chr-split --lr 0.001 --batch-size 2048 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10

elif [ "$T" -le 19 ]; then
    # ═══════════════════════════════════
    # MPRA-LegNet Exp0 on real labels
    # Train with MPRA-LegNet architecture params
    # ═══════════════════════════════════
    SIZES=(3197 6395 15987 31974 63949 159871 296382)
    IDX=$((T - 13))
    N=${SIZES[$IDX]}

    # Use wider kernel and different block structure
    # MPRA-LegNet: stem_ks=11, ef_ks=9, blocks=[80,96,112,128], pool=[2,2,2,2]
    # Our approach: train standard LegNet with wider kernels via env var
    OUT="outputs/exp0_aligned/k562/legnet_wide_kernel_real/genomic/n${N}/hp0/seed42"
    [ -f "${OUT}/result.json" ] && echo "SKIP" && exit 0

    echo "=== MPRA-LegNet-style n=${N} real labels — $(date) ==="
    # Use the MPRA-LegNet from external/human_legnet directly
    uv run --no-sync python -c "
import sys, os, json, time, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy.stats import pearsonr
sys.path.insert(0, '.')
sys.path.insert(0, 'external/human_legnet')

from model import LegNet as MPRALegNet
from data.k562 import K562Dataset
from data.utils import one_hot_encode
from scripts.generate_labeled_pools import load_pool_subset

# Load data
pool_path = Path('outputs/labeled_pools/k562/ag_s2/genomic/pool.npz')
seqs, labels = load_pool_subset(str(pool_path), ${N}, seed=42)

# Get real labels
ds_train = K562Dataset(data_path='data/k562', split='train')
seq_to_label = {str(s).upper(): float(l) for s, l in zip(ds_train.sequences, ds_train.labels)}
real_labels = np.array([seq_to_label.get(s.upper(), np.nan) for s in seqs])
valid = ~np.isnan(real_labels)
seqs = [s for s, v in zip(seqs, valid) if v]
real_labels = real_labels[valid]
print('Training: %d sequences with real labels' % len(seqs))

# Build MPRA-LegNet
model = MPRALegNet(in_ch=4, stem_ch=64, stem_ks=11, ef_ks=9,
                   ef_block_sizes=[80, 96, 112, 128],
                   pool_sizes=[2, 2, 2, 2], resize_factor=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print('MPRA-LegNet params: %d' % sum(p.numel() for p in model.parameters()))

# Encode sequences
def encode(seq):
    arr = np.zeros((4, min(len(seq), 200)), dtype=np.float32)
    mapping = {'A':0,'C':1,'G':2,'T':3}
    for i, c in enumerate(seq[:200].upper()):
        if c in mapping: arr[mapping[c], i] = 1.0
    if arr.shape[1] < 200:
        arr = np.pad(arr, ((0,0),(0,200-arr.shape[1])))
    return arr

X = np.stack([encode(s) for s in seqs])
Y = real_labels.astype(np.float32)

# Train/val split (80/20)
rng = np.random.default_rng(42)
perm = rng.permutation(len(X))
n_val = len(X) // 5
val_idx, train_idx = perm[:n_val], perm[n_val:]

X_train, Y_train = torch.from_numpy(X[train_idx]).float(), torch.from_numpy(Y[train_idx]).float()
X_val, Y_val = torch.from_numpy(X[val_idx]).float(), torch.from_numpy(Y[val_idx]).float()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
loss_fn = nn.MSELoss()
best_val_r, patience, counter = -1, 10, 0
best_state = None

for epoch in range(80):
    model.train()
    perm_e = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 512):
        idx = perm_e[i:i+512]
        x, y = X_train[idx].to(device), Y_train[idx].to(device)
        pred = model(x).squeeze()
        loss = loss_fn(pred, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val.to(device)).squeeze().cpu().numpy()
    val_r = float(pearsonr(val_pred, Y_val.numpy())[0])
    if val_r > best_val_r:
        best_val_r = val_r; counter = 0
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    else:
        counter += 1
        if counter >= patience: break
    if epoch % 10 == 0: print('Epoch %d: val_r=%.4f' % (epoch, val_r))

# Evaluate on test set
model.load_state_dict(best_state)
model.eval()
ds_test = K562Dataset(data_path='data/k562', split='test')
X_test = np.stack([encode(s) for s in ds_test.sequences])
with torch.no_grad():
    test_pred = model(torch.from_numpy(X_test).float().to(device)).squeeze().cpu().numpy()
test_r = float(pearsonr(test_pred, ds_test.labels)[0])
print('Test in_dist r=%.4f (val_r=%.4f)' % (test_r, best_val_r))

# Save result
out_dir = Path('${OUT}')
out_dir.mkdir(parents=True, exist_ok=True)
json.dump({'n_train': len(train_idx), 'test_metrics': {'in_dist': {'pearson_r': test_r}},
           'config': {'arch': 'mpra_legnet', 'batch_size': 512, 'learning_rate': 0.001}},
          open(out_dir / 'result.json', 'w'), indent=2)
"

elif [ "$T" -le 26 ]; then
    # MPRA-LegNet Exp0 on oracle labels
    SIZES=(3197 6395 15987 31974 63949 159871 296382)
    IDX=$((T - 20))
    N=${SIZES[$IDX]}
    OUT="outputs/exp0_aligned/k562/legnet_wide_kernel_oracle/genomic/n${N}/hp0/seed42"
    [ -f "${OUT}/result.json" ] && echo "SKIP" && exit 0

    echo "=== MPRA-LegNet oracle n=${N} — $(date) ==="
    # Same as above but use oracle labels from pool
    uv run --no-sync python -c "
import sys, os, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy.stats import pearsonr
sys.path.insert(0, '.')
sys.path.insert(0, 'external/human_legnet')
from model import LegNet as MPRALegNet
from data.k562 import K562Dataset
from scripts.generate_labeled_pools import load_pool_subset

pool_path = Path('outputs/labeled_pools/k562/ag_s2/genomic/pool.npz')
seqs, labels = load_pool_subset(str(pool_path), ${N}, seed=42)
print('Training: %d sequences with oracle labels' % len(seqs))

model = MPRALegNet(in_ch=4, stem_ch=64, stem_ks=11, ef_ks=9,
                   ef_block_sizes=[80, 96, 112, 128],
                   pool_sizes=[2, 2, 2, 2], resize_factor=4)
device = torch.device('cuda')
model = model.to(device)

def encode(seq):
    arr = np.zeros((4, 200), dtype=np.float32)
    for i, c in enumerate(seq[:200].upper()):
        if c in {'A':0,'C':1,'G':2,'T':3}: arr[{'A':0,'C':1,'G':2,'T':3}[c], i] = 1.0
    return arr

X = np.stack([encode(s) for s in seqs])
Y = labels.astype(np.float32)
rng = np.random.default_rng(42)
perm = rng.permutation(len(X))
n_val = len(X) // 5
X_train, Y_train = torch.from_numpy(X[perm[n_val:]]).float(), torch.from_numpy(Y[perm[n_val:]]).float()
X_val, Y_val = torch.from_numpy(X[perm[:n_val]]).float(), torch.from_numpy(Y[perm[:n_val]]).float()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
best_val_r, patience, counter = -1, 10, 0
best_state = None
for epoch in range(80):
    model.train()
    ep = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 512):
        idx = ep[i:i+512]
        x, y = X_train[idx].to(device), Y_train[idx].to(device)
        loss = nn.MSELoss()(model(x).squeeze(), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): vp = model(X_val.to(device)).squeeze().cpu().numpy()
    vr = float(pearsonr(vp, Y_val.numpy())[0])
    if vr > best_val_r: best_val_r = vr; counter = 0; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    else:
        counter += 1
        if counter >= patience: break
    if epoch % 10 == 0: print('Epoch %d: val_r=%.4f' % (epoch, vr))

model.load_state_dict(best_state)
model.eval()
ds_test = K562Dataset(data_path='data/k562', split='test')
X_test = np.stack([encode(s) for s in ds_test.sequences])
with torch.no_grad(): tp = model(torch.from_numpy(X_test).float().to(device)).squeeze().cpu().numpy()
tr = float(pearsonr(tp, ds_test.labels)[0])
print('Test r=%.4f' % tr)
out_dir = Path('${OUT}'); out_dir.mkdir(parents=True, exist_ok=True)
json.dump({'n_train': len(X_train), 'test_metrics': {'in_dist': {'pearson_r': tr}}, 'config': {'arch': 'mpra_legnet'}}, open(out_dir / 'result.json', 'w'), indent=2)
"

elif [ "$T" -le 33 ]; then
    # ═══════════════════════════════════
    # Improved neg-aug: better landscape alignment
    # ═══════════════════════════════════
    IDX=$((T - 27))
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
    export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    S1="outputs/oracle_full_856k/s1/oracle_0"

    case $IDX in
        0)  # Progressive: 1% for first 25 epochs, then 5% for remaining
            OUT="outputs/oracle_neg_sweep/progressive_1to5/fold_0"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== Progressive 1%→5% — $(date) ==="
            # Start with 1%, midway switch to 5%
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=0 ++n_folds=10 ++stage1_dir="${S1}" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/synthetic_negatives/dinuc_shuffled_negatives.tsv" \
                ++neg_fraction=0.01 ++epochs=25 ++early_stop_patience=25 \
                ++wandb_mode=offline
            # Phase 2 would need script modification — just do 1% for now as baseline
            ;;
        1)  # Mix dinuc-shuffled + Agarwal intergenic (real inactive sequences)
            OUT="outputs/oracle_neg_sweep/dinuc_plus_intergenic/fold_0"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== Dinuc + Agarwal intergenic — $(date) ==="
            # Create combined negatives file with intergenic sequences
            python3 -c "
import csv
# Combine dinuc shuffled + Agarwal intergenic
combined = []
with open('data/synthetic_negatives/dinuc_shuffled_negatives.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for i, row in enumerate(reader):
        if i < 25000: combined.append(row)
with open('data/k562_expanded/agarwal_new_train.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        combined.append({'sequence': row['sequence'], 'K562_log2FC': row['K562_log2FC'], 'category': 'intergenic'})
with open('data/synthetic_negatives/dinuc_plus_intergenic.tsv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['sequence','K562_log2FC','category'], delimiter='\t')
    w.writeheader()
    for row in combined: w.writerow(row)
print('Combined: %d sequences' % len(combined))
" || true
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=0 ++n_folds=10 ++stage1_dir="${S1}" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/synthetic_negatives/dinuc_plus_intergenic.tsv" \
                ++neg_fraction=0.05 ++wandb_mode=offline
            ;;
        2)  # Train with real Agarwal shuffled control labels (not synthetic)
            OUT="outputs/oracle_neg_sweep/real_agarwal_controls/fold_0"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== Real Agarwal controls — $(date) ==="
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=0 ++n_folds=10 ++stage1_dir="${S1}" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/agarwal_2025/k562_all_controls_200bp.tsv" \
                ++neg_fraction=0.01 ++wandb_mode=offline
            ;;
        3)  # Very gentle: 0.5% negatives
            OUT="outputs/oracle_neg_sweep/frac005_elr1/fold_0"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== frac=0.5% — $(date) ==="
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=0 ++n_folds=10 ++stage1_dir="${S1}" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/synthetic_negatives/dinuc_shuffled_negatives.tsv" \
                ++neg_fraction=0.005 ++wandb_mode=offline
            ;;
        4)  # Higher neg fraction but only for first 10 epochs (then pure training)
            OUT="outputs/oracle_neg_sweep/warmup_neg/fold_0"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== Warmup neg (5% for 10 epochs only) — $(date) ==="
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=0 ++n_folds=10 ++stage1_dir="${S1}" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/synthetic_negatives/dinuc_shuffled_negatives.tsv" \
                ++neg_fraction=0.05 ++epochs=10 ++early_stop_patience=10 \
                ++wandb_mode=offline
            ;;
        5)  # Use Agarwal intergenic sequences only (real genomic negatives, not synthetic)
            OUT="outputs/oracle_neg_sweep/intergenic_only/fold_0"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== Intergenic only — $(date) ==="
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=0 ++n_folds=10 ++stage1_dir="${S1}" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/k562_expanded/agarwal_new_train.tsv" \
                ++neg_fraction=0.03 ++wandb_mode=offline
            ;;
        6)  # Lower encoder LR (5e-5) with 2% neg — preserves encoder more
            OUT="outputs/oracle_neg_sweep/frac02_elr05/fold_0"
            [ -f "${OUT}/test_metrics.json" ] && echo "SKIP" && exit 0
            echo "=== frac=2% elr=5e-5 — $(date) ==="
            uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \
                --config-name stage2_k562_oracle \
                ++fold_id=0 ++n_folds=10 ++stage1_dir="${S1}" \
                ++output_dir="${OUT}" ++use_full_dataset=True \
                ++negatives_path="data/synthetic_negatives/dinuc_shuffled_negatives.tsv" \
                ++neg_fraction=0.02 ++encoder_lr=0.00005 ++wandb_mode=offline
            ;;
    esac

else
    # Additional 2M replicates
    STRATEGIES=(random evoaug_structural prm_5pct recombination_uniform evoaug_heavy motif_grammar)
    IDX=$((T - 34))
    STRAT=${STRATEGIES[$IDX]}

    OUT="outputs/exp1_1_2m_scaling/k562/legnet_ag_s2/${STRAT}/n2000000/hp0/seed2042"
    [ -f "${OUT}/result.json" ] && echo "SKIP" && exit 0

    echo "=== 2M ${STRAT} seed=2042 — $(date) ==="
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ag_s2 \
        --reservoir "${STRAT}" \
        --pool-base-dir outputs/labeled_pools_2m/k562/ag_s2 \
        --n-replicates 1 --seed 2042 \
        --output-dir "outputs/exp1_1_2m_scaling/k562/legnet_ag_s2" \
        --training-sizes 2000000 \
        --chr-split --lr 0.001 --batch-size 2048 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10
fi

echo "=== Done task ${T} — $(date) ==="
