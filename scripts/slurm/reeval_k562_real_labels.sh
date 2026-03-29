#!/bin/bash
# Re-evaluate K562 models that were evaluated against oracle labels.
# Three models need re-evaluation against real (ground truth) labels:
#   - DREAM-RNN ens3 (outputs/dream_rnn_k562_with_preds)
#   - Malinois (outputs/malinois_k562_with_preds)
#   - NTv3-post S1 (outputs/ntv3_post_k562_3seeds)
#
# Also re-evaluates AG all-folds S1 HepG2/SK-N-SH SNV using cached embeddings.
#
# Array:
#   0: DREAM-RNN ens3 K562
#   1: Malinois K562
#   2: NTv3-post S1 K562
#   3: AG all-folds S1 HepG2 SNV (cached)
#   4: AG all-folds S1 SKNSH SNV (cached)
#
#SBATCH --job-name=reeval_real
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== Re-eval real labels task=${T} ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

case ${T} in
0)
    echo "DREAM-RNN ens3 K562 — re-eval against real labels"
    # Load the saved model and evaluate on real test sets
    uv run --no-sync python3 -c "
import sys, json, torch, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, '.')
from data.k562 import K562Dataset
from models.dream_rnn_student import DREAMRNNStudent, TrainConfig

# Load model
result_dir = Path('outputs/dream_rnn_k562_with_preds/seed_42/seed_42/fraction_1.0000')
ckpt = torch.load(result_dir / 'best_model.pt', map_location='cpu')
student = DREAMRNNStudent(
    input_channels=5, sequence_length=200, task_mode='k562',
    ensemble_size=ckpt.get('ensemble_size', 3),
    train_config=TrainConfig()
)
for i, sd in enumerate(ckpt['model_state_dicts']):
    student.models[i].load_state_dict(sd)

# Evaluate on real test sets
from experiments.exp1_1_scaling import _evaluate_ground_truth_test, evaluate_predictions
metrics = _evaluate_ground_truth_test(student, 'k562', evaluate_predictions)
print('Metrics:', json.dumps({k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in metrics.items()}, indent=2))

# Patch result
rj = result_dir / 'result.json'
r = json.loads(rj.read_text())
r['test_metrics'] = metrics
rj.write_text(json.dumps(r, indent=2, default=str))
print('Patched', rj)
"
    ;;

1)
    echo "Malinois K562 — re-eval against real labels"
    uv run --no-sync python3 -c "
import sys, json, torch, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, '.')
from data.k562 import K562Dataset
from models.basset_branched import BassetBranched
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

device = torch.device('cuda')
result_dir = Path('outputs/malinois_k562_with_preds/seed_42/seed_42')
ckpt = torch.load(result_dir / 'best_model.pt', map_location='cpu', weights_only=False)
model = BassetBranched()
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.to(device).eval()

f5, f3 = MPRA_UPSTREAM[-200:], MPRA_DOWNSTREAM[:200]
mapping = {'A':0,'C':1,'G':2,'T':3}

def predict(seqs, bs=256):
    preds = []
    for i in range(0, len(seqs), bs):
        batch = seqs[i:i+bs]
        enc = []
        for seq in batch:
            full = f5 + seq[:200] + f3
            oh = np.zeros((4,600), dtype=np.float32)
            for j,c in enumerate(full[:600]):
                if c in mapping: oh[mapping[c],j] = 1.0
            enc.append(oh)
        x = torch.tensor(np.stack(enc), device=device)
        with torch.no_grad():
            preds.append(model(x).cpu().numpy().reshape(-1))
    return np.concatenate(preds)

# Load test data
ds = K562Dataset('data/k562', split='test', label_column='K562_log2FC')
seqs = list(ds.sequences)
labels = ds.labels.astype(np.float32)

print('Evaluating on %d test sequences...' % len(seqs))
preds = predict(seqs)

def safe_corr(x, y, fn):
    m = np.isfinite(x) & np.isfinite(y)
    return float(fn(x[m], y[m])[0]) if m.sum() > 2 else 0.0

metrics = {
    'in_distribution': {
        'pearson_r': safe_corr(preds, labels, pearsonr),
        'spearman_r': safe_corr(preds, labels, spearmanr),
        'mse': float(np.mean((preds[np.isfinite(labels)] - labels[np.isfinite(labels)])**2)),
        'n': int(np.isfinite(labels).sum()),
    }
}

# SNV
import pandas as pd
snv = pd.read_csv('data/k562/test_sets/test_snv_pairs_hashfrag.tsv', sep='\t')
ref_p = predict(snv['sequence_ref'].tolist())
alt_p = predict(snv['sequence_alt'].tolist())
alt_true = snv['K562_log2FC_alt'].to_numpy(dtype=np.float32)
delta_true = snv['delta_log2FC'].to_numpy(dtype=np.float32)
metrics['snv_abs'] = {
    'pearson_r': safe_corr(alt_p, alt_true, pearsonr),
    'spearman_r': safe_corr(alt_p, alt_true, spearmanr),
    'mse': float(np.mean((alt_p - alt_true)**2)),
    'n': len(alt_true),
}
metrics['snv_delta'] = {
    'pearson_r': safe_corr(alt_p - ref_p, delta_true, pearsonr),
    'spearman_r': safe_corr(alt_p - ref_p, delta_true, spearmanr),
    'mse': float(np.mean((alt_p - ref_p - delta_true)**2)),
    'n': len(delta_true),
}

# OOD
ood = pd.read_csv('data/k562/test_sets/test_ood_designed_k562.tsv', sep='\t')
ood_p = predict(ood['sequence'].tolist())
ood_true = ood['K562_log2FC'].to_numpy(dtype=np.float32)
metrics['ood'] = {
    'pearson_r': safe_corr(ood_p, ood_true, pearsonr),
    'spearman_r': safe_corr(ood_p, ood_true, spearmanr),
    'mse': float(np.mean((ood_p[np.isfinite(ood_true)] - ood_true[np.isfinite(ood_true)])**2)),
    'n': int(np.isfinite(ood_true).sum()),
}

print('Results:', json.dumps({k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in metrics.items()}, indent=2))

rj = result_dir / 'result.json'
r = json.loads(rj.read_text())
r['test_metrics'] = metrics
rj.write_text(json.dumps(r, indent=2, default=str))
print('Patched', rj)
"
    ;;

2)
    echo "NTv3-post S1 K562 — re-eval using cached embedding approach"
    # NTv3 S1 was trained on cached embeddings, so we use the same approach
    uv run --no-sync python3 scripts/build_and_eval_snv_cache.py \
        --model ntv3_post --cell-line k562 \
        --cache-dir outputs/ntv3_post_k562_cached/embedding_cache \
        --result-dirs outputs/ntv3_post_k562_3seeds/seed_*/seed_* \
        --skip-cache
    ;;

3)
    echo "AG all-folds S1 HepG2 SNV — build cache and eval"
    uv run --no-sync python3 scripts/build_and_eval_snv_cache.py \
        --model enformer --cell-line hepg2 \
        --cache-dir outputs/enformer_k562_cached/embedding_cache \
        --result-dirs outputs/ag_hashfrag_hepg2_cached/seed_0 outputs/ag_hashfrag_hepg2_cached/seed_1 \
        --skip-cache
    echo "NOTE: AG uses different embedding format - may need dedicated script"
    ;;

4)
    echo "AG all-folds S1 SKNSH SNV — placeholder"
    echo "AG all-folds uses AlphaGenome embeddings, not foundation model embeddings"
    echo "The SNV re-eval needs to use the AG encoder, not Enformer"
    echo "This should be handled by a dedicated AG SNV eval script"
    ;;

esac

echo "Done: $(date)"
