#!/bin/bash
# Fix remaining bar plot anomalies that need GPU compute.
#
# Array:
#   0: AG all-folds S1 HepG2 SNV (needs AG encoder to build SNV cache)
#   1: AG all-folds S1 SKNSH SNV
#   2: Borzoi HepG2 OOD (needs Borzoi encoder for cell-specific OOD sequences)
#   3: Borzoi SKNSH OOD
#
#SBATCH --job-name=fix_anom
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

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

T=$SLURM_ARRAY_TASK_ID
echo "=== Fix anomaly task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

case ${T} in
0|1)
    # AG all-folds S1 HepG2/SKNSH SNV — need AlphaGenome encoder
    CELLS=("hepg2" "sknsh")
    CELL="${CELLS[$T]}"
    echo "AG all-folds S1 ${CELL} SNV — building AG SNV embeddings"

    uv run --no-sync python3 -c "
import sys, json, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import pandas as pd
sys.path.insert(0, '.')

CELL = '${CELL}'
fc_col = {'hepg2': 'HepG2_log2FC', 'sknsh': 'SKNSH_log2FC'}[CELL]

# Load SNV data
snv_df = pd.read_csv('data/%s/test_sets/test_snv_pairs_hashfrag.tsv' % CELL, sep='\t')
ref_seqs = snv_df['sequence_ref'].tolist()
alt_seqs = snv_df['sequence_alt'].tolist()
alt_true = snv_df[fc_col + '_alt'].to_numpy(dtype=np.float32)
delta_true = snv_df['delta_' + fc_col].to_numpy(dtype=np.float32)

print('SNV data: %d pairs' % len(ref_seqs))

# Use AlphaGenome to encode sequences
import jax
import jax.numpy as jnp
from alphagenome_ft import create_model_with_heads
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

# Load AG model
head_name = 'alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4'
register_s2f_head(head_name=head_name, arch='boda-flatten-512-512', task_mode='human', num_tracks=1, dropout_rate=0.1)

weights = '/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1'
model = create_model_with_heads('all_folds', heads=[head_name], checkpoint_path=weights, use_encoder_output=True, detach_backbone=True)

# Load trained checkpoint for this cell
import orbax.checkpoint as ocp
from collections.abc import Mapping

# Try all seeds
for seed in [0, 1, 2]:
    rd = Path('outputs/ag_hashfrag_%s_cached/seed_%d' % (CELL, seed))
    ckpt_path = (rd / 'best_model' / 'checkpoint').resolve()
    if not ckpt_path.exists():
        print('  Skipping seed %d: no checkpoint' % seed)
        continue

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored = checkpointer.restore(str(ckpt_path), item=model._params)
        if isinstance(restored, Mapping):
            model._params = dict(restored)
        else:
            model._params = restored
    except Exception as e:
        print('  Skipping seed %d: %s' % (seed, e))
        continue

    # Encode sequences
    f5 = MPRA_UPSTREAM[-200:]
    f3 = MPRA_DOWNSTREAM[:200]
    mapping = {'A':0,'C':1,'G':2,'T':3}

    def encode_one(seq):
        seq = seq.upper()
        if len(seq) < 200:
            pad = 200 - len(seq)
            seq = 'N'*(pad//2) + seq + 'N'*(pad-pad//2)
        elif len(seq) > 200:
            st = (len(seq)-200)//2
            seq = seq[st:st+200]
        full = f5 + seq + f3
        oh = np.zeros((600, 5), dtype=np.float32)
        for i, c in enumerate(full):
            if c in mapping: oh[i, mapping[c]] = 1.0
        oh[:, 4] = 1.0
        return oh

    @jax.jit
    def predict_step(params, state, seqs):
        return model._predict(params, state, seqs,
            jnp.zeros(len(seqs), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
            strand_reindexing=None)[head_name]

    def predict_batch(seqs, bs=64):
        preds = []
        for i in range(0, len(seqs), bs):
            oh = np.stack([encode_one(s) for s in seqs[i:i+bs]])
            p = predict_step(model._params, model._state, jnp.array(oh))
            preds.append(np.array(jnp.squeeze(p)).reshape(-1))
            if i % (bs*50) == 0 and i > 0:
                print('    %d/%d' % (i, len(seqs)))
        return np.concatenate(preds)

    print('  Seed %d: predicting ref+alt (%d each)...' % (seed, len(ref_seqs)))
    ref_p = predict_batch(ref_seqs)
    alt_p = predict_batch(alt_seqs)
    delta_p = alt_p - ref_p

    def safe_corr(x, y, fn):
        m = np.isfinite(x) & np.isfinite(y)
        return float(fn(x[m], y[m])[0]) if m.sum() > 2 else 0.0

    snv_abs_r = safe_corr(alt_p, alt_true, pearsonr)
    snv_delta_r = safe_corr(delta_p, delta_true, pearsonr)

    print('  Seed %d: snv_abs=%.4f snv_delta=%.4f' % (seed, snv_abs_r, snv_delta_r))

    # Patch
    rj = rd / 'test_metrics.json'
    if rj.exists():
        r = json.loads(rj.read_text())
        r['snv_abs'] = {'pearson_r': snv_abs_r, 'spearman_r': safe_corr(alt_p, alt_true, spearmanr), 'n': len(alt_true)}
        r['snv_delta'] = {'pearson_r': snv_delta_r, 'spearman_r': safe_corr(delta_p, delta_true, spearmanr), 'n': len(delta_true)}
        rj.write_text(json.dumps(r, indent=2, default=str))
        print('  Patched %s' % rj)
"
    ;;

2|3)
    # Borzoi HepG2/SKNSH OOD — need to encode cell-specific OOD sequences
    CELLS=("hepg2" "sknsh")
    CELL="${CELLS[$((T-2))]}"
    echo "Borzoi ${CELL} OOD — encoding cell-specific OOD sequences"

    uv run --no-sync python3 scripts/build_and_eval_snv_cache.py \
        --model borzoi \
        --cell-line "${CELL}" \
        --cache-dir "outputs/borzoi_k562_cached/embedding_cache" \
        --result-dir "outputs/borzoi_${CELL}_cached/seed_0/seed_0" \
        --skip-cache

    echo "NOTE: Borzoi OOD requires encoding OOD sequences, not using SNV cache."
    echo "The build_and_eval_snv_cache only handles SNV, not OOD."
    echo "OOD fix needs dedicated encoding of cell-specific OOD sequences."
    ;;

esac

echo "=== Done: $(date) ==="
