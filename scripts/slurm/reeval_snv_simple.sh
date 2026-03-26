#!/bin/bash
# Simple SNV re-evaluation: use the original training scripts' eval functions
# with the correct cell-specific test_set_dir.
#
# The key fix: pass data/{cell}/test_sets/ instead of data/k562/test_sets/
# so the SNV file has cell-specific label columns.
#
# This script re-evaluates all S1 cached models and patches result JSONs.
#
#SBATCH --job-name=reeval_snv2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

echo "=== Re-evaluating SNV metrics with cell-specific labels ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Use the eval_ood_multicell.py infrastructure but call each model's eval function
# with SNV sequences instead of OOD sequences. We do this by calling the original
# scripts' evaluate functions directly.

uv run --no-sync python3 -c "
import json, sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

REPO = Path('.')
os.chdir(str(REPO))
sys.path.insert(0, str(REPO))

def safe_corr(x, y, fn):
    m = np.isfinite(x) & np.isfinite(y)
    return float(fn(x[m], y[m])[0]) if m.sum() > 2 else 0.0

def compute_metrics(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    return {
        'pearson_r': safe_corr(pred, true, pearsonr),
        'spearman_r': safe_corr(pred, true, spearmanr),
        'mse': float(np.mean((pred[m] - true[m])**2)),
        'n': int(m.sum()),
    }

def patch(path, snv_abs, snv_delta):
    d = json.loads(path.read_text())
    c = d.get('test_metrics', d)
    if snv_abs: c['snv_abs'] = snv_abs
    if snv_delta: c['snv_delta'] = snv_delta
    path.write_text(json.dumps(d, indent=2, default=str) + '\n')
    print('  Patched', path)

for cell in ['hepg2', 'sknsh']:
    fc = {'hepg2': 'HepG2_log2FC', 'sknsh': 'SKNSH_log2FC'}[cell]
    snv_path = REPO / 'data' / cell / 'test_sets' / 'test_snv_pairs_hashfrag.tsv'
    if not snv_path.exists():
        print('SKIP', cell, ': no SNV file')
        continue
    snv_df = pd.read_csv(snv_path, sep='\t')
    alt_col = fc + '_alt'
    delta_col = 'delta_' + fc
    ref_seqs = snv_df['sequence_ref'].tolist()
    alt_seqs = snv_df['sequence_alt'].tolist()
    alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
    delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
    print('Loaded SNV for', cell, ':', len(ref_seqs), 'pairs')

    # -- Malinois --
    print('\n=== Malinois', cell, '===')
    try:
        from scripts.eval_ood_multicell import eval_malinois
        for s in [0, 1, 2]:
            rd = REPO / 'outputs' / ('malinois_%s_3seeds' % cell) / ('seed_%d/seed_%d' % (s, s))
            if not rd.exists(): continue
            # eval_malinois loads model and predicts - but we need ref+alt not just ood
            # Use its internal loading but predict on our sequences
            # Load model directly
            import torch
            ckpt_path = None
            for p in rd.rglob('best_model.pt'):
                ckpt_path = p; break
            if ckpt_path is None: continue

            device = torch.device('cuda')
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

            from models.basset_branched import BassetBranched
            model = BassetBranched()
            model.load_state_dict(ckpt)
            model.to(device).eval()

            from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
            f5 = MPRA_UPSTREAM[-200:]
            f3 = MPRA_DOWNSTREAM[:200]
            mapping = {'A':0,'C':1,'G':2,'T':3}

            def predict_mal(seqs, bs=256):
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

            ref_p = predict_mal(ref_seqs)
            alt_p = predict_mal(alt_seqs)
            snv_abs = compute_metrics(alt_p, alt_true)
            snv_delta = compute_metrics(alt_p - ref_p, delta_true)
            print('  seed %d: snv_abs=%.4f snv_delta=%.4f' % (s, snv_abs['pearson_r'], snv_delta['pearson_r']))
            for jn in ['result.json', 'test_metrics.json']:
                jp = rd / jn
                if jp.exists(): patch(jp, snv_abs, snv_delta); break
            del model; torch.cuda.empty_cache()
    except Exception as e:
        print('  ERROR:', e)
        import traceback; traceback.print_exc()

    # -- AG S1 --
    print('\n=== AG S1', cell, '===')
    try:
        from experiments.train_oracle_alphagenome_hashfrag_cached import (
            evaluate_all_test_sets as ag_eval,
        )
        # AG S1 eval reads from test_set_dir, so we just pass the cell-specific dir
        # But it also needs the model loaded... too complex to refactor here.
        # Instead: just use the AG prediction infrastructure from eval_ood
        import jax, jax.numpy as jnp
        from alphagenome_ft import create_model_with_heads
        from models.alphagenome_heads import register_s2f_head
        from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

        f5 = MPRA_UPSTREAM[-200:]
        f3 = MPRA_DOWNSTREAM[:200]
        mapping = {'A':0,'C':1,'G':2,'T':3}

        def encode_one(seq):
            seq = seq.upper()
            if len(seq) < 200:
                pad = 200 - len(seq); seq = 'N'*(pad//2) + seq + 'N'*(pad-pad//2)
            elif len(seq) > 200:
                st = (len(seq)-200)//2; seq = seq[st:st+200]
            full = f5 + seq + f3
            oh = np.zeros((600,5), dtype=np.float32)
            for i,c in enumerate(full):
                if c in mapping: oh[i,mapping[c]] = 1.0
            oh[:,4] = 1.0
            return oh

        for s in [0, 1, 2]:
            rd = REPO / 'outputs' / ('ag_hashfrag_%s_cached' % cell) / ('seed_%d' % s)
            if not rd.exists(): continue
            jn = rd / 'test_metrics.json'
            if not jn.exists(): continue
            rdata = json.loads(jn.read_text())
            head_name = rdata.get('head_name', 'alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4')
            register_s2f_head(head_name=head_name, arch='boda-flatten-512-512', task_mode='human', num_tracks=1, dropout_rate=0.1)

            weights = '/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1'
            model = create_model_with_heads('all_folds', heads=[head_name], checkpoint_path=weights, use_encoder_output=True, detach_backbone=True)

            # Load checkpoint
            import orbax.checkpoint as ocp
            ckpt_path = (rd / 'best_model' / 'checkpoint').resolve()
            if ckpt_path.exists():
                checkpointer = ocp.PyTreeCheckpointer()
                restored = checkpointer.restore(str(ckpt_path), item=model._params)
                model._params = dict(restored) if isinstance(restored, dict) else restored

            @jax.jit
            def pred_step(params, state, seqs):
                return model._predict(params, state, seqs,
                    jnp.zeros(len(seqs), dtype=jnp.int32),
                    negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
                    strand_reindexing=None)[head_name]

            def predict_ag(seqs, bs=64):
                preds = []
                for i in range(0, len(seqs), bs):
                    oh = np.stack([encode_one(s) for s in seqs[i:i+bs]])
                    p = pred_step(model._params, model._state, jnp.array(oh))
                    preds.append(np.array(jnp.squeeze(p)).reshape(-1))
                return np.concatenate(preds)

            ref_p = predict_ag(ref_seqs)
            alt_p = predict_ag(alt_seqs)
            snv_abs = compute_metrics(alt_p, alt_true)
            snv_delta = compute_metrics(alt_p - ref_p, delta_true)
            print('  seed %d: snv_abs=%.4f snv_delta=%.4f' % (s, snv_abs['pearson_r'], snv_delta['pearson_r']))
            patch(jn, snv_abs, snv_delta)
            del model; jax.clear_caches()
    except Exception as e:
        print('  ERROR:', e)
        import traceback; traceback.print_exc()

print('\nDone.')
"

echo "Done: $(date)"
