#!/usr/bin/env python
import sys
import json
from pathlib import Path
import pandas as pd
from alphagenome_ft import create_model_with_heads
import jax.numpy as jnp
import jax
import numpy as np
from scipy.stats import pearsonr, spearmanr

def _safe_corr(pred, target, fn):
    if pred.size < 2 or target.size < 2 or np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])

def _center_pad_4ch(seq_str, target_len=384):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    arr = np.zeros((len(seq_str), 4), dtype=np.float32)
    for i, c in enumerate(seq_str):
        if c in mapping: arr[i, mapping[c]] = 1.0
        
    curr_len = arr.shape[0]
    if curr_len == target_len: return arr
    if curr_len > target_len:
        start = (curr_len - target_len) // 2
        return arr[start:start+target_len]
        
    pad = np.zeros((target_len, 4), dtype=np.float32)
    left = (target_len - curr_len) // 2
    pad[left:left+curr_len, :] = arr
    return pad

def rc_seq(seq):
    comp = {'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N'}
    return "".join(comp.get(c, 'N') for c in reversed(seq))

def _standardize_yeast_sequence(sequence: str) -> str:
    """Pad a core 110bp yeast sequence up to 384bp identically to the YeastDataset."""
    FLANK_5_PRIME = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCTCG"
    FLANK_3_PRIME = "GGTTACGGCTGTT"
    partial_5_prime = FLANK_5_PRIME[-17:]
    
    seq = sequence
    if seq.endswith(FLANK_3_PRIME):
        seq = seq[:-len(FLANK_3_PRIME)]
    if seq.startswith(partial_5_prime):
        seq = seq[len(partial_5_prime):]
        
    core_len = 80
    if len(seq) < core_len:
        seq = seq + "N" * (core_len - len(seq))
    elif len(seq) > core_len:
        seq = seq[:core_len]
        
    seq_150 = FLANK_5_PRIME + seq + FLANK_3_PRIME
    
    ALPHAGENOME_FLANK_5_PRIME = "GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATC"
    ALPHAGENOME_FLANK_3_PRIME = "GGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA"
    
    total_valid = len(ALPHAGENOME_FLANK_5_PRIME) + 150 + len(ALPHAGENOME_FLANK_3_PRIME)
    left_pad = (384 - total_valid) // 2
    right_pad = 384 - total_valid - left_pad
    
    expanded = "N" * left_pad + ALPHAGENOME_FLANK_5_PRIME + seq_150 + ALPHAGENOME_FLANK_3_PRIME + "N" * right_pad
    return expanded

def main():
    if len(sys.argv) < 3:
        print("Usage: eval_yeast_ag.py <ckpt_dir> <head_name>")
        sys.exit(1)
        
    ckpt_dir = sys.argv[1]
    head_name = sys.argv[2]
    
    from albench.models.alphagenome_heads import register_s2f_head
    arch = "pool-flatten" if "pool" in head_name else "mlp-512-512"
    register_s2f_head(head_name=head_name, arch=arch, task_mode="yeast", num_tracks=18)
    
    # Resolve checkpoint path safely assuming we're on the HPC or locally
    base_weights = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    if not Path(base_weights).exists():
        base_weights = str(Path("checkpoints/alphagenome-jax-all_folds-v1").resolve())
        
    model = create_model_with_heads("all_folds", heads=[head_name], checkpoint_path=base_weights, use_encoder_output=True)

    def merge_nested_dicts(base, override):
        from collections.abc import Mapping
        if not isinstance(override, Mapping):
            return override
        if not isinstance(base, Mapping):
            return override
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                merged[k] = merge_nested_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    import orbax.checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    loaded_params, _ = checkpointer.restore(Path(ckpt_dir).resolve() / 'checkpoint')
    model._params = jax.device_put(merge_nested_dicts(model._params, loaded_params))
    
    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(params, state, sequences, jnp.zeros(len(sequences), dtype=jnp.int32), negative_strand_mask=jnp.zeros(len(sequences), dtype=bool), strand_reindexing=None)[head_name]

    bin_values = jnp.arange(18, dtype=jnp.float32)

    def _predict(seqs_str):
        if not seqs_str: return np.array([])
        x_fwd = np.stack([_center_pad_4ch(_standardize_yeast_sequence(s)) for s in seqs_str])
        x_rev = np.stack([_center_pad_4ch(rc_seq(_standardize_yeast_sequence(s))) for s in seqs_str])
        
        preds_fwd, preds_rev = [], []
        # Batch evaluation
        for i in range(0, len(x_fwd), 256):
            batch_params = (model._params, model._state)
            logits_fwd = predict_step(*batch_params, jnp.array(x_fwd[i:i+256]))
            logits_rev = predict_step(*batch_params, jnp.array(x_rev[i:i+256]))
            
            # Predict mean expression using classification bins for Yeast
            probs_fwd = jax.nn.softmax(logits_fwd, axis=-1)
            probs_rev = jax.nn.softmax(logits_rev, axis=-1)
            
            expr_fwd = jnp.sum(probs_fwd * bin_values, axis=-1)
            expr_rev = jnp.sum(probs_rev * bin_values, axis=-1)
            
            preds_fwd.append(np.array(expr_fwd))
            preds_rev.append(np.array(expr_rev))
            
        return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0

    print("Loading MAUDE expression map...")
    maude_df = pd.read_csv("data/yeast/filtered_test_data_with_MAUDE_expression.txt", sep="\t", header=None, names=["sequence", "expression"])
    seq_to_exp = dict(zip(maude_df["sequence"], maude_df["expression"]))
    
    subset_dir = Path("data/yeast/test_subset_ids")
    metrics = {}
    
    if (subset_dir / "all_random_seqs.csv").exists():
        print("Evaluating ID (random)...")
        df_id = pd.read_csv(subset_dir / "all_random_seqs.csv")
        df_id["exp"] = df_id["sequence"].map(seq_to_exp)
        df_id = df_id.dropna(subset=["exp"])
        
        preds = _predict(df_id["sequence"].tolist())
        targets = df_id["exp"].values
        pr = _safe_corr(preds, targets, pearsonr)
        metrics["id_pearson_r"] = pr
        print(f"ID Pearson R: {pr:.4f}")

    if (subset_dir / "yeast_seqs.csv").exists():
        print("Evaluating OOD (native yeast)...")
        df_ood = pd.read_csv(subset_dir / "yeast_seqs.csv")
        df_ood["exp"] = df_ood["sequence"].map(seq_to_exp)
        df_ood = df_ood.dropna(subset=["exp"])
        
        preds = _predict(df_ood["sequence"].tolist())
        targets = df_ood["exp"].values
        pr = _safe_corr(preds, targets, pearsonr)
        metrics["ood_pearson_r"] = pr
        print(f"OOD Pearson R: {pr:.4f}")
        
    if (subset_dir / "all_SNVs_seqs.csv").exists():
        print("Evaluating SNVs...")
        df_snv = pd.read_csv(subset_dir / "all_SNVs_seqs.csv")
        df_snv["alt_exp"] = df_snv["alt_sequence"].map(seq_to_exp)
        df_snv["ref_exp"] = df_snv["ref_sequence"].map(seq_to_exp)
        df_snv = df_snv.dropna(subset=["alt_exp", "ref_exp"])
        
        pred_alt = _predict(df_snv["alt_sequence"].tolist())
        pred_ref = _predict(df_snv["ref_sequence"].tolist())
        
        pred_delta = pred_alt - pred_ref
        true_delta = df_snv["alt_exp"].values - df_snv["ref_exp"].values
        
        pr = _safe_corr(pred_delta, true_delta, pearsonr)
        metrics["snv_pearson_r"] = pr
        print(f"SNV (Delta) Pearson R: {pr:.4f}")

    out_path = Path(ckpt_dir) / f"{head_name}_eval_yeast_metrics.json"
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    main()
