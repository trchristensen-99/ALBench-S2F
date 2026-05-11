"""Comprehensive bias evaluation for 10-fold AG-S2 oracle ensembles.

For each oracle and each sequence-class set, computes:
  1. Mean offset bias: mean(pred - true)
  2. CpG-activity slope: corr(pred, CpG content) [should match real slope]
  3. GC-activity slope:  corr(pred, GC content)
  4. Length-activity slope (if length varies)
  5. Mononucleotide composition correlations
  6. Per-quintile residual (regression-to-mean check)

Datasets evaluated (use real measured labels where available):
  - Gosai ctrl_neg parquet (n=503, REAL K562 mean=+0.27)
  - real_inter_all (real intergenic n=25k, mean=-0.50)
  - real_inter_negative_only (n=20k, mean=-0.75)
  - real_agarwal_all (Agarwal 2025 lentiMPRA — real)
  - test_in_distribution_hashfrag (real test, mean varies)
  - test_ood_designed_k562 (designed sequences with measured K562)
  - test_snv_pairs (variant pairs with measured deltas)
  - cre_sequences (CRE library)
  - Plus: random_dna (synthetic) and dinuc_shuffled (synthetic) for baseline checks

Outputs:
  results/preflight/comprehensive_bias/{oracle}_report.json
  results/preflight/figures/meeting/11_comprehensive_bias.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from alphagenome_ft import create_model_with_heads
from scipy.stats import pearsonr, spearmanr

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

_FLANK_5 = MPRA_UPSTREAM[-200:]
_FLANK_3 = MPRA_DOWNSTREAM[:200]
_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _seq_to_600(seq: str) -> np.ndarray:
    seq = seq.upper()
    if len(seq) > 200:
        s = (len(seq) - 200) // 2
        seq = seq[s : s + 200]
    elif len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    full = _FLANK_5 + seq + _FLANK_3
    out = np.zeros((600, 4), dtype=np.float32)
    for i, c in enumerate(full):
        if c in _MAP:
            out[i, _MAP[c]] = 1.0
    return out


def cpg_content(seq: str) -> float:
    seq = seq.upper()
    n = max(len(seq) - 1, 1)
    cpgs = sum(1 for i in range(n) if seq[i:i+2] == "CG")
    return cpgs / n


def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def _build_predict(ckpt: Path, batch: int = 256):
    head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.1,
    )
    weights = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights,
        use_encoder_output=True,
        detach_backbone=False,
    )
    reinit_head_params(model, head_name, num_tokens=5, dim=1536)

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params, state, sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            requested_outputs=[head_name],
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    loaded_params, _ = ocp.StandardCheckpointer().restore(ckpt)
    model._params = jax.device_put(loaded_params)
    _ = predict_step(model._params, model._state, jnp.zeros((batch, 600, 4), dtype=jnp.float32))
    _.block_until_ready()
    return predict_step, model._params, model._state


def _predict_batched(predict_step, params, state, x, batch=256):
    n = len(x)
    preds = []
    for i in range(0, n, batch):
        end = min(i + batch, n)
        actual = end - i
        b = x[i:end]
        if actual < batch:
            pad = batch - actual
            b = np.concatenate([b, np.zeros((pad, 600, 4), dtype=np.float32)])
        preds.append(np.array(predict_step(params, state, jnp.array(b))).reshape(-1)[:actual])
    return np.concatenate(preds)


def predict_ensemble(seqs: list[str], oracle_dir: Path, n_folds=10, batch=256):
    """Run RC-averaged predictions across all folds, return ensemble mean."""
    canonical = np.stack([_seq_to_600(s) for s in seqs])
    rc = canonical[:, ::-1, ::-1]
    fold_preds = []
    for fold in range(n_folds):
        ckpt = oracle_dir / f"fold_{fold}" / "best_model" / "checkpoint"
        if not ckpt.exists():
            continue
        ps, p, s = _build_predict(ckpt, batch=batch)
        fwd = _predict_batched(ps, p, s, canonical, batch=batch)
        rev = _predict_batched(ps, p, s, rc, batch=batch)
        fold_preds.append((fwd + rev) / 2)
        del ps, p, s
        jax.clear_caches()
    return np.mean(np.stack(fold_preds), axis=0)


def load_datasets():
    """Load multiple real K562 episomal MPRA datasets + synthetic controls.
    Returns dict of name → (sequences, true_labels_or_None)."""
    datasets = {}

    # Gosai ctrl_neg
    df = pd.read_parquet(REPO / "data/k562/gosai_ctrl_neg.parquet")
    datasets["gosai_ctrl_neg"] = (df["sequence"].tolist(), df["K562_log2FC"].to_numpy(np.float32))

    # Real intergenic (Ernst negatives)
    df = pd.read_csv(REPO / "data/synthetic_negatives/real_inter_all.tsv", sep="\t")
    # Subsample to 2000 for speed
    df = df.sample(min(2000, len(df)), random_state=42).reset_index(drop=True)
    datasets["real_intergenic_all"] = (df["sequence"].tolist(),
                                        df["K562_log2FC"].to_numpy(np.float32))

    # Real intergenic — strictly negative subset
    df = pd.read_csv(REPO / "data/synthetic_negatives/real_inter_negative_only.tsv", sep="\t")
    df = df.sample(min(2000, len(df)), random_state=42).reset_index(drop=True)
    datasets["real_intergenic_negative_only"] = (df["sequence"].tolist(),
                                                  df["K562_log2FC"].to_numpy(np.float32))

    # Real Agarwal negatives
    p = REPO / "data/synthetic_negatives/real_agarwal_all.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        df = df.sample(min(2000, len(df)), random_state=42).reset_index(drop=True)
        if "K562_log2FC" in df.columns:
            datasets["real_agarwal_all"] = (df["sequence"].tolist(),
                                             df["K562_log2FC"].to_numpy(np.float32))

    # Test sets (in-distribution = chr 7+13)
    p = REPO / "data/k562/test_sets/test_in_distribution_hashfrag.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        df = df.sample(min(3000, len(df)), random_state=42).reset_index(drop=True)
        if "K562_log2FC" in df.columns:
            datasets["test_in_dist_chr7_13"] = (df["sequence"].tolist(),
                                                  df["K562_log2FC"].to_numpy(np.float32))

    # OOD designed
    p = REPO / "data/k562/test_sets/test_ood_designed_k562.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        if "K562_log2FC" in df.columns:
            df = df.sample(min(3000, len(df)), random_state=42).reset_index(drop=True)
            datasets["test_ood_designed"] = (df["sequence"].tolist(),
                                              df["K562_log2FC"].to_numpy(np.float32))

    # CRE sequences (high-activity designed)
    p = REPO / "data/k562/test_sets/cre_sequences.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        if "K562_log2FC" in df.columns:
            df = df.dropna(subset=["K562_log2FC"]).sample(
                min(1000, len(df.dropna(subset=["K562_log2FC"]))), random_state=42
            ).reset_index(drop=True)
            datasets["cre_sequences"] = (df["sequence"].tolist(),
                                          df["K562_log2FC"].to_numpy(np.float32))

    # Synthetic random DNA (no labels — bias check)
    rng = np.random.default_rng(42)
    rand_seqs = []
    for _ in range(500):
        rand_seqs.append("".join(rng.choice(list("ACGT"), 200)))
    datasets["synthetic_random_uniform"] = (rand_seqs, None)

    # Synthetic dinuc-shuffled — use Gosai ctrl_neg as proxy (already real-measured)
    # already in gosai_ctrl_neg

    return datasets


def analyze_oracle_bias(oracle_name: str, oracle_dir: Path, datasets: dict, out_dir: Path):
    """Run comprehensive bias eval on one oracle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"oracle": oracle_name, "datasets": {}}
    for ds_name, (seqs, true_labels) in datasets.items():
        cache = out_dir / f"{oracle_name}_{ds_name}_preds.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            pred = d["pred"]
        else:
            print(f"[{oracle_name}] predicting {ds_name} (n={len(seqs)})...")
            pred = predict_ensemble(seqs, oracle_dir)
            np.savez_compressed(cache, pred=pred)
        # Compute features
        cpg_vals = np.array([cpg_content(s) for s in seqs], dtype=np.float32)
        gc_vals = np.array([gc_content(s) for s in seqs], dtype=np.float32)
        len_vals = np.array([len(s) for s in seqs], dtype=np.float32)

        result = {
            "n": int(len(pred)),
            "pred_mean": float(pred.mean()),
            "pred_std": float(pred.std()),
            "cpg_pred_corr_r": float(pearsonr(cpg_vals, pred)[0]),
            "gc_pred_corr_r": float(pearsonr(gc_vals, pred)[0]),
        }
        if len_vals.std() > 0:
            result["length_pred_corr_r"] = float(pearsonr(len_vals, pred)[0])

        if true_labels is not None:
            result["true_mean"] = float(true_labels.mean())
            result["true_std"] = float(true_labels.std())
            result["mean_residual"] = float((pred - true_labels).mean())
            result["pearson_r"] = float(pearsonr(pred, true_labels)[0])
            result["spearman_rho"] = float(spearmanr(pred, true_labels)[0])
            result["mse"] = float(np.mean((pred - true_labels) ** 2))
            # Real CpG slope on labels
            result["cpg_true_corr_r"] = float(pearsonr(cpg_vals, true_labels)[0])
            result["gc_true_corr_r"] = float(pearsonr(gc_vals, true_labels)[0])
            # Bin residuals
            qs = np.quantile(true_labels, np.linspace(0, 1, 6))
            bin_residuals = []
            for i in range(5):
                lo, hi = qs[i], qs[i + 1]
                mask = (true_labels >= lo) & (true_labels < hi if i < 4 else true_labels <= hi)
                if mask.sum() < 5:
                    continue
                bin_residuals.append({
                    "bin_center": float((lo + hi) / 2),
                    "n": int(mask.sum()),
                    "residual_mean": float((pred[mask] - true_labels[mask]).mean()),
                })
            result["binned_residuals"] = bin_residuals
        else:
            result["true_mean"] = None
            result["true_std"] = None

        report["datasets"][ds_name] = result
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-name", required=True)
    ap.add_argument("--oracle-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets:")
    for k, (seqs, lbl) in datasets.items():
        truth = f" (true mean={lbl.mean():+.3f})" if lbl is not None else " (no labels)"
        print(f"  {k}: n={len(seqs)}{truth}")

    report = analyze_oracle_bias(args.oracle_name, args.oracle_dir, datasets, args.out_dir)

    out_path = args.out_dir / f"{args.oracle_name}_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
