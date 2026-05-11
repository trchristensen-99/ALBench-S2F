"""Comprehensive bias evaluation v2 with mutation-rate sensitivity sweeps
and consolidated paper-figure groupings.

Eval sets generated:

  Tier 1 (real labels, accuracy metrics):
    - genomic_test_chr7_13
    - snv_pairs (delta R from SNV)
    - high_activity_designed (OOD)
    - gosai_ctrl_neg (dinuc-shuffled, mean=+0.27)
    - real_intergenic (mean=-0.50)

  Tier 2 (mutation sensitivity vs reference; no measured labels):
    - sub_1pct, sub_5pct, sub_10pct, sub_20pct  — substitution rates
    - indel_5pct, indel_20pct                    — insertion/deletion
    - inversion_small, inversion_large           — segment reversals
    - translocation_few, translocation_many      — within-seq shuffles

  Tier 3 (no labels, raw bias check):
    - synthetic_random_uniform

Consolidated groupings (for paper figure):
    - inactive_combined        = avg over {random, gosai_ctrl_neg, real_intergenic}
    - low_substitution         = avg over {sub_1, sub_5}
    - high_substitution        = avg over {sub_10, sub_20}
    - low_nonsub_mutation      = avg over {indel_5, inversion_small, translocation_few}
    - high_nonsub_mutation     = avg over {indel_20, inversion_large, translocation_many}

NOTE: Agarwal lentiMPRA REMOVED — different MPRA assay, true labels not comparable.
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
    return sum(1 for i in range(n) if seq[i:i+2] == "CG") / n


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


# ── Mutation generators ──────────────────────────────────────────────────────
def mutate_substitution(seq: str, rate: float, rng) -> str:
    """Replace `rate` fraction of bases with random non-self nucleotides."""
    seq_list = list(seq.upper())
    n = len(seq_list)
    n_mut = max(1, int(n * rate))
    positions = rng.choice(n, size=n_mut, replace=False)
    for p in positions:
        if seq_list[p] in "ACGT":
            alt = rng.choice([b for b in "ACGT" if b != seq_list[p]])
            seq_list[p] = alt
    return "".join(seq_list)


def mutate_indel(seq: str, rate: float, rng) -> str:
    """Insert random base or delete a base at `rate` fraction of positions.
    Preserves total length by balancing insertions and deletions."""
    seq_list = list(seq.upper())
    n_events = max(2, int(len(seq_list) * rate))
    for _ in range(n_events):
        if not seq_list:
            break
        pos = int(rng.integers(0, len(seq_list)))
        if rng.random() < 0.5 and len(seq_list) > 1:
            # delete
            del seq_list[pos]
        else:
            # insert
            seq_list.insert(pos, rng.choice(list("ACGT")))
    # Restore to original length
    target = len(seq)
    if len(seq_list) > target:
        seq_list = seq_list[:target]
    elif len(seq_list) < target:
        seq_list = seq_list + ["N"] * (target - len(seq_list))
    return "".join(seq_list)


def mutate_inversion(seq: str, size_range: tuple[int, int], n_inv: int, rng) -> str:
    """Reverse-complement `n_inv` random segments of size in size_range."""
    seq_list = list(seq.upper())
    rc_map = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    n = len(seq_list)
    for _ in range(n_inv):
        size = int(rng.integers(*size_range))
        start = int(rng.integers(0, max(1, n - size)))
        segment = seq_list[start : start + size]
        # Reverse complement
        inverted = [rc_map.get(b, b) for b in segment[::-1]]
        seq_list[start : start + size] = inverted
    return "".join(seq_list)


def mutate_translocation(seq: str, n_swaps: int, segment_size: int, rng) -> str:
    """Swap `n_swaps` pairs of segments within the sequence."""
    seq_list = list(seq.upper())
    n = len(seq_list)
    for _ in range(n_swaps):
        if n < 2 * segment_size + 4:
            break
        p1 = int(rng.integers(0, n - segment_size))
        p2 = int(rng.integers(0, n - segment_size))
        if abs(p1 - p2) < segment_size:
            continue
        a = seq_list[p1 : p1 + segment_size]
        b = seq_list[p2 : p2 + segment_size]
        seq_list[p1 : p1 + segment_size] = b
        seq_list[p2 : p2 + segment_size] = a
    return "".join(seq_list)


# ── Prediction helpers ──────────────────────────────────────────────────────
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


# ── Eval set builders ───────────────────────────────────────────────────────
def load_eval_sets(n_mutation_ref=1500):
    """Build all eval sets. Returns {name → (sequences, true_labels_or_None, ref_for_delta_or_None)}."""
    sets = {}

    # Tier 1 (real labels)
    df = pd.read_csv(REPO / "data/k562/test_sets/test_in_distribution_hashfrag.tsv", sep="\t")
    df = df.sample(min(3000, len(df)), random_state=42).reset_index(drop=True)
    sets["genomic_test_chr7_13"] = (df["sequence"].tolist(),
                                      df["K562_log2FC"].to_numpy(np.float32), None)

    df = pd.read_csv(REPO / "data/k562/test_sets/test_ood_designed_k562.tsv", sep="\t")
    if "K562_log2FC" in df.columns:
        df = df.sample(min(3000, len(df)), random_state=42).reset_index(drop=True)
        sets["high_activity_designed"] = (df["sequence"].tolist(),
                                            df["K562_log2FC"].to_numpy(np.float32), None)

    df = pd.read_parquet(REPO / "data/k562/gosai_ctrl_neg.parquet")
    sets["gosai_ctrl_neg_dinuc"] = (df["sequence"].tolist(),
                                     df["K562_log2FC"].to_numpy(np.float32), None)

    df = pd.read_csv(REPO / "data/synthetic_negatives/real_inter_all.tsv", sep="\t")
    df = df.sample(min(2000, len(df)), random_state=42).reset_index(drop=True)
    sets["real_intergenic"] = (df["sequence"].tolist(),
                                df["K562_log2FC"].to_numpy(np.float32), None)

    # Synthetic random (no labels)
    rng = np.random.default_rng(42)
    rand_seqs = ["".join(rng.choice(list("ACGT"), 200)) for _ in range(500)]
    sets["synthetic_random"] = (rand_seqs, None, None)

    # Tier 2: mutation eval sets — built from a fixed reference set of genomic test sequences
    ref_df = pd.read_csv(REPO / "data/k562/test_sets/test_in_distribution_hashfrag.tsv", sep="\t")
    ref_df = ref_df.sample(n_mutation_ref, random_state=43).reset_index(drop=True)
    ref_seqs = ref_df["sequence"].tolist()
    sets["_mutation_ref"] = (ref_seqs, None, None)  # internal — predicted as baseline for delta

    rng = np.random.default_rng(20260511)
    for rate, name in [(0.01, "sub_1pct"), (0.05, "sub_5pct"),
                       (0.10, "sub_10pct"), (0.20, "sub_20pct")]:
        mutated = [mutate_substitution(s, rate, rng) for s in ref_seqs]
        sets[name] = (mutated, None, "_mutation_ref")

    for rate, name in [(0.05, "indel_5pct"), (0.20, "indel_20pct")]:
        mutated = [mutate_indel(s, rate, rng) for s in ref_seqs]
        sets[name] = (mutated, None, "_mutation_ref")

    for size_range, n_inv, name in [((10, 30), 1, "inversion_small"),
                                      ((50, 100), 3, "inversion_large")]:
        mutated = [mutate_inversion(s, size_range, n_inv, rng) for s in ref_seqs]
        sets[name] = (mutated, None, "_mutation_ref")

    for n_swaps, seg_size, name in [(1, 15, "translocation_few"),
                                      (5, 20, "translocation_many")]:
        mutated = [mutate_translocation(s, n_swaps, seg_size, rng) for s in ref_seqs]
        sets[name] = (mutated, None, "_mutation_ref")

    return sets


# Consolidated groupings (for paper figure)
CONSOLIDATIONS = {
    "inactive_combined": ["synthetic_random", "gosai_ctrl_neg_dinuc", "real_intergenic"],
    "low_substitution_mut": ["sub_1pct", "sub_5pct"],
    "high_substitution_mut": ["sub_10pct", "sub_20pct"],
    "low_nonsub_mut": ["indel_5pct", "inversion_small", "translocation_few"],
    "high_nonsub_mut": ["indel_20pct", "inversion_large", "translocation_many"],
}


def evaluate(oracle_name: str, oracle_dir: Path, eval_sets: dict, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Predict on each set (cache to npz)
    preds = {}
    for name, (seqs, _, _) in eval_sets.items():
        cache = cache_dir / f"{oracle_name}_{name}.npz"
        if cache.exists():
            preds[name] = np.load(cache)["pred"]
        else:
            print(f"[{oracle_name}] predicting {name} (n={len(seqs)})...")
            preds[name] = predict_ensemble(seqs, oracle_dir)
            np.savez_compressed(cache, pred=preds[name])

    # Compute metrics per set
    per_set = {}
    for name, (seqs, true, ref_key) in eval_sets.items():
        if name == "_mutation_ref":
            continue
        pred = preds[name]
        cpg_vals = np.array([cpg_content(s) for s in seqs], dtype=np.float32)
        gc_vals = np.array([gc_content(s) for s in seqs], dtype=np.float32)
        info = {
            "n": int(len(pred)),
            "pred_mean": float(pred.mean()),
            "pred_std": float(pred.std()),
            "cpg_pred_corr": float(pearsonr(cpg_vals, pred)[0]),
            "gc_pred_corr": float(pearsonr(gc_vals, pred)[0]),
        }
        if true is not None:
            info["true_mean"] = float(true.mean())
            info["mean_residual"] = float((pred - true).mean())
            info["abs_mean_residual"] = float(abs((pred - true).mean()))
            info["pearson_r"] = float(pearsonr(pred, true)[0])
            info["spearman_rho"] = float(spearmanr(pred, true)[0])
            info["mse"] = float(np.mean((pred - true) ** 2))
            info["cpg_true_corr"] = float(pearsonr(cpg_vals, true)[0])
        if ref_key and ref_key in preds:
            ref_pred = preds[ref_key]
            delta = pred - ref_pred
            info["delta_pred_mean"] = float(delta.mean())
            info["delta_pred_abs_mean"] = float(np.abs(delta).mean())
            info["delta_pred_std"] = float(delta.std())
        per_set[name] = info

    # Consolidated groupings (for paper figure)
    consolidated = {}
    for group_name, member_names in CONSOLIDATIONS.items():
        members = [per_set[m] for m in member_names if m in per_set]
        if not members:
            continue
        # Average key metrics across members
        out = {"n_members": len(members), "members": member_names}
        # For inactive_combined: average pred_mean (with their respective targets)
        if "delta_pred_abs_mean" in members[0]:
            # Mutation groups: average |Δpred|
            out["delta_pred_abs_mean"] = float(np.mean([m["delta_pred_abs_mean"] for m in members]))
            out["delta_pred_std"] = float(np.mean([m["delta_pred_std"] for m in members]))
            out["delta_pred_mean"] = float(np.mean([m["delta_pred_mean"] for m in members]))
        if "abs_mean_residual" in members[0]:
            out["abs_mean_residual"] = float(np.mean([m["abs_mean_residual"] for m in members]))
        # pred_mean (raw)
        out["pred_mean"] = float(np.mean([m["pred_mean"] for m in members]))
        consolidated[group_name] = out

    return {
        "oracle": oracle_name,
        "per_set": per_set,
        "consolidated": consolidated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-name", required=True)
    ap.add_argument("--oracle-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    eval_sets = load_eval_sets()
    print(f"Loaded {len(eval_sets)} eval sets")
    for k, (s, t, ref) in eval_sets.items():
        truth = f", true mean={t.mean():+.3f}" if t is not None else ""
        delta_ref = f", delta-vs-{ref}" if ref else ""
        print(f"  {k}: n={len(s)}{truth}{delta_ref}")

    report = evaluate(args.oracle_name, args.oracle_dir, eval_sets, args.out_dir / "cache")

    out_path = args.out_dir / f"{args.oracle_name}_v2_report.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out_path}")

    # Print consolidated summary
    print(f"\n=== Consolidated metrics for {args.oracle_name} ===")
    for group, vals in report["consolidated"].items():
        print(f"  {group}: {vals}")


if __name__ == "__main__":
    main()
