"""Pre-flight Task 8: one-cycle acquisition sanity check.

For one acquisition method, sample ``d_acquired`` sequences from the
``d_init``-sized chromosome-split K562 pool, compute the Jaccard distance
to a random selection of the same size, and write a small JSON report.
The acceptance criterion (per the pre-flight plan): the method must run
without errors AND Jaccard distance to random > 0.3 (i.e., the method
selects materially different sequences than random — confirming it isn't
silently identity-mapping).

Method categories supported here:
  - **reservoir** methods (``prm_5``, ``prm_20``, ``gc_matched``,
    ``dinuc_shuffle``, ``motif_grammar``): plumbed to the existing
    ``albench.reservoir`` infrastructure via the same dispatch as
    ``experiments/exp1_2_acquisition.py::_generate_from_reservoir``.
  - **model-based** methods (``uncertainty_ensemble``, ``uncertainty_mc_dropout``,
    ``diversity_kmeans``, ``diversity_max_distance``): require a trained
    student ensemble. Until the main-sweep students are trained, we use
    a sequence-feature proxy (k-mer-based) that's deterministic and
    differentiable from random — sufficient for the runs-without-errors
    + non-trivial-Jaccard sanity check, NOT for any quantitative claim
    about method efficacy.

Outputs (under --output_dir):
    selected_idx.npy   — sequence indices (relative to the pool) that
                         the method picked
    random_idx.npy     — the random baseline indices (same size, same seed)
    jaccard.json       — {jaccard_distance, jaccard_index, method,
                          method_class, n_selected, n_overlap}

Usage:
    uv run --no-sync python scripts/preflight/acquire_one_cycle.py \\
        --method prm_5 --d_init 600000 --d_acquired 4000 --seed 42 \\
        --output_dir results/preflight/task8_acquisition_sanity/prm_5/seed42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

RESERVOIR_METHODS = {
    "prm_5": "prm_5pct",
    "prm_20": "prm_20pct",
    "prm_10": "prm_10pct",
    "prm_1": "prm_1pct",
    "gc_matched": "gc_matched",
    "dinuc_shuffle": "dinuc_shuffle",
    "motif_grammar": "motif_grammar",
    "evoaug_heavy": "evoaug_heavy",
    "evoaug_structural": "evoaug_structural",
    "recombination": "recombination_uniform",
}
MODEL_BASED_METHODS = {
    "uncertainty_ensemble",
    "uncertainty_mc_dropout",
    "diversity_kmeans",
    "diversity_max_distance",
}


def _load_pool() -> tuple[list[str], np.ndarray]:
    """Load the K562 chromosome-split train pool from the new ref+alt cache
    if present, falling back to legacy pool. Returns (sequences, labels)."""
    new_pool = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt" / "pool" / "train.parquet"
    if new_pool.exists():
        import pandas as pd

        df = pd.read_parquet(new_pool)
        seqs = df["sequence"].astype(str).tolist()
        labels = df["K562_log2FC"].to_numpy(dtype=np.float32)
        print(f"  Loaded {len(seqs):,} train sequences from ref+alt parquet pool.")
        return seqs, labels
    raise SystemExit(
        f"Pool not found at {new_pool}. Build it first via "
        "scripts/preflight/build_k562_refalt_pool.py."
    )


def _kmer_counts(seqs: list[str], k: int = 5, max_seqs: int | None = None) -> np.ndarray:
    """Return (N, 4**k) k-mer count matrix. For diversity proxies."""
    if max_seqs is not None and len(seqs) > max_seqs:
        seqs = seqs[:max_seqs]
    bases = "ACGT"
    kmer_to_idx = {
        "".join(b): i for i, b in enumerate(__import__("itertools").product(bases, repeat=k))
    }
    out = np.zeros((len(seqs), 4**k), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        for j in range(len(s) - k + 1):
            kmer = s[j : j + k]
            if kmer in kmer_to_idx:
                out[i, kmer_to_idx[kmer]] += 1.0
    # L2 normalize so distance-based methods aren't dominated by length
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _select_reservoir(
    method: str, pool_seqs: list[str], pool_labels: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """Use the existing reservoir dispatch to select (or generate) n examples.

    For reservoir methods, the canonical use is sequence GENERATION (not
    pool selection). For the Task 8 sanity check, we treat this as a
    pool-selection problem: generate ``n`` sequences from the reservoir
    seeded by the pool, then for each generated seq find the closest
    pool sequence (k-mer cosine) and return that index. This way the
    method's output is a subset of the pool, consistent with what the
    main sweep will measure as "data acquired".
    """
    spec = importlib.util.spec_from_file_location(
        "exp1_2", REPO / "experiments" / "exp1_2_acquisition.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res_name = RESERVOIR_METHODS[method]
    print(f"  Using reservoir '{res_name}' to generate {n:,} candidate sequences …")
    candidate_seqs = mod._generate_from_reservoir(
        reservoir_name=res_name,
        n=n,
        seed=seed,
        task="k562",
        pool_seqs=pool_seqs,
        pool_labels=pool_labels,
        oracle=None,
    )
    # Map each generated seq to the nearest pool seq via 5-mer cosine.
    # For efficiency, only embed a sub-sampled pool (else (600k, 1024) is heavy).
    pool_idx = np.arange(len(pool_seqs))
    if len(pool_seqs) > 100_000:
        rng = np.random.default_rng(seed)
        pool_idx = rng.choice(len(pool_seqs), size=100_000, replace=False)
    pool_subset = [pool_seqs[i] for i in pool_idx]
    print(f"  Computing 5-mer features for {len(pool_subset):,} pool seqs and {n:,} candidates …")
    K = 5
    pool_kmer = _kmer_counts(pool_subset, k=K)
    cand_kmer = _kmer_counts(candidate_seqs, k=K)
    # Nearest pool seq per candidate (cosine sim = dot on L2-normalized vectors)
    print("  Mapping candidates to nearest pool seq …")
    selected = np.zeros(n, dtype=np.int64)
    chunk = 1000
    for i in range(0, len(candidate_seqs), chunk):
        end = min(i + chunk, len(candidate_seqs))
        sims = cand_kmer[i:end] @ pool_kmer.T  # (chunk, |pool_subset|)
        best = sims.argmax(axis=1)
        selected[i:end] = pool_idx[best]
    # Deduplicate; if collisions cause < n unique, top-up with random non-overlap
    unique_sel = np.unique(selected)
    if len(unique_sel) < n:
        rng = np.random.default_rng(seed + 1)
        remaining = np.setdiff1d(np.arange(len(pool_seqs)), unique_sel, assume_unique=True)
        topup = rng.choice(remaining, size=n - len(unique_sel), replace=False)
        selected = np.concatenate([unique_sel, topup])
    else:
        selected = unique_sel[:n]
    return selected


def _select_proxy_uncertainty(pool_seqs: list[str], n: int, seed: int) -> np.ndarray:
    """Proxy for ``uncertainty_*``: pick sequences with highest k-mer
    entropy (heuristic for diverse / hard-to-predict regions). NOT a
    real uncertainty score — sufficient only for the sanity check.

    The main-sweep uncertainty acquisition will use a trained student
    ensemble's prediction variance.
    """
    print(f"  PROXY mode: scoring pool by 5-mer entropy (NOT real uncertainty).")
    K = 5
    sample_n = min(len(pool_seqs), 200_000)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(pool_seqs), size=sample_n, replace=False)
    bases = "ACGT"
    kmer_to_idx = {
        "".join(b): i for i, b in enumerate(__import__("itertools").product(bases, repeat=K))
    }
    entropies = np.zeros(sample_n, dtype=np.float32)
    for i, idx in enumerate(sample_idx):
        s = pool_seqs[idx].upper()
        cnt = Counter(s[j : j + K] for j in range(len(s) - K + 1) if s[j : j + K] in kmer_to_idx)
        if not cnt:
            entropies[i] = 0.0
            continue
        total = sum(cnt.values())
        probs = np.array([c / total for c in cnt.values()])
        entropies[i] = float(-(probs * np.log(probs + 1e-12)).sum())
    # Top-n by entropy
    top = np.argsort(-entropies)[:n]
    return sample_idx[top]


def _select_proxy_diversity(method: str, pool_seqs: list[str], n: int, seed: int) -> np.ndarray:
    """Proxy for ``diversity_*`` that ranks pool sequences by a deterministic
    feature-space measure. Real diversity acquisition uses learned
    student embeddings — not implemented here."""
    print(f"  PROXY mode: 5-mer cosine farthest-first (NOT real diversity).")
    K = 5
    sample_n = min(len(pool_seqs), 50_000)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(pool_seqs), size=sample_n, replace=False)
    feats = _kmer_counts([pool_seqs[i] for i in sample_idx], k=K)
    # Greedy farthest-first start from a deterministic seed point
    if method == "diversity_kmeans":
        # Quick proxy: hash-bucket into n buckets, pick centroid-nearest per bucket
        n_buckets = n
        bucket_idx = np.arange(sample_n) % n_buckets
        chosen = np.zeros(n, dtype=np.int64)
        for b in range(n_buckets):
            mask = bucket_idx == b
            if mask.sum() == 0:
                chosen[b] = sample_idx[0]
                continue
            mean = feats[mask].mean(axis=0)
            sims = feats[mask] @ mean
            best_in_mask = np.where(mask)[0][int(sims.argmax())]
            chosen[b] = sample_idx[best_in_mask]
        return chosen
    # diversity_max_distance: greedy farthest-first
    chosen_local = [0]
    min_sim_to_chosen = feats @ feats[0]
    for _ in range(n - 1):
        # Pick the seq with smallest max similarity to chosen → farthest
        next_i = int(min_sim_to_chosen.argmin())
        chosen_local.append(next_i)
        new_sim = feats @ feats[next_i]
        min_sim_to_chosen = np.maximum(min_sim_to_chosen, new_sim)
    return sample_idx[np.array(chosen_local)]


def _jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Return (distance = 1 - index, index, n_overlap)."""
    sa = set(a.tolist())
    sb = set(b.tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0, 1.0, 0
    j_idx = inter / union
    return 1.0 - j_idx, j_idx, inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="One of the supported method names.")
    ap.add_argument("--d_init", type=int, default=600_000)
    ap.add_argument("--d_acquired", type=int, default=4_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_seqs, pool_labels = _load_pool()
    if args.d_init > len(pool_seqs):
        raise SystemExit(f"d_init={args.d_init} > pool size {len(pool_seqs):,}")

    rng = np.random.default_rng(args.seed)
    init_idx = rng.choice(len(pool_seqs), size=args.d_init, replace=False)
    init_seqs = [pool_seqs[i] for i in init_idx]
    init_labels = pool_labels[init_idx]

    # Acquisition
    method = args.method
    if method in RESERVOIR_METHODS:
        method_class = "reservoir"
        local_sel = _select_reservoir(method, init_seqs, init_labels, args.d_acquired, args.seed)
    elif method in MODEL_BASED_METHODS:
        method_class = "model_proxy"
        if method.startswith("uncertainty"):
            local_sel = _select_proxy_uncertainty(init_seqs, args.d_acquired, args.seed)
        else:
            local_sel = _select_proxy_diversity(method, init_seqs, args.d_acquired, args.seed)
    else:
        raise SystemExit(
            f"Unknown method '{method}'. Supported: "
            f"{sorted(RESERVOIR_METHODS)} + {sorted(MODEL_BASED_METHODS)}"
        )
    selected_idx = init_idx[local_sel]

    # Random baseline (different seed to make Jaccard non-degenerate)
    rng_b = np.random.default_rng(args.seed + 17)
    random_local_sel = rng_b.choice(len(init_idx), size=args.d_acquired, replace=False)
    random_idx = init_idx[random_local_sel]

    distance, index, overlap = _jaccard(selected_idx, random_idx)

    np.save(out_dir / "selected_idx.npy", selected_idx)
    np.save(out_dir / "random_idx.npy", random_idx)
    summary = {
        "method": method,
        "method_class": method_class,
        "d_init": args.d_init,
        "d_acquired": args.d_acquired,
        "n_selected": int(len(selected_idx)),
        "n_overlap": int(overlap),
        "jaccard_distance": float(distance),
        "jaccard_index": float(index),
        "passes_sanity_threshold": bool(distance > 0.3),
        "seed": args.seed,
    }
    (out_dir / "jaccard.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
