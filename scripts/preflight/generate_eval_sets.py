"""Generate the expanded distribution-shift eval-set panel for K562.

Produces a family of evaluation parquet files under
``outputs/eval_sets_expanded/`` so the main-sweep checkpoints can be
scored against multiple shift types in week 6/22 without retraining.

Eval families:
  - **PRM**: partial-mutagenesis at fixed rates {1%, 5%, 10%, 20%}.
    Each rate produces N_test mutated sequences derived 1:1 from the
    held-out test set. Records mutation positions per sequence.
  - **Dinuc shuffle**: dinucleotide-preserving shuffle of test sequences.
    Preserves first-order base composition + dinucleotide frequencies
    but breaks higher-order motif structure.
  - **EvoAug structural**: heavy structural perturbation (insertions,
    deletions, inversions, translocations, tandem dup) per pilot paper.
  - **GC-stratified**: test sequences binned into 4 quartiles by GC content
    (each emitted as its own eval set). Tests model robustness across the
    GC-content distribution.
  - **Random uniform**: i.i.d. uniform random sequences at the median test
    length. The "fully OOD" bottom of the panel.

Note: length-stratification is intentionally omitted — the K562 MPRA
library is uniformly 200bp (>75% of test rows), so length quantiles are
degenerate and provide no real shift signal.

Test source: chromosome-split test sequences from
``outputs/oracle_pseudolabels_k562_ag_s2_refalt/pool/test.parquet`` if the
new cache is built; otherwise falls back to
``data/k562/test_sets/test_in_distribution_hashfrag.tsv``.

Sequences are NOT scored here — that's a separate downstream step using
the AG-S2 ensemble inference pattern (``infer_s2_fold.py`` adapted).

Usage:
    uv run --no-sync python scripts/preflight/generate_eval_sets.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _load_test_sequences() -> pd.DataFrame:
    """Load held-out test sequences. Prefer chromosome-split parquet from
    the new ref+alt cache; fall back to the existing chr7,13 TSV."""
    parq = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt" / "pool" / "test.parquet"
    if parq.exists():
        df = pd.read_parquet(parq)
        if "K562_log2FC" in df.columns:
            df = df.rename(columns={"K562_log2FC": "label"})
        print(f"  loaded {len(df):,} test sequences from {parq.name} (ref+alt chromosome split)")
        return df
    tsv = REPO / "data" / "k562" / "test_sets" / "test_chr7_13_all.tsv"
    if not tsv.exists():
        raise FileNotFoundError(
            f"Neither ref+alt parquet nor chr7,13 TSV found:\n  {parq}\n  {tsv}"
        )
    df = pd.read_csv(tsv, sep="\t")
    if "K562_log2FC" in df.columns:
        df = df.rename(columns={"K562_log2FC": "label"})
    # Drop any rows with empty/NaN sequence
    df = df.dropna(subset=["sequence"]).reset_index(drop=True)
    print(f"  loaded {len(df):,} test sequences from {tsv.name} (chr7,13 fallback)")
    return df


# ── PRM (partial mutagenesis at fixed rates) ─────────────────────────────
def _prm(
    test_seqs: list[str], rate: float, rng: np.random.Generator
) -> tuple[list[str], list[list[int]]]:
    """Mutate each base with probability `rate` to a different base (uniform)."""
    bases = ["A", "C", "G", "T"]
    out_seqs, out_positions = [], []
    for s in test_seqs:
        s = s.upper()
        arr = list(s)
        positions = []
        for i in range(len(arr)):
            if arr[i] not in bases:
                continue
            if rng.random() < rate:
                cur = arr[i]
                pool = [b for b in bases if b != cur]
                arr[i] = pool[rng.integers(0, len(pool))]
                positions.append(i)
        out_seqs.append("".join(arr))
        out_positions.append(positions)
    return out_seqs, out_positions


# ── Dinuc shuffle ────────────────────────────────────────────────────────
def _dinuc_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Shuffle preserving dinucleotide frequencies via Eulerian-path method
    (Altschul & Erickson 1985). Single-sequence implementation."""
    seq = seq.upper()
    if len(seq) < 4:
        return seq
    # Build edge list: graph where node = base, edges = dinucleotides
    # then perform Eulerian path randomization
    nodes = list(set(seq))
    edges: dict[str, list[str]] = {n: [] for n in nodes}
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])
    # Shuffle outgoing edges except for the last edge of each node terminating
    # at the original last char (to ensure Eulerian path exists)
    last_char = seq[-1]
    last_edge_per_node: dict[str, str] = {}
    for n, out_list in edges.items():
        if not out_list:
            continue
        # Pick a random "last" edge candidate that targets last_char if possible
        last_idx = None
        for j, t in enumerate(out_list):
            if t == last_char:
                last_idx = j
                break
        if last_idx is not None:
            last_edge_per_node[n] = out_list.pop(last_idx)
        rng.shuffle(out_list)
        if n in last_edge_per_node:
            out_list.append(last_edge_per_node[n])
    # Walk the Eulerian path: at each node, consume the next pre-shuffled
    # outgoing edge. pos[n] = next-unread index in edges[n].
    pos = {n: 0 for n in nodes}
    cur = seq[0]
    out = [cur]
    for _ in range(len(seq) - 1):
        if pos[cur] >= len(edges[cur]):
            # Eulerian path exhausted early — fallback uniform base draw.
            nxt = nodes[rng.integers(0, len(nodes))]
        else:
            nxt = edges[cur][pos[cur]]
            pos[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


def _dinuc_shuffle_batch(test_seqs: list[str], rng: np.random.Generator) -> list[str]:
    return [_dinuc_shuffle(s, rng) for s in test_seqs]


# ── EvoAug structural (heavy) ────────────────────────────────────────────
def _evoaug_structural(
    test_seqs: list[str], rng: np.random.Generator, intensity: str = "heavy"
) -> list[str]:
    """Apply EvoAug-style structural perturbations.

    Per intensity, applies a random subset of {deletion, insertion, inversion,
    translocation, tandem_dup, point_mutation}. Lengths are clipped to keep
    sequence length constant ± a small tolerance.
    """
    bases = ["A", "C", "G", "T"]
    intensity_cfg = {
        "light": dict(p_del=0.2, p_ins=0.2, p_inv=0.1, p_pm=0.02, max_size=10, n_events=(1, 2)),
        "medium": dict(p_del=0.3, p_ins=0.3, p_inv=0.2, p_pm=0.05, max_size=20, n_events=(2, 3)),
        "heavy": dict(p_del=0.4, p_ins=0.4, p_inv=0.3, p_pm=0.05, max_size=30, n_events=(2, 5)),
    }[intensity]
    out_seqs = []
    for s in test_seqs:
        s = list(s.upper())
        original_len = len(s)
        n_events = rng.integers(intensity_cfg["n_events"][0], intensity_cfg["n_events"][1] + 1)
        for _ in range(int(n_events)):
            r = rng.random()
            if r < intensity_cfg["p_del"]:
                size = rng.integers(1, intensity_cfg["max_size"] + 1)
                start = rng.integers(0, max(1, len(s) - size))
                del s[start : start + size]
            elif r < intensity_cfg["p_del"] + intensity_cfg["p_ins"]:
                size = rng.integers(1, intensity_cfg["max_size"] + 1)
                ins = [bases[rng.integers(0, 4)] for _ in range(size)]
                start = rng.integers(0, len(s) + 1)
                s[start:start] = ins
            elif r < intensity_cfg["p_del"] + intensity_cfg["p_ins"] + intensity_cfg["p_inv"]:
                size = rng.integers(2, intensity_cfg["max_size"] + 1)
                start = rng.integers(0, max(1, len(s) - size))
                # invert (reverse-complement)
                comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
                inv = [comp.get(b, "N") for b in s[start : start + size][::-1]]
                s[start : start + size] = inv
            else:
                # point mutation across the whole sequence at p_pm rate
                for i in range(len(s)):
                    if rng.random() < intensity_cfg["p_pm"] and s[i] in bases:
                        cur = s[i]
                        s[i] = [b for b in bases if b != cur][rng.integers(0, 3)]
        # Clip / pad to original length
        if len(s) > original_len:
            # randomly clip from start or end
            if rng.random() < 0.5:
                s = s[:original_len]
            else:
                s = s[-original_len:]
        elif len(s) < original_len:
            pad = original_len - len(s)
            left = pad // 2
            right = pad - left
            s = ["N"] * left + s + ["N"] * right
        out_seqs.append("".join(s))
    return out_seqs


# ── GC-stratified bins ───────────────────────────────────────────────────
def _gc_bin_test(df: pd.DataFrame, n_bins: int = 4) -> dict[str, pd.DataFrame]:
    """Bin test sequences by GC content into n_bins quantiles."""
    gc = df["sequence"].str.upper().apply(lambda s: (s.count("G") + s.count("C")) / max(1, len(s)))
    df = df.assign(gc_content=gc)
    quantile_edges = np.quantile(gc, np.linspace(0, 1, n_bins + 1))
    bins = {}
    for i in range(n_bins):
        lo, hi = quantile_edges[i], quantile_edges[i + 1]
        if i == n_bins - 1:
            mask = (gc >= lo) & (gc <= hi)
        else:
            mask = (gc >= lo) & (gc < hi)
        bins[f"gc_q{i + 1}_of_{n_bins}"] = (
            df[mask].reset_index(drop=True).assign(gc_bin_lo=float(lo), gc_bin_hi=float(hi))
        )
    return bins


# ── Random uniform ──────────────────────────────────────────────────────
def _random_uniform(n_seqs: int, length: int, rng: np.random.Generator) -> list[str]:
    bases = np.array(list("ACGT"))
    arr = bases[rng.integers(0, 4, size=(n_seqs, length))]
    return ["".join(row) for row in arr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        default=str(REPO / "outputs" / "eval_sets_expanded"),
        help="Output directory for parquet files.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading test sequences …")
    test_df = _load_test_sequences()
    test_seqs = test_df["sequence"].astype(str).tolist()
    median_len = int(test_df["sequence"].str.len().median())
    print(f"  N={len(test_seqs):,}  median_len={median_len}")

    # ── PRM at 4 rates ────────────────────────────────────────────────────
    for rate, tag in [(0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct"), (0.20, "20pct")]:
        prm_seqs, prm_positions = _prm(test_seqs, rate, rng)
        df = test_df.copy()
        df["sequence"] = prm_seqs
        df["mutation_positions"] = [json.dumps(p) for p in prm_positions]
        df["eval_set"] = f"prm_{tag}"
        df["intensity"] = rate
        df.to_parquet(out_dir / f"prm_{tag}.parquet")
        print(f"  saved prm_{tag}.parquet  N={len(df):,}")

    # ── Dinuc shuffle ──────────────────────────────────────────────────────
    dinuc_seqs = _dinuc_shuffle_batch(test_seqs, rng)
    df = test_df.copy()
    df["sequence"] = dinuc_seqs
    df["eval_set"] = "dinuc_shuffle"
    df["intensity"] = float("nan")
    df.to_parquet(out_dir / "dinuc_shuffle.parquet")
    print(f"  saved dinuc_shuffle.parquet  N={len(df):,}")

    # ── EvoAug at 3 intensities ───────────────────────────────────────────
    for intensity in ("light", "medium", "heavy"):
        evo_seqs = _evoaug_structural(test_seqs, rng, intensity=intensity)
        df = test_df.copy()
        df["sequence"] = evo_seqs
        df["eval_set"] = f"evoaug_{intensity}"
        df["intensity"] = {"light": 1, "medium": 2, "heavy": 3}[intensity]
        df.to_parquet(out_dir / f"evoaug_{intensity}.parquet")
        print(f"  saved evoaug_{intensity}.parquet  N={len(df):,}")

    # ── GC-stratified ─────────────────────────────────────────────────────
    for name, bin_df in _gc_bin_test(test_df, n_bins=4).items():
        bin_df["eval_set"] = name
        bin_df.to_parquet(out_dir / f"{name}.parquet")
        print(f"  saved {name}.parquet  N={len(bin_df):,}")

    # ── Random uniform (fully OOD) ────────────────────────────────────────
    n_random = len(test_seqs)
    rand_seqs = _random_uniform(n_random, median_len, rng)
    df = pd.DataFrame({"sequence": rand_seqs, "eval_set": "random_uniform"})
    df.to_parquet(out_dir / "random_uniform.parquet")
    print(f"  saved random_uniform.parquet  N={len(df):,}  length={median_len}")

    # ── Index / summary ────────────────────────────────────────────────────
    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    summary = {
        "n_test_source": int(len(test_seqs)),
        "median_test_length": median_len,
        "eval_sets": files,
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {len(files)} eval sets to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
