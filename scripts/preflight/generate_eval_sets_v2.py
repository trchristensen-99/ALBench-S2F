"""Extended eval-set panel generator (V2): covers single-base + multi-base
indels, structural-only mutations, GC-stratified random, and activity-
stratified test subsets.

Adds to the existing 13 panels in outputs/eval_sets_expanded/. Preserves
those panels (only writes new files; doesn't overwrite).

Panels added:

  Single-mutation-type:
    - prm_50pct.parquet — PRM at 50% (extreme)
    - del_1bp / del_5bp / del_10bp — single-bp / 5-bp / 10-bp deletions
    - ins_1bp / ins_5bp / ins_10bp — single-bp / 5-bp / 10-bp insertions

  Structural-only (no point mutations mixed in):
    - inversion_only / inversion_only_long
    - tandem_dup
    - translocation_pair (split each test seq + swap halves with another seq)

  GC-stratified random (matches bias eval levels):
    - random_gc_25/35/45/55/65/75 — i.i.d. random at fixed GC
    Note: these correspond to what bias_eval.json scored numerically;
    saving as parquets lets us run student inference on them.

  Activity-stratified test subsets:
    - test_activity_q1..q10 — test sequences binned by truth-label decile

  Motif-presence (placeholder):
    - left for follow-up; would use HOMER / JASPAR scanning

Usage:
  uv run --no-sync python scripts/preflight/generate_eval_sets_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TEST_PARQUET = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt" / "pool" / "test.parquet"
OUT_DIR = REPO / "outputs" / "eval_sets_expanded"

BASES = ["A", "C", "G", "T"]


def _gen_random_at_gc(n: int, length: int, gc: float, rng: np.random.Generator) -> list[str]:
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    arr = rng.choice(np.array(BASES), size=(n, length), p=p)
    return ["".join(row) for row in arr]


def _del_at_random_pos(seq: str, n_del: int, rng: np.random.Generator) -> str:
    if n_del >= len(seq):
        return seq
    start = int(rng.integers(0, len(seq) - n_del + 1))
    return seq[:start] + seq[start + n_del :]


def _ins_at_random_pos(seq: str, n_ins: int, rng: np.random.Generator) -> str:
    start = int(rng.integers(0, len(seq) + 1))
    insert = "".join(rng.choice(BASES, size=n_ins))
    return seq[:start] + insert + seq[start:]


def _inversion(seq: str, rng: np.random.Generator, max_size: int = 50) -> str:
    if len(seq) < 4:
        return seq
    size = int(rng.integers(2, min(max_size, len(seq)) + 1))
    start = int(rng.integers(0, len(seq) - size + 1))
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    inv = "".join(comp.get(b, "N") for b in seq[start : start + size][::-1])
    return seq[:start] + inv + seq[start + size :]


def _tandem_dup(seq: str, rng: np.random.Generator, max_size: int = 30) -> str:
    if len(seq) < 4:
        return seq
    size = int(rng.integers(2, min(max_size, len(seq) // 2) + 1))
    start = int(rng.integers(0, len(seq) - size + 1))
    return seq[: start + size] + seq[start : start + size] + seq[start + size :]


def _translocation_pair(seq_a: str, seq_b: str, rng: np.random.Generator) -> str:
    """Swap a random window between seq_a and seq_b → returns new seq_a."""
    if len(seq_a) < 10 or len(seq_b) < 10:
        return seq_a
    L = min(len(seq_a), len(seq_b))
    start = int(rng.integers(0, L // 2))
    size = int(rng.integers(10, L // 4))
    return seq_a[:start] + seq_b[start : start + size] + seq_a[start + size :]


def _pad_or_clip(seq: str, target_len: int) -> str:
    if len(seq) == target_len:
        return seq
    if len(seq) > target_len:
        offset = (len(seq) - target_len) // 2
        return seq[offset : offset + target_len]
    pad = target_len - len(seq)
    return "N" * (pad // 2) + seq + "N" * (pad - pad // 2)


def main():
    if not TEST_PARQUET.exists():
        raise SystemExit(f"missing {TEST_PARQUET}")
    test_df = pd.read_parquet(TEST_PARQUET).reset_index(drop=True)
    test_seqs = test_df["sequence"].astype(str).tolist()
    target_len = int(np.median([len(s) for s in test_seqs]))
    print(f"Loaded {len(test_seqs):,} test seqs, median_len={target_len}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # ── PRM 50% ────────────────────────────────────────────────────────
    print("  PRM 50% …")
    prm50 = []
    for s in test_seqs:
        s = s.upper()
        arr = list(s)
        for i in range(len(arr)):
            if arr[i] in BASES and rng.random() < 0.5:
                cur = arr[i]
                arr[i] = [b for b in BASES if b != cur][rng.integers(0, 3)]
        prm50.append("".join(arr))
    df = test_df.copy()
    df["sequence"] = prm50
    df["eval_set"] = "prm_50pct"
    df.to_parquet(OUT_DIR / "prm_50pct.parquet")

    # ── Indels ──────────────────────────────────────────────────────────
    for n_bp in (1, 5, 10):
        print(f"  del_{n_bp}bp …")
        out = [_pad_or_clip(_del_at_random_pos(s.upper(), n_bp, rng), target_len) for s in test_seqs]
        df = test_df.copy(); df["sequence"] = out; df["eval_set"] = f"del_{n_bp}bp"
        df["mutation_size"] = n_bp
        df.to_parquet(OUT_DIR / f"del_{n_bp}bp.parquet")
        print(f"  ins_{n_bp}bp …")
        out = [_pad_or_clip(_ins_at_random_pos(s.upper(), n_bp, rng), target_len) for s in test_seqs]
        df = test_df.copy(); df["sequence"] = out; df["eval_set"] = f"ins_{n_bp}bp"
        df["mutation_size"] = n_bp
        df.to_parquet(OUT_DIR / f"ins_{n_bp}bp.parquet")

    # ── Structural-only (no point mutations) ────────────────────────────
    print("  inversion_only (max 30bp) …")
    inv_seqs = [_pad_or_clip(_inversion(s.upper(), rng, max_size=30), target_len) for s in test_seqs]
    df = test_df.copy(); df["sequence"] = inv_seqs; df["eval_set"] = "inversion_only"
    df.to_parquet(OUT_DIR / "inversion_only.parquet")

    print("  inversion_only_long (max 80bp) …")
    inv_long = [_pad_or_clip(_inversion(s.upper(), rng, max_size=80), target_len) for s in test_seqs]
    df = test_df.copy(); df["sequence"] = inv_long; df["eval_set"] = "inversion_only_long"
    df.to_parquet(OUT_DIR / "inversion_only_long.parquet")

    print("  tandem_dup …")
    tdup = [_pad_or_clip(_tandem_dup(s.upper(), rng, max_size=20), target_len) for s in test_seqs]
    df = test_df.copy(); df["sequence"] = tdup; df["eval_set"] = "tandem_dup"
    df.to_parquet(OUT_DIR / "tandem_dup.parquet")

    print("  translocation_pair …")
    rng2 = np.random.default_rng(43)
    partner_idx = rng2.permutation(len(test_seqs))
    trans_seqs = [
        _pad_or_clip(_translocation_pair(test_seqs[i].upper(), test_seqs[partner_idx[i]].upper(), rng), target_len)
        for i in range(len(test_seqs))
    ]
    df = test_df.copy(); df["sequence"] = trans_seqs; df["eval_set"] = "translocation_pair"
    df.to_parquet(OUT_DIR / "translocation_pair.parquet")

    # ── GC-stratified random (matches bias_eval levels) ────────────────
    for gc in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
        n = 5000
        print(f"  random_gc_{int(gc * 100):02d}pct (n={n}) …")
        seqs = _gen_random_at_gc(n, target_len, gc, rng)
        pd.DataFrame({"sequence": seqs, "eval_set": f"random_gc_{int(gc * 100):02d}pct", "gc_content": gc}).to_parquet(
            OUT_DIR / f"random_gc_{int(gc * 100):02d}pct.parquet"
        )

    # ── Activity-stratified test subsets (10 deciles by truth label) ─
    if "K562_log2FC" in test_df.columns:
        print("  test_activity_q1..q10 …")
        deciles = np.quantile(test_df["K562_log2FC"], np.linspace(0, 1, 11))
        for i in range(10):
            lo, hi = deciles[i], deciles[i + 1]
            mask = (test_df["K562_log2FC"] >= lo) & (
                test_df["K562_log2FC"] <= hi if i == 9 else test_df["K562_log2FC"] < hi
            )
            sub = test_df[mask].reset_index(drop=True).copy()
            sub["eval_set"] = f"test_activity_q{i + 1}"
            sub["truth_decile_lo"] = float(lo)
            sub["truth_decile_hi"] = float(hi)
            sub.to_parquet(OUT_DIR / f"test_activity_q{i + 1}.parquet")
            print(f"    q{i + 1}: n={len(sub):,}  truth ∈ [{lo:+.2f}, {hi:+.2f}]")

    # ── Summary ────────────────────────────────────────────────────────
    files = sorted([p.name for p in OUT_DIR.glob("*.parquet")])
    summary = {"n_panels": len(files), "panels": files, "n_test_source": len(test_seqs)}
    (OUT_DIR / "summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {len(files)} eval-set panels under {OUT_DIR}/")


if __name__ == "__main__":
    main()
