"""Empirical assay ceiling from repeated measurements inside Table S2.

The lfcSE-based ceiling is a MODEL of the noise. A sequence measured twice gives the
noise directly: correlation between independent measurements of the same sequence IS
the test-retest reliability, and sqrt of it bounds any model's achievable r.

Crucially this also gives what the lfcSE route could not: the ref/alt error
correlation rho_e, which sets the ceiling on SNV EFFECT prediction. If ref and alt
are measured in the same library their errors share batch effects, and we can
estimate that directly from variants whose ref and alt are both repeated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import os

# Repo root: env override first, else derived from this file's location.
# Never a literal -- the path differs on every machine that runs this.
_REPO_ROOT = Path(os.environ.get("ALBENCH_REPO") or Path(__file__).resolve().parents[2])


REPO = Path(_REPO_ROOT)
t = pd.read_csv(REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False)
print("columns:", list(t.columns))
print(f"rows: {len(t):,}")
for c in ("OL", "data_project"):
    if c in t:
        v = t[c].value_counts()
        print(f"\n{c}: {len(v)} distinct -> {dict(list(v.items())[:8])}")

# ---- repeated SEQUENCES -----------------------------------------------------
t["seq"] = t["sequence"].astype(str).str.upper()
dup = t[t.duplicated("seq", keep=False)]
print(
    f"\nrows whose sequence appears more than once: {len(dup):,} "
    f"({dup.seq.nunique():,} distinct sequences)"
)

if len(dup):
    g = dup.groupby("seq")
    # Are the repeats INDEPENDENT measurements (differing values) or copied rows?
    spread = g["K562_log2FC"].agg(lambda s: s.max() - s.min())
    identical = (spread < 1e-9).sum()
    print(
        f"  of {len(spread):,} repeated sequences, {identical:,} have IDENTICAL "
        f"log2FC (copied rows, not replicates)"
    )
    print(f"  {len(spread) - identical:,} differ -> independent measurements")

    real = spread[spread >= 1e-9].index
    sub = dup[dup.seq.isin(real)]
    pairs = sub.groupby("seq").head(2).groupby("seq")
    first, second = [], []
    ol_same, ol_diff = [], []
    for seq, grp in sub.groupby("seq"):
        if len(grp) < 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        if not (np.isfinite(a.K562_log2FC) and np.isfinite(b.K562_log2FC)):
            continue
        first.append(a.K562_log2FC)
        second.append(b.K562_log2FC)
        (ol_same if a.get("OL") == b.get("OL") else ol_diff).append((a.K562_log2FC, b.K562_log2FC))
    first, second = np.array(first), np.array(second)
    if first.size > 20:
        r = pearsonr(first, second)[0]
        print(f"\n  TEST-RETEST on {first.size:,} repeated sequences: r = {r:.4f}")
        print(
            f"    => empirical reliability {r:.4f}, ceiling on model r = {np.sqrt(max(r, 0)):.4f}"
        )
        print(f"    (lfcSE-based estimate was reliability 0.890, ceiling 0.943)")
        d = first - second
        print(f"    SD of the difference between repeats = {d.std():.4f}")
        print(f"    => per-measurement noise SD ~ {d.std() / np.sqrt(2):.4f}")
        for nm, arr in (("same OL", ol_same), ("different OL", ol_diff)):
            if len(arr) > 20:
                a = np.array(arr)
                print(f"    {nm:<13} n={len(arr):>6}  r={pearsonr(a[:, 0], a[:, 1])[0]:.4f}")

# ---- repeated VARIANT PAIRS: estimate rho_e ---------------------------------
parts = t.IDs.astype(str).str.split(":", expand=True)
t["key"] = parts[0] + ":" + parts[1] + ":" + parts[2] + ":" + parts[3]
t["allele"] = parts[4] if parts.shape[1] > 4 else ""
t["is_sub"] = (parts[2].astype(str).str.len() == 1) & (parts[3].astype(str).str.len() == 1)

ref = t[(t.allele == "R") & t.is_sub]
alt = t[(t.allele == "A") & t.is_sub]
rd = ref[ref.duplicated("key", keep=False)]
ad = alt[alt.duplicated("key", keep=False)]
shared = set(rd.key) & set(ad.key)
print(f"\nvariants with BOTH alleles measured more than once: {len(shared):,}")

# The same variant measured in more than one OLIGO CONTEXT. Not a technical
# replicate -- the flanking sequence differs -- so this bounds how reproducible a
# variant's effect is ACROSS contexts, which is the quantity a model that sees only
# the oligo can possibly predict. Vectorised: the per-key loop was O(n^2).
if len(shared) > 30:
    sub_r = ref[ref.key.isin(shared)].sort_values("key")
    sub_a = alt[alt.key.isin(shared)].sort_values("key")
    r2 = sub_r.groupby("key")["K562_log2FC"].agg(list)
    a2 = sub_a.groupby("key")["K562_log2FC"].agg(list)
    keys = [k for k in r2.index if len(r2[k]) >= 2 and len(a2.get(k, [])) >= 2]
    R = np.array([[r2[k][0], r2[k][1]] for k in keys], float)
    A = np.array([[a2[k][0], a2[k][1]] for k in keys], float)
    ok = np.isfinite(R).all(1) & np.isfinite(A).all(1)
    R, A = R[ok], A[ok]
    print(f"  usable variants with 2 contexts for both alleles: {len(R):,}")

    if len(R) > 20:
        # reproducibility of the LEVEL across contexts
        rl = pearsonr(np.r_[R[:, 0], A[:, 0]], np.r_[R[:, 1], A[:, 1]])[0]
        print(f"\n  LEVEL across contexts : r = {rl:.4f}  -> ceiling {np.sqrt(max(rl, 0)):.4f}")
        # reproducibility of the DELTA across contexts
        d1, d2 = A[:, 0] - R[:, 0], A[:, 1] - R[:, 1]
        rdlt = pearsonr(d1, d2)[0]
        print(f"  DELTA across contexts : r = {rdlt:.4f}  -> ceiling {np.sqrt(max(rdlt, 0)):.4f}")
        print(
            f"    (our oracle measures 0.430 on SNV effect; the lfcSE route implied a "
            f"ceiling of 0.51-0.69 depending on rho_e)"
        )
        print(f"  SD of delta: context1 {d1.std():.4f}  context2 {d2.std():.4f}")
        print(f"  SD of (delta1 - delta2) = {(d1 - d2).std():.4f}")

        # rho_e: do the ref and alt DISCREPANCIES between contexts move together?
        er, ea = R[:, 0] - R[:, 1], A[:, 0] - A[:, 1]
        rho_e = pearsonr(er, ea)[0]
        print(f"\n  ref-discrepancy vs alt-discrepancy correlation = {rho_e:+.4f}")
        print(f"    This is the empirical analogue of rho_e. Our SNV-effect result")
        print(f"    implied rho_e >= 0.434 as a LOWER BOUND; compare directly.")

        # stratify by effect size, matching the oracle analysis
        print(f"\n  DELTA reproducibility by |effect| (mean of the two contexts):")
        m = (np.abs(d1) + np.abs(d2)) / 2
        for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)]:
            sel = (m >= lo) & (m < hi)
            if sel.sum() > 30:
                print(
                    f"    |delta| in [{lo:.2f},{hi if hi < 90 else np.inf:.2f})  "
                    f"n={int(sel.sum()):>6}  r={pearsonr(d1[sel], d2[sel])[0]:>7.4f}"
                )
