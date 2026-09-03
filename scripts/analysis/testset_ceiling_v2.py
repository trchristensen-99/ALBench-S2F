"""Per-test-set reliability read from each test set's OWN lfcSE, plus a direct mono-vs-multi test.

Two corrections to the previous pass:
  1. It matched test-set sequences back to Table S2 to recover lfcSE. That silently dropped the
     designed/OOD set entirely (0 of 22,962 matched, because those sequences are not in the table).
     Every battery TSV in fact ships its own SE columns, so read those - they are authoritative.
  2. The mono-vs-multi explanation for the historical SNV number was ASSERTED from ceilings, not
     measured. The archived oracle predictions on the oversized 45,543-pair set let us measure the
     achieved correlation on the same pairs, split by mono/multi, and settle it directly.

Everything here is a TENTATIVE ceiling: lfcSE is an estimate, the derivation assumes independent
unbiased additive noise, and finite-sample Var(y) is itself noisy.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def rel(tag, y, se, achieved=None):
    y, se = np.asarray(y, float), np.asarray(se, float)
    m = np.isfinite(y) & np.isfinite(se)
    y, se = y[m], se[m]
    vo, vn = float(np.var(y, ddof=1)), float(np.mean(se**2))
    rho = (vo - vn) / vo if vo > 0 else np.nan
    rmax = float(np.sqrt(max(0.0, rho)))
    s = (
        f"{tag:<34} n={len(y):>7,}  sd={np.sqrt(vo):.3f}  Var(noise)={vn:.4f}  "
        f"rho={rho:.3f}  r_max={rmax:.4f}"
    )
    if achieved is not None:
        s += f"  achieved={achieved:.4f} ({achieved / max(rmax, 1e-9):.0%} of ceiling)"
    print(s)
    return rmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="data/k562/test_sets")
    ap.add_argument("--battery", default="data/k562/test_sets_ag_s2_chrsplit")
    a = ap.parse_args()

    print("=== ABSOLUTE ACTIVITY (SE read from each test set's own file) ===")
    ood = pd.read_csv(os.path.join(a.ts, "test_ood_designed_k562.tsv"), sep="\t")
    rel("OOD designed high-activity", ood["K562_log2FC"], ood["K562_lfcSE"], achieved=0.8398)
    print(
        f"{'':<34} activity mean={ood['K562_log2FC'].mean():.3f} "
        f"vs genomic-typical ~0.0; methods={ood['method'].nunique()}"
    )
    for meth, g in ood.groupby("method"):
        if len(g) > 200:
            rel(f"  OOD / {meth[:26]}", g["K562_log2FC"], g["K562_lfcSE"])

    ind = pd.read_csv(os.path.join(a.ts, "test_chr7_13_ref_only.tsv"), sep="\t")
    col = "K562_lfcSE" if "K562_lfcSE" in ind.columns else None
    if col:
        rel("WT / genomic in-dist", ind["K562_log2FC"], ind[col], achieved=0.9496)

    print("\n=== SNV DELTA: ceiling AND achieved, mono vs multi, on the SAME 45,543 pairs ===")
    snv = pd.read_csv(os.path.join(a.ts, "test_snv_pairs.tsv"), sep="\t")
    snv["dse"] = np.sqrt(snv["K562_lfcSE_ref"] ** 2 + snv["K562_lfcSE_alt"] ** 2)
    # variant_key = chr:pos:ref:alt from the ref ID; multi-context = key appears more than once
    vk = snv["IDs_ref"].astype(str).str.split(":", expand=True)
    snv["vk"] = vk[0] + ":" + vk[1] + ":" + vk[2] + ":" + vk[3]
    cnt = snv["vk"].map(snv["vk"].value_counts())
    snv["mono"] = cnt == 1
    print(
        f"[set] {len(snv):,} pairs; strict-mono {int(snv.mono.sum()):,}, "
        f"multi-context {int((~snv.mono).sum()):,}"
    )

    legacy = os.path.join(a.battery, "snv_oracle_legacy_oversized_45543.npz")
    pred = None
    if os.path.exists(legacy):
        z = np.load(legacy, allow_pickle=True)
        keys = list(z.files)
        dm = "delta_mean" if "delta_mean" in keys else None
        if dm and len(z[dm]) == len(snv):
            pred = np.asarray(z[dm], float)
            print(f"[legacy] archived oracle predictions found ({legacy.split('/')[-1]})")
        else:
            print(f"[legacy] shape mismatch or no delta_mean; keys={keys}")

    def blk(mask, tag):
        y, se = snv.loc[mask, "delta_log2FC"], snv.loc[mask, "dse"]
        ach = None
        if pred is not None:
            mm = mask.to_numpy() & np.isfinite(pred) & np.isfinite(snv["delta_log2FC"].to_numpy())
            if mm.sum() > 50:
                ach = float(pearsonr(snv["delta_log2FC"].to_numpy()[mm], pred[mm])[0])
        rel(tag, y, se, achieved=ach)

    blk(pd.Series(True, index=snv.index), "delta ALL 45,543 pairs")
    blk(snv.mono, "delta strict-MONO")
    blk(~snv.mono, "delta MULTI-context")
    print("\nIf ACHIEVED is higher on MULTI than on MONO, the historical figure came from the")
    print("inclusive set and monoallelic filtering made the test harder. If the reverse, the")
    print("historical figure was already the strict one and something else explains the gap.")


if __name__ == "__main__":
    main()
