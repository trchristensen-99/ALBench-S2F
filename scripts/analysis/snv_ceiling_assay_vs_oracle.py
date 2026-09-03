"""Is the SNV-effect ceiling set by the ORACLE or by the ASSAY?

The oracle reaches only r = 0.419 on Delta log2FC while hitting 0.974 on absolute activity. Two
explanations: (a) the oracle systematically compresses variant effects, or (b) real Delta is
dominated by irreducible measurement noise so NO model could do better. These are distinguishable
without any new experiment, because Gosai Table S2 ships a per-oligo standard error (lfcSE).

For a measurement y = t + e with Var(e) known, the reliability is
    rho = (Var(y) - Var(e)) / Var(y)
and the maximum correlation any perfect predictor of t can achieve against the OBSERVED y is
    r_max = sqrt(rho).
Delta = alt - ref is a difference of two independent measurements, so its noise variance is the SUM
of the two per-oligo variances while its TRUE variance is much smaller than that of absolute
activity -- which is exactly why differences are so much harder to predict than levels.

Cross-check that needs no error model at all: variants measured in SEVERAL oligo contexts give two
independent Delta estimates of the same variant. Their correlation is an empirical reliability
bound (it conflates genuine context effects with noise, so it is a LOWER bound). These are the
multi-context pairs the canonical mono test set deliberately discards.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

CELLS = ("K562", "HepG2", "SKNSH")


def reliability(y, se):
    var_obs = np.var(y, ddof=1)
    var_noise = np.mean(se**2)
    rho = (var_obs - var_noise) / var_obs
    return var_obs, var_noise, rho, np.sqrt(max(0.0, rho))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    ap.add_argument("--cell", default="K562", choices=CELLS)
    ap.add_argument(
        "--oracle_delta_r",
        type=float,
        default=0.419,
        help="measured oracle Pearson r on Delta, for the comparison line",
    )
    args = ap.parse_args()

    lab, sec = f"{args.cell}_log2FC", f"{args.cell}_lfcSE"
    df = pd.read_csv(args.table, sep="\t", usecols=["IDs", "chr", "OL", lab, sec], low_memory=False)
    p = df["IDs"].astype(str).str.split(":", expand=True)
    df["variant_key"] = p[0] + ":" + p[1] + ":" + p[2] + ":" + p[3]
    df["allele"] = p[4]
    df = df[df["allele"].isin(["R", "A"])].dropna(subset=[lab, sec])

    refs = df[df.allele == "R"][["variant_key", "OL", lab, sec]]
    alts = df[df.allele == "A"][["variant_key", "OL", lab, sec]]
    m = refs.merge(alts, on=["variant_key", "OL"], suffixes=("_ref", "_alt"))
    m["delta"] = m[f"{lab}_alt"] - m[f"{lab}_ref"]
    # independent measurements -> variances add
    m["delta_se"] = np.sqrt(m[f"{sec}_ref"] ** 2 + m[f"{sec}_alt"] ** 2)
    n_ctx = m.groupby("variant_key")["OL"].transform("nunique")
    m["mono"] = n_ctx == 1

    print(f"=== {args.cell} ===")
    print(
        f"pairs {len(m):,}  variants {m.variant_key.nunique():,}  "
        f"strict-mono {int(m.mono.sum()):,}  multi-context {int((~m.mono).sum()):,}\n"
    )

    # ---- absolute activity, for scale
    a = df.drop_duplicates(subset=["IDs"])
    vo, vn, rho, rmax = reliability(a[lab].values, a[sec].values)
    print(
        f"{'ABSOLUTE activity':<26} var_obs={vo:.4f} var_noise={vn:.4f} "
        f"reliability={rho:.4f} r_max={rmax:.4f}"
    )

    # ---- delta, all pairs and mono only
    for tag, sub in (("DELTA (all pairs)", m), ("DELTA (strict-mono)", m[m.mono])):
        vo, vn, rho, rmax = reliability(sub["delta"].values, sub["delta_se"].values)
        print(
            f"{tag:<26} var_obs={vo:.4f} var_noise={vn:.4f} "
            f"reliability={rho:.4f} r_max={rmax:.4f}  n={len(sub):,}"
        )

    # ---- empirical cross-context reliability (no error model)
    mc = m[~m.mono]
    if len(mc):
        first, second = [], []
        for _, g in mc.groupby("variant_key"):
            if len(g) >= 2:
                d = g["delta"].values
                first.append(d[0])
                second.append(d[1])
        if len(first) > 30:
            r_ctx = pearsonr(first, second)[0]
            print(
                f"\ncross-context Delta agreement (same variant, 2 oligo contexts): "
                f"r={r_ctx:.4f} on n={len(first):,} variants"
            )
            print(
                f"  -> implied single-measurement reliability ~ {r_ctx:.4f}, "
                f"r_max ~ {np.sqrt(max(0, r_ctx)):.4f}  (LOWER bound: includes real context effects)"
            )

    # ---- verdict
    vo, vn, rho, rmax = reliability(m[m.mono]["delta"].values, m[m.mono]["delta_se"].values)
    print(
        f"\nVERDICT  assay r_max on mono Delta = {rmax:.3f} vs oracle achieved "
        f"{args.oracle_delta_r:.3f}"
    )
    if rmax > args.oracle_delta_r + 0.10:
        print(
            f"  ORACLE-LIMITED. The assay supports up to r~{rmax:.2f}; the oracle leaves "
            f"{rmax - args.oracle_delta_r:.2f} of headroom on the table."
        )
    elif rmax < args.oracle_delta_r + 0.05:
        print(
            "  ASSAY-LIMITED. The oracle is already near the noise ceiling; no model can do "
            "much better on these labels."
        )
    else:
        print("  MIXED. Both contribute materially.")

    # ---- how much of the gap is pure difference-taking?
    print(
        "\nWHY DELTA IS HARDER: noise variance roughly doubles (two oligos) while true variance "
        "collapses (a 1 bp change moves activity very little), so the signal-to-noise ratio "
        "falls on both sides at once."
    )


if __name__ == "__main__":
    main()
