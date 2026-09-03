"""Per-test-set assay reliability, computed on the EXACT sequences in each test set.

An earlier pass computed one reliability from the whole Gosai table (798k rows) and applied it to
every test set. That is wrong, and it showed: out-of-fold genomic r = 0.9496 EXCEEDED the supposed
ceiling of 0.9429, which is impossible for a true ceiling.

Reliability depends on the SUBSET's own signal-to-noise:
    rho   = (Var(y) - E[SE^2]) / Var(y)          r_max = sqrt(rho)
A subset with a wider dynamic range (larger Var(y)) at similar measurement noise has HIGHER
reliability and therefore a HIGHER ceiling. The designed high-activity set spans far more range
than random intergenic controls, so a single pooled number cannot describe both.

These are TENTATIVE ceilings, not hard limits, because:
  * lfcSE is itself an estimate; if it is conservative, rho is understated and the ceiling is too low
  * it assumes independent, unbiased, additive noise
  * Var(y) on a finite test set is itself noisy
  * a model can exceed the estimate by exploiting correlated error structure lfcSE does not capture
So "model r above tentative ceiling" is a FLAG to investigate (usually in-sample contamination or an
understated SE), not proof of impossibility.
"""

import argparse
import os

import numpy as np
import pandas as pd

CELL = "K562"


def reliability(y, se):
    y, se = np.asarray(y, float), np.asarray(se, float)
    m = np.isfinite(y) & np.isfinite(se)
    y, se = y[m], se[m]
    var_obs = float(np.var(y, ddof=1))
    var_noise = float(np.mean(se**2))
    rho = (var_obs - var_noise) / var_obs if var_obs > 0 else np.nan
    return dict(
        n=len(y),
        var_obs=var_obs,
        var_noise=var_noise,
        rho=rho,
        r_max=float(np.sqrt(max(0.0, rho))),
        sd=float(np.sqrt(var_obs)),
    )


def show(tag, d, achieved=None):
    line = (
        f"{tag:<30} n={d['n']:>7,}  sd={d['sd']:.3f}  Var(obs)={d['var_obs']:.4f}  "
        f"Var(noise)={d['var_noise']:.4f}  rho={d['rho']:.3f}  r_max={d['r_max']:.4f}"
    )
    if achieved is not None:
        line += f"  | achieved={achieved:.4f}"
        if achieved > d["r_max"]:
            line += "  ** EXCEEDS -> SE likely conservative, or contamination"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    ap.add_argument("--battery_dir", default="data/k562/test_sets_ag_s2_chrsplit")
    args = ap.parse_args()

    lab, sec = f"{CELL}_log2FC", f"{CELL}_lfcSE"
    df = pd.read_csv(args.table, sep="\t", usecols=["IDs", "sequence", lab, sec], low_memory=False)
    df = df.dropna(subset=["sequence"])
    # sequence -> (label, se); duplicates collapse to the first occurrence
    seq2 = {}
    for s, y, e in zip(df["sequence"].astype(str), df[lab], df[sec]):
        if s not in seq2:
            seq2[s] = (y, e)
    print(f"[table] {len(seq2):,} unique sequences with labels\n")

    print("=== ABSOLUTE-ACTIVITY TEST SETS (ceiling computed on each set's own sequences) ===")
    achieved = {
        "genomic_oracle.npz": 0.9496,
        "ood_oracle.npz": 0.8398,
        "ctrl_neg_oracle.npz": 0.8231,
    }
    for fn, tag in (
        ("genomic_oracle.npz", "WT / genomic in-dist"),
        ("ood_oracle.npz", "Synthetic / OOD high-act"),
        ("ctrl_neg_oracle.npz", "ctrl_neg intergenic"),
    ):
        p = os.path.join(args.battery_dir, fn)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        seqs = [str(x) for x in z["sequences"]]
        hit = [(seq2[s][0], seq2[s][1]) for s in seqs if s in seq2]
        if len(hit) < 50:
            print(f"{tag:<30} only {len(hit)} of {len(seqs):,} matched the table -- SKIPPED")
            continue
        y, se = zip(*hit)
        d = reliability(y, se)
        d["matched_frac"] = len(hit) / len(seqs)
        show(tag, d, achieved.get(fn))
        print(f"{'':<30} matched {len(hit):,}/{len(seqs):,} ({d['matched_frac']:.1%})")

    print("\n=== POOLED (the earlier, WRONG basis) ===")
    show("all table rows", reliability(df[lab], df[sec]))

    print("\n=== SNV DELTA ===")
    p = df["IDs"].astype(str).str.split(":", expand=True)
    d2 = df.assign(vk=p[0] + ":" + p[1] + ":" + p[2] + ":" + p[3], allele=p[4])
    # OL is needed to pair within an oligo context; re-read it
    ol = pd.read_csv(args.table, sep="\t", usecols=["OL"], low_memory=False)["OL"]
    d2 = d2.assign(OL=ol)
    d2 = d2[d2.allele.isin(["R", "A"])].dropna(subset=[lab, sec])
    refs = d2[d2.allele == "R"][["vk", "OL", lab, sec]]
    alts = d2[d2.allele == "A"][["vk", "OL", lab, sec]]
    m = refs.merge(alts, on=["vk", "OL"], suffixes=("_r", "_a"))
    m["delta"] = m[f"{lab}_a"] - m[f"{lab}_r"]
    m["dse"] = np.sqrt(m[f"{sec}_r"] ** 2 + m[f"{sec}_a"] ** 2)
    nctx = m.groupby("vk")["OL"].transform("nunique")
    m["mono"] = nctx == 1
    show("delta ALL pairs", reliability(m["delta"], m["dse"]))
    show("delta strict-MONO", reliability(m[m.mono]["delta"], m[m.mono]["dse"]), 0.2886)
    show("delta MULTI-context", reliability(m[~m.mono]["delta"], m[~m.mono]["dse"]))
    r_mono = reliability(m[m.mono]["delta"], m[m.mono]["dse"])["r_max"]
    r_multi = reliability(m[~m.mono]["delta"], m[~m.mono]["dse"])["r_max"]
    print(f"\n  multi/mono ceiling ratio = {r_multi / max(r_mono, 1e-9):.3f}")
    print("  => an evaluation run on the FULL (non-mono) SNV set is measuring an intrinsically")
    print("     EASIER problem. If a historical number used all pairs while ours uses strict-mono,")
    print("     that difference alone shifts the achievable correlation upward.")


if __name__ == "__main__":
    main()
