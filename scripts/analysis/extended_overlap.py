"""Extended Agarwal↔Gosai K562 sequence matching:
- Exact forward + reverse complement
- Hamming distance ≤ 2 (substitutions) via chunked-index strategy

Then fits multiple mapping models (linear, quadratic, isotonic).
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

RC_MAP = str.maketrans("ACGTN", "TGCAN")
def rc(s): return s.translate(RC_MAP)[::-1]


def main():
    # --- Load Agarwal (seq + label) ---
    print("Loading Agarwal K562 large-scale...")
    xls = pd.ExcelFile(REPO / "data/agarwal_2025/Table_S3_large_scale_lib_design.xlsx")
    ag_lib = pd.read_excel(xls, "K562 large-scale", header=2)
    name_col = ag_lib.columns[0]; seq_col = ag_lib.columns[-1]
    ag_lib = ag_lib[[name_col, seq_col]].dropna()
    ag_lib.columns = ["name", "seq230"]
    ag_lib["seq200"] = ag_lib["seq230"].astype(str).str[15:215].str.upper()
    enc = pd.read_csv(REPO / "data/agarwal_2025/ENCFF252GNM.tsv", sep="\t")
    enc = enc[enc["condition"] == "K562"]
    enc_agg = enc.groupby("name")["log2"].mean().reset_index()
    enc_agg.columns = ["name", "log2_mean"]
    ag_full = ag_lib.merge(enc_agg, on="name", how="inner")
    print(f"  Agarwal: n={len(ag_full)}")

    # --- Load Gosai ---
    print("\nLoading Gosai 798k...")
    gos = pd.read_csv(REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t",
                      usecols=["sequence", "K562_log2FC"], low_memory=False)
    gos = gos.dropna()
    gos["sequence"] = gos["sequence"].astype(str).str.upper()
    print(f"  Gosai: n={len(gos)}")

    # --- Exact match (forward) ---
    print("\n=== Exact forward matches ===")
    ag_seqs = ag_full["seq200"].values
    gos_seq_to_label = dict(zip(gos["sequence"].values, gos["K562_log2FC"].values))
    ag_set = set(ag_seqs)

    exact_fwd = []
    for s, l in gos_seq_to_label.items():
        if s in ag_set:
            ag_label = ag_full.loc[ag_full["seq200"] == s, "log2_mean"].iloc[0]
            exact_fwd.append((s, ag_label, l, 0, "fwd"))
    print(f"  exact forward: {len(exact_fwd)}")

    # --- Exact match (reverse complement) ---
    print("\n=== Exact reverse-complement matches ===")
    ag_rc_set = {rc(s) for s in ag_seqs}
    exact_rc = []
    for s, l in gos_seq_to_label.items():
        if s in ag_rc_set:
            ag_label = ag_full.loc[ag_full["seq200"] == rc(s), "log2_mean"].iloc[0]
            exact_rc.append((s, ag_label, l, 0, "rc"))
    print(f"  exact RC: {len(exact_rc)}")

    # --- Hamming ≤ 2 via chunked index ---
    # Split each 200bp seq into 3 windows of ~67bp. With hamming ≤ 2, at least one window
    # is exact match. Build hash: window_seq → list of (ag_idx, window_idx, orientation).
    print("\n=== Building chunked hash index for hamming ≤ 2 search ===")
    t0 = time.time()
    chunk_size = 67
    chunks = [(0, 67), (67, 134), (134, 200)]
    # Index Agarwal seqs (both fwd and RC)
    idx = defaultdict(list)
    for ag_i, ag_s in enumerate(ag_seqs):
        for w_i, (a, b) in enumerate(chunks):
            idx[ag_s[a:b]].append((ag_i, w_i, "fwd"))
            idx[rc(ag_s)[a:b]].append((ag_i, w_i, "rc"))
    print(f"  built index in {time.time()-t0:.1f}s, {len(idx)} unique chunks")

    # Query Gosai seqs
    seen_pairs = set()
    near_matches = []
    for g_i, (gs, gl) in enumerate(gos_seq_to_label.items()):
        # already exact?
        if gs in ag_set:
            continue
        if gs in ag_rc_set:
            continue
        cands = set()
        for w_i, (a, b) in enumerate(chunks):
            chunk = gs[a:b]
            for cand in idx.get(chunk, []):
                cands.add(cand)
        # Verify hamming for each candidate
        for ag_i, _, orient in cands:
            ag_s = ag_seqs[ag_i] if orient == "fwd" else rc(ag_seqs[ag_i])
            ham = sum(1 for c1, c2 in zip(gs, ag_s) if c1 != c2)
            if ham <= 2 and ham >= 1:
                key = (gs, ag_i, orient)
                if key in seen_pairs: continue
                seen_pairs.add(key)
                ag_label = ag_full["log2_mean"].iloc[ag_i]
                near_matches.append((gs, ag_label, gl, ham, orient))
        if g_i % 100000 == 0 and g_i > 0:
            print(f"  scanned {g_i}/{len(gos_seq_to_label)} Gosai seqs, near_matches so far: {len(near_matches)}")

    print(f"\n  hamming-1: {sum(1 for x in near_matches if x[3]==1)}")
    print(f"  hamming-2: {sum(1 for x in near_matches if x[3]==2)}")

    # --- Combine all paired matches ---
    all_pairs = exact_fwd + exact_rc + near_matches
    paired = pd.DataFrame(all_pairs, columns=["sequence", "agarwal_label", "gosai_label", "hamming", "orientation"])
    paired = paired.drop_duplicates(subset=["sequence"])
    print(f"\n=== Combined paired dataset: n={len(paired)} ===")
    for h in [0, 1, 2]:
        sub = paired[paired.hamming == h]
        print(f"  hamming={h}: n={len(sub)}")
        if len(sub) > 0:
            print(f"     Agarwal: mean={sub.agarwal_label.mean():+.3f}, std={sub.agarwal_label.std():.3f}")
            print(f"     Gosai:   mean={sub.gosai_label.mean():+.3f}, std={sub.gosai_label.std():.3f}")
            print(f"     Pearson(Ag, Gos): {sub[['agarwal_label', 'gosai_label']].corr().iloc[0,1]:.4f}")

    # --- Fit multiple mapping models on combined paired data ---
    print(f"\n=== Mapping models on n={len(paired)} paired sequences ===")
    x = paired.agarwal_label.values
    y = paired.gosai_label.values
    from scipy.stats import linregress
    from sklearn.isotonic import IsotonicRegression

    # Linear
    lin = linregress(x, y)
    print(f"  Linear: Gosai = {lin.slope:+.3f}*Ag + {lin.intercept:+.3f}  R²={lin.rvalue**2:.3f}")
    # Polynomial degree 2
    poly2 = np.polyfit(x, y, 2)
    y_pred_p2 = np.polyval(poly2, x)
    r2_p2 = 1 - np.sum((y - y_pred_p2)**2) / np.sum((y - y.mean())**2)
    print(f"  Quadratic: y = {poly2[0]:+.3f}*x² + {poly2[1]:+.3f}*x + {poly2[2]:+.3f}  R²={r2_p2:.3f}")
    # Polynomial degree 3
    poly3 = np.polyfit(x, y, 3)
    y_pred_p3 = np.polyval(poly3, x)
    r2_p3 = 1 - np.sum((y - y_pred_p3)**2) / np.sum((y - y.mean())**2)
    print(f"  Cubic:    R²={r2_p3:.3f}")
    # Isotonic (monotonic non-parametric)
    iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    y_pred_iso = iso.predict(x)
    r2_iso = 1 - np.sum((y - y_pred_iso)**2) / np.sum((y - y.mean())**2)
    print(f"  Isotonic: R²={r2_iso:.3f}")

    # Save
    out_dir = REPO / "outputs/agarwal_gosai_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_dir / "paired_extended.csv", index=False)
    np.save(out_dir / "iso_model_x.npy", iso.X_thresholds_)
    np.save(out_dir / "iso_model_y.npy", iso.y_thresholds_)
    summary = {
        "n_total": len(paired),
        "n_exact_fwd": len(exact_fwd),
        "n_exact_rc": len(exact_rc),
        "n_ham1": int(sum(1 for x in near_matches if x[3]==1)),
        "n_ham2": int(sum(1 for x in near_matches if x[3]==2)),
        "linear": {"slope": lin.slope, "intercept": lin.intercept, "r2": lin.rvalue**2},
        "quadratic": {"coefs": poly2.tolist(), "r2": float(r2_p2)},
        "cubic": {"coefs": poly3.tolist(), "r2": float(r2_p3)},
        "isotonic_r2": float(r2_iso),
    }
    (out_dir / "extended_mapping_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_dir / 'paired_extended.csv'}")
    print(f"Saved {out_dir / 'extended_mapping_summary.json'}")


if __name__ == "__main__":
    main()
