"""K562-accessibility subsetting for the genomic-vs-random GC-bias investigation.

Question: the genomic reservoir is predicted (and oracle-labelled) LOWER than
random. Is that partly because genomic TEST windows are mostly inaccessible? If we
restrict genomic to K562-accessible windows, does their activity/prediction rise
toward random, closing the genomic-vs-random gap?
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

from albench.composition import cpg_obs_exp, gc_content

ROOT = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
TESTDIR = ROOT / "data/k562/test_sets_ag_s2_chrsplit"
TSV = ROOT / "data/k562/test_sets/test_chr7_13_all.tsv"
PEAKS = ROOT / "data/encode_accessibility/K562_peaks_merged.bed"
CELL = ROOT / "outputs/cadence/genomic_llm2x30"
OUTDIR = ROOT / "outputs/gc_bias"
WIN = 100  # half-window: variant pos +/- 100 -> 200bp window

def main() -> None:
    g = np.load(TESTDIR / "genomic_oracle.npz", allow_pickle=True)
    seqs = [str(s).upper() for s in g["sequences"]]
    oracle = np.asarray(g["oracle_mean"], dtype=np.float64)
    true = np.asarray(g["true_label"], dtype=np.float64)
    n = len(seqs)
    print(f"genomic test windows: {n}")

    # --- recover coords by exact sequence match to the chr7/13 source tsv ---
    seq2cp: dict[str, tuple[str, int]] = {}
    with open(TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            seq2cp[row["sequence"].upper()] = ("chr" + row["chr"], int(row["pos"]))
    chrom = np.array(["?"] * n, dtype=object)
    pos = np.full(n, -1, dtype=np.int64)
    for i, s in enumerate(seqs):
        cp = seq2cp.get(s)
        if cp is not None:
            chrom[i], pos[i] = cp
    matched = pos >= 0
    print(f"coords recovered: {matched.sum()}/{n} ({matched.mean()*100:.1f}%)")

    # --- load K562 merged peaks, build per-chrom sorted interval arrays ---
    peaks: dict[str, list[tuple[int, int]]] = {}
    with open(PEAKS) as f:
        for line in f:
            c, a, b = line.split()[:3]
            peaks.setdefault(c, []).append((int(a), int(b)))
    peak_arr = {c: np.array(sorted(v)) for c, v in peaks.items()}

    def overlaps(c: str, lo: int, hi: int) -> bool:
        arr = peak_arr.get(c)
        if arr is None:
            return False
        starts = arr[:, 0]
        # candidate peaks whose start < window hi
        j = np.searchsorted(starts, hi)
        if j == 0:
            return False
        # check the nearest few peaks ending >= window lo
        seg = arr[:j]
        return bool(np.any(seg[:, 1] > lo))

    accessible = np.zeros(n, dtype=bool)
    for i in np.where(matched)[0]:
        accessible[i] = overlaps(chrom[i], pos[i] - WIN, pos[i] + WIN)
    acc = matched & accessible
    inacc = matched & ~accessible
    print(f"accessible: {acc.sum()}  inaccessible: {inacc.sum()}  "
          f"frac accessible (of matched): {acc.sum()/matched.sum()*100:.1f}%")

    # --- ensemble model predictions on genomic + random (avg over cell models) ---
    files = [p for p in sorted(glob.glob(str(CELL / "r*_s0.npz")))
             if "meta" not in p]
    pg = np.zeros(n); pr = None; k = 0
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if "test_pred_genomic" not in d.files:
            continue
        pg += np.asarray(d["test_pred_genomic"], dtype=np.float64)
        rr = np.asarray(d["test_pred_random_32k"], dtype=np.float64)
        pr = rr if pr is None else pr + rr
        k += 1
    pred_genomic = pg / k
    pred_random = pr / k
    print(f"ensembled over {k} models")

    # --- random baseline oracle ---
    r32 = np.load(TESTDIR / "random_32k_oracle.npz", allow_pickle=True)
    oracle_random = np.asarray(r32["oracle_mean"], dtype=np.float64)

    # --- composition ---
    gc = np.array([gc_content(s) for s in seqs])
    cpg = np.array([cpg_obs_exp(s) for s in seqs])
    gc_random = np.array([gc_content(str(s)) for s in r32["sequences"]])
    cpg_random = np.array([cpg_obs_exp(str(s)) for s in r32["sequences"]])

    def med(x, m):
        return float(np.median(x[m])) if m.sum() else float("nan")

    groups = {
        "all_genomic": matched | ~matched,  # everything
        "accessible": acc,
        "inaccessible": inacc,
    }
    rows = {}
    for name, m in groups.items():
        rows[name] = dict(
            n=int(m.sum()),
            oracle=med(oracle, m), true=med(true, m),
            pred=med(pred_genomic, m), gc=med(gc, m), cpg=med(cpg, m),
        )
    rows["random_32k"] = dict(
        n=int(oracle_random.size),
        oracle=float(np.median(oracle_random)), true=float("nan"),
        pred=float(np.median(pred_random)), gc=float(np.median(gc_random)),
        cpg=float(np.median(cpg_random)),
    )

    print("\n=== MEDIANS ===")
    hdr = f"{'group':14s} {'n':>7s} {'oracle':>8s} {'true':>8s} {'pred':>8s} {'GC':>7s} {'CpGo/e':>7s}"
    print(hdr)
    for name, rr in rows.items():
        print(f"{name:14s} {rr['n']:7d} {rr['oracle']:8.3f} {rr['true']:8.3f} "
              f"{rr['pred']:8.3f} {rr['gc']:7.3f} {rr['cpg']:7.3f}")

    gap_all = rows["random_32k"]["pred"] - rows["all_genomic"]["pred"]
    gap_acc = rows["random_32k"]["pred"] - rows["accessible"]["pred"]
    o_gap_all = rows["random_32k"]["oracle"] - rows["all_genomic"]["oracle"]
    o_gap_acc = rows["random_32k"]["oracle"] - rows["accessible"]["oracle"]
    print(f"\nPRED gap random-genomic: all={gap_all:.3f}  accessible={gap_acc:.3f}  "
          f"closed {(1-gap_acc/gap_all)*100:.0f}%")
    print(f"ORACLE gap random-genomic: all={o_gap_all:.3f}  accessible={o_gap_acc:.3f}  "
          f"closed {(1-o_gap_acc/o_gap_all)*100:.0f}%")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTDIR / "accessibility_subset.npz",
        chrom=chrom, pos=pos, matched=matched, accessible=accessible,
        oracle=oracle, true=true, pred_genomic=pred_genomic, gc=gc, cpg=cpg,
        oracle_random=oracle_random, pred_random=pred_random,
        gc_random=gc_random, cpg_random=cpg_random,
        summary=json.dumps(rows), n_models=k,
    )

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.linspace(-1.5, 3.5, 60)

    def dist(a, data, groups_spec, title, xlabel):
        for lbl, vals, color in groups_spec:
            a.hist(vals, bins=bins, density=True, histtype="step", lw=2,
                   color=color, label=f"{lbl} (med {np.median(vals):.2f})")
        a.set_title(title); a.set_xlabel(xlabel); a.legend(fontsize=8)

    dist(ax[0], None, [
        ("random", pred_random, "#4C72B0"),
        ("accessible genomic", pred_genomic[acc], "#55A868"),
        ("inaccessible genomic", pred_genomic[inacc], "#C44E52"),
    ], "Model PREDICTED activity", "predicted activity")
    dist(ax[1], None, [
        ("random", oracle_random, "#4C72B0"),
        ("accessible genomic", oracle[acc], "#55A868"),
        ("inaccessible genomic", oracle[inacc], "#C44E52"),
    ], "ORACLE activity", "oracle_mean")

    # accessibility vs GC/CpG
    ax[2].scatter(gc[inacc], cpg[inacc], s=3, alpha=0.2, color="#C44E52", label="inaccessible")
    ax[2].scatter(gc[acc], cpg[acc], s=3, alpha=0.3, color="#55A868", label="accessible")
    ax[2].set_xlabel("GC content"); ax[2].set_ylabel("CpG O/E")
    ax[2].set_title(f"accessible GC {np.median(gc[acc]):.2f}/CpG {np.median(cpg[acc]):.2f}  "
                    f"vs inacc GC {np.median(gc[inacc]):.2f}/CpG {np.median(cpg[inacc]):.2f}")
    ax[2].legend(fontsize=8); ax[2].set_ylim(0, 1.5)
    fig.suptitle("K562-accessibility subset of genomic test windows (chr7/chr13)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "accessibility_subset.png", dpi=130)
    print("saved", OUTDIR / "accessibility_subset.png")

if __name__ == "__main__":
    main()
