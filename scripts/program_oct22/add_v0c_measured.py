"""Measured K562 labels for the V0c validation sequences (extra file; the V0c file itself stays byte-identical).

V0c carries oracle_v3 labels only. Its sequences are real Gosai et al. library members (chr19/21/X), so the measured K562 log2FC and its standard error come from joining on the sequence to Table S2.
Writes val_V0c_measured.npz next to val_V0c.npz with `sequences` (same order), `measured_log2fc`, `measured_lfcse` (NaN where a sequence is not in Table S2) and `oracle_labels` (copied, for convenience), and prints
the match rate and the oracle-vs-measured correlation.

    python scripts/program_oct22/add_v0c_measured.py [--shared /grid/koo/home/shared/albench/data] [--table data/k562/DATA-Table_S2__MPRA_dataset.txt]      (cpuq)
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", default="/grid/koo/home/shared/albench/data")
    ap.add_argument("--table", default="data/k562/DATA-Table_S2__MPRA_dataset.txt")
    a = ap.parse_args()
    z = np.load(f"{a.shared}/val_V0c.npz", allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    want = {s: i for i, s in enumerate(seqs)}
    lfc = np.full(len(seqs), np.nan, np.float32)
    se = np.full(len(seqs), np.nan, np.float32)
    with open(a.table) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        js, jl, je = hdr.index("sequence"), hdr.index("K562_log2FC"), hdr.index("K562_lfcSE")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            i = want.get(p[js].strip().upper())
            if i is None:
                continue
            try:
                lfc[i], se[i] = float(p[jl]), float(p[je])
            except ValueError:
                pass
    ok = np.isfinite(lfc)
    orc = z["oracle_labels"].astype(np.float64)
    r = float(np.corrcoef(lfc[ok], orc[ok])[0, 1])
    print(f"matched {ok.sum():,} of {len(seqs):,} V0c sequences; corr(oracle, measured) = {r:.4f}")
    np.savez_compressed(
        f"{a.shared}/val_V0c_measured.npz",
        sequences=np.asarray(seqs),
        measured_log2fc=lfc,
        measured_lfcse=se,
        oracle_labels=z["oracle_labels"],
    )
    print("wrote", f"{a.shared}/val_V0c_measured.npz")


if __name__ == "__main__":
    main()
