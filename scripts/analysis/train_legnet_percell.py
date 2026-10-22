"""Train one MPRA-LegNet per cell type on real Gosai labels with chr splits.

Needed for the GC-bias model comparison: Malinois and AG-MPRA already have a
separate model per cell type (K562/HepG2/SK-N-SH); LegNet only had K562. Trains
all three identically (same architecture, same split) so the three LegNet models
are mutually comparable. test=chr7/13, val=chr19/21/X, train=the rest.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
TABLE = f"{REPO}/data/hepg2/DATA-Table_S2__MPRA_dataset.txt"
TEST_CHR = {"7", "13"}
VAL_CHR = {"19", "21", "X"}
CELL_COL = {"k562": "K562_log2FC", "hepg2": "HepG2_log2FC", "sknsh": "SKNSH_log2FC"}


def load_split(cell):
    df = pd.read_csv(TABLE, sep="\t", usecols=["chr", "sequence", CELL_COL[cell]])
    df = df.rename(columns={CELL_COL[cell]: "y"}).dropna(subset=["y", "sequence"])
    df["chr"] = df["chr"].astype(str).str.replace("chr", "", regex=False)
    df = df[df["sequence"].str.len() == 200]
    tr = df[~df["chr"].isin(TEST_CHR | VAL_CHR)]
    va = df[df["chr"].isin(VAL_CHR)]
    te = df[df["chr"].isin(TEST_CHR)]
    return tr, va, te


def main():
    cell = sys.argv[1]
    from models.legnet_student import LegNetStudent, TrainConfig
    import torch

    tr, va, te = load_split(cell)
    print(f"  {cell}: train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    tc = TrainConfig(lr=5e-3, batch_size=1024, epochs=40, early_stopping_patience=8,
                     num_workers=4, optimizer="adamw")
    st = LegNetStudent(train_config=tc, ensemble_size=1, device="cuda")
    st.fit(tr["sequence"].tolist(), tr["y"].to_numpy(np.float32),
           va["sequence"].tolist(), va["y"].to_numpy(np.float32))
    pred = st.predict(te["sequence"].tolist())
    r = float(np.corrcoef(pred, te["y"].to_numpy(np.float32))[0, 1])
    print(f"  {cell}: chr7/13 test Pearson = {r:.4f}", flush=True)
    out = f"{REPO}/outputs/gc_bias/legnet_percell/legnet_{cell}.pt"
    torch.save(st.models[0], out)  # whole module; our own reuse, weights_only=False
    np.savez(f"{REPO}/outputs/gc_bias/legnet_percell/legnet_{cell}_meta.npz",
             cell=cell, test_pearson=r, n_train=len(tr))
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
