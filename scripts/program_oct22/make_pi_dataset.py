"""Package the 30k Gosai K562 genomic training set with its validation and test sets for an outside user of the study (a PI who wants to run his own search on it).

Everything is copied from the lab's shared albench folder (the files the study itself uses), so numbers on this package are comparable with the study's. Two additions:
`measured_log2fc` (the Gosai et al. 2024 measurement, joined by sequence from the full 314,981-sequence pool; NaN if a sequence is not in it) next to the oracle_v3 label, and plain TSV copies
of every npz. The script refuses to write if any training sequence appears in the validation or any test set.

    python scripts/program_oct22/make_pi_dataset.py --out /grid/koo/home/shared/data/episomal_mpra/gosai_2024_k562_30k_autoresearch   (cpuq)
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat

import numpy as np

SH = "/grid/koo/home/shared"
SRC = f"{SH}/albench/data"
POOL = f"{SH}/data/lentimpra/k562_genomic_reservoir/genomic_pool_314981.npz"


def tsv(path, cols):
    names = list(cols)
    n = len(cols[names[0]])
    with open(path, "w") as f:
        f.write("\t".join(names) + "\n")
        for i in range(n):
            f.write(
                "\t".join(
                    str(cols[k][i])
                    if not isinstance(cols[k][i], (float, np.floating))
                    else f"{cols[k][i]:.6g}"
                    for k in names
                )
                + "\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tr = np.load(f"{SRC}/genomic_start/genomic_d30000.npz", allow_pickle=True)
    seqs = np.array([str(s) for s in tr["sequences"]], dtype=object)
    orc = np.asarray(tr["oracle_labels"], np.float32)
    assert (
        str(tr["oracle_id"]) == "oracle_v3"
        and len(seqs) == 30000
        and len(set(seqs.tolist())) == 30000
    )
    pool = np.load(POOL, allow_pickle=True)
    real = dict(
        zip((str(s) for s in pool["sequences"]), np.asarray(pool["real_labels"], np.float32))
    )
    meas = np.array([real.get(s, np.nan) for s in seqs], np.float32)
    n_found = int(np.isfinite(meas).sum())

    val = np.load(f"{SRC}/val_V0c.npz", allow_pickle=True)
    tests = {
        n: np.load(f"{SRC}/test_sets/{n}_oracle.npz", allow_pickle=True)
        for n in ("genomic", "ood", "random_32k")
    }
    bad = {}
    s_tr = set(seqs.tolist())
    bad["val_V0c"] = len(s_tr & {str(s) for s in val["sequences"]})
    for n, z in tests.items():
        bad[f"test_{n}"] = len(s_tr & {str(s) for s in z["sequences"]})
    print(
        "training sequences found in the measured pool:",
        n_found,
        "of",
        len(seqs),
        "| overlaps with validation/test:",
        bad,
        flush=True,
    )
    assert all(v == 0 for v in bad.values()), f"training sequences appear in held-out sets: {bad}"
    assert n_found > 0.99 * len(seqs), "the measured labels could not be joined"

    np.savez_compressed(
        f"{a.out}/train_30k.npz",
        sequences=seqs,
        oracle_labels=orc,
        measured_log2fc=meas,
        oracle_id=np.array("oracle_v3"),
        source=np.array(
            "first 30,000 of the canonical nested genomic pool (seed-42 permutation of the 314,981 holdout-clean Gosai K562 sequences); see README.md"
        ),
    )
    tsv(f"{a.out}/train_30k.tsv", {"sequence": seqs, "oracle_label": orc, "measured_log2fc": meas})
    shutil.copy(f"{SRC}/val_V0c.npz", f"{a.out}/val_V0c.npz")
    tsv(
        f"{a.out}/val_V0c.tsv",
        {
            "sequence": [str(s) for s in val["sequences"]],
            "oracle_label": np.asarray(val["oracle_labels"], np.float32),
        },
    )
    os.makedirs(f"{a.out}/test_sets", exist_ok=True)
    for n, z in tests.items():
        shutil.copy(f"{SRC}/test_sets/{n}_oracle.npz", f"{a.out}/test_sets/{n}_oracle.npz")
        cols = {
            "sequence": [str(s) for s in z["sequences"]],
            "oracle_label": np.asarray(z["oracle_mean"], np.float32),
        }
        if "true_label" in z.files:
            cols["measured_log2fc"] = np.asarray(z["true_label"], np.float32)
        tsv(f"{a.out}/test_sets/{n}.tsv", cols)
    for root, dirs, files in os.walk(a.out):
        os.chmod(root, 0o2775)
        for f in files:
            os.chmod(os.path.join(root, f), 0o664)
    print("wrote", a.out, sorted(os.listdir(a.out)), flush=True)


if __name__ == "__main__":
    main()
