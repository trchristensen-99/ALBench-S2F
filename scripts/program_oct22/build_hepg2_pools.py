"""Turn the multicell relabel sidecars into HepG2- and differential-labelled replicate pools and validation caches for the student retrains.

    python scripts/program_oct22/build_hepg2_pools.py [--reps 1]

Reads  outputs/oracle_v3_hepg2/relabel/reps_r{r}_d30000/{strategy}_r{r}_d30000.multicell.npz   (label_pool_multicell.py relabel-reps)
       outputs/oracle_v3_hepg2/relabel/val/val_V0.multicell.npz                               (label_pool_multicell.py label --in val_V0.npz --mode oof)
Writes outputs/grid_pools/k562_hepg2/{hepg2,diff}/{strategy}_r{r}_d30000.npz and outputs/val_variants/val_V0_{hepg2,diff}.npz, each stamped
oracle_id = oracle_v3_<target> (the training loaders refuse non-canonical stamps unless HP_ALLOW_UNSTAMPED=1, which the retrain cells set on purpose).
Every file is checked row by row against the file it was derived from, and the K562 label in the sidecar must reproduce the stored label.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_lib import GP, REPS, SEED, R, save_pool  # noqa: E402

RES = ["genomic", "random", "motif_ct_enriched"]
MC = f"{R}/outputs/oracle_v3_hepg2/relabel"


def check(z, m, what, key="label"):
    n = len(z["sequences"])
    assert len(m["hepg2_label"]) == n, (what, n, len(m["hepg2_label"]))
    if "sequences" in m.files:
        assert all(str(a) == str(b) for a, b in zip(z["sequences"], m["sequences"], strict=True)), (
            f"{what}: sequence order differs"
        )
    stored = z["oracle_labels"].astype(np.float64)
    r = np.corrcoef(stored, m[f"k562_{key}"])[0, 1]
    assert r > 0.999 and np.isfinite(m["hepg2_label"]).all(), (what, r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    for r in args.reps:
        for st in RES:
            z = np.load(f"{REPS}/{st}_r{r}_d30000.npz", allow_pickle=True)
            m = np.load(f"{MC}/reps_r{r}_d30000/{st}_r{r}_d30000.multicell.npz", allow_pickle=True)
            rr = check(z, m, f"{st} r{r}")
            for tgt in ("hepg2", "diff"):
                out = f"{GP}/k562_hepg2/{tgt}"
                os.makedirs(out, exist_ok=True)
                save_pool(
                    f"{out}/{st}_r{r}_d30000.npz",
                    list(z["sequences"]),
                    m[f"{tgt}_label"],
                    f"oracle_v3_{tgt}",
                    st,
                    SEED(r),
                )
            print(
                f"{st} r{r}: K562 fidelity r={rr:.5f}; hepg2 label mean {m['hepg2_label'].mean():.3f} sd {m['hepg2_label'].std():.3f}; K562-HepG2 corr {np.corrcoef(m['k562_label'], m['hepg2_label'])[0, 1]:.3f}",
                flush=True,
            )
    z = np.load(f"{R}/outputs/val_variants/val_V0.npz", allow_pickle=True)
    m = np.load(f"{MC}/val/val_V0.multicell.npz", allow_pickle=True)
    rr = check(
        z, m, "val_V0", key="ens"
    )  # val_V0 labels are the plain 10-fold ensemble mean (not out-of-fold), so the HepG2 / diff val labels must be too
    for tgt in ("hepg2", "diff"):
        np.savez(
            f"{R}/outputs/val_variants/val_V0_{tgt}.tmp.npz",
            sequences=z["sequences"],
            oracle_labels=m[f"{tgt}_ens"].astype(np.float32),
            oracle_id=f"oracle_v3_{tgt}",
        )
        os.replace(
            f"{R}/outputs/val_variants/val_V0_{tgt}.tmp.npz",
            f"{R}/outputs/val_variants/val_V0_{tgt}.npz",
        )
    print(
        f"val_V0: K562 fidelity r={rr:.5f}; wrote val_V0_hepg2 / val_V0_diff ({len(z['sequences']):,} sequences)"
    )


if __name__ == "__main__":
    main()
