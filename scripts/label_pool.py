"""Label one generated cell with the oracle ensemble, atomically.

Extracted from a heredoc that lived inside a site-specific sbatch file, where it was
invisible to readers, untestable, and would have been lost when that file was retired.

Two properties matter more than they look:

  ATOMIC WRITE. The output is written to a temporary file and renamed. A rename is
  atomic on POSIX, so a job killed mid-write cannot leave a truncated file that later
  passes the `is it already labelled?` check and silently poisons a training set.

  PROVENANCE. `oracle_id` is stamped into the output. Pools labelled by a different
  oracle are indistinguishable from good ones once written, and this project has
  already had to audit its way back out of exactly that.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", required=True, help="generated cell .npz")
    ap.add_argument("--out", dest="dst", required=True, help="destination .npz")
    ap.add_argument("--task", default="k562")
    ap.add_argument("--oracle", default="ag_s2")
    ap.add_argument(
        "--oracle-id",
        default="full856k_clean",
        help="provenance stamp written into the output; must name the ensemble actually used",
    )
    args = ap.parse_args()

    if Path(args.dst).exists() and Path(args.dst).stat().st_size > 0:
        print(f"SKIP already labelled: {args.dst}")
        return 0

    from experiments.exp1_1_scaling import _load_oracle
    from scripts.generate_labeled_pools import _label_sequences

    z = np.load(args.src, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    oracle = _load_oracle(args.task, oracle_type=args.oracle)
    lab = np.asarray(_label_sequences(oracle, seqs), dtype=np.float32).ravel()

    if lab.shape[0] != len(seqs):
        raise SystemExit(f"oracle returned {lab.shape[0]} labels for {len(seqs)} sequences")
    if not np.isfinite(lab).all():
        n = int((~np.isfinite(lab)).sum())
        raise SystemExit(
            f"{n} non-finite labels; refusing to write. A NaN here becomes a training "
            f"set that looks fine and a student that learns nothing from those rows."
        )

    tmp = args.dst + ".tmp.npz"
    np.savez_compressed(
        tmp,
        sequences=np.array(seqs, dtype=object),
        oracle_labels=lab,
        strategy=z["strategy"],
        params=z["params"],
        seed=z["seed"],
        oracle_id=args.oracle_id,
    )
    os.replace(tmp, args.dst)
    print(f"wrote {len(lab):,} labels -> {args.dst}  mean={lab.mean():.3f} sd={lab.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
