"""Select a D-sequence training corpus from one or more candidate reservoirs.

Built for the two questions the acquisition figure has to answer, both of which hold the
TRAINING SET SIZE fixed and vary only what the rule had to choose from:

  (a) how much does having more to choose from help?   one reservoir, candidate pool swept
      from D (where selection is a no-op and the arm is just a random draw) upward.
  (b) does a mixture of reservoirs beat a single one?   several reservoirs, equal shares,
      summing to one fixed candidate-pool size so pool size is not confounded with mixture.

Every arm is scored by the SAME bank -- the 30k genomic one -- so the comparison is between
candidate sets, not between rankers. That is the realistic single-round setting too: you
have the Gosai data, you train one pilot model, you select once, you order.

    python scratchpad/nov25/select_corpus.py --out P.npz --select 30000 \
        --bank outputs/wed41/bench38_ln_e500_wd2_30k_c20 --method uncertainty \
        --source random=150000 --source motif_shared_core=150000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/program_oct22"))

NESTED = REPO / "outputs/grid_pools/k562_nested"
# 300k is the largest size every family has; genomic physically cannot exceed ~315k.
SRC_SIZES = (300_000, 1_000_000)


def load_source(name: str, n: int, rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """First `n` rows of a reservoir's nested pool. Nested pools are prefixes, so taking the
    first n is the canonical subset and keeps every mixture share drawn the same way."""
    for size in SRC_SIZES:
        p = NESTED / f"{name}_d{size}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            seqs = [str(s) for s in z["sequences"]]
            y = np.asarray(z["oracle_labels"], dtype=np.float32)
            if len(seqs) < n:
                continue
            return seqs[:n], y[:n]
    raise SystemExit(f"no pool for {name} with at least {n:,} rows under {NESTED}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", type=int, default=30_000)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--method", default="uncertainty")
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="name=count, repeatable; counts sum to the candidate-pool size",
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and out.stat().st_size > 0:
        try:
            with np.load(out, allow_pickle=True) as z:
                _ = z["sequences"]
            print(f"SKIP already present: {out}")
            return 0
        except Exception as e:
            print(f"present but unreadable ({type(e).__name__}), rebuilding")

    rng = np.random.default_rng(a.seed)
    seqs: list[str] = []
    ys: list[np.ndarray] = []
    src_tag: list[str] = []
    for spec in a.source:
        name, _, cnt = spec.partition("=")
        s, y = load_source(name, int(cnt), rng)
        seqs += s
        ys.append(y)
        src_tag += [name] * len(s)
    y = np.concatenate(ys)
    print(f"candidate pool: {len(seqs):,} from {len(a.source)} source(s)", flush=True)

    if a.select >= len(seqs):
        # Selection is a no-op: the arm IS the pool. This is the x = D point of the sweep
        # and it must be kept, because it is the random-draw baseline the curve rises from.
        idx = np.arange(len(seqs))
        print("pool <= select: taking everything (no-op selection)", flush=True)
    else:
        from albench.acquisition.diversity import DiversityAcquisition
        from albench.acquisition.ensemble_acq import ExpectedActivityAcquisition
        from albench.acquisition.legnet_bank import LegNetBank
        from albench.acquisition.random_acq import RandomAcquisition
        from albench.acquisition.uncertainty import UncertaintyAcquisition
        from albench.acquisition.uncertainty_variants import (
            ActivityNormalisedUncertaintyAcquisition,
        )

        rules = {
            "random": lambda: RandomAcquisition(seed=a.seed),
            "uncertainty": UncertaintyAcquisition,
            "activity_normalized": lambda: ActivityNormalisedUncertaintyAcquisition(seed=a.seed),
            "expected_activity": lambda: ExpectedActivityAcquisition(seed=a.seed),
            "diversity_kmeans": lambda: DiversityAcquisition(seed=a.seed),
        }
        if a.method not in rules:
            raise SystemExit(f"unknown method {a.method}; have {sorted(rules)}")
        student = LegNetBank(a.bank)
        idx = np.asarray(rules[a.method]().select(student, seqs, a.select), dtype=int)
    sel = np.sort(idx)

    from stage2_lib import save_pool  # noqa: E402

    out.parent.mkdir(parents=True, exist_ok=True)
    save_pool(
        str(out),
        [seqs[i] for i in sel],
        y[sel],
        "oracle_v3",
        "+".join(s.split("=")[0] for s in a.source),
        a.seed,
    )
    tags, counts = np.unique([src_tag[i] for i in sel], return_counts=True)
    mix = {str(t): int(c) for t, c in zip(tags, counts, strict=True)}
    # The realised mixture is the result, not bookkeeping: when a rule picks from several
    # reservoirs it is CHOOSING the mixture, and this is what it chose.
    Path(str(out).replace(".npz", "_mix.json")).write_text(
        json.dumps(
            {
                "sources": a.source,
                "method": a.method,
                "n_candidates": len(seqs),
                "selected": int(len(sel)),
                "realised_mixture": mix,
                "label_mean": round(float(y[sel].mean()), 4),
                "pool_label_mean": round(float(y.mean()), 4),
            },
            indent=2,
        )
    )
    print(f"wrote {out}  mixture {mix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
