"""Does mutagenising REPEATEDLY differ from one stronger pass? Build the pools to find out.

The objection is that partial random mutagenesis is applied once per sequence and that the
gains come from mutagenising repeatedly. The sampler does indeed do a single pass -- that
much is settled by reading it. What is not settled is whether repetition WITHOUT selection
is a different thing from a single pass at a higher rate, and that question is cheap to
answer while the expensive one (mutate-score-select, i.e. directed evolution) is not.

The arithmetic says they should be close but not identical. Under k independent passes at
per-base rate p, a base ends up changed with probability (3/4)(1 - (1 - 4p/3)^k), which
saturates at 3/4 rather than growing linearly -- repeated rounds can revert an earlier
mutation. So this builds two matched families:

    accum_k    k successive passes at rate p, each on the previous round's output
    once_r     a single pass at the rate r that gives the SAME expected divergence

If the pair trains equally well, repetition without selection adds nothing and the only
version worth paying for is the one with selection in the loop. If they differ, the
difference is in the per-sequence mutation-count DISTRIBUTION rather than its mean, which
is a finding about what kind of variation a corpus needs.

Mutation counts are recorded per sequence so the claim "these two are matched" is checked
against the sequences rather than against the formula.

    python scratchpad/nov25/build_prm_ladder.py --out-dir <dir> --n 30000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ALBENCH_REPO", Path(__file__).resolve().parents[2]))
NEST = REPO / "outputs/grid_pools/k562_nested"
ROUNDS = (1, 2, 5, 10)
BASE_RATE = 0.05


def effective_rate(p: float, k: int) -> float:
    """Expected per-base divergence from the ORIGINAL after k passes at rate p.

    Derived from what the sampler actually does, which is not "redraw a base from ACGT":
    it picks round(p*L) DISTINCT positions and shifts each by 1, 2 or 3 mod 4, so every
    chosen position is guaranteed to change. One pass therefore diverges by exactly p.

    Over k passes a position is hit X ~ Binomial(k, p) times and ends up different unless
    the shifts happen to sum to 0 mod 4. The shift is uniform on {1,2,3}, so
    P(sum == 0 mod 4 | X=m) = (1 + 3(-1/3)^m)/4, and averaging over X gives
    E[(3/4)(1 - (-1/3)^X)] = (3/4)(1 - (1 - 4p/3)^k). At k=1 that is exactly p, which is
    the check that the algebra matches the implementation.
    """
    return 0.75 * (1.0 - (1.0 - 4.0 * p / 3.0) ** k)


def rate_for_divergence(d: float) -> float:
    """The single-pass rate giving divergence d. One pass diverges by its own rate."""
    return min(0.75, d)


def mutate_aligned(seqs: list[str], rate: float, rng: np.random.Generator) -> list[str]:
    """One pass of the sampler's mutation, but row i stays a descendant of row i.

    PartialMutagenesisSampler.generate draws its parents with
    ``rng.choice(n_base, size=n, replace=True)``, so its output is a random resample of
    mutants rather than a row-aligned one. For building a pool that is fine; for an
    accumulation ladder it severs the lineage, and the first version of this script
    silently compared unrelated sequences and reported 0.748 divergence for every arm --
    which is simply the divergence of two random 200-mers. This reproduces the per-pass
    semantics exactly (round(rate*L) distinct positions, each shifted by 1-3 mod 4) while
    keeping the correspondence.
    """
    L = len(seqs[0])
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), L)
    lut = np.full(256, 255, dtype=np.uint8)
    for i, c in enumerate(b"ACGT"):
        lut[c] = i
    code = lut[arr]
    n_mut = max(1, int(round(rate * L)))
    # argsort of uniform noise gives distinct positions per row without a Python loop.
    pos = np.argsort(rng.random((len(seqs), L)), axis=1)[:, :n_mut]
    rows = np.arange(len(seqs))[:, None]
    shifts = rng.integers(1, 4, size=pos.shape, dtype=np.uint8)
    valid = code[rows, pos] < 4
    code[rows, pos] = np.where(valid, (code[rows, pos] + shifts) % 4, code[rows, pos])
    out = np.frombuffer(b"ACGT", dtype=np.uint8)[np.clip(code, 0, 3)]
    out = np.where(code < 4, out, arr)  # keep N and anything else untouched
    return [row.tobytes().decode("ascii") for row in out]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--rate", type=float, default=BASE_RATE)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parents are real CREs from TRAINING chromosomes only: the nested genomic pool already
    # excludes chr7/13 (test) and chr19/21/X (validation), so taking its prefix keeps the
    # holdout intact. A parent from a held-out chromosome would leak it into every child.
    z = np.load(NEST / "genomic_d300000.npz", allow_pickle=True)
    parents = [str(s) for s in z["sequences"][: a.n]]
    print(f"{len(parents):,} parents from genomic_d300000 (training chromosomes)")

    manifest = {}
    for k in ROUNDS:
        # --- accumulate: k passes, each on the previous output ----------------
        cur = parents
        for r in range(k):
            cur = mutate_aligned(cur, a.rate, np.random.default_rng(a.seed + 1000 * k + r))
        acc = cur
        d_acc = divergence(parents, acc)

        # --- match: ONE pass at the rate predicted to give the same divergence -
        r1 = rate_for_divergence(effective_rate(a.rate, k))
        one = mutate_aligned(parents, r1, np.random.default_rng(a.seed + 7 * k))
        d_one = divergence(parents, one)

        for name, seqs, d in (("accum", acc, d_acc), ("once", one, d_one)):
            tag = f"prm_{name}{k}" if name == "accum" else f"prm_once_k{k}"
            p = out / f"{tag}_raw.npz"
            np.savez_compressed(
                p,
                sequences=np.array(seqs, dtype=object),
                strategy=np.array(tag),
                seed=np.array(a.seed),
                params=np.array(
                    json.dumps(
                        {
                            "rounds": k if name == "accum" else 1,
                            "rate": a.rate if name == "accum" else round(r1, 5),
                            "parents": "genomic_d300000[:n]",
                        }
                    )
                ),
            )
            manifest[tag] = {
                "rounds": k if name == "accum" else 1,
                "rate": a.rate if name == "accum" else round(r1, 5),
                "mean_divergence": round(float(d.mean()), 5),
                "sd_divergence": round(float(d.std()), 5),
                "file": p.name,
            }
        gap = abs(d_acc.mean() - d_one.mean()) / max(1e-9, d_acc.mean())
        print(
            f"k={k:>2}  accumulate {d_acc.mean():.4f} +- {d_acc.std():.4f}   "
            f"once@{r1:.4f} {d_one.mean():.4f} +- {d_one.std():.4f}   "
            f"mean gap {gap:.1%}   spread accum {d_acc.std():.4f} vs once {d_one.std():.4f}"
        )
        # The two arms are matched on MEAN divergence by construction; the spread is where
        # they differ, because a single pass gives every sequence exactly round(rate*L)
        # mutations while k passes give a binomial spread around that. That difference is
        # the experiment, so it is printed rather than summarised as a ratio against ~0.
        # The pair is only a control if the means really do match. A formula that is
        # wrong here would make every downstream comparison a comparison of two different
        # mutation loads wearing the same label.
        if gap > 0.05:
            raise SystemExit(
                f"k={k}: matched single pass is {gap:.1%} off the accumulated divergence; "
                "the two arms are not matched and would not be a control"
            )
    (out / "prm_ladder_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n-> {out}/prm_ladder_manifest.json")
    return 0


def divergence(a: list[str], b: list[str]) -> np.ndarray:
    """Per-sequence fraction of positions that differ."""
    A = np.frombuffer("".join(a).encode(), dtype=np.uint8).reshape(len(a), -1)
    B = np.frombuffer("".join(b).encode(), dtype=np.uint8).reshape(len(b), -1)
    return (A != B).mean(axis=1)


if __name__ == "__main__":
    raise SystemExit(main())
