"""Does an acquisition rule do the same thing on every reservoir, or a different thing?

This is the question the plus-shaped design turns on. If a rule's behaviour is a property of
the RULE, the acquisition axis can be explored on one reservoir and the result carries. If it
is a property of the rule x reservoir PAIR, every arm has to be crossed and the design blows
up. ab43 gave one encouraging data point -- epistemic minus random was +0.0423 on
motif_shared_core and +0.0421 on the 1.97M union -- but that is one rule and two pools.

Training a student for every cell is ~3 GPU-h each and the full cross is 40+ cells. The
selection itself costs no training at all: a rule is a function of the bank's predictions
over the candidates, so the whole cross can be profiled by inference in minutes and only the
cells where something actually changes need a student. That is what this does.

TWO BANKS PER POOL, which is the decomposition that matters. Scoring a pool with a bank
trained on the SAME reservoir is the realistic setting, but it confounds two things: the
candidates changed AND the model ranking them changed. Scoring every pool with one COMMON
bank (the 30k genomic one) holds the ranker fixed, so the difference between the two columns
separates "this pool is different" from "the model you get from this pool is different".

Beyond what acq_activity_profile.py reports, two composition measures, because the label
mean does not say whether a rule HOMOGENISES the corpus:

    kmer_spread   total variance of 4-mer frequencies in the selection, over the same
                  quantity for the whole candidate pool. <1 means the rule bought a
                  narrower slice of sequence space than random would have, which is the
                  obvious mechanism for a corpus that trains a bad student.
    gc_shift      mean GC of the selection minus the pool's, in points.

    python scratchpad/nov25/acq_pool_interaction.py --pool P.npz --banks B1 B2 --out O.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_matrix(seqs: list[str], k: int = 4) -> np.ndarray:
    """(n, 4**k) 4-mer frequency matrix. Vectorised: these are 200k-sequence pools."""
    n, L = len(seqs), len(seqs[0])
    codes = np.full((n, L), 255, np.uint8)
    arr = np.frombuffer("".join(seqs).encode(), np.uint8).reshape(n, L)
    for b, v in _BASES.items():
        codes[arr == ord(b)] = v
    ok = codes < 4
    out = np.zeros((n, 4**k), np.float32)
    idx = np.zeros(n * (L - k + 1), np.int64)
    valid = np.ones(n * (L - k + 1), bool)
    for j in range(k):
        c = codes[:, j : L - k + 1 + j]
        idx = idx * 4 + c.reshape(-1).astype(np.int64)
        valid &= ok[:, j : L - k + 1 + j].reshape(-1)
    rows = np.repeat(np.arange(n), L - k + 1)
    np.add.at(out, (rows[valid], idx[valid]), 1.0)
    tot = out.sum(1, keepdims=True)
    return out / np.maximum(tot, 1.0)


def spread(M: np.ndarray) -> float:
    """Total variance of the k-mer profile: a scalar 'how much of sequence space'."""
    return float(M.var(axis=0).sum())


def gc_of(seqs: list[str]) -> np.ndarray:
    arr = np.frombuffer("".join(seqs).encode(), np.uint8).reshape(len(seqs), len(seqs[0]))
    gc = (arr == ord("G")) | (arr == ord("C"))
    return gc.mean(axis=1) * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--banks", nargs="+", required=True)
    ap.add_argument("--seed-size", type=int, default=30_000)
    ap.add_argument("--select", type=int, default=30_000)
    ap.add_argument("--cap", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # albench.acquisition.legnet_bank.LegNetBank, NOT albench.ensemble.bank.LegNetBank:
    # two classes share the name and only this one has predict()/uncertainty().
    from albench.acquisition.diversity import DiversityAcquisition
    from albench.acquisition.ensemble_acq import ExpectedActivityAcquisition
    from albench.acquisition.legnet_bank import LegNetBank
    from albench.acquisition.random_acq import RandomAcquisition
    from albench.acquisition.uncertainty import UncertaintyAcquisition
    from albench.acquisition.uncertainty_variants import ActivityNormalisedUncertaintyAcquisition

    z = np.load(a.pool, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    y = np.asarray(z["oracle_labels"], dtype=float)

    rng = np.random.default_rng(0)
    cand = np.arange(a.seed_size, len(seqs))
    if len(cand) > a.cap:
        cand = np.sort(rng.choice(cand, a.cap, replace=False))
    cs = [seqs[i] for i in cand]
    cy = y[cand]
    print(f"{len(cs):,} candidates from {Path(a.pool).name}", flush=True)

    K = kmer_matrix(cs)
    pool_spread = spread(K)
    pool_gc = gc_of(cs)

    res: dict = {
        "pool": Path(a.pool).name,
        "n_candidates": len(cs),
        "pool_label_mean": round(float(cy.mean()), 4),
        "pool_label_sd": round(float(cy.std()), 4),
        "pool_gc": round(float(pool_gc.mean()), 3),
        "banks": {},
    }

    for bank in a.banks:
        student = LegNetBank(bank)
        pred = np.asarray(student.predict(cs), dtype=float).ravel()
        unc = np.asarray(student.uncertainty(cs), dtype=float).ravel()
        tag = Path(bank).name
        print(f"\n-- bank {tag}", flush=True)
        entry: dict = {
            "pearson_unc_vs_pred": round(float(np.corrcoef(pred, unc)[0, 1]), 4),
            # How well this bank actually predicts THIS pool. A rule can only be as good as
            # the ranking it is given, so a common bank on a far-away pool is a different
            # experiment from a matched bank, and this number says how different.
            "bank_r_on_pool": round(float(np.corrcoef(pred, cy)[0, 1]), 4),
            "rules": {},
        }
        rules = {
            "random": RandomAcquisition(seed=0),
            "epistemic": UncertaintyAcquisition(),
            "activity_normalized": ActivityNormalisedUncertaintyAcquisition(seed=0),
            "expected_activity": ExpectedActivityAcquisition(seed=0),
            "diversity_kmeans": DiversityAcquisition(seed=0),
        }
        picks: dict[str, np.ndarray] = {}
        for name, rule in rules.items():
            try:
                idx = np.asarray(rule.select(student, cs, a.select), dtype=int)
            except Exception as e:  # a rule that cannot run must say so, not vanish
                entry["rules"][name] = {"error": f"{type(e).__name__}: {e}"}
                print(f"   {name}: FAILED {e}", flush=True)
                continue
            picks[name] = idx
            sel = cy[idx]
            entry["rules"][name] = {
                "label_mean": round(float(sel.mean()), 4),
                "label_sd": round(float(sel.std()), 4),
                "label_z_vs_pool": round(float((sel.mean() - cy.mean()) / (cy.std() or 1)), 4),
                "unc_mean": round(float(unc[idx].mean()), 4),
                "kmer_spread_ratio": round(float(spread(K[idx]) / (pool_spread or 1)), 4),
                "gc_shift": round(float(pool_gc[idx].mean() - pool_gc.mean()), 3),
                "frac_above_pool_mean": round(float((sel > cy.mean()).mean()), 4),
            }
            print(
                "   %-20s label z %+0.3f  spread %.3f  gc %+0.2f  unc %.4f"
                % (
                    name,
                    entry["rules"][name]["label_z_vs_pool"],
                    entry["rules"][name]["kmer_spread_ratio"],
                    entry["rules"][name]["gc_shift"],
                    entry["rules"][name]["unc_mean"],
                ),
                flush=True,
            )
        entry["overlap"] = {}
        names = sorted(picks)
        for i, A in enumerate(names):
            for B in names[i + 1 :]:
                entry["overlap"][f"{A}|{B}"] = round(
                    float(len(np.intersect1d(picks[A], picks[B])) / a.select), 4
                )
        res["banks"][tag] = entry

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
