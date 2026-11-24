"""Two architecturally independent yeast oracles on the same held-out test set.

THE PROBLEM THIS MEASURES. The yeast oracle is a DREAM-RNN: a CNN-plus-LSTM trained from
scratch on the 6.07M random-promoter library, with none of AlphaGenome's pre-training. It
scores r = 0.9647 on the DREAM 2022 test set, which is at or above the published
post-challenge model, so as a labeller it is good. But a simple recurrent model is also a
much easier function for a student to copy than an AlphaGenome-derived one, and a student
that copies the labeller looks excellent on oracle-labelled data while having learned the
labeller's idiosyncrasies rather than yeast regulatory grammar. Nothing in an
oracle-labelled benchmark can detect that, because the oracle defines the target.

Two numbers make it detectable, and both come from the AlphaGenome yeast ensemble that
already exists in outputs/oracle_alphagenome_yeast_ensemble and has never been compared
with the RNN:

  1. AG's own r on the DREAM test set -- is it a competent second opinion, or a worse
     model whose disagreement would mean nothing?
  2. r(AG, RNN) on those same sequences, and how it compares with r(RNN, truth). Where
     two independent architectures agree, the agreement is about yeast. Where they
     diverge, at least one is extrapolating, and that is precisely the set of sequences
     on which a student's apparent skill should not be believed.

The output supports the mitigation this is for: label the reservoir pools with the RNN
ensemble (it is the better and cheaper labeller), but hold out the AG ensemble as an
independent judge and report a corpus gain only where it survives being scored against
the second oracle too. The same logic as the two-oracle disagreement work on K562.

    python scratchpad/nov24/ye45_cross_oracle.py --limit 71103
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))


def pearson(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes-npz", default="data/yeast/eval_classes_v2.npz")
    ap.add_argument("--rnn-pred", default="outputs/yeast_prod/dream_eval_pred.npz")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (for a smoke run)")
    ap.add_argument("--out", default="outputs/yeast_prod/cross_oracle.json")
    a = ap.parse_args()

    z = np.load(REPO / a.classes_npz, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    y = np.asarray(z["labels"], np.float64)
    classes = {k[4:]: np.asarray(z[k]) for k in z.files if k.startswith("cls_")}

    # The RNN ensemble's predictions, already computed by ye44 rather than recomputed, so
    # the two oracles are being compared on exactly the numbers that were reported.
    rp = np.load(REPO / a.rnn_pred, allow_pickle=True)
    key = "ensemble" if "ensemble" in rp.files else rp.files[0]
    rnn = np.asarray(rp[key], np.float64).ravel()
    if len(rnn) != len(seqs):
        raise SystemExit(
            f"{len(rnn)} RNN predictions for {len(seqs)} sequences: "
            f"{a.rnn_pred} was written for a different eval set"
        )

    if a.limit:
        seqs, y, rnn = seqs[: a.limit], y[: a.limit], rnn[: a.limit]
        classes = {k: v[v < a.limit] for k, v in classes.items()}
    print(f"{len(seqs):,} DREAM test sequences, {len(classes)} classes", flush=True)

    from exp1_1_scaling import _load_oracle

    t0 = time.time()
    ag = _load_oracle("yeast", oracle_type="ag")
    print(f"AG yeast ensemble loaded in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    preds = []
    B = 2048
    for s in range(0, len(seqs), B):
        preds.append(np.asarray(ag.predict(seqs[s : s + B]), np.float64).ravel())
        if s % (B * 8) == 0:
            done = min(s + B, len(seqs))
            print(
                f"  {done:,}/{len(seqs):,}  ({done / max(time.time() - t0, 1e-6):.0f} seq/s)",
                flush=True,
            )
    agp = np.concatenate(preds)[: len(seqs)]
    print(f"scored in {time.time() - t0:.0f}s", flush=True)

    res = {
        "n": len(seqs),
        "rnn_vs_truth": {
            "pearson": round(pearson(rnn, y), 4),
            "r2": round(pearson(rnn, y) ** 2, 4),
            "spearman": round(spearman(rnn, y), 4),
        },
        "ag_vs_truth": {
            "pearson": round(pearson(agp, y), 4),
            "r2": round(pearson(agp, y) ** 2, 4),
            "spearman": round(spearman(agp, y), 4),
        },
        "ag_vs_rnn": {
            "pearson": round(pearson(agp, rnn), 4),
            "spearman": round(spearman(agp, rnn), 4),
        },
    }

    print("\n%-22s %10s %10s %10s" % ("", "pearson", "r2", "spearman"))
    for k in ("rnn_vs_truth", "ag_vs_truth"):
        v = res[k]
        print("%-22s %10.4f %10.4f %10.4f" % (k, v["pearson"], v["r2"], v["spearman"]))
    print(
        "%-22s %10.4f %10s %10.4f"
        % ("ag_vs_rnn", res["ag_vs_rnn"]["pearson"], "-", res["ag_vs_rnn"]["spearman"])
    )

    # WHERE THE TWO DISAGREE is the output that gets used. A student's gain on these rows
    # is not evidence about yeast; it is evidence about whichever oracle labelled them.
    d = agp - rnn
    dz = (d - d.mean()) / (d.std() + 1e-9)
    hard = np.flatnonzero(np.abs(dz) > 2)
    res["disagreement"] = {
        "sd_of_difference": round(float(d.std()), 4),
        "n_beyond_2sd": int(len(hard)),
        "frac_beyond_2sd": round(float(len(hard)) / len(seqs), 4),
        "rnn_r_on_agreeing": round(pearson(rnn[np.abs(dz) <= 2], y[np.abs(dz) <= 2]), 4),
        "rnn_r_on_disagreeing": round(pearson(rnn[hard], y[hard]), 4),
        "ag_r_on_disagreeing": round(pearson(agp[hard], y[hard]), 4),
    }
    print("\n--- where the two oracles disagree (|z| > 2 on their difference) ---")
    for k, v in res["disagreement"].items():
        print("  %-24s %s" % (k, v))

    print("\n--- per sequence class ---")
    print("%-24s %8s %10s %10s %10s" % ("class", "n", "rnn_r", "ag_r", "ag_vs_rnn"))
    res["per_class"] = {}
    for name, idx in sorted(classes.items()):
        if len(idx) < 30:
            continue
        row = {
            "n": int(len(idx)),
            "rnn_r": round(pearson(rnn[idx], y[idx]), 4),
            "ag_r": round(pearson(agp[idx], y[idx]), 4),
            "ag_vs_rnn": round(pearson(agp[idx], rnn[idx]), 4),
        }
        res["per_class"][name] = row
        print(
            "%-24s %8d %10.4f %10.4f %10.4f"
            % (name, row["n"], row["rnn_r"], row["ag_r"], row["ag_vs_rnn"])
        )

    out = REPO / a.out
    os.makedirs(out.parent, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(str(out).replace(".json", "_pred.npz"), ag=agp, rnn=rnn, truth=y)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
