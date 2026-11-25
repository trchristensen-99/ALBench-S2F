"""Train an AlphaGenome S2 yeast oracle: joint encoder + head fine-tuning.

WHY. The existing AlphaGenome yeast oracle is S1 -- a head trained on a FROZEN
human/mouse-pretrained encoder -- and it scores r = 0.6076 on the DREAM test set against
the DREAM-RNN's 0.9647, anti-correlating (-0.4652) where the two disagree. Nothing makes a
frozen human encoder linearly predict yeast promoter strength, so S1 is the obvious
explanation and S2 is the obvious fix. It has never been tried for the oracle, although
`S2_CONFIG["yeast"]` has been configured for it all along: all six downres blocks
unfrozen, 50 epochs, 5 warmup epochs head-only before the encoder moves.

WHY IT MATTERS BEYOND ORACLE QUALITY. Our yeast students are DREAM-RNNs. An RNN oracle and
an RNN student share an inductive bias, so a student can score well by reproducing the
oracle's idiosyncrasies rather than yeast biology -- the copying hazard. An AlphaGenome
oracle is a conv-transformer: architecturally distant from every yeast student we train,
so copying stops being the cheap path. Replacing the oracle therefore fixes the measurement
problem and the copying problem at once, and matches the human setup, where the oracle is
AG-FT and the students are not.

THE PRECONDITION IS BINDING. This is only worth adopting if S2 reaches roughly RNN parity.
A worse oracle that is architecturally convenient is still a worse oracle, and every
reservoir comparison in yeast would inherit its errors. The decision rule, fixed before the
number is seen (Pearson on the held-out DREAM test set, against the RNN's 0.9647):

    < 0.90      unusable. Keep the RNN as sole labeller and say so.
    0.90-0.95   RNN labels; AG-S2 becomes the independent judge for the copying diagnostic.
    >= ~0.96    parity. Adopt AG-S2 as the labeller, for consistency with the human task.

STAGED. Run ONE fold first (--fold-id 0) and read it against that rule before spending ten
folds' worth of GPU on an ensemble.

    python experiments/train_ag_yeast_s2_oracle.py --fold-id 0 --out-dir outputs/yeast_ag_s2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold-id", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--folds-npy", default="data/yeast/oracle_folds_v2.npy")
    ap.add_argument("--data-path", default="data/yeast")
    ap.add_argument("--out-dir", default="outputs/yeast_ag_s2")
    ap.add_argument("--encoder-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--s1-checkpoint",
        default=None,
        help="S1 fold to warm-start the head from; default is the matching fold of "
        "outputs/oracle_alphagenome_yeast_ensemble. Warm starting is the agreed setup "
        "(S2 from the S1 baseline, not from scratch as the AG-FT repo does).",
    )
    ap.add_argument("--eval-classes", default="data/yeast/eval_classes_v2.npz")
    ap.add_argument("--max-train", type=int, default=0, help="debug: cap training rows")
    a = ap.parse_args()

    out = Path(a.out_dir) / f"fold_{a.fold_id}_s2"
    out.mkdir(parents=True, exist_ok=True)

    # --- data -----------------------------------------------------------------
    # context_mode="alphagenome384": the AG encoder reads a wider window than the
    # DREAM-RNN's 150 bp, and the dataset knows how to build it. Passing dream150 here
    # would train the encoder on sequences shaped for a different model, which is the kind
    # of mismatch that shows up only as mediocre accuracy.
    from data.yeast import YeastDataset

    ds = YeastDataset(a.data_path, split="train", context_mode="alphagenome384")
    seqs = np.asarray([str(s) for s in ds.sequences])
    labels = np.asarray(ds.labels, dtype=np.float32)

    folds = np.load(a.folds_npy)
    if len(folds) != len(seqs):
        raise SystemExit(
            f"fold map has {len(folds):,} entries but the training set has {len(seqs):,}. "
            "The folds were built by scripts/build_yeast_folds.py against a specific "
            "dataset build; they are not interchangeable."
        )
    test_fold = a.fold_id
    val_fold = (a.fold_id + 1) % a.n_folds
    tr = ~np.isin(folds, [test_fold, val_fold])
    va = folds == val_fold
    if a.max_train:
        idx = np.flatnonzero(tr)[: a.max_train]
        tr = np.zeros_like(tr)
        tr[idx] = True
    print(
        f"fold {a.fold_id}: train {tr.sum():,} | val {va.sum():,} "
        f"(held-out test fold {test_fold}, val fold {val_fold})",
        flush=True,
    )

    s1 = a.s1_checkpoint
    if s1 is None:
        cand = (
            REPO
            / "outputs/oracle_alphagenome_yeast_ensemble"
            / f"oracle_{a.fold_id}"
            / "best_model"
        )
        s1 = str(cand) if cand.exists() else None
    print(f"S1 warm start: {s1 or 'NONE (cold start)'}", flush=True)

    # --- train ----------------------------------------------------------------
    from exp1_1_scaling import _train_ag_s2_student

    t0 = time.time()
    model = _train_ag_s2_student(
        "yeast",
        list(seqs[tr]),
        labels[tr],
        a.encoder_lr,
        a.batch_size,
        a.seed,
        s1_checkpoint=s1,
    )
    train_s = time.time() - t0
    print(f"trained in {train_s / 60:.1f} min", flush=True)

    # --- evaluate on the REAL DREAM test set ----------------------------------
    # The same 71,103 measured sequences ye45 used, so this number is directly comparable
    # to the RNN's 0.9647 and S1's 0.6076 rather than to a fold-internal split.
    res: dict = {
        "fold_id": a.fold_id,
        "train_n": int(tr.sum()),
        "train_minutes": round(train_s / 60, 1),
        "s1_checkpoint": s1,
        "encoder_lr": a.encoder_lr,
    }
    ev = REPO / a.eval_classes
    if ev.exists():
        z = np.load(ev, allow_pickle=True)
        es = [str(s) for s in z["sequences"]]
        ey = np.asarray(z["labels"], dtype=np.float64).ravel()
        preds = []
        B = 2048
        for s in range(0, len(es), B):
            preds.append(np.asarray(model.predict(es[s : s + B]), np.float64).ravel())
        p = np.concatenate(preds)[: len(es)]
        res["dream_test"] = {
            "n": int(len(es)),
            "pearson": round(pearson(p, ey), 4),
            "r2": round(pearson(p, ey) ** 2, 4),
        }
        r = res["dream_test"]["pearson"]
        print(f"\nDREAM test set: n={len(es):,} pearson={r:.4f}", flush=True)
        print("  reference: DREAM-RNN 0.9647, AlphaGenome S1 0.6076", flush=True)
        verdict = (
            "UNUSABLE - keep the RNN as sole labeller"
            if r < 0.90
            else "JUDGE ONLY - RNN labels, AG-S2 is the independent check"
            if r < 0.955
            else "PARITY - adopt AG-S2 as the labeller"
        )
        print(f"  verdict under the pre-agreed rule: {verdict}", flush=True)
        res["verdict"] = verdict
        if "classes" in z.files:
            cls = np.asarray(z["classes"])
            res["dream_test"]["by_class"] = {
                str(c): round(pearson(p[cls == c], ey[cls == c]), 4)
                for c in sorted(set(cls.tolist()))
            }
    else:
        print(f"WARNING: {ev} not found; no DREAM-test number for this fold", flush=True)

    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
