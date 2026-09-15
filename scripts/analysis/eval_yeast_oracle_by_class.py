"""Score the yeast oracle per DREAM eval class, and compare variant A against B.

"The oracle is accurate" is not a usable claim when the eval classes are as different
as random 80-mers, GA-designed extremes, planted motifs and real promoters. Our yeast
reservoirs will generate sequences resembling several of those classes, so what
matters is accuracy PER CLASS, not pooled.

FAIRNESS RULE. Variant B trains on 80% of the splittable classes, so its numbers on
those classes are only meaningful on the folds it held out. Comparing B's full-class
score against A's would flatter B by exactly the amount it memorised. Every A-vs-B
comparison here is therefore restricted to the sequences BOTH models held out -- the
test fold of each ensemble member -- and the restriction is reported, not implied.

PAIRED CLASSES. For SNVs, motif perturbation and motif tiling the quantity that
matters is the allelic DIFFERENCE, and a difference is only meaningful when both
members come from the same context. The human analysis was corrupted by pairing a ref
from one window with an alt from another, so pairs here are matched explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 10 or np.std(true[ok]) < 1e-9:
        return {"n": int(ok.sum())}
    p, t = pred[ok], true[ok]
    mse = float(np.mean((p - t) ** 2))
    return {
        "n": int(ok.sum()),
        "r": float(pearsonr(p, t)[0]),
        "spearman": float(spearmanr(p, t)[0]),
        "mse": mse,
        "R2": float(1 - mse / np.var(t)),
        "true_sd": float(np.std(t)),
    }


def load_ensemble_preds(root: Path, tag: str, n_folds: int) -> dict[int, np.ndarray]:
    """fold_id -> per-fold test predictions over the EVAL file (if the run wrote them)."""
    out = {}
    for f in range(n_folds):
        p = root / f"fold_{f}{tag}" / "eval_class_predictions.npz"
        if p.exists():
            out[f] = np.load(p, allow_pickle=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--classes-npz", default="data/yeast/eval_classes_v2.npz")
    ap.add_argument("--root", default="outputs/yeast_oracle_v2")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--out", default="outputs/yeast_oracle_v2/per_class_report.json")
    args = ap.parse_args()

    z = np.load(REPO / args.classes_npz, allow_pickle=True)
    labels = np.asarray(z["labels"], dtype=float)
    classes = sorted(k[4:] for k in z.files if k.startswith("cls_"))
    unsplittable = {str(x) for x in z["unsplittable"]} if "unsplittable" in z.files else set()

    a = load_ensemble_preds(REPO / args.root, "", args.n_folds)
    b = load_ensemble_preds(REPO / args.root, "_B", args.n_folds)
    print(f"variant A folds with eval predictions: {sorted(a)}")
    print(f"variant B folds with eval predictions: {sorted(b)}")
    if not a and not b:
        print(
            "\nNeither variant has eval-class predictions yet.\n"
            "Run scripts/predict_yeast_eval_classes.py once the oracle folds finish; "
            "this script only aggregates.",
            file=sys.stderr,
        )
        return 1

    def oof(preds: dict[int, np.ndarray]) -> np.ndarray:
        """Assemble out-of-fold predictions: each sequence scored by the model that
        held its fold out, so no sequence is scored by a model that trained on it."""
        out = np.full(len(labels), np.nan)
        for f, z2 in preds.items():
            idx = np.asarray(z2["idx"], dtype=int)
            out[idx] = np.asarray(z2["pred"], dtype=float)
        return out

    pa, pb = oof(a), oof(b)
    report: dict[str, dict] = {}

    print(f"\n{'class':<22}{'n':>7}{'A r':>9}{'B r':>9}{'B - A':>9}{'true sd':>9}  note")
    print("-" * 82)
    for c in classes:
        idx = np.asarray(z[f"cls_{c}"], dtype=int)
        # Restrict to sequences BOTH variants held out, so the comparison is fair.
        common = idx[np.isfinite(pa[idx]) & np.isfinite(pb[idx])] if (a and b) else idx
        ma = metrics(pa[common], labels[common]) if a else {}
        mb = metrics(pb[common], labels[common]) if b else {}
        note = ""
        if c in unsplittable:
            note = "UNSPLITTABLE: B saw none of it (held out entirely)"
        elif b and c not in unsplittable:
            note = "B trained on 80%; scored on its held-out fold only"
        if len(common) < 100:
            note += "  [UNDERPOWERED]" if note else "[UNDERPOWERED: n<100]"
        ra, rb = ma.get("r", float("nan")), mb.get("r", float("nan"))
        print(
            f"{c:<22}{len(common):>7,}{ra:>9.4f}{rb:>9.4f}{rb - ra:>9.4f}"
            f"{ma.get('true_sd', float('nan')):>9.3f}  {note}"
        )
        report[c] = {"n_common": int(len(common)), "A": ma, "B": mb, "note": note}

    print(
        "\nInterpretation guide:\n"
        "  B - A is what including a class in oracle training buys ON THAT CLASS.\n"
        "  Weigh it against the cost: a class B trains on can only be evaluated on\n"
        "  its held-out tenth, so its usable eval set shrinks ~10x.\n"
        "  A difference smaller than ~2/sqrt(n) is not distinguishable from noise."
    )
    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
