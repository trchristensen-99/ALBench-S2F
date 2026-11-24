"""Score the yeast oracle on the DREAM 2022 test set, the number the challenge reported.

Every yeast oracle metric this study has quoted -- 0.6451 truncated, 0.7635 at 12 epochs --
is Pearson r on a random tenth of the 6.07M random-promoter TRAINING library. That is not
the number the DREAM challenge leaderboard reported, and the two are not comparable: the
challenge scored 71,103 separately-measured sequences whose expression estimates come from
far deeper sampling, so their measurement ceiling is much higher than a single held-out
library row's. Comparing our 0.76 against a published 0.94 across that gap says nothing.

`data/yeast/eval_classes_v2.npz` IS that test set (71,103 sequences, the nine DREAM
sequence classes). Every production fold trained with --include-eval-classes empty, so not
one of these sequences was in any fold's training data -- which means the full 10-model
ensemble may score all 71,103, exactly as a challenge submission would.

Reported per fold and for the ensemble: Pearson r, Pearson r^2 (the challenge's headline
metric) and Spearman rho, overall and within each sequence class. The train-overlap check
is not a formality -- it is the one thing that would invalidate the comparison.

    python scratchpad/nov23/ye44_dream_eval.py --out outputs/yeast_prod/dream_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 30 or pred[m].std() == 0:
        return {"n": int(m.sum()), "pearson": None, "pearson_r2": None, "spearman": None}
    r = float(np.corrcoef(pred[m], y[m])[0, 1])
    return {
        "n": int(m.sum()),
        "pearson": round(r, 4),
        "pearson_r2": round(r * r, 4),
        "spearman": round(float(spearmanr(pred[m], y[m]).statistic), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/yeast_prod")
    ap.add_argument("--tag", default="_e12_prod", help="suffix after fold_{i} in the dir name")
    ap.add_argument("--classes-npz", default="data/yeast/eval_classes_v2.npz")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--check-overlap", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from data.yeast import YeastDataset
    from models.dream_rnn import create_dream_rnn

    z = np.load(REPO / a.classes_npz, allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    y = np.asarray(z["labels"], np.float64)
    classes = {k[4:]: np.asarray(z[k]) for k in z.files if k.startswith("cls_")}
    print(f"{len(seqs):,} DREAM eval sequences, {len(classes)} classes", flush=True)

    # The plasmid context and encoding the oracle was trained with, taken from the dataset
    # object rather than rebuilt, so the eval cannot drift from training.
    ds = YeastDataset(
        data_path=str(REPO / "data" / "yeast"),
        split="train",
        subset_size=1000,
        context_mode="dream150",
    )
    processed = [str(x) for x in ds._add_plasmid_context(np.array(seqs))]

    if a.check_overlap:
        # If any eval sequence is also a training row, the comparison is void. Cheap to
        # check against the raw library and the only check that can falsify the headline.
        train_path = REPO / "data" / "yeast" / "train.txt"
        have = set()
        with open(train_path) as fh:
            for line in fh:
                have.add(line.split("\t", 1)[0].strip())
        n_ov = sum(1 for s in seqs if s in have)
        print(f"train-overlap: {n_ov:,} of {len(seqs):,} eval sequences appear in train.txt")
    else:
        n_ov = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_fold, preds = {}, []
    for f in range(10):
        ckpt_path = REPO / a.root / f"fold_{f}fold_{f}{a.tag}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  fold {f}: no checkpoint at {ckpt_path}", flush=True)
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = create_dream_rnn(
            input_channels=6,
            sequence_length=ds.get_sequence_length(),
            task_mode="yeast",
            hidden_dim=ckpt["args"]["hidden_dim"],
            cnn_filters=ckpt["args"]["cnn_filters"],
            dropout_cnn=ckpt["args"]["dropout_cnn"],
            dropout_lstm=ckpt["args"]["dropout_lstm"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(processed), a.batch_size):
                chunk = processed[i : i + a.batch_size]
                x = np.stack([ds.encode_sequence(s, {"is_singleton": 0.0}) for s in chunk])
                out.append(model(torch.from_numpy(x).float().to(device)).cpu().numpy().ravel())
        p = np.concatenate(out).astype(np.float64)
        preds.append(p)
        per_fold[f] = metrics(p, y)
        print(
            f"  fold {f}: r={per_fold[f]['pearson']}  r2={per_fold[f]['pearson_r2']}  "
            f"rho={per_fold[f]['spearman']}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    if not preds:
        print("no folds scored", file=sys.stderr)
        return 1
    ens = np.mean(preds, axis=0)
    res = {
        "n_folds_scored": len(preds),
        "train_overlap": n_ov,
        "per_fold": per_fold,
        "ensemble_overall": metrics(ens, y),
        "ensemble_by_class": {k: metrics(ens[i], y[i]) for k, i in classes.items()},
        "single_fold_mean_pearson": round(
            float(np.mean([v["pearson"] for v in per_fold.values()])), 4
        ),
    }
    np.savez_compressed(
        REPO / a.out.replace(".json", "_pred.npz"),
        ensemble=ens.astype(np.float32),
        per_fold=np.stack(preds).astype(np.float32),
        labels=y.astype(np.float32),
    )
    with open(REPO / a.out, "w") as fh:
        json.dump(res, fh, indent=1)

    e = res["ensemble_overall"]
    print(f"\n=== ENSEMBLE of {len(preds)} folds on the DREAM test set ({e['n']:,} seqs)")
    print(f"    Pearson r   {e['pearson']}")
    print(f"    Pearson r^2 {e['pearson_r2']}   <- the challenge's headline metric")
    print(f"    Spearman    {e['spearman']}")
    print(f"    single-fold mean Pearson {res['single_fold_mean_pearson']}")
    print("\n--- by sequence class ---")
    print("%-22s %8s %8s %8s %8s" % ("class", "n", "r", "r2", "rho"))
    for k, v in sorted(res["ensemble_by_class"].items(), key=lambda kv: -kv[1]["n"]):
        print("%-22s %8d %8s %8s %8s" % (k, v["n"], v["pearson"], v["pearson_r2"], v["spearman"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
