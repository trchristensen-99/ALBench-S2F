"""Yeast DREAM-RNN oracle, 10-fold with ROTATING TEST FOLDS.

Matches the human AG oracle v2 protocol so the two oracles are comparable:

    test fold = fold_id
    val fold  = (fold_id + 1) % n_folds
    train     = the remaining eight folds

The rotation is the point. The archived trainer
(experiments/archive/train_oracle_dream_rnn.py) split train/val only, so its reported
number was the one early stopping selected on -- optimistic by construction. With a
rotating test fold every model has a split it never saw, and the ten test folds tile
the dataset exactly once, giving a complete out-of-fold evaluation.

The fold map is built once by scripts/build_yeast_folds.py and read here, so all ten
jobs agree on the split without coordinating. That also means identical sequences
were already grouped into one fold; this script does not re-derive the split.

Usage:
    python experiments/train_yeast_oracle_v2.py --fold-id 0 --out-dir outputs/yeast_oracle_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.yeast import YeastDataset  # noqa: E402
from models.dream_rnn import create_dream_rnn  # noqa: E402
from models.loss_utils import YeastKLLoss  # noqa: E402
from models.training_base import create_optimizer_and_scheduler  # noqa: E402


class EvalClassDataset(Dataset):
    """DREAM eval-set sequences, encoded exactly like the training data.

    The singleton channel is label-derived and is set to 0 here, matching what the
    base dataset does at inference. That is the honest choice, but note it means an
    added eval sequence is distinguishable from a bulk training sequence by that
    channel alone -- worth remembering when interpreting any gain from variant B.
    """

    def __init__(self, base, sequences: list[str], labels: np.ndarray):
        self.base = base
        # The base dataset stores PREPROCESSED sequences: _add_plasmid_context strips
        # any existing flanks, normalises the random region to a fixed length and
        # re-adds the full plasmid flanks to make exactly SEQUENCE_LENGTH. Passing raw
        # eval sequences to encode_sequence produced tensors of a different length,
        # which surfaced only as a collate error ("storage that is not resizable")
        # once a batch mixed the two sources.
        self.sequences = [str(x) for x in base._add_plasmid_context(np.array(sequences))]
        self.labels = np.asarray(labels, dtype=np.float32)
        enc = base.encode_sequence(self.sequences[0], {"is_singleton": 0.0})
        ref = base.encode_sequence(str(base.sequences[0]), {"is_singleton": 0.0})
        if enc.shape != ref.shape:
            raise ValueError(
                f"encoded eval sequences are {enc.shape} but training sequences are "
                f"{ref.shape}; they cannot share a batch"
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
        enc = self.base.encode_sequence(self.sequences[i], {"is_singleton": 0.0})
        return torch.from_numpy(enc).float(), torch.tensor(self.labels[i], dtype=torch.float32)


def load_eval_classes(path: Path, classes: list[str], test_fold: int, val_fold: int):
    """Return (train, val, test) index lists over the eval file for the named classes.

    Refuses a class the split builder marked UNSPLITTABLE: those have fewer connected
    ref/alt components than folds, so holding a fold out would split pairs and destroy
    the very quantity the paired sets measure.
    """
    z = np.load(path, allow_pickle=True)
    fold = z["fold"]
    unsplittable = {str(x) for x in z["unsplittable"]} if "unsplittable" in z.files else set()
    bad = sorted(set(classes) & unsplittable)
    if bad:
        raise SystemExit(
            f"cannot include {bad}: the split builder marked them unsplittable "
            f"(fewer ref/alt components than folds). Including them means ALL of the "
            f"class, which removes it as an evaluation set. Drop them or decide "
            f"deliberately to sacrifice that evaluation."
        )
    keep = np.zeros(len(fold), dtype=bool)
    for c in classes:
        key = f"cls_{c}"
        if key not in z.files:
            raise SystemExit(
                f"unknown eval class {c!r}; have {[k[4:] for k in z.files if k.startswith('cls_')]}"
            )
        keep[z[key]] = True
    idx = np.flatnonzero(keep)
    seqs = [str(s) for s in z["sequences"]]
    labs = np.asarray(z["labels"], dtype=np.float32)
    tr = idx[(fold[idx] != test_fold) & (fold[idx] != val_fold)]
    va = idx[fold[idx] == val_fold]
    te = idx[fold[idx] == test_fold]
    return (seqs, labs, tr, va, te)


def match_eval_labels_to_train_scale(
    eval_labels: np.ndarray, random_idx: np.ndarray, train_labels: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Put MAUDE-scale eval labels on the bulk training label scale.

    THE BUG THIS FIXES. The bulk training file (data/yeast/train.txt) carries binned
    expression on [0, 17] with mean ~11.15. The DREAM eval file carries MAUDE
    expression on [-1.40, 1.66] with mean ~0.16. Variant B mixed both in one training
    set under one loss, so it was taught to predict ~0.16 for anything resembling an
    eval sequence and ~11 for everything else. That model scored r=0.17 on SNVs it had
    TRAINED on, against 0.87 for variant A which never saw them -- memorisation cannot
    make a model worse, which is what flagged the artefact.

    WHY AN AFFINE MAP FITTED ON THE `random` CLASS. Both files measure the same
    quantity on different scales, so the correction is affine, not rank-based:
    quantile-matching the whole eval set onto the train marginal would compress the
    deliberately-extreme designed classes (high_expression, low_expression) toward the
    middle and destroy the very signal they exist to provide. The `random` class is the
    one eval subset drawn from the same distribution as the bulk data, so it is the
    honest place to fit the scale. The designed extremes then land outside the fitted
    range, which is correct -- they genuinely are extreme.
    """
    ref = np.asarray(eval_labels, dtype=np.float64)[random_idx]
    if len(ref) < 500:
        raise ValueError(
            f"only {len(ref)} `random`-class eval sequences; too few to fit a label "
            f"scale reliably. Widen the reference set or pass it explicitly."
        )
    tr = np.asarray(train_labels, dtype=np.float64)
    scale = tr.std() / ref.std()
    shift = tr.mean() - ref.mean() * scale
    out = (np.asarray(eval_labels, dtype=np.float64) * scale + shift).astype(np.float32)
    assert scale > 0, "label scale must be positive; check the reference class"
    stats = {
        "scale": float(scale),
        "shift": float(shift),
        "n_reference": int(len(ref)),
        "eval_mean_before": float(np.mean(eval_labels)),
        "eval_mean_after": float(out.mean()),
        "train_mean": float(tr.mean()),
        "train_sd": float(tr.std()),
    }
    return out, stats


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict[str, float]:
    """Pearson/Spearman/loss on a loader, using the scalar expression head."""
    from scipy.stats import pearsonr, spearmanr

    model.eval()
    preds, trues, losses = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model.get_logits(x)
        losses.append(criterion(logits, y).item())
        # The scalar activity is the bin-weighted mean; comparing scalars keeps this
        # metric comparable with the human oracle's Pearson r.
        p = torch.softmax(logits.float(), dim=-1)
        centers = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
        preds.append((p * centers).sum(-1).cpu().numpy())
        ty = y.float()
        trues.append((ty * centers).sum(-1).cpu().numpy() if ty.ndim > 1 else ty.cpu().numpy())
    model.train()
    pr = np.concatenate(preds)
    tr = np.concatenate(trues)
    ok = np.isfinite(pr) & np.isfinite(tr)
    return {
        "n": int(ok.sum()),
        "loss": float(np.mean(losses)),
        "pearson": float(pearsonr(pr[ok], tr[ok])[0]),
        "spearman": float(spearmanr(pr[ok], tr[ok])[0]),
        "_pred": pr,
        "_true": tr,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold-id", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--folds-npy", default="data/yeast/oracle_folds_v2.npy")
    ap.add_argument("--out-dir", default="outputs/yeast_oracle_v2")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lr-lstm", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=320)
    ap.add_argument("--cnn-filters", type=int, default=256)
    ap.add_argument("--dropout-cnn", type=float, default=0.2)
    ap.add_argument("--dropout-lstm", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--context-mode", default="dream150")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=None, help="cap dataset size (smoke tests)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--include-eval-classes",
        default="",
        help="comma-separated DREAM eval classes to ADD to training at 80/10/10 "
        "(variant B). Empty = variant A, random data only. Unsplittable classes "
        "(motif_perturbation, motif_tiling) are refused.",
    )
    ap.add_argument("--eval-classes-npz", default="data/yeast/eval_classes_v2.npz")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    args = ap.parse_args()

    set_seed(args.seed + args.fold_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir) / f"fold_{args.fold_id}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    done = out / "test_metrics.json"
    if args.resume and done.exists():
        print(f"SKIP fold {args.fold_id}: {done} already exists")
        return 0

    ds = YeastDataset(
        data_path=str(REPO / "data" / "yeast"),
        split="train",
        subset_size=args.subset,
        context_mode=args.context_mode,
    )
    fold = np.load(REPO / args.folds_npy)
    if args.subset:
        fold = fold[: len(ds)]
    if len(fold) != len(ds):
        raise SystemExit(
            f"fold map has {len(fold):,} entries but the dataset has {len(ds):,}. "
            f"Rebuild with scripts/build_yeast_folds.py."
        )

    test_fold = args.fold_id
    val_fold = (args.fold_id + 1) % args.n_folds
    test_idx = np.where(fold == test_fold)[0]
    val_idx = np.where(fold == val_fold)[0]
    train_idx = np.where((fold != test_fold) & (fold != val_fold))[0]
    print(
        f"fold {args.fold_id}: train={len(train_idx):,} "
        f"val={len(val_idx):,} (fold {val_fold}) test={len(test_idx):,} (fold {test_fold})",
        flush=True,
    )
    assert not (set(train_idx) & set(test_idx)), "train/test overlap"
    assert not (set(val_idx) & set(test_idx)), "val/test overlap"

    extra = {"train": 0, "val": 0, "test": 0}
    add_tr = add_va = add_te = None
    if args.include_eval_classes.strip():
        names = [c.strip() for c in args.include_eval_classes.split(",") if c.strip()]
        seqs_e, labs_e, tr_e, va_e, te_e = load_eval_classes(
            REPO / args.eval_classes_npz, names, test_fold, val_fold
        )
        # The eval file is MAUDE-scale, the bulk file is binned [0, 17]. Mixing them
        # raw teaches the model two different answers to the same question; see
        # match_eval_labels_to_train_scale for what that cost the first variant B.
        _z = np.load(REPO / args.eval_classes_npz, allow_pickle=True)
        if "cls_random" not in _z.files:
            raise SystemExit(
                "eval_classes npz has no `random` class, which is the reference used "
                "to fit the label scale; rebuild it with scripts/build_yeast_eval_classes.py"
            )
        labs_e, _scale_stats = match_eval_labels_to_train_scale(
            labs_e, np.asarray(_z["cls_random"], dtype=int), ds.labels
        )
        print(
            f"  eval labels rescaled to the training scale: "
            f"x{_scale_stats['scale']:.3f} {_scale_stats['shift']:+.3f} "
            f"(mean {_scale_stats['eval_mean_before']:.3f} -> "
            f"{_scale_stats['eval_mean_after']:.3f}, train mean "
            f"{_scale_stats['train_mean']:.3f}, fitted on "
            f"{_scale_stats['n_reference']:,} random-class sequences)",
            flush=True,
        )
        add_tr = EvalClassDataset(ds, [seqs_e[i] for i in tr_e], labs_e[tr_e])
        add_va = EvalClassDataset(ds, [seqs_e[i] for i in va_e], labs_e[va_e])
        add_te = EvalClassDataset(ds, [seqs_e[i] for i in te_e], labs_e[te_e])
        extra = {"train": len(add_tr), "val": len(add_va), "test": len(add_te)}
        print(
            f"  variant B: adding eval classes {names} -> "
            f"+{extra['train']:,} train, +{extra['val']:,} val, +{extra['test']:,} test",
            flush=True,
        )

    def loader(idx, add, shuffle):
        base = Subset(ds, idx)
        data = ConcatDataset([base, add]) if add is not None and len(add) else base
        return DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    train_loader = loader(train_idx, add_tr, True)
    val_loader = loader(val_idx, add_va, False)
    test_loader = loader(test_idx, add_te, False)

    model = create_dream_rnn(
        input_channels=6,
        sequence_length=ds.get_sequence_length(),
        task_mode="yeast",
        hidden_dim=args.hidden_dim,
        cnn_filters=args.cnn_filters,
        dropout_cnn=args.dropout_cnn,
        dropout_lstm=args.dropout_lstm,
    ).to(device)
    criterion = YeastKLLoss(reduction="batchmean")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        train_loader=train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        lr_lstm=args.lr_lstm,
    )

    best_val, best_state, best_epoch = -np.inf, None, -1
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.get_logits(x), y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            tot += loss.item()  # .item() not float(): the latter keeps the graph alive
            nb += 1
        v = evaluate(model, val_loader, device, criterion)
        print(
            f"  epoch {epoch + 1}/{args.epochs} train_loss={tot / max(nb, 1):.4f} "
            f"val_r={v['pearson']:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if v["pearson"] > best_val:
            # Keep the best parameters in memory: the test evaluation below must use
            # the checkpoint VAL selected, not whatever the last epoch happened to be.
            best_val, best_epoch = v["pearson"], epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, out / "best_model.pt")

    t = evaluate(model, test_loader, device, criterion)
    np.savez_compressed(
        out / "test_predictions.npz",
        idx=test_idx,
        y_true=t.pop("_true"),
        y_pred=t.pop("_pred"),
    )
    v = evaluate(model, val_loader, device, criterion)
    v.pop("_pred"), v.pop("_true")
    metrics = {
        "fold_id": args.fold_id,
        "val_fold": val_fold,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": best_epoch + 1,
        "best_val_pearson": float(best_val),
        "val_metrics": v,
        "test_metrics": t,
        "args": vars(args),
        "eval_classes_added": extra,
        "variant": "B" if args.include_eval_classes.strip() else "A",
    }
    done.write_text(json.dumps(metrics, indent=2, default=str))
    print(
        f"fold {args.fold_id} DONE: val r={v['pearson']:.4f} test r={t['pearson']:.4f} "
        f"(val-test = {v['pearson'] - t['pearson']:+.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
