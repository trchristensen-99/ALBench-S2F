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
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.yeast import YeastDataset  # noqa: E402
from models.dream_rnn import create_dream_rnn  # noqa: E402
from models.loss_utils import YeastKLLoss  # noqa: E402
from models.training_base import create_optimizer_and_scheduler  # noqa: E402


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
    args = ap.parse_args()

    set_seed(args.seed + args.fold_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir) / f"fold_{args.fold_id}"
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

    mk = lambda idx, sh: DataLoader(  # noqa: E731
        Subset(ds, idx),
        batch_size=args.batch_size,
        shuffle=sh,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader, val_loader, test_loader = (
        mk(train_idx, True),
        mk(val_idx, False),
        mk(test_idx, False),
    )

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
    }
    done.write_text(json.dumps(metrics, indent=2, default=str))
    print(
        f"fold {args.fold_id} DONE: val r={v['pearson']:.4f} test r={t['pearson']:.4f} "
        f"(val-test = {v['pearson'] - t['pearson']:+.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
