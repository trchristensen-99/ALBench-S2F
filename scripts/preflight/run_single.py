#!/usr/bin/env python
"""Pre-flight single-run trainer for ALBench-S2F.

Trains a specified architecture (LegNet / DREAM-RNN / DREAM-ATTN) on a
random subset of K562 training sequences (oracle = AlphaGenome
pseudolabels) at the given training budget D, with HPs passed via CLI.

This is the reusable launch utility referenced in Task 1: takes
(arch, d_train, hp_overrides, seed) and runs a single training job.

Outputs (under output_dir):
    best.pt              — best-val-loss checkpoint
    history.json         — per-epoch train/val/test loss
    result.json          — summary (test MSE @ best val, flags, GPU-hrs, …)
    config.json          — exact resolved config used

W&B logging (offline by default — sync later with wandb sync) tags follow
the pre-flight schema: phase=preflight, arch=<>, sweep=<>, seed=<>,
d_train=<>.

Usage:
    uv run --no-sync python scripts/preflight/run_single.py \\
        --arch legnet --d_train 1000 --seed 42 --epochs 5 \\
        --output_dir results/preflight/_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


# ── Architecture priors (per-arch HP defaults — sweep grid centers) ──────
ARCH_PRIORS: dict[str, dict[str, Any]] = {
    "legnet": {
        "lr": 5e-3,
        "batch_size": 1024,
        "weight_decay": 0.1,
        "in_channels": 4,
        "block_sizes": [256, 256, 128, 128, 64, 64, 32, 32],
        "ks": 5,
        "dropout": 0.0,
    },
    "dream_rnn": {
        "lr": 1e-3,
        "batch_size": 1024,
        "weight_decay": 0.01,
        "in_channels": 5,
        "hidden_dim": 320,
        "cnn_filters": 160,
        "dropout_cnn": 0.2,
        "dropout_lstm": 0.3,
    },
    "dream_attn": {
        "lr": 3e-4,
        "batch_size": 512,
        "weight_decay": 0.01,
        "in_channels": 5,
        "embedding_dim": 256,
        "num_blocks": 4,
        "kernel_size": 15,
        "num_heads": 4,
        "first_block_dropout": 0.1,
        "core_dropout": 0.1,
        "head_dropout": 0.1,
    },
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── One-hot encoding (4 nt + optional RC orientation channel) ────────────
_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(sequences: list[str], seq_len: int, in_channels: int) -> np.ndarray:
    n = len(sequences)
    out = np.zeros((n, in_channels, seq_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        if len(seq) < seq_len:
            pad = seq_len - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > seq_len:
            start = (len(seq) - seq_len) // 2
            seq = seq[start : start + seq_len]
        for j, nuc in enumerate(seq[:seq_len]):
            idx = _NUC_TO_IDX.get(nuc)
            if idx is not None:
                out[i, idx, j] = 1.0
    # Channel 4 (when in_channels >= 5): RC orientation flag, fwd=0
    return out


def _rc_flip(x: torch.Tensor) -> torch.Tensor:
    """Reverse-complement: flip ACGT and reverse along sequence."""
    out = x.flip(dims=[2]).clone()
    out[:, [0, 1, 2, 3]] = out[:, [3, 2, 1, 0]]
    if out.shape[1] >= 5:
        out[:, 4] = 1.0 - out[:, 4]  # toggle RC indicator if present
    return out


# ── Build models ─────────────────────────────────────────────────────────
def build_model(arch: str, hp: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if arch == "legnet":
        from models.legnet import LegNet

        return LegNet(
            in_channels=hp["in_channels"],
            block_sizes=hp.get("block_sizes"),
            ks=hp.get("ks", 5),
            dropout=hp.get("dropout", 0.0),
            task_mode="k562",
        ).to(device)
    if arch == "dream_rnn":
        from models.dream_rnn import DREAMRNN

        return DREAMRNN(
            input_channels=hp["in_channels"],
            sequence_length=hp.get("sequence_length", 200),
            task_mode="k562",
            hidden_dim=hp["hidden_dim"],
            cnn_filters=hp["cnn_filters"],
            dropout_cnn=hp["dropout_cnn"],
            dropout_lstm=hp["dropout_lstm"],
        ).to(device)
    if arch == "dream_attn":
        from models.dream_attn import DREAMATTN

        return DREAMATTN(
            in_channels=hp["in_channels"],
            sequence_length=hp.get("sequence_length", 200),
            embedding_dim=hp["embedding_dim"],
            num_blocks=hp["num_blocks"],
            kernel_size=hp["kernel_size"],
            num_heads=hp["num_heads"],
            first_block_dropout=hp["first_block_dropout"],
            core_dropout=hp["core_dropout"],
            head_dropout=hp["head_dropout"],
            task_mode="k562",
        ).to(device)
    raise ValueError(f"Unknown arch: {arch}")


# ── Data loading (K562 + AG oracle pseudolabels) ─────────────────────────
def load_data(
    d_train: int,
    seed: int,
    in_channels: int,
    seq_len: int = 200,
    label_source: str = "ag_oracle",
):
    """Sample d_train from K562 train pool, with chosen label source.

    Splits are **chromosome-based** (not hashFrag) per pre-flight spec:
    train = autosomes minus held-out, val = chr19/21/X (per existing
    project convention), test = held-out chromosomes. This matches what
    existing AG-oracle pseudolabel npz files were generated against
    (~316k train rows on chromosome split vs ~296k on hashFrag), so the
    sequence-keyed lookup recovers nearly full alignment.

    label_source:
      - "ag_oracle"  → AG-oracle pseudolabels from the cached npz files,
        looked up by *sequence*. Sequences without a cached pseudolabel
        are dropped.
      - "real"       → K562_log2FC real labels.

    Train uses out-of-fold oracle predictions (``oof_oracle``); val/test
    use the full-ensemble mean (``oracle_mean``).
    """
    from data.k562 import K562Dataset

    _kw = dict(use_hashfrag=False, use_chromosome_fallback=True)
    ds_train = K562Dataset(data_path=str(REPO / "data" / "k562"), split="train", **_kw)
    ds_val = K562Dataset(data_path=str(REPO / "data" / "k562"), split="val", **_kw)
    ds_test = K562Dataset(data_path=str(REPO / "data" / "k562"), split="test", **_kw)

    if label_source == "real":
        train_pool_seqs = [str(s) for s in ds_train.sequences]
        train_pool_lbl = ds_train.labels.astype(np.float32)
        val_seqs = [str(s) for s in ds_val.sequences]
        val_labels = ds_val.labels.astype(np.float32)
        test_seqs = [str(s) for s in ds_test.sequences]
        test_labels = ds_test.labels.astype(np.float32)
    elif label_source == "ag_oracle":
        # Build sequence → label lookup from the cached AG-oracle npz files.
        # We use chromosome-split datasets (the same config the cache was
        # generated against — 316k train rows vs npz 319,742, ~99% overlap),
        # and look up labels by SEQUENCE so any minor misalignment doesn't
        # poison row indexing.
        cache = REPO / "outputs" / "oracle_pseudolabels_k562_ag"
        seq2label: dict[str, float] = {}
        npz = np.load(cache / "train_oracle_labels.npz", allow_pickle=True)
        n = min(len(ds_train.sequences), len(npz["oof_oracle"]))
        for i in range(n):
            seq2label[str(ds_train.sequences[i]).upper()] = float(npz["oof_oracle"][i])
        for split_name, split_ds, npz_name, key in [
            ("val", ds_val, "val_oracle_labels.npz", "oracle_mean"),
            ("test", ds_test, "test_in_dist_oracle_labels.npz", "oracle_mean"),
        ]:
            split_npz = np.load(cache / npz_name, allow_pickle=True)
            n = min(len(split_ds.sequences), len(split_npz[key]))
            for i in range(n):
                seq2label[str(split_ds.sequences[i]).upper()] = float(split_npz[key][i])

        # Filter each split to sequences with cached pseudolabels
        def _filter(seqs, _real_lbl):
            keep_seqs, keep_lbl = [], []
            for s in seqs:
                u = str(s).upper()
                if u in seq2label:
                    keep_seqs.append(str(s))
                    keep_lbl.append(seq2label[u])
            return keep_seqs, np.array(keep_lbl, dtype=np.float32)

        train_pool_seqs, train_pool_lbl = _filter(ds_train.sequences, ds_train.labels)
        val_seqs, val_labels = _filter(ds_val.sequences, ds_val.labels)
        test_seqs, test_labels = _filter(ds_test.sequences, ds_test.labels)
        print(
            f"  AG-oracle cache match: train {len(train_pool_seqs):,}/{len(ds_train.sequences):,}  "
            f"val {len(val_seqs):,}/{len(ds_val.sequences):,}  "
            f"test {len(test_seqs):,}/{len(ds_test.sequences):,}"
        )
    else:
        raise ValueError(f"unknown label_source {label_source!r}")

    n_pool = len(train_pool_seqs)
    if d_train > n_pool:
        raise ValueError(f"d_train={d_train} > train pool size {n_pool}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pool, size=d_train, replace=False)
    train_seqs = [train_pool_seqs[i] for i in idx]
    train_labels = train_pool_lbl[idx]

    Xtr = one_hot(train_seqs, seq_len, in_channels)
    Xva = one_hot(val_seqs, seq_len, in_channels)
    Xte = one_hot(test_seqs, seq_len, in_channels)
    return (Xtr, train_labels), (Xva, val_labels), (Xte, test_labels)


# ── Training loop with best-val checkpointing ────────────────────────────
def _eval_loss(model, loader, device, augment_rc: bool) -> tuple[float, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yhat = model(xb).reshape(-1)
            if augment_rc:
                yhat_rc = model(_rc_flip(xb)).reshape(-1)
                yhat = 0.5 * (yhat + yhat_rc)
            preds.append(yhat.detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return float(np.mean((p - t) ** 2)), p


def train(args: argparse.Namespace, hp: dict[str, Any]) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Data
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_data(
        args.d_train,
        args.seed,
        in_channels=hp["in_channels"],
        seq_len=hp.get("sequence_length", 200),
        label_source=args.label_source,
    )
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()
    Xva_t = torch.from_numpy(Xva).float()
    yva_t = torch.from_numpy(yva).float()
    Xte_t = torch.from_numpy(Xte).float()
    yte_t = torch.from_numpy(yte).float()

    from torch.utils.data import DataLoader, TensorDataset

    # drop_last=False so very small D (e.g. D=500 with bs=1024) still
    # trains on the available partial batch — the D_min sweep needs to
    # see *some* gradient update at every D, otherwise the val-R² check
    # is trivially failed.
    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(Xva_t, yva_t),
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(Xte_t, yte_t),
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model & optimizer
    model = build_model(args.arch, hp, device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )
    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hp["lr"],
        total_steps=total_steps,
        pct_start=hp.get("warmup_pct", 0.1),
        anneal_strategy="cos",
    )

    augment_rc = args.augmentations in ("rev_complement", "rc_shift", "rc_shift_evoaug")
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []}
    best_val = math.inf
    best_epoch = -1
    best_test_mse = math.nan
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for epoch in range(args.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if augment_rc and torch.rand(1).item() < 0.5:
                xb = _rc_flip(xb)
            with torch.amp.autocast("cuda", enabled=args.use_amp):
                yhat = model(xb).reshape(-1)
                loss = criterion(yhat, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)

        val_loss, _ = _eval_loss(model, val_loader, device, augment_rc=augment_rc)
        test_loss, _ = _eval_loss(model, test_loader, device, augment_rc=augment_rc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_test_mse = test_loss
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "hp": hp},
                out_dir / "best.pt",
            )

        print(
            f"  ep {epoch + 1}/{args.epochs}  train={train_loss:.4f}  "
            f"val={val_loss:.4f}  test={test_loss:.4f}  lr={history['lr'][-1]:.5f}"
        )

    elapsed = time.time() - t_start

    # Post-hoc flag: did min val loss occur in the final 10% of epochs?
    final_window_start = int(args.epochs * (1 - args.report_min_val_in_final_pct))
    in_final_window = best_epoch >= final_window_start

    # GPU-hrs
    gpu_hrs = elapsed / 3600.0

    summary = {
        "arch": args.arch,
        "d_train": args.d_train,
        "seed": args.seed,
        "epochs": args.epochs,
        "n_params": n_params,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "test_mse_at_best_val": best_test_mse,
        "min_val_in_final_pct_window": in_final_window,
        "report_min_val_in_final_pct": args.report_min_val_in_final_pct,
        "gpu_hrs": gpu_hrs,
        "augmentations": args.augmentations,
        "hp": hp,
        "wandb_run_id": None,  # filled by W&B if used
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse k=v overrides where v is auto-parsed (int, float, bool, str)."""
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"override must be k=v, got {it!r}")
        k, v = it.split("=", 1)
        # type coerce
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="ALBench-S2F pre-flight single-run trainer")
    ap.add_argument("--arch", required=True, choices=list(ARCH_PRIORS.keys()))
    ap.add_argument("--d_train", required=True, type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument(
        "--augmentations",
        default="rev_complement",
        choices=["none", "rev_complement", "rc_shift", "rc_shift_evoaug"],
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_amp", action="store_true", default=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--label_source",
        default="ag_oracle",
        choices=["ag_oracle", "real"],
        help="ag_oracle = AG-oracle pseudolabels (cached npz, sequence-keyed); "
        "real = K562_log2FC real labels.",
    )
    ap.add_argument("--report_min_val_in_final_pct", type=float, default=0.1)
    ap.add_argument("--sweep_name", default=None, help="W&B tag value for sweep=<>")
    ap.add_argument(
        "--hp",
        nargs="*",
        default=[],
        help="HP overrides as k=v (e.g. lr=3e-3 batch_size=512 dropout=0.1)",
    )
    args = ap.parse_args()

    # Build HPs from priors then apply overrides
    hp = dict(ARCH_PRIORS[args.arch])
    overrides = parse_overrides(args.hp)
    hp.update(overrides)

    # W&B (optional, offline)
    try:
        import wandb

        tags = [
            "phase=preflight",
            f"arch={args.arch}",
            f"seed={args.seed}",
            f"d_train={args.d_train}",
        ]
        if args.sweep_name:
            tags.append(f"sweep={args.sweep_name}")
        wandb.init(
            project="albench-s2f",
            name=f"preflight_{args.arch}_d{args.d_train}_s{args.seed}",
            tags=tags,
            config={**vars(args), **hp},
            mode=os.environ.get("WANDB_MODE", "offline"),
        )
        run_id = wandb.run.id if wandb.run else None
    except Exception:
        run_id = None

    print(f"=== arch={args.arch} d_train={args.d_train} seed={args.seed} epochs={args.epochs} ===")
    print(f"    HPs: {hp}")
    summary = train(args, hp)
    summary["wandb_run_id"] = run_id

    # Persist resolved config
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "config.json").write_text(
        json.dumps({"args": vars(args), "hp": hp}, indent=2, default=str)
    )

    # Final summary line for grep / sacct downstream
    print(
        f"=== DONE  best_val={summary['best_val_mse']:.4f}  "
        f"test={summary['test_mse_at_best_val']:.4f}  "
        f"best_epoch={summary['best_epoch']}/{args.epochs}  "
        f"in_final_10pct={summary['min_val_in_final_pct_window']}  "
        f"params={summary['n_params']:,}  "
        f"gpu_hrs={summary['gpu_hrs']:.2f}"
    )

    try:
        import wandb

        if wandb.run:
            wandb.summary.update(
                {
                    "best_val_mse": summary["best_val_mse"],
                    "test_mse_at_best_val": summary["test_mse_at_best_val"],
                    "best_epoch": summary["best_epoch"],
                    "min_val_in_final_pct_window": summary["min_val_in_final_pct_window"],
                    "n_params": summary["n_params"],
                    "gpu_hrs": summary["gpu_hrs"],
                }
            )
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
