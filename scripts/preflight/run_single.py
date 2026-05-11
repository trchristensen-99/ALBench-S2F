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
        "num_lstm_layers": 1,
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

# Canonical MPRA adapter constants from alphagenome_FT_MPRA/oracle.py.
# Used by shift augmentation to provide real flanking context that the
# sliding-window crop can reveal — without these, the existing torch.roll
# would wrap payload tail to head, which is biologically meaningless.
LEFT_ADAPTER = "AGGACCGGATCAACT"  # 15 bp
RIGHT_ADAPTER = "CATTGCGTGAACCGA"  # 15 bp


def one_hot(
    sequences: list[str],
    seq_len: int,
    in_channels: int,
    pad_with_adapters: bool = False,
) -> np.ndarray:
    """One-hot encode sequences.

    Args:
        sequences: input payload sequences (any length; will be center-
            padded with N or center-truncated to ``seq_len``).
        seq_len: target output length per sample. When
            ``pad_with_adapters=True`` this is the *payload* length only;
            the actual output tensor is ``seq_len + len(LEFT_ADAPTER) +
            len(RIGHT_ADAPTER)`` wide.
        in_channels: 4 for ACGT-only, 5 for ACGT + RC flag (channel 4 = 0
            for forward strand).
        pad_with_adapters: if True, prepend ``LEFT_ADAPTER`` and append
            ``RIGHT_ADAPTER`` one-hots to each payload one-hot. The output
            shape is ``(N, in_channels, seq_len + L + R)``.
    """
    n = len(sequences)
    L = len(LEFT_ADAPTER) if pad_with_adapters else 0
    R = len(RIGHT_ADAPTER) if pad_with_adapters else 0
    full_len = seq_len + L + R
    out = np.zeros((n, in_channels, full_len), dtype=np.float32)
    if pad_with_adapters:
        for j, nuc in enumerate(LEFT_ADAPTER):
            out[:, _NUC_TO_IDX[nuc], j] = 1.0
        for j, nuc in enumerate(RIGHT_ADAPTER):
            out[:, _NUC_TO_IDX[nuc], L + seq_len + j] = 1.0
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
                out[i, idx, L + j] = 1.0
    return out


def _rc_flip(x: torch.Tensor) -> torch.Tensor:
    """Reverse-complement: flip ACGT and reverse along sequence."""
    out = x.flip(dims=[2]).clone()
    out[:, [0, 1, 2, 3]] = out[:, [3, 2, 1, 0]]
    if out.shape[1] >= 5:
        out[:, 4] = 1.0 - out[:, 4]  # toggle RC indicator if present
    return out


def _shift_window_crop(
    x: torch.Tensor, payload_len: int, max_shift: int, training: bool
) -> torch.Tensor:
    """Sliding-window crop of an adapter-padded one-hot batch.

    Args:
        x: (B, C, L) where ``L = payload_len + 2 * max_shift``. Layout is
            ``[left_adapter (max_shift), payload (payload_len),
            right_adapter (max_shift)]``.
        payload_len: number of bp the model expects to see at its input.
        max_shift: half-width of the shift window. Must satisfy
            ``max_shift <= min(len(LEFT_ADAPTER), len(RIGHT_ADAPTER))``;
            this is enforced by the caller in ``train()``.
        training: if False, return the deterministic center crop (the
            canonical payload window, same as if no aug were applied).
            If True, 50% of samples get a random offset in
            ``[0, 2 * max_shift]`` and 50% stay at center.

    Returns: (B, C, payload_len) — what the model actually consumes.
    """
    B, C, L = x.shape
    expected = payload_len + 2 * max_shift
    if L != expected:
        raise ValueError(
            f"_shift_window_crop expected L={expected} (payload_len + 2*max_shift); got {L}"
        )
    if not training or max_shift == 0:
        return x[:, :, max_shift : max_shift + payload_len]
    rand_offsets = torch.randint(0, 2 * max_shift + 1, (B,), device=x.device)
    use_aug = torch.rand(B, device=x.device) > 0.5
    offsets = torch.where(use_aug, rand_offsets, torch.full_like(rand_offsets, max_shift))
    idx = offsets[:, None] + torch.arange(payload_len, device=x.device)[None, :]
    idx = idx[:, None, :].expand(B, C, payload_len)
    return x.gather(2, idx)


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
            num_lstm_layers=hp.get("num_lstm_layers", 1),
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
    pad_with_adapters: bool = False,
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
        # Prefer the new ref+alt+boda2 AG-S2 cache if available (parquet pool +
        # row-aligned npz labels). Falls back to the legacy
        # outputs/oracle_pseudolabels_k562_ag cache via sequence-keyed lookup.
        new_cache = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt"
        if (new_cache / "train_oracle_labels.npz").exists():
            import pandas as pd

            pool = new_cache / "pool"
            train_df = pd.read_parquet(pool / "train.parquet")
            val_df = pd.read_parquet(pool / "val.parquet")
            test_df = pd.read_parquet(pool / "test.parquet")
            train_npz = np.load(new_cache / "train_oracle_labels.npz")
            val_npz = np.load(new_cache / "val_oracle_labels.npz")
            test_npz = np.load(new_cache / "test_oracle_labels.npz")
            train_pool_seqs = [str(s) for s in train_df["sequence"]]
            train_pool_lbl = train_npz["oof_oracle"].astype(np.float32)
            val_seqs = [str(s) for s in val_df["sequence"]]
            val_labels = val_npz["oracle_mean"].astype(np.float32)
            test_seqs = [str(s) for s in test_df["sequence"]]
            test_labels = test_npz["oracle_mean"].astype(np.float32)
            print(
                f"  AG-S2 ref+alt cache:   train {len(train_pool_seqs):,}  "
                f"val {len(val_seqs):,}  test {len(test_seqs):,}"
            )
        else:
            # Legacy fallback (sequence-keyed lookup against older cache)
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
                f"  legacy AG cache match: train {len(train_pool_seqs):,}/{len(ds_train.sequences):,}  "
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

    Xtr = one_hot(train_seqs, seq_len, in_channels, pad_with_adapters=pad_with_adapters)
    Xva = one_hot(val_seqs, seq_len, in_channels, pad_with_adapters=pad_with_adapters)
    Xte = one_hot(test_seqs, seq_len, in_channels, pad_with_adapters=pad_with_adapters)
    return (Xtr, train_labels), (Xva, val_labels), (Xte, test_labels)


# ── Training loop with best-val checkpointing ────────────────────────────
def _eval_loss(
    model,
    loader,
    device,
    augment_rc: bool,
    payload_len: int = 200,
    max_shift: int = 0,
) -> tuple[float, np.ndarray]:
    """Evaluate. If max_shift>0, inputs are adapter-padded — apply the
    deterministic center crop so the model sees the canonical payload
    window. RC is applied AFTER the crop, so the RC view aligns to the
    same canonical window."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if max_shift > 0:
                xb = _shift_window_crop(xb, payload_len, max_shift, training=False)
            yhat = model(xb).reshape(-1)
            if augment_rc:
                yhat_rc = model(_rc_flip(xb)).reshape(-1)
                yhat = 0.5 * (yhat + yhat_rc)
            preds.append(yhat.detach().cpu().numpy())
            targets.append(yb.detach().cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return float(np.mean((p - t) ** 2)), p


def train(args: argparse.Namespace, hp: dict[str, Any], epoch_callback=None) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Augmentation policy. RC is a per-sample channel flip; shift is a
    # sliding-window crop over an adapter-padded one-hot tensor with
    # max_shift = min(len(LEFT_ADAPTER), len(RIGHT_ADAPTER)).
    augment_rc = args.augmentations in ("rev_complement", "rc_shift", "rc_shift_evoaug")
    use_shift = args.augmentations in ("rc_shift", "rc_shift_evoaug")
    payload_len = hp.get("sequence_length", 200)
    max_shift = min(len(LEFT_ADAPTER), len(RIGHT_ADAPTER)) if use_shift else 0

    # Data
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_data(
        args.d_train,
        args.seed,
        in_channels=hp["in_channels"],
        seq_len=payload_len,
        label_source=args.label_source,
        pad_with_adapters=use_shift,
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

    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    # Build optional training-time EvoAug transform
    evoaug_transform = None
    if getattr(args, "evoaug_intensity", None):
        from models.evoaug_transform import EvoAugTransform

        evoaug_transform = EvoAugTransform(
            intensity=args.evoaug_intensity,
            apply_prob=args.evoaug_prob,
            seed=args.seed,
            target_length=payload_len,
        )
        print(
            f"  EvoAug training-time aug enabled: intensity={args.evoaug_intensity}, "
            f"apply_prob={args.evoaug_prob}"
        )

    history = {"train_loss": [], "val_loss": [], "test_loss": [], "lr": []}
    best_val = math.inf
    best_epoch = -1
    best_test_mse = math.nan
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Checkpoint resume: if last.pt exists from a prior pre-emption, load
    # model/optimizer/scheduler state and resume from the saved epoch.
    # This makes long-running Task 4 (240 epochs at D=600k) resilient to
    # slow_nice preemption — without it, every preemption resets to ep 1.
    start_epoch = 0
    last_ckpt = out_dir / "last.pt"
    if last_ckpt.exists():
        try:
            ckpt = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val = float(ckpt["best_val"])
            best_epoch = int(ckpt["best_epoch"])
            best_test_mse = float(ckpt["best_test_mse"])
            history = ckpt.get("history", history)
            print(
                f"  Resumed from epoch {start_epoch} (best_val={best_val:.4f} @ ep {best_epoch + 1})"
            )
        except Exception as e:
            print(f"  WARN: last.pt exists but failed to load ({e}); starting fresh")
            start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_shift:
                # Sliding-window crop on the adapter-padded input. Returns
                # the canonical payload_len window with random sub-bp
                # offset on 50% of samples (±max_shift bp).
                xb = _shift_window_crop(xb, payload_len, max_shift, training=True)
            if augment_rc and torch.rand(1).item() < 0.5:
                xb = _rc_flip(xb)
            if evoaug_transform is not None:
                xb = evoaug_transform(xb)
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

        val_loss, _ = _eval_loss(
            model,
            val_loader,
            device,
            augment_rc=augment_rc,
            payload_len=payload_len,
            max_shift=max_shift,
        )
        test_loss, _ = _eval_loss(
            model,
            test_loader,
            device,
            augment_rc=augment_rc,
            payload_len=payload_len,
            max_shift=max_shift,
        )

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

        # Save resume checkpoint every epoch (overwrites last.pt). This
        # is what enables preemption recovery — without it, slow_nice
        # preemption forces a fresh restart from epoch 1.
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_test_mse": best_test_mse,
                "history": history,
                "hp": hp,
            },
            out_dir / "last.pt",
        )

        # Per-epoch W&B logging — gracefully no-ops if wandb not active.
        try:
            import wandb as _wb

            if _wb.run:
                _wb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "test_loss": test_loss,
                        "lr": history["lr"][-1],
                        "best_val_so_far": best_val,
                        "best_epoch_so_far": best_epoch + 1,
                    },
                    step=epoch + 1,
                )
        except Exception:
            pass

        print(
            f"  ep {epoch + 1}/{args.epochs}  train={train_loss:.4f}  "
            f"val={val_loss:.4f}  test={test_loss:.4f}  lr={history['lr'][-1]:.5f}"
        )

        if epoch_callback is not None:
            try:
                epoch_callback(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "test_loss": test_loss,
                        "best_val_so_far": best_val,
                        "best_epoch_so_far": best_epoch + 1,
                    }
                )
            except Exception:
                pass

        if args.early_stop_patience > 0:
            since_best = epoch - best_epoch
            if since_best >= args.early_stop_patience:
                print(
                    f"  early stop: no val improvement in {args.early_stop_patience} epochs "
                    f"(best @ ep {best_epoch + 1})"
                )
                break

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
    """Parse k=v overrides where v is auto-parsed (int, float, bool, list, str).

    List literals: ``block_sizes=[128,128,64,64]`` → ``[128, 128, 64, 64]``.
    """
    import ast

    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"override must be k=v, got {it!r}")
        k, v = it.split("=", 1)
        # type coerce
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v.startswith("[") and v.endswith("]"):
            # list/tuple literal — use ast.literal_eval (safe vs eval)
            try:
                out[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                out[k] = v
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
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop if val_loss has not improved for N consecutive epochs. 0 = disabled.",
    )
    ap.add_argument(
        "--evoaug_intensity",
        type=str,
        default=None,
        choices=[None, "light", "medium", "heavy"],
        help="Training-time EvoAug intensity (light/medium/heavy). None disables.",
    )
    ap.add_argument(
        "--evoaug_prob",
        type=float,
        default=0.5,
        help="Per-sample probability of applying EvoAug per batch (0.0-1.0).",
    )
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
        # Entity defaults to the user's primary account (set via wandb
        # login → ~/.netrc); we don't hardcode it because the user's
        # primary entity is a team name (trchristensen99-cold-spring-
        # harbor-laboratory), not the username. Override with WANDB_ENTITY
        # env var if needed.
        wandb_kwargs = dict(
            project=os.environ.get("WANDB_PROJECT", "albench-s2f"),
            name=f"preflight_{args.arch}_d{args.d_train}_s{args.seed}",
            tags=tags,
            config={**vars(args), **hp},
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        if os.environ.get("WANDB_ENTITY"):
            wandb_kwargs["entity"] = os.environ["WANDB_ENTITY"]
        wandb.init(**wandb_kwargs)
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
