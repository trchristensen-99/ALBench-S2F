#!/usr/bin/env python
"""DEFAULT oracle — Stage-2 encoder fine-tune, full-dataset random 10-fold CV.

Initialises from the matching Stage-1 fold checkpoint
(``outputs/oracle_full856k_clean/s1/oracle_{fold}``), unfreezes the top encoder
downres blocks (proven ``s2c`` config: encoder_lr=1e-4, head_lr=1e-3, unfreeze
downres_block_4,5), and fine-tunes on the FULL MPRA dataset
(ref + alt + OOD designed, 856,252 rows) under the identical deterministic
random 10-fold split (seed=42 permutation) used by Stage 1, so S2 fold ``k``
validates on exactly the rows S1 fold ``k`` held out.

Raw sequences come from ``build_full_oracle_cache.load_all_sequences`` (cache
order: ref → alt → OOD); labels come from ``cache_dir/all_labels.npy`` (the
authoritative array Stage 1 trained against). Sequences are flanked to 600 bp
with the MPRA context and trained with RC + shift augmentation.

  --max-idx restricts the pool to rows [0:max_idx). Default (None) = all 856,252
  (the standard DEFAULT oracle). Pass 833290 to exclude the 22,962 OOD designed
  high-activity sequences (the COMPARISON oracle).

Usage (one fold)::

    uv run --no-sync python experiments/train_oracle_s2_fullcv.py \
        --cache-dir outputs/oracle_full856k_clean/embedding_cache \
        --stage1-dir outputs/oracle_full856k_clean/s1/oracle_0 \
        --output-dir outputs/oracle_full856k_clean/s2/fold_0 \
        --fold-id 0 --n-folds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Head name MUST match Stage 1 so the S1 checkpoint head params merge correctly.
HEAD_NAME = "oracle_k562_fullcv_boda_flatten_512_512_v4"

_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def _merge(base, override):
    """Recursively merge *override* into *base*, returning a new dict."""
    if not isinstance(override, Mapping) or not isinstance(base, Mapping):
        return override
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
            merged[k] = _merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _one_hot_200(seq: str) -> np.ndarray:
    seq = seq[:200].upper()
    if len(seq) < 200:
        seq = seq + "N" * (200 - len(seq))
    ohe = np.zeros((5, 200), dtype=np.float32)
    for i, c in enumerate(seq):
        j = _MAPPING.get(c)
        if j is not None:
            ohe[j, i] = 1.0
    return ohe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-id", required=True, type=int)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument(
        "--max-idx",
        type=int,
        default=None,
        help="Restrict pool to rows [0:max_idx). None=all 856,252.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--head-arch", default="boda-flatten-512-512")
    parser.add_argument("--num-tracks", type=int, default=1)
    parser.add_argument(
        "--unfreeze-blocks",
        default="4,5",
        help="Comma-separated downres block indices to unfreeze (s2c=4,5).",
    )
    parser.add_argument(
        "--folds-npy",
        type=Path,
        default=None,
        help="pool-aligned fold ids (oracle_poolmap_v2.npy). When given, the split comes from "
        "chromosome-based folds with a rotating val/test instead of a random permutation: "
        "test=fold_id, val=(fold_id+1)%n_folds, train=the remaining folds. A real test fold is "
        "the point - a val fold is what early stopping selected on, so it is optimistic.",
    )
    parser.add_argument(
        "--shift-mode",
        choices=["crop", "roll_n", "roll"],
        default="crop",
        help="crop: take the 600bp window at an offset from the full 300+200+300 assembly, so "
        "shifted-in bases are REAL plasmid context. roll_n: circular roll then mask the wrapped "
        "edge as N (matches the reference implementation). roll: circular roll with no mask, "
        "which splices the 3-prime flank onto the 5-prime start (what v1 did).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the last saved best_model if progress.json exists. slow_nice is "
        "preemptible, and without this an evicted run restarts from epoch 0. Params are restored; "
        "the optimizer state is NOT saved (Adam moments for a fully-unfrozen encoder would be "
        "~3.4 GB per epoch of I/O), so it restarts fresh - a brief re-warmup, not a correctness "
        "issue.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights-path",
        default="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "test_metrics.json"
    if result_path.exists():
        logger.info("Already completed (%s exists), skipping.", result_path)
        return

    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    import torch
    from alphagenome_ft import create_model_with_heads
    from scipy.stats import pearsonr, spearmanr
    from torch.utils.data import DataLoader, Subset

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params
    from scripts.build_full_oracle_cache import load_all_sequences

    def _safe_corr(y_true, y_pred, fn) -> float:
        if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
            return 0.0
        return float(fn(y_true, y_pred)[0])

    # ── 600 bp MPRA flanks ────────────────────────────────────────────────────
    # Keep the FULL 300 bp of plasmid context on each side. The 600 bp model input is a centred
    # window of the 800 bp assembly, which leaves 100 bp of real sequence on each side to shift
    # into - so a shift augmentation never has to invent bases.
    CTX = 300
    PAD = CTX - 200  # 100 bp of spare context per side
    flank5_full = np.zeros((CTX, 4), dtype=np.float32)
    for _i, _c in enumerate(MPRA_UPSTREAM[-CTX:]):
        if _c in _MAPPING:
            flank5_full[_i, _MAPPING[_c]] = 1.0
    flank3_full = np.zeros((CTX, 4), dtype=np.float32)
    for _i, _c in enumerate(MPRA_DOWNSTREAM[:CTX]):
        if _c in _MAPPING:
            flank3_full[_i, _MAPPING[_c]] = 1.0

    flank5 = np.zeros((200, 4), dtype=np.float32)
    for _i, _c in enumerate(MPRA_UPSTREAM[-200:]):
        if _c in _MAPPING:
            flank5[_i, _MAPPING[_c]] = 1.0
    flank3 = np.zeros((200, 4), dtype=np.float32)
    for _i, _c in enumerate(MPRA_DOWNSTREAM[:200]):
        if _c in _MAPPING:
            flank3[_i, _MAPPING[_c]] = 1.0

    def collate(batch, augment: bool):
        bsz = len(batch)
        x = np.zeros((bsz, 600, 4), dtype=np.float32)
        y = np.zeros(bsz, dtype=np.float32)
        for i, (seq_5ch, label) in enumerate(batch):
            core = np.asarray(seq_5ch)[:4, :].T  # (200, 4)
            shift = 0
            if augment and args.max_shift > 0 and np.random.rand() > 0.5:
                shift = int(np.random.randint(-args.max_shift, args.max_shift + 1))

            if args.shift_mode == "crop":
                # window of the 800 bp assembly, offset by `shift`; PAD bounds the offset so the
                # crop always stays inside real sequence
                s = PAD + shift
                s = max(0, min(2 * PAD, s))
                wide = np.concatenate([flank5_full, core, flank3_full], axis=0)  # (800, 4)
                full = wide[s:s + 600]
            else:
                full = np.concatenate([flank5, core, flank3], axis=0)  # (600, 4)
                if shift:
                    full = np.roll(full, shift, axis=0)
                    if args.shift_mode == "roll_n":
                        if shift > 0:
                            full[:shift, :] = 0.25
                        else:
                            full[shift:, :] = 0.25
            if augment and np.random.rand() > 0.5:
                full = full[::-1, ::-1]
            x[i] = full
            y[i] = float(label)
        return {
            "sequences": x,
            "targets": y,
            "organism_index": np.zeros(bsz, dtype=np.int32),
        }

    np.random.seed(args.seed)

    # ── Model: frozen base + S1 head, encoder grads enabled ───────────────────
    register_s2f_head(
        head_name=HEAD_NAME,
        arch=args.head_arch,
        task_mode="human",
        num_tracks=args.num_tracks,
        dropout_rate=args.dropout_rate,
    )
    if not Path(args.weights_path).exists():
        raise FileNotFoundError(f"AlphaGenome weights not found: {args.weights_path}")
    model = create_model_with_heads(
        "all_folds",
        heads=[HEAD_NAME],
        checkpoint_path=str(args.weights_path),
        use_encoder_output=True,
        detach_backbone=False,  # encoder gradients flow
    )
    reinit_head_params(model, HEAD_NAME, num_tokens=5, dim=1536, rng=args.seed)

    # ── Restore Stage-1 fold checkpoint (full model: base encoder + trained head)
    # orbax requires an ABSOLUTE path; S1 saved a StandardCheckpointer checkpoint
    # at best_model/checkpoint via model.save_checkpoint(save_full_model=True).
    s1_ckpt = (args.stage1_dir / "best_model" / "checkpoint").resolve()
    if not s1_ckpt.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {s1_ckpt}")
    # S1's save_checkpoint() stored a TUPLE (params, state) via StandardCheckpointer.
    # orbax serializes that tuple as a top-level list, so a dict-target restore fails
    # with a dict-vs-list metadata mismatch. Restore with no target and unpack.
    s1_params, s1_state = ocp.StandardCheckpointer().restore(str(s1_ckpt))
    model._params = jax.device_put(s1_params)
    model._state = jax.device_put(s1_state)
    logger.info("Loaded Stage-1 checkpoint from %s", s1_ckpt)

    # ── Per-group optimizer (head / encoder / frozen) ─────────────────────────
    unfreeze_all = args.unfreeze_blocks.strip().lower() == "all"
    unfreeze_set = set() if unfreeze_all else {
        f"downres_block_{b.strip()}" for b in args.unfreeze_blocks.split(",") if b.strip()
    }
    logger.info("Unfreezing encoder: %s", "ALL blocks" if unfreeze_all else sorted(unfreeze_set))

    def _label_fn(path, _leaf):
        s = "/".join(str(p.key if hasattr(p, "key") else p) for p in path)
        if HEAD_NAME in s:
            return "head"
        if "sequence_encoder" in s:
            if unfreeze_all:
                return "encoder"
            return "encoder" if any(b in s for b in unfreeze_set) else "frozen"
        return "frozen"

    param_labels = jax.tree_util.tree_map_with_path(_label_fn, model._params)
    optimizer = optax.multi_transform(
        {
            "head": optax.adamw(learning_rate=args.head_lr, weight_decay=args.weight_decay),
            "encoder": optax.adamw(learning_rate=args.encoder_lr, weight_decay=args.weight_decay),
            "frozen": optax.set_to_zero(),
        },
        param_labels,
    )
    opt_state = optimizer.init(model._params)
    counts = {"head": 0, "encoder": 0, "frozen": 0}
    for lab, leaf in zip(
        jax.tree_util.tree_leaves(param_labels), jax.tree_util.tree_leaves(model._params)
    ):
        counts[lab] = counts.get(lab, 0) + leaf.size
    logger.info(
        "Param groups — head=%s encoder=%s frozen=%s",
        f"{counts['head']:,}",
        f"{counts['encoder']:,}",
        f"{counts['frozen']:,}",
    )

    # ── Data: full 856k raw sequences + authoritative S1 labels ───────────────
    all_seqs, recomputed_labels = load_all_sequences()
    all_labels = np.load(args.cache_dir / "all_labels.npy").astype(np.float32)
    assert len(all_seqs) == len(all_labels), (
        f"seq/label mismatch: {len(all_seqs)} vs {len(all_labels)}"
    )
    logger.info("Loaded %s sequences + labels", f"{len(all_seqs):,}")

    class FullSeqDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(all_seqs)

        def __getitem__(self, idx):
            return _one_hot_200(all_seqs[idx]), float(all_labels[idx])

    ds = FullSeqDataset()

    pool = len(all_labels) if args.max_idx is None else int(args.max_idx)
    test_idx = np.array([], dtype=np.int64)
    if args.folds_npy is not None:
        fmap = np.load(args.folds_npy)[:pool]
        test_fold = args.fold_id
        val_fold = (args.fold_id + 1) % args.n_folds
        test_idx = np.where(fmap == test_fold)[0]
        val_idx = np.where(fmap == val_fold)[0]
        train_idx = np.where((fmap >= 0) & (fmap != test_fold) & (fmap != val_fold))[0]
        logger.info(
            "Fold %d: test=fold %d (%s) val=fold %d (%s) train=%s from %d folds",
            args.fold_id, test_fold, f"{len(test_idx):,}", val_fold, f"{len(val_idx):,}",
            f"{len(train_idx):,}", args.n_folds - 2,
        )
        val_start = val_end = 0  # unused on this path
        perm = None
    else:
        perm = np.random.default_rng(seed=42).permutation(pool)
        fold_size = pool // args.n_folds
        val_start = args.fold_id * fold_size
        val_end = val_start + fold_size if args.fold_id < args.n_folds - 1 else pool
    if perm is not None:
        val_idx = perm[val_start:val_end]
        train_idx = np.concatenate([perm[:val_start], perm[val_end:]])
    logger.info(
        "Fold %d/%d (pool=%s) — train=%s val=%s",
        args.fold_id,
        args.n_folds,
        f"{pool:,}",
        f"{len(train_idx):,}",
        f"{len(val_idx):,}",
    )

    train_loader = DataLoader(
        Subset(ds, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(b, augment=True),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(b, augment=False),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ── JIT steps ─────────────────────────────────────────────────────────────
    @jax.jit
    def train_step(params, current_opt_state, sequences, targets, org_idx):
        def loss_fn(p):
            preds = model._predict(
                p,
                model._state,
                sequences,
                org_idx,
                negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
                strand_reindexing=None,
                requested_outputs=[HEAD_NAME],
            )[HEAD_NAME]
            pred = jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds
            return jnp.mean((pred - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    @jax.jit
    def eval_step(params, sequences, org_idx):
        preds = model._predict(
            params,
            model._state,
            sequences,
            org_idx,
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
            requested_outputs=[HEAD_NAME],
        )[HEAD_NAME]
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_pearson = -1.0
    best_params = None
    start_epoch = 0
    progress_path = args.output_dir / "progress.json"
    if args.resume and progress_path.exists():
        prog = json.loads(progress_path.read_text())
        ck = (args.output_dir / "best_model" / "checkpoint").resolve()
        if ck.exists():
            restored = ocp.StandardCheckpointer().restore(ck)
            # save_checkpoint stored (params, state); orbax may hand it back as a list
            pr = restored[0] if isinstance(restored, (tuple, list)) else restored
            model._params = jax.device_put(pr)
            best_params = jax.device_get(model._params)
            start_epoch = int(prog["next_epoch"])
            best_val_pearson = float(prog["best_val_pearson"])
            best_epoch = int(prog["best_epoch"])
            epochs_no_improve = int(prog.get("epochs_no_improve", 0))
            logger.info(
                "RESUMED at epoch %d/%d (best_val_pearson=%.4f from epoch %d)",
                start_epoch + 1, args.epochs, best_val_pearson, best_epoch + 1,
            )
        else:
            logger.warning("progress.json present but no checkpoint; starting fresh")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        train_losses: list[float] = []
        for batch in train_loader:
            model._params, opt_state, loss = train_step(
                model._params,
                opt_state,
                jnp.array(batch["sequences"]),
                jnp.array(batch["targets"]),
                jnp.array(batch["organism_index"]),
            )
            train_losses.append(float(loss))

        y_true_all, y_pred_all = [], []
        for batch in val_loader:
            preds = eval_step(
                model._params,
                jnp.array(batch["sequences"]),
                jnp.array(batch["organism_index"]),
            )
            y_pred_all.append(np.array(preds).reshape(-1))
            y_true_all.append(np.array(batch["targets"]).reshape(-1))
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        pear = _safe_corr(y_true, y_pred, pearsonr)
        spear = _safe_corr(y_true, y_pred, spearmanr)
        logger.info(
            "Epoch %d: train_loss=%.4f val_pearson=%.4f val_spearman=%.4f%s",
            epoch + 1,
            avg_train,
            pear,
            spear,
            " *" if pear > best_val_pearson else "",
        )

        if pear > best_val_pearson:
            best_val_pearson = pear
            best_epoch = epoch
            epochs_no_improve = 0
            model.save_checkpoint(str(args.output_dir / "best_model"), save_full_model=True)
            # Hold the selected params on the host as well. Reloading them through orbax for the
            # test pass is fragile - the round trip can hand back a list rather than the mapping
            # haiku expects - and an in-memory copy removes that failure mode entirely.
            best_params = jax.device_get(model._params)
            progress_path.write_text(json.dumps({
                "next_epoch": epoch + 1, "best_val_pearson": best_val_pearson,
                "best_epoch": best_epoch, "epochs_no_improve": 0,
            }))
        else:
            epochs_no_improve += 1
            progress_path.write_text(json.dumps({
                "next_epoch": epoch + 1, "best_val_pearson": best_val_pearson,
                "best_epoch": best_epoch, "epochs_no_improve": epochs_no_improve,
            }))
            if epochs_no_improve >= args.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (best=%d, val_pearson=%.4f)",
                    epoch + 1,
                    best_epoch + 1,
                    best_val_pearson,
                )
                break

    # Evaluate the TEST fold with the checkpoint that val selected. This is the number to report:
    # the val fold drove early stopping, so it is optimistic by construction.
    test_metrics = None
    if len(test_idx):
        test_loader = DataLoader(
            Subset(ds, test_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate(b, augment=False),
            pin_memory=True,
        )
        if best_params is not None:
            model._params = jax.device_put(best_params)
            logger.info("Test pass uses the val-selected params from epoch %d", best_epoch + 1)
        else:
            logger.warning("No improving epoch recorded; testing with final params")
        yt, yp = [], []
        for batch in test_loader:
            p = eval_step(model._params, jnp.array(batch["sequences"]),
                          jnp.array(batch["organism_index"]))
            yp.append(np.array(p).reshape(-1))
            yt.append(np.array(batch["targets"]).reshape(-1))
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        test_metrics = {
            "n": int(len(yt)),
            "pearson": _safe_corr(yt, yp, pearsonr),
            "spearman": _safe_corr(yt, yp, spearmanr),
            "mse": float(np.mean((yt - yp) ** 2)),
        }
        np.savez_compressed(args.output_dir / "test_predictions.npz",
                            idx=test_idx, y_true=yt, y_pred=yp)
        logger.info("TEST fold %d: pearson=%.4f mse=%.4f n=%d", args.fold_id,
                    test_metrics["pearson"], test_metrics["mse"], test_metrics["n"])

    result = {
        "fold_id": args.fold_id,
        "n_folds": args.n_folds,
        "max_idx": args.max_idx,
        "pool": pool,
        "best_val_pearson": best_val_pearson,
        "best_epoch": best_epoch,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "unfreeze_blocks": sorted(unfreeze_set),
        "head_name": HEAD_NAME,
        "stage1_dir": str(args.stage1_dir),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "test_metrics": test_metrics,
        "val_fold": (args.fold_id + 1) % args.n_folds if args.folds_npy else None,
        "folds_npy": str(args.folds_npy) if args.folds_npy else None,
        "shift_mode": args.shift_mode,
        "max_shift": args.max_shift,
        "unfreeze_all": unfreeze_all,
        "seed": args.seed,
    }
    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved %s (best_val_pearson=%.4f)", result_path, best_val_pearson)


if __name__ == "__main__":
    main()
