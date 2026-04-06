#!/usr/bin/env python
"""Generate predictions.npz for AlphaGenome S1 and S2 checkpoints.

Discovers AG checkpoint directories, loads the model + checkpoint,
runs RC-averaged forward pass on chr-split test sequences (chr7+13),
and saves predictions.npz with in_dist_pred, in_dist_true, ood_pred,
ood_true, snv_ref_pred, snv_alt_pred, snv_alt_true, snv_delta_pred,
snv_delta_true arrays.

Both S1 (frozen encoder, head-only checkpoint) and S2 (full model
checkpoint with fine-tuned encoder) are supported.

Usage (HPC SLURM)::

    sbatch scripts/slurm/generate_ag_predictions.sh

Usage (local with GPU)::

    uv run python scripts/generate_ag_predictions.py --cell k562
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ── MPRA flanking constants ──────────────────────────────────────────────────

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM  # noqa: E402

_FLANK_5_STR: str = MPRA_UPSTREAM[-200:]
_FLANK_3_STR: str = MPRA_DOWNSTREAM[:200]
_NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

_FLANK_5_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(_FLANK_5_STR):
    if _c in _NUC_MAP:
        _FLANK_5_ENC[_i, _NUC_MAP[_c]] = 1.0

_FLANK_3_ENC: np.ndarray = np.zeros((200, 4), dtype=np.float32)
for _i, _c in enumerate(_FLANK_3_STR):
    if _c in _NUC_MAP:
        _FLANK_3_ENC[_i, _NUC_MAP[_c]] = 1.0


# ── Sequence encoding ────────────────────────────────────────────────────────


def _seq_str_to_600bp(seq_str: str) -> np.ndarray:
    """Encode a 200bp sequence string to 600bp one-hot (with MPRA flanks)."""
    seq_str = seq_str.upper()
    target_len = 200
    if len(seq_str) < target_len:
        pad = target_len - len(seq_str)
        seq_str = "N" * (pad // 2) + seq_str + "N" * (pad - pad // 2)
    elif len(seq_str) > target_len:
        start = (len(seq_str) - target_len) // 2
        seq_str = seq_str[start : start + target_len]
    core = np.zeros((200, 4), dtype=np.float32)
    for i, c in enumerate(seq_str):
        if c in _NUC_MAP:
            core[i, _NUC_MAP[c]] = 1.0
    return np.concatenate([_FLANK_5_ENC, core, _FLANK_3_ENC], axis=0)  # (600, 4)


def _safe_corr(pred, true, fn):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 3:
        return 0.0
    return float(fn(pred[mask], true[mask])[0])


# ── Model loading ────────────────────────────────────────────────────────────


def _merge_params(base, override):
    """Recursively merge override into base (for head-only checkpoint loading)."""
    if not isinstance(override, Mapping):
        return override
    if not isinstance(base, Mapping):
        return override
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
            merged[k] = _merge_params(merged[k], v)
        else:
            merged[k] = v
    return merged


def _detect_head_name(loaded_params: dict) -> str | None:
    """Detect the head name from loaded checkpoint params.

    Walks the nested param dict looking for keys under
    ``alphagenome/~_custom_heads/<head_name>`` or known head name patterns.
    """

    def _find_head_names(d, prefix=""):
        names = set()
        if isinstance(d, dict):
            for k, v in d.items():
                path = f"{prefix}/{k}" if prefix else k
                # Pattern: alphagenome/~_custom_heads/<head_name>
                if "_custom_heads" in str(k):
                    if isinstance(v, dict):
                        names.update(v.keys())
                # Known S1/S2 head name patterns
                if any(
                    pat in str(k)
                    for pat in ["exp1_s1_", "s2f_exp1_s2_", "head_hashfrag", "alphagenome_"]
                ):
                    if isinstance(v, dict):
                        names.add(k)
                names.update(_find_head_names(v, path))
        return names

    candidates = _find_head_names(loaded_params)
    # Filter to likely head names, in priority order
    for pattern in ["exp1_s1_", "s2f_exp1_s2_", "head_hashfrag", "alphagenome_"]:
        matches = [n for n in candidates if pattern in n]
        if matches:
            return matches[0]
    return candidates.pop() if candidates else None


def load_ag_model_and_restore(
    ckpt_path: Path,
    stage: str,
    cell_line: str = "k562",
):
    """Load AG model, detect head name from checkpoint, and restore params.

    For S1 checkpoints (head-only params), uses _merge to overlay head params.
    For S2 checkpoints (full model params), does direct assignment.

    Returns (model, predict_step_fn).
    """
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head

    # Load checkpoint once
    checkpointer = ocp.StandardCheckpointer()
    loaded = checkpointer.restore(ckpt_path.resolve())
    loaded_params = loaded[0] if isinstance(loaded, (tuple, list)) else loaded

    # Detect head name from checkpoint params
    detected_name = _detect_head_name(loaded_params)
    if detected_name:
        head_name = detected_name
        print(f"  Detected head name: {head_name}")
    else:
        # Fallback: use conventional names based on stage
        if stage == "s1":
            head_name = "exp1_s1_k562"
        else:
            head_name = "s2f_exp1_s2_k562_0"
        print(f"  WARNING: Could not detect head name, using fallback: {head_name}")

    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.1,
    )

    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )
    detach = stage == "s1"
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=detach,
    )

    if stage == "s1":
        # S1: head-only checkpoint — merge into model (preserves encoder weights)
        model._params = jax.device_put(_merge_params(model._params, loaded_params))
    else:
        # S2: full model checkpoint — direct assignment
        model._params = jax.device_put(loaded_params)

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            requested_outputs=[head_name],
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    return model, predict_step


# ── Prediction ───────────────────────────────────────────────────────────────


def predict_sequences(
    predict_step_fn,
    model_params,
    model_state,
    seqs_str: list[str],
    batch_size: int = 128,
) -> np.ndarray:
    """RC-averaged predictions on raw 200bp strings via 600bp context."""
    import jax.numpy as jnp

    if not seqs_str:
        return np.array([], dtype=np.float32)

    x_fwd = np.stack([_seq_str_to_600bp(s) for s in seqs_str])
    x_rev = x_fwd[:, ::-1, ::-1].copy()  # RC: reverse sequence, swap A<->T, C<->G

    preds_fwd, preds_rev = [], []
    for i in range(0, len(x_fwd), batch_size):
        end = min(i + batch_size, len(x_fwd))
        pf = predict_step_fn(model_params, model_state, jnp.array(x_fwd[i:end]))
        pr = predict_step_fn(model_params, model_state, jnp.array(x_rev[i:end]))
        preds_fwd.append(np.array(pf).reshape(-1))
        preds_rev.append(np.array(pr).reshape(-1))

    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


# ── Test data loading ────────────────────────────────────────────────────────

CELL_LABEL_COLS = {
    "k562": "K562_log2FC",
    "hepg2": "HepG2_log2FC",
    "sknsh": "SKNSH_log2FC",
}


def load_chr_split_test_data(cell_line: str = "k562"):
    """Load chr-split test sequences and labels (chr7+13).

    Returns dict with test set name -> (sequences, labels) tuples.
    """
    import pandas as pd

    from data.k562 import K562Dataset

    label_col = CELL_LABEL_COLS.get(cell_line, "K562_log2FC")
    test_sets = {}

    # In-distribution: K562Dataset test split (chr7+13)
    ds = K562Dataset(
        data_path=str(REPO / "data" / "k562"),
        split="test",
        label_column=label_col,
        use_hashfrag=False,
        use_chromosome_fallback=True,
        include_alt_alleles=True,
    )
    test_sets["in_dist"] = {
        "sequences": list(ds.sequences),
        "labels": ds.labels.astype(np.float32),
    }

    # SNV pairs (filter to chr7+13 for chr-split)
    test_dir = REPO / "data" / "k562" / "test_sets"
    cell_test_dir = REPO / "data" / cell_line / "test_sets"

    snv_path = cell_test_dir / "test_snv_pairs_hashfrag.tsv"
    if not snv_path.exists():
        snv_path = test_dir / "test_snv_pairs_hashfrag.tsv"
    if snv_path.exists():
        snv_df = pd.read_csv(snv_path, sep="\t")
        # Filter to test chromosomes (chr7+13) for chr-split
        if "IDs_ref" in snv_df.columns:
            test_chrs = {"7", "13", "chr7", "chr13"}
            chroms = snv_df["IDs_ref"].str.split(":", expand=True)[0]
            snv_df = snv_df[chroms.isin(test_chrs)].reset_index(drop=True)

        alt_col = f"{label_col}_alt"
        if alt_col not in snv_df.columns and cell_line == "k562":
            alt_col = "K562_log2FC_alt"
        delta_col = f"delta_{label_col}"
        if delta_col not in snv_df.columns and cell_line == "k562":
            delta_col = "delta_log2FC"

        test_sets["snv"] = {
            "ref_sequences": snv_df["sequence_ref"].tolist(),
            "alt_sequences": snv_df["sequence_alt"].tolist(),
            "alt_true": snv_df[alt_col].to_numpy(dtype=np.float32)
            if alt_col in snv_df.columns
            else None,
            "delta_true": snv_df[delta_col].to_numpy(dtype=np.float32)
            if delta_col in snv_df.columns
            else None,
        }

    # OOD designed sequences
    ood_path = cell_test_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists():
        ood_path = test_dir / f"test_ood_designed_{cell_line}.tsv"
    if not ood_path.exists() and cell_line == "k562":
        ood_path = test_dir / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path, sep="\t")
        ood_col = label_col if label_col in ood_df.columns else "K562_log2FC"
        test_sets["ood"] = {
            "sequences": ood_df["sequence"].tolist(),
            "labels": ood_df[ood_col].to_numpy(dtype=np.float32),
        }

    return test_sets


# ── Checkpoint discovery ─────────────────────────────────────────────────────


def discover_checkpoints(cell_line: str = "k562"):
    """Find all AG S1 and S2 checkpoint directories.

    Returns list of dicts with keys: path, stage, cell, label.
    """
    checkpoints = []

    # bar_final AG S1 checkpoints
    s1_dir = REPO / "outputs" / "bar_final" / cell_line / "ag_s1_pred"
    if s1_dir.exists():
        for ckpt in sorted(s1_dir.rglob("best_model/checkpoint")):
            run_dir = ckpt.parent.parent
            checkpoints.append(
                {
                    "ckpt_path": ckpt,
                    "run_dir": run_dir,
                    "stage": "s1",
                    "cell": cell_line,
                    "label": f"bar_final/{cell_line}/ag_s1_pred/{run_dir.relative_to(s1_dir)}",
                }
            )

    # bar_final AG S2 checkpoints
    s2_dir = REPO / "outputs" / "bar_final" / cell_line / "ag_s2_rc_shift"
    if s2_dir.exists():
        for ckpt in sorted(s2_dir.rglob("best_model/checkpoint")):
            run_dir = ckpt.parent.parent
            checkpoints.append(
                {
                    "ckpt_path": ckpt,
                    "run_dir": run_dir,
                    "stage": "s2",
                    "cell": cell_line,
                    "label": f"bar_final/{cell_line}/ag_s2_rc_shift/{run_dir.relative_to(s2_dir)}",
                }
            )

    # chr_split AG S1 checkpoints
    for ag_dir_name in ["ag_all_folds_s1", "ag_fold_1_s1"]:
        ag_dir = REPO / "outputs" / "chr_split" / cell_line / ag_dir_name
        if ag_dir.exists():
            for ckpt in sorted(ag_dir.rglob("best_model/checkpoint")):
                run_dir = ckpt.parent.parent
                checkpoints.append(
                    {
                        "ckpt_path": ckpt,
                        "run_dir": run_dir,
                        "stage": "s1",
                        "cell": cell_line,
                        "label": f"chr_split/{cell_line}/{ag_dir_name}/{run_dir.relative_to(ag_dir)}",
                    }
                )

    # chr_split AG S2 checkpoints
    for ag_dir_name in ["ag_all_folds_s2", "ag_fold_1_s2"]:
        ag_dir = REPO / "outputs" / "chr_split" / cell_line / ag_dir_name
        if ag_dir.exists():
            for ckpt in sorted(ag_dir.rglob("best_model/checkpoint")):
                run_dir = ckpt.parent.parent
                checkpoints.append(
                    {
                        "ckpt_path": ckpt,
                        "run_dir": run_dir,
                        "stage": "s2",
                        "cell": cell_line,
                        "label": f"chr_split/{cell_line}/{ag_dir_name}/{run_dir.relative_to(ag_dir)}",
                    }
                )

    # Also check bar_final AG S1 without _pred suffix
    s1_dir2 = REPO / "outputs" / "bar_final" / cell_line / "ag_s1"
    if s1_dir2.exists():
        for ckpt in sorted(s1_dir2.rglob("best_model/checkpoint")):
            run_dir = ckpt.parent.parent
            checkpoints.append(
                {
                    "ckpt_path": ckpt,
                    "run_dir": run_dir,
                    "stage": "s1",
                    "cell": cell_line,
                    "label": f"bar_final/{cell_line}/ag_s1/{run_dir.relative_to(s1_dir2)}",
                }
            )

    return checkpoints


# ── Main ─────────────────────────────────────────────────────────────────────


def generate_predictions_for_checkpoint(
    ckpt_info: dict,
    test_sets: dict,
    batch_size: int = 128,
    force: bool = False,
) -> None:
    """Generate and save predictions.npz for a single checkpoint."""
    run_dir = ckpt_info["run_dir"]
    pred_path = run_dir / "predictions.npz"
    if pred_path.exists() and not force:
        print(f"  SKIP (exists): {ckpt_info['label']}")
        return

    stage = ckpt_info["stage"]
    ckpt_path = ckpt_info["ckpt_path"]
    cell = ckpt_info["cell"]
    print(f"\n{'=' * 70}")
    print(f"  Stage: {stage.upper()} | Cell: {cell}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output: {pred_path}")
    print(f"{'=' * 70}")

    if stage not in ("s1", "s2"):
        print(f"  ERROR: Unknown stage '{stage}'")
        return

    model, predict_step = load_ag_model_and_restore(ckpt_path, stage, cell)

    params, state = model._params, model._state
    arrays = {}
    metrics = {}

    # In-distribution
    if "in_dist" in test_sets:
        td = test_sets["in_dist"]
        print(f"  Predicting in_dist ({len(td['sequences'])} seqs)...", flush=True)
        pred = predict_sequences(predict_step, params, state, td["sequences"], batch_size)
        true = td["labels"]
        arrays["in_dist_pred"] = pred
        arrays["in_dist_true"] = true
        mask = np.isfinite(pred) & np.isfinite(true)
        if mask.sum() >= 3:
            metrics["in_dist"] = {
                "pearson_r": _safe_corr(pred, true, pearsonr),
                "spearman_r": _safe_corr(pred, true, spearmanr),
                "mse": float(np.mean((pred[mask] - true[mask]) ** 2)),
                "n": int(mask.sum()),
            }
            print(
                f"    in_dist: pearson={metrics['in_dist']['pearson_r']:.4f}, "
                f"mse={metrics['in_dist']['mse']:.4f}, n={metrics['in_dist']['n']}"
            )

    # SNV pairs
    if "snv" in test_sets:
        td = test_sets["snv"]
        print(f"  Predicting SNV ref ({len(td['ref_sequences'])} seqs)...", flush=True)
        ref_pred = predict_sequences(predict_step, params, state, td["ref_sequences"], batch_size)
        print(f"  Predicting SNV alt ({len(td['alt_sequences'])} seqs)...", flush=True)
        alt_pred = predict_sequences(predict_step, params, state, td["alt_sequences"], batch_size)
        arrays["snv_ref_pred"] = ref_pred
        arrays["snv_alt_pred"] = alt_pred

        if td["alt_true"] is not None:
            arrays["snv_alt_true"] = td["alt_true"]
            metrics["snv_abs"] = {
                "pearson_r": _safe_corr(alt_pred, td["alt_true"], pearsonr),
                "spearman_r": _safe_corr(alt_pred, td["alt_true"], spearmanr),
                "mse": float(np.mean((alt_pred - td["alt_true"]) ** 2)),
                "n": len(td["alt_true"]),
            }
            print(f"    snv_abs: pearson={metrics['snv_abs']['pearson_r']:.4f}")

        delta_pred = alt_pred - ref_pred
        arrays["snv_delta_pred"] = delta_pred
        if td["delta_true"] is not None:
            arrays["snv_delta_true"] = td["delta_true"]
            metrics["snv_delta"] = {
                "pearson_r": _safe_corr(delta_pred, td["delta_true"], pearsonr),
                "spearman_r": _safe_corr(delta_pred, td["delta_true"], spearmanr),
                "mse": float(np.mean((delta_pred - td["delta_true"]) ** 2)),
                "n": len(td["delta_true"]),
            }
            print(f"    snv_delta: pearson={metrics['snv_delta']['pearson_r']:.4f}")

    # OOD
    if "ood" in test_sets:
        td = test_sets["ood"]
        print(f"  Predicting OOD ({len(td['sequences'])} seqs)...", flush=True)
        pred = predict_sequences(predict_step, params, state, td["sequences"], batch_size)
        true = td["labels"]
        arrays["ood_pred"] = pred
        arrays["ood_true"] = true
        mask = np.isfinite(pred) & np.isfinite(true)
        if mask.sum() >= 3:
            metrics["ood"] = {
                "pearson_r": _safe_corr(pred, true, pearsonr),
                "spearman_r": _safe_corr(pred, true, spearmanr),
                "mse": float(np.mean((pred[mask] - true[mask]) ** 2)),
                "n": int(mask.sum()),
            }
            print(
                f"    ood: pearson={metrics['ood']['pearson_r']:.4f}, "
                f"mse={metrics['ood']['mse']:.4f}"
            )

    # Save predictions
    np.savez_compressed(pred_path, **arrays)
    print(f"  Saved: {pred_path} ({pred_path.stat().st_size / 1024:.0f} KB)")

    # Update result.json with metrics
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        result["test_metrics"] = metrics
    else:
        result = {
            "model": f"alphagenome_{stage}",
            "cell": cell,
            "test_metrics": metrics,
        }
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Updated: {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions.npz for AlphaGenome S1/S2 checkpoints"
    )
    parser.add_argument(
        "--cell",
        default="k562",
        choices=["k562", "hepg2", "sknsh"],
        help="Cell line (default: k562)",
    )
    parser.add_argument(
        "--stage",
        default=None,
        choices=["s1", "s2"],
        help="Only process S1 or S2 checkpoints (default: both)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for prediction (default: 128)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing predictions.npz files",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Process a single checkpoint directory (must contain best_model/checkpoint)",
    )
    parser.add_argument(
        "--ckpt-stage",
        type=str,
        default="s1",
        choices=["s1", "s2"],
        help="Stage for --ckpt-dir (default: s1)",
    )
    args = parser.parse_args()

    print(f"Cell line: {args.cell}")
    print(f"Stage filter: {args.stage or 'all'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Force overwrite: {args.force}")

    # Load test data once
    print("\nLoading chr-split test data...", flush=True)
    test_sets = load_chr_split_test_data(args.cell)
    for name, data in test_sets.items():
        if "sequences" in data:
            print(f"  {name}: {len(data['sequences'])} sequences")
        elif "ref_sequences" in data:
            print(f"  {name}: {len(data['ref_sequences'])} pairs")

    if args.ckpt_dir:
        # Process a single checkpoint
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_path = ckpt_dir / "best_model" / "checkpoint"
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / "checkpoint"
        if not ckpt_path.exists():
            print(f"ERROR: No checkpoint found in {ckpt_dir}")
            sys.exit(1)
        ckpt_info = {
            "ckpt_path": ckpt_path,
            "run_dir": ckpt_dir,
            "stage": args.ckpt_stage,
            "cell": args.cell,
            "label": str(ckpt_dir),
        }
        generate_predictions_for_checkpoint(ckpt_info, test_sets, args.batch_size, args.force)
    else:
        # Discover and process all checkpoints
        checkpoints = discover_checkpoints(args.cell)
        if args.stage:
            checkpoints = [c for c in checkpoints if c["stage"] == args.stage]

        print(f"\nFound {len(checkpoints)} checkpoints:")
        for c in checkpoints:
            exists = (c["run_dir"] / "predictions.npz").exists()
            status = " [EXISTS]" if exists else ""
            print(f"  [{c['stage'].upper()}] {c['label']}{status}")

        if not checkpoints:
            print("No checkpoints found. Nothing to do.")
            return

        for i, ckpt_info in enumerate(checkpoints):
            print(f"\n[{i + 1}/{len(checkpoints)}] {ckpt_info['label']}")
            try:
                generate_predictions_for_checkpoint(
                    ckpt_info, test_sets, args.batch_size, args.force
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
