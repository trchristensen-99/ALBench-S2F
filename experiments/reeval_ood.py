#!/usr/bin/env python
"""Re-evaluate the OOD test set for existing trained models.

Scans result.json files and re-runs the OOD evaluation where either:
  - OOD is absent (Citra cached models had no OOD file)
  - OOD used the old 14,086-sequence file (n != 22862)

Updates result.json in-place with corrected OOD metrics.

Usage (scan a directory tree):
  uv run python experiments/reeval_ood.py \
      --scan-dir outputs/exp0_k562_scaling_alphagenome \
      --k562-data-path data/k562 \
      --weights-path /path/to/alphagenome_weights

Usage (single run directory):
  uv run python experiments/reeval_ood.py \
      --run-dir outputs/exp0_k562_scaling_alphagenome/fraction_0.0100/seed_xxx \
      --head-name alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4 \
      --head-arch boda-flatten-512-512 \
      --k562-data-path data/k562 \
      --weights-path /path/to/alphagenome_weights
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from alphagenome_ft import create_model_with_heads
from scipy.stats import pearsonr, spearmanr

from models.alphagenome_heads import register_s2f_head

# ── constants ──────────────────────────────────────────────────────────────────
CORRECT_OOD_N = 22862
OOD_FNAME = "test_ood_designed_k562.tsv"
BATCH_SIZE = 256

_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def _one_hot(seq: str, length: int = 200) -> np.ndarray:
    seq = seq.upper()
    if len(seq) < length:
        pad = length - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > length:
        start = (len(seq) - length) // 2
        seq = seq[start : start + length]
    arr = np.zeros((length, 4), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in _MAPPING:
            arr[i, _MAPPING[c]] = 1.0
    return arr


def _head_name_from_result(result: dict, run_dir: Path) -> tuple[str, str]:
    """Infer head_name and head_arch from result.json content."""
    arch = result.get("head_arch", "boda-flatten-512-512")
    arch_slug = arch.replace("-", "_")

    # Full-encoder runs store head_arch but not head_name prefix
    # Cached runs store neither (use config default)
    # Distinguish by the presence of a 'seed' or 'rng_int' key (full-encoder)
    # vs cached runs which have 'training_time_seconds' only
    if "seed" in result:
        # Old-style full-encoder (had explicit seed)
        base = "alphagenome_k562_head_hashfrag"
    elif "rng_int" in result:
        # New-style full-encoder (rng_int key)
        base = "alphagenome_k562_head_hashfrag"
    else:
        # Cached run
        base = "alphagenome_k562_head_hashfrag_scaling_cached"

    head_name = f"{base}_{arch_slug}_v4"
    return head_name, arch


def _load_checkpoint(model, ckpt_path: Path) -> None:
    from collections.abc import Mapping

    def _merge(base, override):
        if not isinstance(override, Mapping):
            return override
        if not isinstance(base, Mapping):
            return override
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                merged[k] = _merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    checkpointer = ocp.StandardCheckpointer()
    loaded_params, _ = checkpointer.restore(ckpt_path)
    model._params = jax.device_put(_merge(model._params, loaded_params))


def _predict_sequences(predict_fn, params, state, seqs: list[str]) -> np.ndarray:
    """RC-averaged inference for a list of raw sequence strings."""
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    flank5 = MPRA_UPSTREAM[-200:]
    flank3 = MPRA_DOWNSTREAM[:200]

    def encode(seq_str):
        core = _one_hot(seq_str, 200)
        f5 = _one_hot(flank5, 200)
        f3 = _one_hot(flank3, 200)
        return np.concatenate([f5, core, f3], axis=0)  # (600, 4)

    x_fwd = np.stack([encode(s) for s in seqs])
    x_rev = x_fwd[:, ::-1, ::-1]
    n = len(seqs)
    preds_fwd, preds_rev = [], []
    for i in range(0, n, BATCH_SIZE):
        preds_fwd.append(
            np.array(predict_fn(params, state, jnp.array(x_fwd[i : i + BATCH_SIZE]))).reshape(-1)
        )
        preds_rev.append(
            np.array(predict_fn(params, state, jnp.array(x_rev[i : i + BATCH_SIZE]))).reshape(-1)
        )
    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def evaluate_ood(
    run_dir: Path,
    ood_path: Path,
    weights_path: str,
    head_name: str,
    head_arch: str,
    device,
) -> dict:
    """Load checkpoint in run_dir and evaluate on OOD test set."""
    ckpt_path = run_dir / "best_model" / "checkpoint"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    register_s2f_head(
        head_name=head_name,
        arch=head_arch,
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.0,
    )
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
        device=device,
    )

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
        )[head_name]

    _load_checkpoint(model, ckpt_path)
    params, state = model._params, model._state

    df = pd.read_csv(ood_path, sep="\t")
    seqs = df["sequence"].tolist()
    labels = df["K562_log2FC"].to_numpy(dtype=np.float32)
    print(f"  OOD: {len(seqs):,} sequences", flush=True)

    preds = _predict_sequences(predict_step, params, state, seqs)
    r = pearsonr(labels, preds)[0]
    rho = spearmanr(labels, preds)[0]
    mse = float(np.mean((preds - labels) ** 2))
    return {"pearson_r": float(r), "spearman_r": float(rho), "mse": mse, "n": len(seqs)}


def needs_reeval(result: dict) -> bool:
    ood = result.get("test_metrics", {}).get("ood", {})
    return ood.get("n", 0) != CORRECT_OOD_N


def process_run(
    run_dir: Path,
    ood_path: Path,
    weights_path: str,
    device,
    head_name_override: str | None = None,
    head_arch_override: str | None = None,
) -> bool:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return False
    result = json.load(open(result_path))
    if not needs_reeval(result):
        print(f"  {run_dir.name}: OOD already correct (n={CORRECT_OOD_N}), skipping.")
        return False

    head_name = head_name_override
    head_arch = head_arch_override
    if head_name is None or head_arch is None:
        head_name, head_arch = _head_name_from_result(result, run_dir)

    print(f"\n=== {run_dir} ===")
    print(f"  head={head_name}  arch={head_arch}")
    old_ood = result.get("test_metrics", {}).get("ood", {})
    print(
        f"  Old OOD: n={old_ood.get('n', 0)}  pearson={old_ood.get('pearson_r', float('nan')):.4f}"
    )

    ood_metrics = evaluate_ood(run_dir, ood_path, weights_path, head_name, head_arch, device)
    print(
        f"  New OOD: n={ood_metrics['n']}  pearson={ood_metrics['pearson_r']:.4f}  spearman={ood_metrics['spearman_r']:.4f}"
    )

    result["test_metrics"]["ood"] = ood_metrics
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Wrote updated result.json", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-dir", type=Path, help="Scan this directory tree for result.json files"
    )
    parser.add_argument("--run-dir", type=Path, help="Evaluate a single run directory")
    parser.add_argument("--head-name", type=str, help="Override head name (for --run-dir)")
    parser.add_argument("--head-arch", type=str, default="boda-flatten-512-512")
    parser.add_argument("--k562-data-path", type=Path, required=True)
    parser.add_argument("--weights-path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    try:
        device = jax.devices("gpu")[0]
        print(f"Using GPU: {device}")
    except RuntimeError:
        device = jax.devices("cpu")[0]
        print(f"No GPU — using CPU: {device}")

    ood_path = args.k562_data_path / "test_sets" / OOD_FNAME
    if not ood_path.exists():
        raise FileNotFoundError(f"OOD file not found: {ood_path}")
    print(f"OOD file: {ood_path} ({sum(1 for _ in open(ood_path)) - 1:,} sequences)")

    weights_path = str(Path(args.weights_path).expanduser().resolve())

    if args.run_dir:
        process_run(args.run_dir, ood_path, weights_path, device, args.head_name, args.head_arch)
    elif args.scan_dir:
        # Find all result.json under scan_dir
        results = sorted(args.scan_dir.glob("**/result.json"))
        print(f"Found {len(results)} result.json files in {args.scan_dir}")
        n_reeval = 0
        for result_path in results:
            run_dir = result_path.parent
            if process_run(run_dir, ood_path, weights_path, device):
                n_reeval += 1
        print(f"\nDone. Re-evaluated {n_reeval} run(s).")
    else:
        parser.error("Provide --scan-dir or --run-dir")


if __name__ == "__main__":
    main()
