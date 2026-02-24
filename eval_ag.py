#!/usr/bin/env python
"""Evaluate a trained AlphaGenome K562 head on ID / OOD / SNV test sets.

Usage:
  python eval_ag.py <ckpt_dir> <head_name> [arch]

Examples:
  # Legacy mlp or pool-flatten (arch inferred)
  python eval_ag.py outputs/oracle_alphagenome_k562/mlp_512_512/last_model alphagenome_k562_head

  # Boda heads: pass full unique head name (must match checkpoint); arch optional.
  python eval_ag.py outputs/ag_flatten/best_model alphagenome_k562_head_boda_flatten_512_512_v4
  python eval_ag.py outputs/ag_sum/best_model alphagenome_k562_head_boda_sum_512_512_v4 boda-sum-512-512
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from alphagenome_ft import create_model_with_heads
from scipy.stats import pearsonr, spearmanr


def _arch_from_head_name(head_name: str) -> str:
    """Infer HeadArch from head name for checkpoint compatibility (v4 + hidden_0/1)."""
    h = head_name.lower()
    if "pool" in h and "flatten" in h:
        return "pool-flatten"
    # More specific patterns must come before generic ones
    if "boda_flatten_1024_512" in h or "boda-flatten-1024-512" in h:
        return "boda-flatten-1024-512"
    if "boda_flatten_1024" in h or "boda-flatten-1024" in h:
        return "boda-flatten-1024-dropout"
    if "boda_flatten_512_256" in h or "boda-flatten-512-256" in h:
        return "boda-flatten-512-256"
    if "boda_flatten" in h or "boda-flatten" in h:
        return "boda-flatten-512-512"
    if "boda_sum_1024" in h or "boda-sum-1024" in h:
        return "boda-sum-1024-dropout"
    if "boda_sum" in h or "boda-sum" in h:
        return "boda-sum-512-512"
    if "boda_mean" in h or "boda-mean" in h:
        return "boda-mean-512-512"
    if "boda_max" in h or "boda-max" in h:
        return "boda-max-512-512"
    if "boda_center" in h or "boda-center" in h:
        return "boda-center-512-512"
    if "encoder_1024" in h or "encoder-1024" in h:
        return "encoder-1024-dropout"
    return "mlp-512-512"


def _safe_corr(pred, target, fn):
    if pred.size < 2 or target.size < 2 or np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def _center_pad(seq_str, target_len=384):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((len(seq_str), 4), dtype=np.float32)
    for i, c in enumerate(seq_str):
        if c in mapping:
            arr[i, mapping[c]] = 1.0

    curr_len = arr.shape[0]
    if curr_len == target_len:
        return arr
    if curr_len > target_len:
        start = (curr_len - target_len) // 2
        return arr[start : start + target_len]

    pad = np.zeros((target_len, 4), dtype=np.float32)
    left = (target_len - curr_len) // 2
    pad[left : left + curr_len, :] = arr
    return pad


def rc_seq(seq):
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    return "".join(comp.get(c, "N") for c in reversed(seq))


def evaluate(ckpt_dir: str, head_name: str, arch: str | None = None) -> dict:
    """Run K562 ID/OOD/SNV eval for a single checkpoint and head."""
    if arch is None:
        arch = _arch_from_head_name(head_name)

    from albench.models.alphagenome_heads import register_s2f_head

    register_s2f_head(head_name=head_name, arch=arch, task_mode="human", num_tracks=1)

    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
        use_encoder_output=True,
    )

    def merge_nested_dicts(base, override):
        from collections.abc import Mapping

        if not isinstance(override, Mapping):
            return override
        if not isinstance(base, Mapping):
            return override
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                merged[k] = merge_nested_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    loaded_params, _ = checkpointer.restore(Path(ckpt_dir).resolve() / "checkpoint")
    model._params = jax.device_put(merge_nested_dicts(model._params, loaded_params))

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

    def _standardize_to_200bp(sequence: str) -> str:
        target_len = 200
        curr_len = len(sequence)
        if curr_len == target_len:
            return sequence
        if curr_len < target_len:
            pad_needed = target_len - curr_len
            left_pad = pad_needed // 2
            right_pad = pad_needed - left_pad
            return "N" * left_pad + sequence + "N" * right_pad
        start = (curr_len - target_len) // 2
        return sequence[start : start + target_len]

    def _predict(seqs_str):
        if not seqs_str:
            return np.array([])
        x_fwd = np.stack([_center_pad(_standardize_to_200bp(s)) for s in seqs_str])
        x_rev = np.stack([_center_pad(rc_seq(_standardize_to_200bp(s))) for s in seqs_str])

        preds_fwd, preds_rev = [], []
        for i in range(0, len(x_fwd), 256):
            batch_params = (model._params, model._state)
            preds_fwd.append(
                np.array(predict_step(*batch_params, jnp.array(x_fwd[i : i + 256]))).reshape(-1)
            )
            preds_rev.append(
                np.array(predict_step(*batch_params, jnp.array(x_rev[i : i + 256]))).reshape(-1)
            )

        return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0

    test_set_dir = Path("data/k562/test_sets")
    metrics: dict[str, dict[str, float]] = {}

    # ID
    in_df = pd.read_csv(test_set_dir / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_pred = _predict(in_df["sequence"].astype(str).tolist())
    in_true = in_df["K562_log2FC"].to_numpy(dtype=np.float32)
    metrics["in_distribution"] = {"pearson_r": _safe_corr(in_pred, in_true, pearsonr)}

    # SNV (Absolute and Delta)
    snv_df = pd.read_csv(test_set_dir / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ref_pred = _predict(snv_df["sequence_ref"].astype(str).tolist())
    alt_pred = _predict(snv_df["sequence_alt"].astype(str).tolist())

    # Absolute metric (Primary)
    snv_abs_pred = np.concatenate([ref_pred, alt_pred])
    ref_true = snv_df["K562_log2FC_ref"].to_numpy(dtype=np.float32)
    alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
    snv_abs_true = np.concatenate([ref_true, alt_true])
    metrics["snv_abs"] = {"pearson_r": _safe_corr(snv_abs_pred, snv_abs_true, pearsonr)}

    # Delta metric (Secondary)
    delta_pred = alt_pred - ref_pred
    metrics["snv_delta"] = {
        "pearson_r": _safe_corr(
            delta_pred, snv_df["delta_log2FC"].to_numpy(dtype=np.float32), pearsonr
        )
    }

    # OOD
    ood_df = pd.read_csv(test_set_dir / "test_ood_cre.tsv", sep="\t")
    ood_pred = _predict(ood_df["sequence"].astype(str).tolist())
    metrics["ood"] = {
        "pearson_r": _safe_corr(
            ood_pred, ood_df["K562_log2FC"].to_numpy(dtype=np.float32), pearsonr
        )
    }

    return metrics


def evaluate_chrom_test(
    ckpt_dir: str,
    head_name: str,
    data_path: str = "data/k562",
    arch: str | None = None,
    cache_dir: str | None = None,
) -> dict:
    """Evaluate a trained AlphaGenome head on the chromosome test split (chr 7, 13).

    Same test set as Malinois eval. Returns pearson_r, spearman_r, mse, n.

    If ``cache_dir`` is provided and contains ``test_canonical.npy`` / ``test_rc.npy``
    (built by ``scripts/analysis/build_test_embedding_cache.py``), head-only inference
    is used — the frozen encoder is skipped entirely, making per-head eval ~10× faster.
    """
    if arch is None:
        arch = _arch_from_head_name(head_name)

    from albench.models.alphagenome_heads import register_s2f_head

    register_s2f_head(head_name=head_name, arch=arch, task_mode="human", num_tracks=1)

    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
        use_encoder_output=True,
    )

    def merge_nested_dicts(base, override):
        from collections.abc import Mapping

        if not isinstance(override, Mapping):
            return override
        if not isinstance(base, Mapping):
            return override
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], Mapping) and isinstance(v, Mapping):
                merged[k] = merge_nested_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    loaded_params, _ = checkpointer.restore(Path(ckpt_dir).resolve() / "checkpoint")
    model._params = jax.device_put(merge_nested_dicts(model._params, loaded_params))

    from albench.data.k562_full import K562FullDataset

    ds = K562FullDataset(data_path, split="test")
    labels = np.array(ds.labels, dtype=np.float32)

    # ── Fast path: head-only eval using pre-built test embedding cache ─────────
    _cache_dir = Path(cache_dir).resolve() if cache_dir else None
    can_path = _cache_dir / "test_canonical.npy" if _cache_dir else None
    rc_path = _cache_dir / "test_rc.npy" if _cache_dir else None

    if can_path is not None and can_path.exists() and rc_path.exists():
        from albench.models.embedding_cache import build_head_only_predict_fn

        head_fn = build_head_only_predict_fn(model, head_name)

        @jax.jit
        def _head_step(params, emb, org):
            out = head_fn(params, emb, org)
            return jnp.squeeze(out, axis=-1) if out.ndim > 1 else out

        test_can = np.load(can_path, mmap_mode="r")
        test_rc_arr = np.load(rc_path, mmap_mode="r")
        n = len(ds)
        preds_can: list[np.ndarray] = []
        preds_rc: list[np.ndarray] = []
        for i in range(0, n, 512):
            c = jnp.array(test_can[i : i + 512].astype(np.float32))
            r = jnp.array(test_rc_arr[i : i + 512].astype(np.float32))
            org = jnp.zeros(len(c), dtype=jnp.int32)
            preds_can.append(np.array(_head_step(model._params, c, org)).reshape(-1))
            preds_rc.append(np.array(_head_step(model._params, r, org)).reshape(-1))
        preds = (np.concatenate(preds_can) + np.concatenate(preds_rc)) / 2.0

    else:
        # ── Full encoder path (original behaviour) ─────────────────────────────
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

        def _predict(seqs_str, seq_len=600):
            if not seqs_str:
                return np.array([])
            # Use full 600bp sequences (with Addgene flanks) to match training distribution.
            x_fwd = np.stack([_center_pad(s, target_len=seq_len) for s in seqs_str])
            x_rev = np.stack([_center_pad(rc_seq(s), target_len=seq_len) for s in seqs_str])
            preds_fwd, preds_rev = [], []
            for i in range(0, len(x_fwd), 256):
                batch_params = (model._params, model._state)
                preds_fwd.append(
                    np.array(predict_step(*batch_params, jnp.array(x_fwd[i : i + 256]))).reshape(-1)
                )
                preds_rev.append(
                    np.array(predict_step(*batch_params, jnp.array(x_rev[i : i + 256]))).reshape(-1)
                )
            return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0

        seqs = [str(ds.sequences[j]) for j in range(len(ds))]
        preds = _predict(seqs)

    n = int(len(preds))
    return {
        "pearson_r": _safe_corr(preds, labels, pearsonr),
        "spearman_r": _safe_corr(preds, labels, spearmanr),
        "mse": float(np.mean((labels - preds) ** 2)),
        "n": n,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    ckpt_dir = sys.argv[1]
    head_name = sys.argv[2]
    arch = sys.argv[3] if len(sys.argv) > 3 else None

    metrics = evaluate(ckpt_dir, head_name, arch)
    print(f"\nEvaluating: {ckpt_dir}\nMetrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
