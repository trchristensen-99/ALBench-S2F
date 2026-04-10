#!/usr/bin/env python
"""Evaluate augmented oracle S1 models on val fold, random DNA, and shuffled controls."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import build_head_only_predict_fn, reinit_head_params

    weights_path = os.environ["ALPHAGENOME_WEIGHTS"]
    cache_dir = Path("outputs/oracle_full_856k/embedding_cache")

    def load_s1(ckpt_dir, hname):
        register_s2f_head(
            head_name=hname,
            arch="boda-flatten-512-512",
            task_mode="k562",
            num_tracks=1,
            hidden_dims=[512, 512],
            dropout_rate=0.1,
        )
        model = create_model_with_heads(
            "all_folds",
            heads=[hname],
            checkpoint_path=weights_path,
            use_encoder_output=True,
            detach_backbone=True,
        )
        reinit_head_params(model, hname, num_tokens=5, dim=1536, rng=42)
        hfn = build_head_only_predict_fn(model, hname)
        mgr = ocp.CheckpointManager(str(Path(ckpt_dir).resolve()))
        params = mgr.restore(0, args=ocp.args.StandardRestore(model._params))
        return params, hfn

    def predict(params, hfn, can, rc, bs=128):
        preds = []
        for i in range(0, len(can), bs):
            e = min(i + bs, len(can))
            bc = jnp.array(can[i:e].astype(np.float32))
            br = jnp.array(rc[i:e].astype(np.float32))
            oi = jnp.zeros(bc.shape[0], dtype=jnp.int32)
            p = (hfn(params, bc, oi) + hfn(params, br, oi)) / 2.0
            preds.append(np.array(p).reshape(-1)[: e - i])
        return np.concatenate(preds)

    # Load models
    configs = [
        ("Original", "outputs/oracle_full_856k/s1/oracle_0/best_model"),
        ("Finetune", "outputs/oracle_neg_curriculum/finetune/oracle_0/best_model"),
        ("Dinuc-dist", "outputs/oracle_neg_dinuc_dist/s1/oracle_0/best_model"),
    ]
    models = {}
    for name, ckpt in configs:
        try:
            p, h = load_s1(ckpt, "h_" + name.lower()[:5])
            models[name] = (p, h)
            print("Loaded %s" % name)
        except Exception as e:
            print("FAILED %s: %s" % (name, e))

    # Validation fold
    all_labels = np.load(cache_dir / "all_labels.npy")
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_labels))
    val_idx = perm[: len(all_labels) // 10]
    val_can = np.load(cache_dir / "train_canonical.npy")[val_idx]
    val_rc = np.load(cache_dir / "train_rc.npy")[val_idx]
    val_lab = all_labels[val_idx]

    print("\n=== VALIDATION FOLD (n=%d) ===" % len(val_idx))
    for name, (p, h) in models.items():
        pred = predict(p, h, val_can, val_rc)
        r = float(pearsonr(pred, val_lab)[0])
        print("  %s: r=%.4f mean=%.3f std=%.3f" % (name, r, np.mean(pred), np.std(pred)))

    # Random DNA + shuffled controls
    neg_cache = Path("outputs/oracle_neg_augmentation/neg_embed_cache")
    rand_can = np.load(neg_cache / "neg_random_canonical.npy")[:1000]
    rand_rc = np.load(neg_cache / "neg_random_rc.npy")[:1000]
    shuf_can = np.load(neg_cache / "neg_dinuc_shuffled_canonical.npy")[:250]
    shuf_rc = np.load(neg_cache / "neg_dinuc_shuffled_rc.npy")[:250]

    print("\n=== RANDOM DNA (n=1000) ===")
    for name, (p, h) in models.items():
        pred = predict(p, h, rand_can, rand_rc)
        print(
            "  %s: mean=%.3f std=%.3f pct_pos=%.1f%%"
            % (name, np.mean(pred), np.std(pred), 100 * np.mean(pred > 0))
        )

    print("\n=== DINUC-SHUFFLED CONTROLS (n=250) ===")
    for name, (p, h) in models.items():
        pred = predict(p, h, shuf_can, shuf_rc)
        print(
            "  %s: mean=%.3f std=%.3f pct_pos=%.1f%%"
            % (name, np.mean(pred), np.std(pred), 100 * np.mean(pred > 0))
        )

    print("\nExpected: shuffled controls mean approx -0.45 on Gosai scale")


if __name__ == "__main__":
    main()
