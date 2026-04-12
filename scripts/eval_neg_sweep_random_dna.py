#!/usr/bin/env python
"""Evaluate oracle neg-aug S2 configs on random DNA, shuffled controls, and intergenic sequences.

For each config:
  1. Load the S2 checkpoint (full encoder + head)
  2. Predict on 500 random DNA sequences (200bp)
  3. Predict on 250 Agarwal dinuc-shuffled controls
  4. Predict on 200 Agarwal ernst_negative (intergenic-like) sequences
  5. Report mean, std, % > 0 for each sequence type
  6. Print existing test metrics (in_dist, OOD, SNV) from test_metrics.json

Usage:
    uv run --no-sync python scripts/eval_neg_sweep_random_dna.py

Environment:
    ALPHAGENOME_WEIGHTS: path to AlphaGenome weights directory
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configs to evaluate ──────────────────────────────────────────────────────
CONFIGS = [
    ("var_tight_i5d2", "outputs/oracle_neg_sweep/var_tight_i5d2/fold_0"),
    ("var_ar1a_w2x", "outputs/oracle_neg_sweep/var_ar1a_w2x/fold_0"),
    ("var_d20_tight", "outputs/oracle_neg_sweep/var_d20_tight/fold_0"),
    ("var_10pct_real_inter", "outputs/oracle_neg_sweep/var_10pct_real_inter/fold_0"),
]


def load_sequences():
    """Load random DNA, shuffled controls, and intergenic sequences."""
    # 1. Random DNA (500 sequences, 200bp)
    rng = np.random.default_rng(42)
    random_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(500)]

    # 2. Agarwal controls: shuffled + ernst_negative (intergenic)
    controls_path = REPO / "data" / "agarwal_2025" / "k562_all_controls_200bp.tsv"
    shuffled_seqs = []
    intergenic_seqs = []

    with open(controls_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["category"] == "shuffled_negative":
                shuffled_seqs.append(row["sequence"])
            elif row["category"] == "ernst_negative":
                intergenic_seqs.append(row["sequence"])

    logger.info(
        "Loaded: %d random, %d shuffled, %d intergenic (ernst_negative)",
        len(random_seqs),
        len(shuffled_seqs),
        len(intergenic_seqs),
    )
    return random_seqs, shuffled_seqs, intergenic_seqs


def load_s2_model(ckpt_dir: str, config_name: str):
    """Load a Stage 2 oracle model from its best_model checkpoint."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params

    weights_path = os.environ["ALPHAGENOME_WEIGHTS"]

    # Head name matches the training config (stage2_k562_oracle.yaml)
    head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"

    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.1,
    )

    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=False,
    )
    reinit_head_params(model, head_name, num_tokens=5, dim=1536, rng=42)

    # Restore best_model checkpoint
    ckpt_path = Path(ckpt_dir) / "best_model" / "checkpoint"
    if ckpt_path.exists():
        checkpointer = ocp.StandardCheckpointer()
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(loaded_params)
        logger.info("Loaded checkpoint: %s", ckpt_path)
    else:
        # Try CheckpointManager format
        ckpt_mgr_path = Path(ckpt_dir) / "best_model"
        if ckpt_mgr_path.exists():
            mgr = ocp.CheckpointManager(str(ckpt_mgr_path.resolve()))
            loaded_params = mgr.restore(
                mgr.latest_step(),
                args=ocp.args.StandardRestore(model._params),
            )
            model._params = jax.device_put(loaded_params)
            logger.info("Loaded checkpoint (mgr): %s", ckpt_mgr_path)
        else:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}/best_model")

    @jax.jit
    def predict_step(params, state, sequences):
        return model._predict(
            params,
            state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
            requested_outputs=[head_name],
        )[head_name]

    return model, predict_step, head_name


def predict_sequences(model, predict_step_fn, seqs: list[str], batch_size: int = 128) -> np.ndarray:
    """RC-averaged predictions on 200bp strings via 600bp flanked context."""
    import jax.numpy as jnp

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]

    def encode_600bp(seq_200):
        core = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(seq_200[:200].upper()):
            if c in mapping:
                core[i, mapping[c]] = 1.0
        f5 = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(flank_5):
            if c in mapping:
                f5[i, mapping[c]] = 1.0
        f3 = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(flank_3):
            if c in mapping:
                f3[i, mapping[c]] = 1.0
        return np.concatenate([f5, core, f3], axis=0)  # (600, 4)

    x_fwd = np.stack([encode_600bp(s) for s in seqs])
    x_rev = x_fwd[:, ::-1, ::-1].copy()

    params, state = model._params, model._state
    preds_fwd, preds_rev = [], []
    for i in range(0, len(x_fwd), batch_size):
        j = min(i + batch_size, len(x_fwd))
        pf = np.array(predict_step_fn(params, state, jnp.array(x_fwd[i:j]))).reshape(-1)[: j - i]
        pr = np.array(predict_step_fn(params, state, jnp.array(x_rev[i:j]))).reshape(-1)[: j - i]
        preds_fwd.append(pf)
        preds_rev.append(pr)
    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def report_stats(name: str, preds: np.ndarray):
    """Print stats for a set of predictions."""
    print(
        "    %-25s  mean=%+.3f  std=%.3f  %%>0=%.1f%%  min=%.3f  max=%.3f"
        % (
            name,
            np.mean(preds),
            np.std(preds),
            100 * np.mean(preds > 0),
            np.min(preds),
            np.max(preds),
        )
    )


def main():
    random_seqs, shuffled_seqs, intergenic_seqs = load_sequences()

    out_dir = REPO / "outputs" / "neg_sweep_random_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for config_name, ckpt_dir in CONFIGS:
        full_ckpt_dir = REPO / ckpt_dir
        print("\n" + "=" * 70)
        print("CONFIG: %s" % config_name)
        print("  ckpt: %s" % ckpt_dir)
        print("=" * 70)

        # Check for existing test metrics
        tm_path = full_ckpt_dir / "test_metrics.json"
        if tm_path.exists():
            with open(tm_path) as f:
                tm = json.load(f)
            test_m = tm.get("test_metrics", tm)
            print("  [Test metrics from training]")
            for k, v in test_m.items():
                if isinstance(v, dict) and "pearson_r" in v:
                    print("    %-20s  pearson=%.4f" % (k, v["pearson_r"]))
        else:
            print("  [No test_metrics.json found]")

        # Load model
        t0 = time.time()
        try:
            model, predict_step_fn, head_name = load_s2_model(str(full_ckpt_dir), config_name)
        except FileNotFoundError as e:
            print("  SKIP: %s" % e)
            continue
        logger.info("Model loaded in %.1f s", time.time() - t0)

        # Predict
        t0 = time.time()
        rand_preds = predict_sequences(model, predict_step_fn, random_seqs)
        shuf_preds = predict_sequences(model, predict_step_fn, shuffled_seqs)
        inter_preds = predict_sequences(model, predict_step_fn, intergenic_seqs)
        logger.info("Predictions done in %.1f s", time.time() - t0)

        print("  [Random/Control predictions]")
        report_stats("Random DNA (n=%d)" % len(random_seqs), rand_preds)
        report_stats("Dinuc-shuffled (n=%d)" % len(shuffled_seqs), shuf_preds)
        report_stats("Intergenic/ernst (n=%d)" % len(intergenic_seqs), inter_preds)

        all_results[config_name] = {
            "random": {
                "mean": float(np.mean(rand_preds)),
                "std": float(np.std(rand_preds)),
                "pct_positive": float(100 * np.mean(rand_preds > 0)),
            },
            "shuffled": {
                "mean": float(np.mean(shuf_preds)),
                "std": float(np.std(shuf_preds)),
                "pct_positive": float(100 * np.mean(shuf_preds > 0)),
            },
            "intergenic": {
                "mean": float(np.mean(inter_preds)),
                "std": float(np.std(inter_preds)),
                "pct_positive": float(100 * np.mean(inter_preds > 0)),
            },
        }

        # Free GPU memory
        del model, predict_step_fn
        import gc

        gc.collect()

    # Save all results
    with open(out_dir / "neg_sweep_random_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(
        "%-22s  %10s %10s %10s  %10s %10s  %10s %10s"
        % (
            "Config",
            "Rand mean",
            "Rand %>0",
            "Rand std",
            "Shuf mean",
            "Shuf %>0",
            "Inter mean",
            "Inter %>0",
        )
    )
    print("-" * 90)
    for name, r in all_results.items():
        print(
            "%-22s  %+10.3f %9.1f%% %10.3f  %+10.3f %9.1f%%  %+10.3f %9.1f%%"
            % (
                name,
                r["random"]["mean"],
                r["random"]["pct_positive"],
                r["random"]["std"],
                r["shuffled"]["mean"],
                r["shuffled"]["pct_positive"],
                r["intergenic"]["mean"],
                r["intergenic"]["pct_positive"],
            )
        )
    print("\nExpected: shuffled controls measured mean ~ -0.53 (Gosai/Agarwal)")
    print("Saved: %s" % (out_dir / "neg_sweep_random_eval.json"))


if __name__ == "__main__":
    main()
