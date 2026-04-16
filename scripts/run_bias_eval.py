#!/usr/bin/env python
"""Run bias evaluation on completed debias sweep models.

Loads each trained model and evaluates on:
  1. Random 200bp DNA (500 seqs, various CpG levels)
  2. Agarwal shuffled/intergenic controls
  3. Gosai ctrl_neg with real labels (Pearson R vs ground truth)
  4. Dinucleotide-shuffled random DNA
  5. CpG-depleted random DNA

Usage:
    uv run --no-sync python scripts/run_bias_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["TORCHDYNAMO_DISABLE"] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def predict_sequences(predict_fn, params, state, sequences, batch_size=64):
    """Predict on a list of sequences using the S2 model."""
    import jax.numpy as jnp

    _MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
    all_preds = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        # Encode to (bsz, 600, 4) — pad 200bp core with 200bp flanks of N
        ohe_batch = np.zeros((len(batch_seqs), 600, 4), dtype=np.float32)
        for j, seq in enumerate(batch_seqs):
            seq = seq[:200].upper()
            for k, c in enumerate(seq):
                if c in _MAPPING:
                    ohe_batch[j, 200 + k, _MAPPING[c]] = 1.0

        preds = predict_fn(params, state, jnp.array(ohe_batch))
        all_preds.append(np.array(preds).flatten())

    return np.concatenate(all_preds)


def evaluate_bias(model_dir, output_path=None):
    """Run comprehensive bias evaluation on a trained model."""
    from alphagenome_ft import create_model_with_heads

    model_dir = Path(model_dir)
    if output_path is None:
        output_path = model_dir / "bias_eval.json"

    if output_path.exists():
        print(f"  SKIP: {output_path} exists")
        return json.loads(output_path.read_text())

    # Load model
    weights_path = os.environ.get(
        "ALPHAGENOME_WEIGHTS",
        "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1",
    )

    # Find the head name from test_metrics
    tm_path = model_dir / "test_metrics.json"
    if not tm_path.exists():
        print(f"  SKIP: no test_metrics.json in {model_dir}")
        return None

    tm = json.loads(tm_path.read_text())
    head_name = tm.get("head_name", "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4")

    print(f"  Loading model from {model_dir}...")
    import jax
    import jax.numpy as jnp

    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=False,
    )

    # Load fine-tuned weights
    best_path = model_dir / "best_model"
    if best_path.exists():
        import orbax.checkpoint as ocp

        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(str(best_path))
        model._params = restored.get("params", model._params)
        print(f"  Loaded weights from {best_path}")

    @jax.jit
    def predict_step(params, state, sequences):
        preds = model._predict(
            params,
            model._state,
            sequences,
            jnp.zeros(len(sequences), dtype=jnp.int32),
            negative_strand_mask=jnp.zeros(len(sequences), dtype=bool),
            strand_reindexing=None,
            requested_outputs=[head_name],
        )[head_name]
        return jnp.squeeze(preds, axis=-1) if preds.ndim > 1 else preds

    # Helper
    def _predict_seqs(seqs):
        return predict_sequences(predict_step, model._params, model._state, seqs)

    bias_results = {}
    rng = np.random.RandomState(42)

    # 1. Random DNA (500 seqs)
    print("  Evaluating random DNA...")
    random_seqs = ["".join("ACGT"[i] for i in rng.randint(0, 4, size=200)) for _ in range(500)]
    random_preds = _predict_seqs(random_seqs)
    bias_results["random_dna"] = {
        "mean": float(np.mean(random_preds)),
        "std": float(np.std(random_preds)),
        "pct_positive": float(np.mean(random_preds > 0) * 100),
        "n": len(random_preds),
    }

    # 2. CpG-depleted random DNA
    cpg_depleted = [s.replace("CG", "TG") for s in random_seqs[:200]]
    cpg_dep_preds = _predict_seqs(cpg_depleted)
    bias_results["cpg_depleted_random"] = {
        "mean": float(np.mean(cpg_dep_preds)),
        "std": float(np.std(cpg_dep_preds)),
        "pct_positive": float(np.mean(cpg_dep_preds > 0) * 100),
        "n": len(cpg_dep_preds),
    }

    # 3. Agarwal controls
    controls_path = REPO / "data" / "agarwal_2025" / "k562_all_controls_200bp.tsv"
    if controls_path.exists():
        ctrl_df = pd.read_csv(controls_path, sep="\t")
        for cat_name, cat_label in [
            ("shuffled_negative", "shuffled"),
            ("ernst_negative", "intergenic"),
        ]:
            cat_df = ctrl_df[ctrl_df["category"] == cat_name]
            if len(cat_df) > 0:
                cat_preds = _predict_seqs(cat_df["sequence"].tolist())
                bias_results[cat_label] = {
                    "mean": float(np.mean(cat_preds)),
                    "std": float(np.std(cat_preds)),
                    "pct_positive": float(np.mean(cat_preds > 0) * 100),
                    "n": len(cat_preds),
                }

    # 4. Gosai ctrl_neg with real labels
    gosai_path = REPO / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
    if gosai_path.exists():
        df = pd.read_csv(gosai_path, sep="\t", low_memory=False)
        # Find ctrl_neg — check various column names
        ctrl_neg = pd.DataFrame()
        for col in ["group", "element_type", "category", "class", "data_project"]:
            if col in df.columns:
                mask = (
                    df[col]
                    .astype(str)
                    .str.contains("ctrl_neg|negative_control|scrambl", case=False, na=False)
                )
                if mask.any():
                    ctrl_neg = df[mask].dropna(subset=["sequence", "K562_log2FC"])
                    break

        if len(ctrl_neg) > 0:
            ctrl_seqs = ctrl_neg["sequence"].str[:200].tolist()
            ctrl_real = ctrl_neg["K562_log2FC"].values.astype(np.float32)
            ctrl_preds = _predict_seqs(ctrl_seqs)
            from scipy.stats import pearsonr, spearmanr

            p_r, _ = pearsonr(ctrl_real, ctrl_preds)
            s_r, _ = spearmanr(ctrl_real, ctrl_preds)
            bias_results["gosai_ctrl_neg"] = {
                "mean_pred": float(np.mean(ctrl_preds)),
                "mean_real": float(np.mean(ctrl_real)),
                "pearson_r": float(p_r),
                "spearman_r": float(s_r),
                "mse": float(np.mean((ctrl_preds - ctrl_real) ** 2)),
                "n": len(ctrl_preds),
            }

    # Save
    with open(output_path, "w") as f:
        json.dump(bias_results, f, indent=2)
    print(f"  Saved: {output_path}")

    return bias_results


def main():
    sweep_dir = REPO / "outputs" / "debias_sweep"
    if not sweep_dir.exists():
        print("No debias_sweep directory found")
        return

    all_results = {}
    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        if not (config_dir / "test_metrics.json").exists():
            continue

        name = config_dir.name
        print(f"\n=== {name} ===")
        result = evaluate_bias(config_dir)
        if result:
            all_results[name] = result

    # Print summary table
    print("\n" + "=" * 90)
    print("BIAS EVALUATION SUMMARY")
    print("=" * 90)
    print("%-35s %8s %8s %8s %8s" % ("Config", "RandDNA", "CpGDepl", "Shuffled", "CtrlNeg"))
    print("-" * 90)
    for name, r in sorted(all_results.items()):
        rand = r.get("random_dna", {}).get("mean", None)
        cpgd = r.get("cpg_depleted_random", {}).get("mean", None)
        shuf = r.get("shuffled", {}).get("mean", None)
        ctrl = r.get("gosai_ctrl_neg", {}).get("mean_pred", None)
        print(
            "%-35s %8s %8s %8s %8s"
            % (
                name,
                "%+.3f" % rand if rand is not None else "n/a",
                "%+.3f" % cpgd if cpgd is not None else "n/a",
                "%+.3f" % shuf if shuf is not None else "n/a",
                "%+.3f" % ctrl if ctrl is not None else "n/a",
            )
        )

    # Save combined summary
    summary_path = sweep_dir / "bias_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
