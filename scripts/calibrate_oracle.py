#!/usr/bin/env python
"""Post-hoc calibration of the AG S1 oracle to fix absolute scale bias.

The baseline AG S1 oracle (hashfrag 10-fold ensemble) achieves excellent ranking
(in-dist Pearson ~0.935) but predicts random DNA at mean ~0.84 instead of the
expected ~-0.53 (Agarwal et al. shuffled controls mean).

This script:
  1. Loads the AG S1 oracle ensemble (checkpoint-only, no encoder forward pass).
  2. Runs oracle predictions on the 249 Agarwal dinucleotide-shuffled controls
     (which are NOT in the K562 training set — they come from a different library).
  3. Fits four calibration functions mapping oracle_pred -> real_MPRA_value:
       a) Isotonic regression (monotonic, non-parametric)
       b) Platt scaling (linear a*x + b)
       c) Quantile mapping (empirical CDF alignment)
       d) Piecewise linear (separate slopes for low/mid/high)
  4. Evaluates each calibration method on:
       - Agarwal shuffled controls (calibration set — sanity check)
       - Random DNA (10K sequences, seed=42)
       - In-distribution test set (hashfrag)
       - OOD designed test set
  5. Reports mean, std, skewness before/after calibration for random DNA.
  6. Reports Pearson r on in-dist and OOD (should be UNCHANGED by calibration).

Usage (local, requires JAX + alphagenome):
    python scripts/calibrate_oracle.py

Usage (SLURM):
    sbatch scripts/slurm/calibrate_oracle.sh
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent

# ── Agarwal data paths ────────────────────────────────────────────────────────
AGARWAL_DIR = REPO / "data" / "agarwal_2025"
SHUFFLED_CONTROLS_CSV = AGARWAL_DIR / "k562_dinucleotide_shuffled_controls.csv"

# ── Oracle checkpoint dir ─────────────────────────────────────────────────────
ORACLE_DIR = REPO / "outputs" / "ag_hashfrag_oracle_cached"

# ── Test set paths ────────────────────────────────────────────────────────────
TEST_SET_DIR = REPO / "data" / "k562" / "test_sets"


# ── Sequence encoding ─────────────────────────────────────────────────────────


def _load_flanks() -> tuple[np.ndarray, np.ndarray]:
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]

    def _enc(seq: str) -> np.ndarray:
        arr = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(seq):
            if c in mapping:
                arr[i, mapping[c]] = 1.0
        return arr

    return _enc(flank_5), _enc(flank_3)


def encode_200bp_with_flanks(
    seq: str,
    f5: np.ndarray,
    f3: np.ndarray,
) -> np.ndarray:
    """Encode a 200 bp insert (padded/clipped) as (600, 4) one-hot."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = seq.upper()
    if len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > 200:
        start = (len(seq) - 200) // 2
        seq = seq[start : start + 200]
    core = np.zeros((200, 4), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in mapping:
            core[i, mapping[c]] = 1.0
    return np.concatenate([f5, core, f3], axis=0)  # (600, 4)


# ── Oracle loader ─────────────────────────────────────────────────────────────


def load_oracle_ensemble():
    """Load the 10-fold AG S1 oracle and return a predict function."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from models.alphagenome_heads import register_s2f_head

    ckpt_paths = sorted(
        p / "best_model" / "checkpoint"
        for p in sorted(ORACLE_DIR.glob("oracle_*"))
        if (p / "best_model" / "checkpoint").exists()
    )
    if not ckpt_paths:
        raise FileNotFoundError(f"No AG oracle checkpoints in {ORACLE_DIR}")
    logger.info("Found %d oracle checkpoints", len(ckpt_paths))

    head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
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
    logger.info("Creating model with weights: %s", weights_path)
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=True,
    )

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
    params_list = []
    for ckpt_path in ckpt_paths:
        loaded_params, _ = checkpointer.restore(ckpt_path)
        model._params = jax.device_put(_merge(model._params, loaded_params))
        params_list.append(jax.device_put(model._params))
    model_state = model._state
    logger.info("Loaded %d oracle parameter sets", len(params_list))

    f5, f3 = _load_flanks()

    def predict(sequences: list[str], batch_size: int = 128) -> np.ndarray:
        """RC-averaged ensemble predictions on 200 bp sequences."""
        n = len(sequences)
        x_fwd = np.stack([encode_200bp_with_flanks(s, f5, f3) for s in sequences])
        x_rev = x_fwd[:, ::-1, ::-1]
        all_preds = []
        for params in params_list:
            pf_batches, pr_batches = [], []
            for i in range(0, n, batch_size):
                cf = jnp.array(x_fwd[i : i + batch_size])
                cr = jnp.array(x_rev[i : i + batch_size])
                pf_batches.append(np.array(predict_step(params, model_state, cf)).reshape(-1))
                pr_batches.append(np.array(predict_step(params, model_state, cr)).reshape(-1))
            fold_pred = (np.concatenate(pf_batches) + np.concatenate(pr_batches)) / 2.0
            all_preds.append(fold_pred)
        return np.stack(all_preds).mean(axis=0).astype(np.float32)

    return predict


# ── Calibration methods ───────────────────────────────────────────────────────


class PlattCalibrator:
    """Linear calibration: y_cal = a * y_raw + b."""

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, y_raw: np.ndarray, y_true: np.ndarray) -> None:
        # Least-squares fit
        X = np.column_stack([y_raw, np.ones(len(y_raw))])
        coeffs, _, _, _ = np.linalg.lstsq(X, y_true, rcond=None)
        self.a, self.b = float(coeffs[0]), float(coeffs[1])
        logger.info("Platt calibration: a=%.4f, b=%.4f", self.a, self.b)

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        return (self.a * y_raw + self.b).astype(np.float32)


class IsotonicCalibrator:
    """Monotone non-parametric calibration via isotonic regression."""

    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_raw: np.ndarray, y_true: np.ndarray) -> None:
        self._iso.fit(y_raw, y_true)
        logger.info(
            "Isotonic calibration: %d knots, pred range [%.3f, %.3f]",
            len(self._iso.X_thresholds_),
            float(self._iso.y_thresholds_.min()),
            float(self._iso.y_thresholds_.max()),
        )

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        return self._iso.predict(y_raw).astype(np.float32)


class QuantileCalibrator:
    """Quantile mapping: align oracle CDF to calibration data CDF."""

    def __init__(self, n_quantiles: int = 100) -> None:
        self.n_quantiles = n_quantiles
        self._raw_q: np.ndarray | None = None
        self._true_q: np.ndarray | None = None

    def fit(self, y_raw: np.ndarray, y_true: np.ndarray) -> None:
        qs = np.linspace(0, 100, self.n_quantiles)
        self._raw_q = np.percentile(y_raw, qs)
        self._true_q = np.percentile(y_true, qs)
        logger.info(
            "Quantile calibration: raw [%.3f, %.3f] -> true [%.3f, %.3f]",
            float(self._raw_q[0]),
            float(self._raw_q[-1]),
            float(self._true_q[0]),
            float(self._true_q[-1]),
        )

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        return np.interp(y_raw, self._raw_q, self._true_q).astype(np.float32)


class PiecewiseLinearCalibrator:
    """Piecewise linear calibration: separate slopes for low / mid / high regions."""

    def __init__(self, breakpoints: list[float] | None = None) -> None:
        # Default breakpoints at oracle prediction scale (approximate)
        self.breakpoints = breakpoints or [0.0, 1.0]
        self._segments: list[tuple[float, float]] = []  # (slope, intercept) per segment

    def fit(self, y_raw: np.ndarray, y_true: np.ndarray) -> None:
        bps = sorted(self.breakpoints)
        # Split data into regions
        regions = []
        prev = -np.inf
        for bp in bps + [np.inf]:
            mask = (y_raw > prev) & (y_raw <= bp)
            if mask.sum() >= 2:
                regions.append((y_raw[mask], y_true[mask]))
            else:
                # Fall back to global fit for empty regions
                regions.append((y_raw, y_true))
            prev = bp

        self._segments = []
        for r_raw, r_true in regions:
            X = np.column_stack([r_raw, np.ones(len(r_raw))])
            coeffs, _, _, _ = np.linalg.lstsq(X, r_true, rcond=None)
            self._segments.append((float(coeffs[0]), float(coeffs[1])))

        logger.info("Piecewise linear: %d segments", len(self._segments))
        for i, (a, b) in enumerate(self._segments):
            logger.info("  Segment %d: slope=%.4f, intercept=%.4f", i, a, b)

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        out = np.empty_like(y_raw, dtype=np.float32)
        bps = sorted(self.breakpoints)
        prev = -np.inf
        for seg_idx, bp in enumerate(bps + [np.inf]):
            mask = (y_raw > prev) & (y_raw <= bp)
            a, b = self._segments[seg_idx]
            out[mask] = a * y_raw[mask] + b
            prev = bp
        return out


# ── Evaluation helpers ────────────────────────────────────────────────────────


def compute_stats(preds: np.ndarray, label: str) -> dict:
    from scipy.stats import pearsonr, skew, spearmanr

    d = {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "skewness": float(skew(preds)),
        "min": float(np.min(preds)),
        "max": float(np.max(preds)),
        "n": int(len(preds)),
    }
    logger.info(
        "  %s: mean=%.4f, std=%.4f, skew=%.4f",
        label,
        d["mean"],
        d["std"],
        d["skewness"],
    )
    return d


def compute_metrics(preds: np.ndarray, true: np.ndarray, label: str) -> dict:
    from scipy.stats import pearsonr, spearmanr

    def _safe(fn, a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(fn(a, b)[0])

    d = {
        "pearson_r": _safe(pearsonr, preds, true),
        "spearman_r": _safe(spearmanr, preds, true),
        "mse": float(np.mean((preds - true) ** 2)),
        "mean_pred": float(np.mean(preds)),
        "std_pred": float(np.std(preds)),
        "n": int(len(preds)),
    }
    logger.info(
        "  %s: pearson=%.4f, spearman=%.4f, mse=%.4f, mean_pred=%.4f",
        label,
        d["pearson_r"],
        d["spearman_r"],
        d["mse"],
        d["mean_pred"],
    )
    return d


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import os

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    output_dir = REPO / "outputs" / "oracle_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load calibration data (Agarwal shuffled controls) ─────────────────
    logger.info("Loading Agarwal shuffled controls from %s", SHUFFLED_CONTROLS_CSV)
    shuf_df = pd.read_csv(SHUFFLED_CONTROLS_CSV)
    # Drop any rows with missing log2_mean
    shuf_df = shuf_df.dropna(subset=["log2_mean", "element_200nt"])
    cal_seqs = shuf_df["element_200nt"].tolist()
    cal_true = shuf_df["log2_mean"].to_numpy(dtype=np.float32)
    logger.info(
        "Calibration set: n=%d, true_mean=%.4f, true_std=%.4f",
        len(cal_seqs),
        float(np.mean(cal_true)),
        float(np.std(cal_true)),
    )

    # ── 2. Load test sets ─────────────────────────────────────────────────────
    logger.info("Loading test sets...")
    in_dist_df = pd.read_csv(TEST_SET_DIR / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_dist_seqs = in_dist_df["sequence"].tolist()
    in_dist_true = in_dist_df["K562_log2FC"].to_numpy(dtype=np.float32)

    ood_df = pd.read_csv(TEST_SET_DIR / "test_ood_designed_k562.tsv", sep="\t")
    ood_seqs = ood_df["sequence"].tolist()
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    logger.info("In-dist test: n=%d", len(in_dist_seqs))
    logger.info("OOD test: n=%d", len(ood_seqs))

    # ── 3. Generate random DNA sequences ─────────────────────────────────────
    logger.info("Generating random DNA sequences (N=10000, seed=42)...")
    rng = np.random.default_rng(42)
    random_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(10_000)]

    # ── 4. Load oracle and run predictions ───────────────────────────────────
    logger.info("Loading oracle ensemble...")
    predict = load_oracle_ensemble()

    logger.info("Predicting on calibration set (shuffled controls, n=%d)...", len(cal_seqs))
    cal_preds_raw = predict(cal_seqs)
    logger.info(
        "Cal preds raw: mean=%.4f, std=%.4f",
        float(np.mean(cal_preds_raw)),
        float(np.std(cal_preds_raw)),
    )

    # Use a subset of in-dist for speed (every 4th = ~10K sequences)
    STEP = 4
    in_sub_seqs = in_dist_seqs[::STEP]
    in_sub_true = in_dist_true[::STEP]
    logger.info("Predicting on in-dist subset (n=%d)...", len(in_sub_seqs))
    in_sub_preds_raw = predict(in_sub_seqs)

    logger.info("Predicting on OOD test (n=%d)...", len(ood_seqs))
    ood_preds_raw = predict(ood_seqs)

    logger.info("Predicting on random DNA (n=%d)...", len(random_seqs))
    random_preds_raw = predict(random_seqs)

    # ── 5. Fit calibration methods ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Fitting calibration methods...")

    calibrators: dict[str, object] = {
        "platt": PlattCalibrator(),
        "isotonic": IsotonicCalibrator(),
        "quantile": QuantileCalibrator(n_quantiles=100),
        "piecewise": PiecewiseLinearCalibrator(breakpoints=[0.0, 1.0]),
    }

    for name, cal in calibrators.items():
        logger.info("Fitting %s calibrator...", name)
        cal.fit(cal_preds_raw, cal_true)

    # ── 6. Evaluate all methods ───────────────────────────────────────────────
    results: dict = {
        "calibration_set_size": len(cal_seqs),
        "calibration_true_mean": float(np.mean(cal_true)),
        "calibration_true_std": float(np.std(cal_true)),
        "raw_oracle": {},
        "calibrated": {},
    }

    # Raw oracle stats
    logger.info("=" * 60)
    logger.info("RAW ORACLE STATS:")
    results["raw_oracle"]["calibration"] = compute_metrics(
        cal_preds_raw, cal_true, "raw/calibration"
    )
    results["raw_oracle"]["in_dist"] = compute_metrics(in_sub_preds_raw, in_sub_true, "raw/in_dist")
    results["raw_oracle"]["ood"] = compute_metrics(ood_preds_raw, ood_true, "raw/ood")
    results["raw_oracle"]["random_dna"] = compute_stats(random_preds_raw, "raw/random_dna")

    # Per-calibrator evaluation
    for name, cal in calibrators.items():
        logger.info("=" * 60)
        logger.info("CALIBRATED ORACLE (%s):", name.upper())
        results["calibrated"][name] = {}

        cal_preds_cal = cal.predict(cal_preds_raw)
        in_preds_cal = cal.predict(in_sub_preds_raw)
        ood_preds_cal = cal.predict(ood_preds_raw)
        rand_preds_cal = cal.predict(random_preds_raw)

        results["calibrated"][name]["calibration"] = compute_metrics(
            cal_preds_cal, cal_true, f"{name}/calibration"
        )
        results["calibrated"][name]["in_dist"] = compute_metrics(
            in_preds_cal, in_sub_true, f"{name}/in_dist"
        )
        results["calibrated"][name]["ood"] = compute_metrics(ood_preds_cal, ood_true, f"{name}/ood")
        results["calibrated"][name]["random_dna"] = compute_stats(
            rand_preds_cal, f"{name}/random_dna"
        )

    # ── 7. Save results ───────────────────────────────────────────────────────
    out_json = output_dir / "calibration_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", out_json)

    # ── 8. Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ORACLE CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"\nCalibration set: {len(cal_seqs)} Agarwal shuffled controls")
    print(f"True mean (shuffled controls): {np.mean(cal_true):.4f}")
    print()

    header = f"{'Method':<14s} {'Rand mean':>10s} {'Rand std':>10s} {'Rand skew':>10s} {'InDist R':>10s} {'OOD R':>10s} {'Cal MSE':>10s}"
    print(header)
    print("-" * len(header))

    # Raw oracle
    rr = results["raw_oracle"]
    print(
        f"{'raw':<14s}"
        f"{rr['random_dna']['mean']:>10.4f}"
        f"{rr['random_dna']['std']:>10.4f}"
        f"{rr['random_dna']['skewness']:>10.4f}"
        f"{rr['in_dist']['pearson_r']:>10.4f}"
        f"{rr['ood']['pearson_r']:>10.4f}"
        f"{rr['calibration']['mse']:>10.4f}"
    )

    # Calibrated
    for name in calibrators:
        cr = results["calibrated"][name]
        print(
            f"{name:<14s}"
            f"{cr['random_dna']['mean']:>10.4f}"
            f"{cr['random_dna']['std']:>10.4f}"
            f"{cr['random_dna']['skewness']:>10.4f}"
            f"{cr['in_dist']['pearson_r']:>10.4f}"
            f"{cr['ood']['pearson_r']:>10.4f}"
            f"{cr['calibration']['mse']:>10.4f}"
        )

    print()
    print("Expected after calibration:")
    print("  random_dna mean ≈ -0.53 (Agarwal shuffled controls mean)")
    print("  in_dist Pearson r ≈ unchanged (calibration should NOT affect ranking)")
    print("  ood Pearson r ≈ unchanged")
    print()

    # ── 9. Save calibration parameters for downstream use ────────────────────
    # Save the best calibrator (Platt is simplest and most interpretable)
    import pickle

    for name, cal in calibrators.items():
        with open(output_dir / f"calibrator_{name}.pkl", "wb") as f:
            pickle.dump(cal, f)
    logger.info("Saved calibrators to %s", output_dir)

    # Save also the raw and calibrated predictions for round 2 analysis
    np.savez_compressed(
        output_dir / "calibration_predictions.npz",
        cal_preds_raw=cal_preds_raw,
        cal_true=cal_true,
        in_sub_preds_raw=in_sub_preds_raw,
        in_sub_true=in_sub_true,
        ood_preds_raw=ood_preds_raw,
        ood_true=ood_true,
        random_preds_raw=random_preds_raw,
    )
    logger.info("Saved raw predictions to %s", output_dir / "calibration_predictions.npz")

    logger.info("Done.")


if __name__ == "__main__":
    main()
