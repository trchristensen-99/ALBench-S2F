#!/usr/bin/env python
"""Round 2: Refined post-hoc calibration of the AG S1 oracle.

Key improvements over Round 1:
  1. Extended calibration data: Agarwal Table_S6 enhancers (~54K sequences
     with real K562 MPRA labels from Agarwal et al. 2025) instead of just
     the 249 shuffled controls.  S6 sequences span the FULL range of oracle
     outputs (min~-3, max~+7), enabling far better calibration than the
     narrow shuffled-control range used in Round 1.

  2. A "constrained Platt" variant that anchors the random-DNA anchor point
     (expected mean=-0.53 from shuffled controls) to ensure the calibrated
     random-DNA mean matches expectations.

  3. Full test-suite evaluation: in-dist, SNV abs, SNV delta, OOD.

  4. Calibration fit diagnosed on separate validation split of S6 data.

Round 1 key finding: Platt (linear) is the ONLY method that preserves
ranking (Pearson r unchanged).  Isotonic/quantile/piecewise all hurt ranking
because they bend the monotonic mapping outside the calibration data range.
=> Round 2 focuses on improving the Platt calibration with better data.

Usage:
    sbatch scripts/slurm/calibrate_oracle_r2.sh

Outputs in outputs/oracle_calibration_r2/:
  calibration_results_r2.json   — summary metrics
  calibration_predictions_r2.npz — raw + calibrated prediction arrays
  calibrator_*.pkl               — fitted calibrator objects
"""

from __future__ import annotations

import json
import logging
import os
import pickle
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

AGARWAL_DIR = REPO / "data" / "agarwal_2025"
SHUFFLED_CONTROLS_CSV = AGARWAL_DIR / "k562_dinucleotide_shuffled_controls.csv"
S3_XLSX = AGARWAL_DIR / "Table_S3_large_scale_lib_design.xlsx"
S6_XLSX = AGARWAL_DIR / "Table_S6_folds_and_performance.xlsx"

ORACLE_DIR = REPO / "outputs" / "ag_hashfrag_oracle_cached"
TEST_SET_DIR = REPO / "data" / "k562" / "test_sets"

OUTPUT_DIR = REPO / "outputs" / "oracle_calibration_r2"


# ── Sequence encoding ─────────────────────────────────────────────────────────


def _load_flanks() -> tuple[np.ndarray, np.ndarray]:
    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _enc(seq: str) -> np.ndarray:
        arr = np.zeros((200, 4), dtype=np.float32)
        for i, c in enumerate(seq):
            if c in mapping:
                arr[i, mapping[c]] = 1.0
        return arr

    return _enc(MPRA_UPSTREAM[-200:]), _enc(MPRA_DOWNSTREAM[:200])


def encode_200bp(seq: str, f5: np.ndarray, f3: np.ndarray) -> np.ndarray:
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
    return np.concatenate([f5, core, f3], axis=0)


# ── Oracle loader ─────────────────────────────────────────────────────────────


def load_oracle_ensemble(batch_size: int = 128):
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

    def predict(sequences: list[str]) -> np.ndarray:
        n = len(sequences)
        x_fwd = np.stack([encode_200bp(s, f5, f3) for s in sequences])
        x_rev = x_fwd[:, ::-1, ::-1]
        all_preds = []
        for params in params_list:
            pf, pr = [], []
            for i in range(0, n, batch_size):
                cf = jnp.array(x_fwd[i : i + batch_size])
                cr = jnp.array(x_rev[i : i + batch_size])
                pf.append(np.array(predict_step(params, model_state, cf)).reshape(-1))
                pr.append(np.array(predict_step(params, model_state, cr)).reshape(-1))
            all_preds.append((np.concatenate(pf) + np.concatenate(pr)) / 2.0)
        return np.stack(all_preds).mean(axis=0).astype(np.float32)

    return predict


# ── Calibration data loader ───────────────────────────────────────────────────


def load_calibration_data() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Load calibration data from two sources.

    Returns:
        train_seqs, train_labels, val_seqs, val_labels
        - train: folds 1-8 of S6 enhancers (calibration fit)
        - val:   folds 9-10 of S6 enhancers (held-out validation)
    """
    logger.info("Loading Table_S3 (sequences)...")
    s3 = pd.read_excel(S3_XLSX, header=2)
    s3.columns = s3.iloc[0].tolist()
    s3 = s3.iloc[1:].reset_index(drop=True)
    seq_col = "230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)"

    logger.info("Loading Table_S6 (labels)...")
    s6 = pd.read_excel(S6_XLSX)

    logger.info("Merging S3 + S6...")
    merged = s6.merge(
        s3[["name", "category", seq_col]],
        left_on="Sequence ID",
        right_on="name",
        how="inner",
    )
    # Keep only enhancers (not promoters — different biology)
    merged = merged[merged["category"] == "potential enhancer"].copy()
    # Extract 200 nt element (skip 15 nt 5' adaptor)
    merged["element_200nt"] = merged[seq_col].str[15:215]
    merged = merged.dropna(subset=["element_200nt", "Observed log2(RNA/DNA)"])

    train_mask = merged["Fold"].isin(range(1, 9))  # folds 1-8
    val_mask = merged["Fold"].isin([9, 10])  # folds 9-10

    train_df = merged[train_mask]
    val_df = merged[val_mask]

    logger.info(
        "Calibration set: train=%d (folds 1-8), val=%d (folds 9-10)",
        len(train_df),
        len(val_df),
    )
    logger.info(
        "Train label stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        float(train_df["Observed log2(RNA/DNA)"].mean()),
        float(train_df["Observed log2(RNA/DNA)"].std()),
        float(train_df["Observed log2(RNA/DNA)"].min()),
        float(train_df["Observed log2(RNA/DNA)"].max()),
    )

    return (
        train_df["element_200nt"].tolist(),
        train_df["Observed log2(RNA/DNA)"].to_numpy(dtype=np.float32),
        val_df["element_200nt"].tolist(),
        val_df["Observed log2(RNA/DNA)"].to_numpy(dtype=np.float32),
    )


# ── Calibration methods ───────────────────────────────────────────────────────


class PlattCalibrator:
    """Linear a*x + b calibration (preserves Pearson r perfectly)."""

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, y_raw: np.ndarray, y_true: np.ndarray) -> None:
        X = np.column_stack([y_raw, np.ones(len(y_raw))])
        coeffs, _, _, _ = np.linalg.lstsq(X, y_true, rcond=None)
        self.a, self.b = float(coeffs[0]), float(coeffs[1])
        logger.info("Platt: a=%.4f, b=%.4f", self.a, self.b)

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        return (self.a * y_raw + self.b).astype(np.float32)

    def __repr__(self) -> str:
        return f"Platt(a={self.a:.4f}, b={self.b:.4f})"


class ConstrainedPlattCalibrator:
    """Platt scaling with an additional anchor point at random DNA.

    Fits a linear a*x + b subject to the constraint that:
      E[a * oracle(random) + b] = target_mean
    where oracle(random) is estimated from the shuffled controls.

    This gives a 2-point system if we fix the slope `a` to 1 and solve for `b`,
    or uses a weighted least-squares where the anchor gets weight `anchor_weight`.
    """

    def __init__(
        self,
        shuffled_controls_csv: Path = SHUFFLED_CONTROLS_CSV,
        anchor_weight: float = 10.0,
    ) -> None:
        self.shuffled_controls_csv = shuffled_controls_csv
        self.anchor_weight = anchor_weight
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, y_raw: np.ndarray, y_true: np.ndarray) -> None:
        # Build the anchor: shuffled control oracle predictions vs true mean
        shuf_df = pd.read_csv(self.shuffled_controls_csv)
        shuf_df = shuf_df.dropna(subset=["log2_mean"])
        shuf_true_mean = float(shuf_df["log2_mean"].mean())

        # We need oracle predictions on the shuffled controls — load from round 1 if available
        r1_npz = REPO / "outputs" / "oracle_calibration" / "calibration_predictions.npz"
        if r1_npz.exists():
            r1 = np.load(r1_npz)
            shuf_pred_mean = float(r1["cal_preds_raw"].mean())
            logger.info(
                "Anchor from round 1: oracle(shuffled)=%.4f -> true=%.4f",
                shuf_pred_mean,
                shuf_true_mean,
            )
        else:
            # Fall back to estimated mean (oracle on shuffled ~0.285 from round 1)
            shuf_pred_mean = 0.285
            logger.warning(
                "Round 1 predictions not found; using estimated shuf pred mean=%.4f",
                shuf_pred_mean,
            )

        # Weighted least squares: concatenate anchor with training data
        # Anchor is replicated `anchor_weight` * N_train times to get weight effect
        n_anchor = max(1, int(self.anchor_weight * len(y_raw)))
        x_aug = np.concatenate([y_raw, np.full(n_anchor, shuf_pred_mean, dtype=np.float32)])
        y_aug = np.concatenate([y_true, np.full(n_anchor, shuf_true_mean, dtype=np.float32)])

        X = np.column_stack([x_aug, np.ones(len(x_aug))])
        coeffs, _, _, _ = np.linalg.lstsq(X, y_aug, rcond=None)
        self.a, self.b = float(coeffs[0]), float(coeffs[1])
        logger.info(
            "ConstrainedPlatt (anchor_w=%.1f): a=%.4f, b=%.4f",
            self.anchor_weight,
            self.a,
            self.b,
        )

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        return (self.a * y_raw + self.b).astype(np.float32)

    def __repr__(self) -> str:
        return f"ConstrainedPlatt(a={self.a:.4f}, b={self.b:.4f})"


class TemperatureScaledCalibrator:
    """Fix bias only (b shift), preserve scale.

    Equivalent to: y_cal = x + b, where b is chosen so that
    E[x + b | shuffled_controls] = target_mean.
    This is the most conservative calibration: zero impact on relative rankings,
    just shifts predictions by a constant.
    """

    def __init__(self, shuffled_controls_csv: Path = SHUFFLED_CONTROLS_CSV) -> None:
        self.shuffled_controls_csv = shuffled_controls_csv
        self.shift: float = 0.0

    def fit(self, y_raw: np.ndarray, y_true: np.ndarray) -> None:
        # Estimate shift as mean(y_true) - mean(y_raw) for shuffled controls
        r1_npz = REPO / "outputs" / "oracle_calibration" / "calibration_predictions.npz"
        if r1_npz.exists():
            r1 = np.load(r1_npz)
            shuf_pred_mean = float(r1["cal_preds_raw"].mean())
        else:
            shuf_pred_mean = 0.285

        shuf_df = pd.read_csv(self.shuffled_controls_csv)
        shuf_true_mean = float(shuf_df["log2_mean"].dropna().mean())

        self.shift = shuf_true_mean - shuf_pred_mean
        logger.info(
            "ShiftOnly: shift=%.4f (shuf_true=%.4f, shuf_pred=%.4f)",
            self.shift,
            shuf_true_mean,
            shuf_pred_mean,
        )

    def predict(self, y_raw: np.ndarray) -> np.ndarray:
        return (y_raw + self.shift).astype(np.float32)

    def __repr__(self) -> str:
        return f"ShiftOnly(shift={self.shift:.4f})"


# ── Evaluation helpers ────────────────────────────────────────────────────────


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


def compute_stats(preds: np.ndarray, label: str) -> dict:
    from scipy.stats import skew

    d = {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "skewness": float(skew(preds)),
        "median": float(np.median(preds)),
        "p5": float(np.percentile(preds, 5)),
        "p95": float(np.percentile(preds, 95)),
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load calibration data ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("LOADING CALIBRATION DATA (Agarwal S6 enhancers)")
    train_seqs, train_true, val_seqs, val_true = load_calibration_data()

    # Also load shuffled controls for stats
    shuf_df = pd.read_csv(SHUFFLED_CONTROLS_CSV).dropna(subset=["log2_mean", "element_200nt"])
    shuf_seqs = shuf_df["element_200nt"].tolist()
    shuf_true = shuf_df["log2_mean"].to_numpy(dtype=np.float32)

    # ── Load test sets ────────────────────────────────────────────────────────
    logger.info("Loading test sets...")
    in_dist_df = pd.read_csv(TEST_SET_DIR / "test_in_distribution_hashfrag.tsv", sep="\t")
    in_dist_seqs = in_dist_df["sequence"].tolist()
    in_dist_true = in_dist_df["K562_log2FC"].to_numpy(dtype=np.float32)

    snv_df = (
        pd.read_csv(TEST_SET_DIR.parent / "test_snv_pairs_hashfrag.tsv", sep="\t")
        if (TEST_SET_DIR.parent / "test_snv_pairs_hashfrag.tsv").exists()
        else None
    )
    if snv_df is None:
        # Try alternative location
        snv_path = list(TEST_SET_DIR.parent.rglob("test_snv_pairs_hashfrag.tsv"))
        if snv_path:
            snv_df = pd.read_csv(snv_path[0], sep="\t")

    ood_df = pd.read_csv(TEST_SET_DIR / "test_ood_designed_k562.tsv", sep="\t")
    ood_seqs = ood_df["sequence"].tolist()
    ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)

    # Random DNA
    rng = np.random.default_rng(42)
    random_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(10_000)]

    logger.info(
        "Test sets: in_dist=%d, ood=%d, random=%d",
        len(in_dist_seqs),
        len(ood_seqs),
        len(random_seqs),
    )
    if snv_df is not None:
        logger.info("SNV pairs: %d", len(snv_df))

    # ── Load oracle ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("LOADING ORACLE ENSEMBLE")
    predict = load_oracle_ensemble()

    # ── Run predictions ───────────────────────────────────────────────────────
    # Check if Round 1 predictions can be reused for shuffled controls
    r1_npz = REPO / "outputs" / "oracle_calibration" / "calibration_predictions.npz"
    if r1_npz.exists():
        logger.info("Reusing Round 1 shuffled control predictions from %s", r1_npz)
        r1 = np.load(r1_npz)
        shuf_preds_raw = r1["cal_preds_raw"]
        logger.info(
            "Shuffled controls (from R1): mean=%.4f, std=%.4f",
            float(shuf_preds_raw.mean()),
            float(shuf_preds_raw.std()),
        )
    else:
        logger.info("Predicting on shuffled controls (n=%d)...", len(shuf_seqs))
        shuf_preds_raw = predict(shuf_seqs)

    # Predict on S6 calibration train/val (this is the main new data)
    logger.info("Predicting on S6 calibration TRAIN set (n=%d)...", len(train_seqs))
    train_preds_raw = predict(train_seqs)
    logger.info(
        "Train preds: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        float(train_preds_raw.mean()),
        float(train_preds_raw.std()),
        float(train_preds_raw.min()),
        float(train_preds_raw.max()),
    )

    logger.info("Predicting on S6 calibration VAL set (n=%d)...", len(val_seqs))
    val_preds_raw = predict(val_seqs)

    logger.info("Predicting on in-dist test (n=%d)...", len(in_dist_seqs))
    in_dist_preds_raw = predict(in_dist_seqs)

    logger.info("Predicting on OOD test (n=%d)...", len(ood_seqs))
    ood_preds_raw = predict(ood_seqs)

    logger.info("Predicting on random DNA (n=%d)...", len(random_seqs))
    random_preds_raw = predict(random_seqs)

    # SNV predictions
    snv_ref_preds_raw = snv_alt_preds_raw = snv_alt_true = None
    if snv_df is not None:
        logger.info("Predicting on SNV ref (n=%d)...", len(snv_df))
        snv_ref_preds_raw = predict(snv_df["sequence_ref"].tolist())
        logger.info("Predicting on SNV alt (n=%d)...", len(snv_df))
        snv_alt_preds_raw = predict(snv_df["sequence_alt"].tolist())
        alt_col = "K562_log2FC_alt" if "K562_log2FC_alt" in snv_df.columns else "K562_log2FC_alt"
        if alt_col in snv_df.columns:
            snv_alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
        delta_col = "delta_K562_log2FC" if "delta_K562_log2FC" in snv_df.columns else "delta_log2FC"
        if delta_col in snv_df.columns:
            snv_delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)
        else:
            snv_delta_true = None

    # ── Fit calibration methods ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FITTING CALIBRATION METHODS (on S6 enhancer train folds 1-8)")

    calibrators: dict[str, object] = {
        "platt_s6": PlattCalibrator(),
        "constrained_platt_w1": ConstrainedPlattCalibrator(anchor_weight=1.0),
        "constrained_platt_w10": ConstrainedPlattCalibrator(anchor_weight=10.0),
        "shift_only": TemperatureScaledCalibrator(),
        # Also refit Platt on shuffled controls only (as in Round 1) for comparison
        "platt_shuffled": PlattCalibrator(),
    }

    calibrators["platt_s6"].fit(train_preds_raw, train_true)
    calibrators["constrained_platt_w1"].fit(train_preds_raw, train_true)
    calibrators["constrained_platt_w10"].fit(train_preds_raw, train_true)
    calibrators["shift_only"].fit(train_preds_raw, train_true)  # Uses shuffled controls internally
    calibrators["platt_shuffled"].fit(shuf_preds_raw, shuf_true)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATING ALL METHODS")

    results: dict = {
        "calibration_data": {
            "source": "Agarwal S6 enhancers (folds 1-8)",
            "train_n": len(train_seqs),
            "val_n": len(val_seqs),
            "train_label_mean": float(np.mean(train_true)),
            "train_label_std": float(np.std(train_true)),
            "shuffled_n": len(shuf_seqs),
            "shuffled_true_mean": float(np.mean(shuf_true)),
        },
        "raw_oracle": {},
        "calibrated": {},
    }

    logger.info("--- RAW ORACLE ---")
    results["raw_oracle"]["s6_train"] = compute_metrics(train_preds_raw, train_true, "raw/s6_train")
    results["raw_oracle"]["s6_val"] = compute_metrics(val_preds_raw, val_true, "raw/s6_val")
    results["raw_oracle"]["in_dist"] = compute_metrics(
        in_dist_preds_raw, in_dist_true, "raw/in_dist"
    )
    results["raw_oracle"]["ood"] = compute_metrics(ood_preds_raw, ood_true, "raw/ood")
    results["raw_oracle"]["shuffled"] = compute_metrics(shuf_preds_raw, shuf_true, "raw/shuffled")
    results["raw_oracle"]["random_dna"] = compute_stats(random_preds_raw, "raw/random_dna")

    if snv_df is not None and snv_alt_true is not None:
        results["raw_oracle"]["snv_abs"] = compute_metrics(
            snv_alt_preds_raw, snv_alt_true, "raw/snv_abs"
        )
        if snv_delta_true is not None:
            delta_pred_raw = snv_alt_preds_raw - snv_ref_preds_raw
            results["raw_oracle"]["snv_delta"] = compute_metrics(
                delta_pred_raw, snv_delta_true, "raw/snv_delta"
            )

    for name, cal in calibrators.items():
        logger.info("--- %s ---", name.upper())
        results["calibrated"][name] = {"calibrator": repr(cal)}

        s6_val_cal = cal.predict(val_preds_raw)
        in_cal = cal.predict(in_dist_preds_raw)
        ood_cal = cal.predict(ood_preds_raw)
        rand_cal = cal.predict(random_preds_raw)
        shuf_cal = cal.predict(shuf_preds_raw)

        results["calibrated"][name]["s6_val"] = compute_metrics(
            s6_val_cal, val_true, f"{name}/s6_val"
        )
        results["calibrated"][name]["in_dist"] = compute_metrics(
            in_cal, in_dist_true, f"{name}/in_dist"
        )
        results["calibrated"][name]["ood"] = compute_metrics(ood_cal, ood_true, f"{name}/ood")
        results["calibrated"][name]["shuffled"] = compute_metrics(
            shuf_cal, shuf_true, f"{name}/shuffled"
        )
        results["calibrated"][name]["random_dna"] = compute_stats(rand_cal, f"{name}/random_dna")

        if snv_df is not None and snv_alt_true is not None:
            snv_alt_cal = cal.predict(snv_alt_preds_raw)
            results["calibrated"][name]["snv_abs"] = compute_metrics(
                snv_alt_cal, snv_alt_true, f"{name}/snv_abs"
            )
            if snv_delta_true is not None:
                delta_cal = cal.predict(snv_alt_preds_raw) - cal.predict(snv_ref_preds_raw)
                results["calibrated"][name]["snv_delta"] = compute_metrics(
                    delta_cal, snv_delta_true, f"{name}/snv_delta"
                )

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = OUTPUT_DIR / "calibration_results_r2.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", out_json)

    # Save calibrators
    for name, cal in calibrators.items():
        with open(OUTPUT_DIR / f"calibrator_{name}.pkl", "wb") as f:
            pickle.dump(cal, f)

    # Save predictions
    save_dict = {
        "train_preds_raw": train_preds_raw,
        "train_true": train_true,
        "val_preds_raw": val_preds_raw,
        "val_true": val_true,
        "in_dist_preds_raw": in_dist_preds_raw,
        "in_dist_true": in_dist_true,
        "ood_preds_raw": ood_preds_raw,
        "ood_true": ood_true,
        "random_preds_raw": random_preds_raw,
        "shuf_preds_raw": shuf_preds_raw,
        "shuf_true": shuf_true,
    }
    if snv_df is not None:
        save_dict["snv_ref_preds_raw"] = snv_ref_preds_raw
        save_dict["snv_alt_preds_raw"] = snv_alt_preds_raw
    np.savez_compressed(OUTPUT_DIR / "calibration_predictions_r2.npz", **save_dict)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ORACLE CALIBRATION ROUND 2 SUMMARY")
    print("=" * 90)
    print(f"\nCalibration data: {len(train_seqs)} Agarwal S6 enhancers (folds 1-8)")
    print(f"Validation: {len(val_seqs)} enhancers (folds 9-10)")
    print(f"Shuffled controls mean: {np.mean(shuf_true):.4f}")
    print(f"Target random DNA mean: {np.mean(shuf_true):.4f}")
    print()

    hdr = (
        f"{'Method':<22s} {'Rand mean':>10s} {'Rand std':>10s} "
        f"{'S6val R':>8s} {'InDist R':>10s} {'OOD R':>8s} {'S6val MSE':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))

    rr = results["raw_oracle"]
    print(
        f"{'raw':<22s}"
        f"{rr['random_dna']['mean']:>10.4f}"
        f"{rr['random_dna']['std']:>10.4f}"
        f"{rr['s6_val']['pearson_r']:>8.4f}"
        f"{rr['in_dist']['pearson_r']:>10.4f}"
        f"{rr['ood']['pearson_r']:>8.4f}"
        f"{rr['s6_val']['mse']:>10.4f}"
    )
    for name in calibrators:
        cr = results["calibrated"][name]
        print(
            f"{name:<22s}"
            f"{cr['random_dna']['mean']:>10.4f}"
            f"{cr['random_dna']['std']:>10.4f}"
            f"{cr['s6_val']['pearson_r']:>8.4f}"
            f"{cr['in_dist']['pearson_r']:>10.4f}"
            f"{cr['ood']['pearson_r']:>8.4f}"
            f"{cr['s6_val']['mse']:>10.4f}"
        )

    print()
    print("Legend:")
    print("  platt_s6           = Platt on 43K S6 enhancers (new, full range)")
    print("  constrained_platt  = Platt with anchor at shuffled controls mean")
    print("  shift_only         = Pure additive shift (preserves all rank metrics)")
    print("  platt_shuffled     = Platt on 249 shuffled controls (Round 1)")
    print()
    print("Key: Pearson r should be IDENTICAL for all linear methods (platt, shift_only)")

    # Calibrator parameters summary
    print()
    print("Calibrator parameters:")
    for name, cal in calibrators.items():
        print(f"  {name}: {cal!r}")


if __name__ == "__main__":
    main()
