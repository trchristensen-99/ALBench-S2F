"""Score the 10-fold AG-S2 oracle ensemble on bias-relevant panels.

Generates synthetic negative controls and scores both them and the
existing eval-set panels through the actual 10-fold ensemble, producing
distribution statistics that are directly comparable to historical
single-fold pareto numbers (this script reports ENSEMBLE means, not
single-fold).

Outputs::

    outputs/oracle_pseudolabels_k562_ag_s2_refalt/bias_eval.json

Bias panels included:
  - random DNA at 5 GC levels (25/35/45/55/65/75%) — CpG shortcut probe.
    Each level: 500 i.i.d. uniform sequences with biased base frequencies
    matching the target GC.
  - dinuc-shuffled real test seqs — should preserve dinuc statistics
    while breaking regulatory motifs; pred should drop from real-seq pred.
  - 200bp uniform random (50% GC) — generic baseline.
  - real K562 test split — for comparison (label-aware).
  - all 13 eval-set parquets — distribution-shift panel.

For each panel, we report:
  - n: count
  - mean / std / pct_positive (=fraction predicted > 0)
  - quantiles (10, 50, 90)
  - if labels present: pearson_r, spearman_r, mse, ensemble_mean_vs_truth_offset

Usage:
    uv run --no-sync python scripts/preflight/score_oracle_bias.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from alphagenome_ft import create_model_with_heads
from scipy.stats import pearsonr, spearmanr

from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from models.alphagenome_heads import register_s2f_head
from models.embedding_cache import reinit_head_params

REPO = Path(__file__).resolve().parents[2]
ORACLE_DIR = REPO / "outputs" / "stage2_k562_oracle"
OUT_DIR = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt"

_FLANK_5 = MPRA_UPSTREAM[-200:]
_FLANK_3 = MPRA_DOWNSTREAM[:200]
_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def _seq_to_600(seq: str) -> np.ndarray:
    seq = seq.upper()
    if len(seq) < 200:
        pad = 200 - len(seq)
        seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
    elif len(seq) > 200:
        s = (len(seq) - 200) // 2
        seq = seq[s : s + 200]
    full = _FLANK_5 + seq + _FLANK_3
    out = np.zeros((600, 4), dtype=np.float32)
    for i, c in enumerate(full):
        if c in _MAP:
            out[i, _MAP[c]] = 1.0
    return out


def _gen_random_at_gc(n: int, gc: float, seed: int) -> list[str]:
    """Sample n i.i.d. 200bp sequences with target GC content."""
    rng = np.random.default_rng(seed)
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    bases = np.array(list("ACGT"))
    arr = rng.choice(4, size=(n, 200), p=p)
    return ["".join(bases[row]) for row in arr]


def _gen_dinuc_shuffle(seqs: list[str], seed: int) -> list[str]:
    """Dinuc-preserving Eulerian-path shuffle (same as eval-set generator)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in seqs:
        s = s.upper()
        if len(s) < 4:
            out.append(s)
            continue
        nodes = list(set(s))
        edges: dict[str, list[str]] = {n: [] for n in nodes}
        for i in range(len(s) - 1):
            edges[s[i]].append(s[i + 1])
        last_char = s[-1]
        last_edge_per_node: dict[str, str] = {}
        for n, out_list in edges.items():
            if not out_list:
                continue
            last_idx = None
            for j, t in enumerate(out_list):
                if t == last_char:
                    last_idx = j
                    break
            if last_idx is not None:
                last_edge_per_node[n] = out_list.pop(last_idx)
            rng.shuffle(out_list)
            if n in last_edge_per_node:
                out_list.append(last_edge_per_node[n])
        pos = {n: 0 for n in nodes}
        cur = s[0]
        new = [cur]
        for _ in range(len(s) - 1):
            if pos[cur] >= len(edges[cur]):
                nxt = nodes[rng.integers(0, len(nodes))]
            else:
                nxt = edges[cur][pos[cur]]
                pos[cur] += 1
            new.append(nxt)
            cur = nxt
        out.append("".join(new))
    return out


def _build_predict_step(fold_id: int, batch_size: int = 256):
    head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode="human",
        num_tracks=1,
        dropout_rate=0.1,
    )
    weights = "/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights,
        use_encoder_output=True,
        detach_backbone=False,
    )
    reinit_head_params(model, head_name, num_tokens=5, dim=1536)

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

    ckpt_path = ORACLE_DIR / f"fold_{fold_id}" / "best_model" / "checkpoint"
    loaded_params, _ = ocp.StandardCheckpointer().restore(ckpt_path)
    model._params = jax.device_put(loaded_params)
    _dummy = jnp.zeros((batch_size, 600, 4), dtype=jnp.float32)
    _ = predict_step(model._params, model._state, _dummy)
    _.block_until_ready()
    return predict_step, model._params, model._state


def _predict_seqs(predict_step, params, state, seqs: list[str], batch_size=256):
    if not seqs:
        return np.array([], dtype=np.float32)
    n = len(seqs)
    x = np.stack([_seq_to_600(s) for s in seqs])
    preds_fwd, preds_rev = [], []
    x_rev = x[:, ::-1, ::-1]
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        actual = end - i
        bf = x[i:end]
        br = x_rev[i:end]
        if actual < batch_size:
            pad = batch_size - actual
            bf = np.concatenate([bf, np.zeros((pad, 600, 4), dtype=np.float32)])
            br = np.concatenate([br, np.zeros((pad, 600, 4), dtype=np.float32)])
        pf = np.array(predict_step(params, state, jnp.array(bf))).reshape(-1)[:actual]
        pr = np.array(predict_step(params, state, jnp.array(br))).reshape(-1)[:actual]
        preds_fwd.append(pf)
        preds_rev.append(pr)
    return (np.concatenate(preds_fwd) + np.concatenate(preds_rev)) / 2.0


def _stats(preds: np.ndarray, labels: np.ndarray | None) -> dict:
    out = {
        "n": int(len(preds)),
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "median": float(np.median(preds)),
        "q10": float(np.quantile(preds, 0.1)),
        "q90": float(np.quantile(preds, 0.9)),
        "pct_positive": float(100.0 * (preds > 0).mean()),
    }
    if labels is not None and len(labels) == len(preds) and labels.std() > 0 and preds.std() > 0:
        out["pearson_r"] = float(pearsonr(preds, labels)[0])
        out["spearman_r"] = float(spearmanr(preds, labels)[0])
        out["mse"] = float(np.mean((preds - labels) ** 2))
        out["mean_residual"] = float(np.mean(preds - labels))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n_random", type=int, default=500, help="Sequences per GC level / control panel."
    )
    ap.add_argument("--folds", type=int, default=10, help="Number of ensemble folds (full = 10).")
    args = ap.parse_args()

    print(f"=== Bias eval on 10-fold AG-S2 ensemble ===")
    print(f"  Oracle dir: {ORACLE_DIR}")
    print(f"  Output: {OUT_DIR}/bias_eval.json")

    # ── Build all bias panels ─────────────────────────────────────────────
    panels: dict[str, dict] = {}

    # 1. Random DNA at 6 GC levels
    for gc in (0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75):
        seqs = _gen_random_at_gc(args.n_random, gc, seed=42 + int(gc * 100))
        panels[f"random_gc_{int(gc * 100):02d}pct"] = {"seqs": seqs, "labels": None}

    # 2. Dinuc-shuffled real test seqs
    test_pool = OUT_DIR / "pool" / "test.parquet"
    if test_pool.exists():
        # Use full test pool (not head=500) so we have enough seqs per
        # activity-decile bin for reliable residual statistics.
        test_df_full = pd.read_parquet(test_pool)
        real_seqs_full = test_df_full["sequence"].astype(str).tolist()
        real_labels_full = test_df_full["K562_log2FC"].to_numpy(np.float32)
        panels["test_real"] = {"seqs": real_seqs_full, "labels": real_labels_full}
        # Dinuc shuffle on a sub-sample (slow) — 500 is plenty
        panels["test_dinuc_shuffled"] = {
            "seqs": _gen_dinuc_shuffle(real_seqs_full[:500], seed=42),
            "labels": None,
        }
        # Cross-cell-type sanity: same test seqs, but score against HepG2
        # and SKNSH labels if present in pool. Oracle should NOT be perfect
        # on these (it's K562-specific) — partial correlation is healthy,
        # near-perfect correlation would indicate the oracle is picking up
        # generic active-sequence features rather than K562-specific.
        for cell in ("HepG2", "SKNSH"):
            col = f"{cell}_log2FC"
            if col in test_df_full.columns:
                panels[f"test_real_{cell}_label"] = {
                    "seqs": real_seqs_full,
                    "labels": test_df_full[col].to_numpy(np.float32),
                }
        # SNV pairs for variant-effect calibration
        snv_pool = OUT_DIR / "pool" / "snv_pairs.parquet"
        if snv_pool.exists():
            snv_df = pd.read_parquet(snv_pool).head(2000)
            ref_col = "ref_sequence" if "ref_sequence" in snv_df.columns else "sequence"
            alt_col = "alt_sequence" if "alt_sequence" in snv_df.columns else None
            if alt_col is None:
                # Pool layout may differ; skip if can't find pair columns
                pass
            else:
                snv_ref_label = (
                    snv_df["ref_log2FC"].to_numpy(np.float32)
                    if "ref_log2FC" in snv_df.columns
                    else None
                )
                snv_alt_label = (
                    snv_df["alt_log2FC"].to_numpy(np.float32)
                    if "alt_log2FC" in snv_df.columns
                    else None
                )
                panels["snv_ref"] = {
                    "seqs": snv_df[ref_col].astype(str).tolist(),
                    "labels": snv_ref_label,
                }
                panels["snv_alt"] = {
                    "seqs": snv_df[alt_col].astype(str).tolist(),
                    "labels": snv_alt_label,
                }

    # 3. Eval-set panels
    eval_dir = REPO / "outputs" / "eval_sets_expanded"
    if eval_dir.exists():
        for parq in sorted(eval_dir.glob("*.parquet")):
            df = pd.read_parquet(parq).head(2000)  # cap each panel for speed
            seqs = df["sequence"].astype(str).tolist()
            label_col = "K562_log2FC" if "K562_log2FC" in df.columns else None
            panels[f"eval_{parq.stem}"] = {
                "seqs": seqs,
                "labels": df[label_col].to_numpy(np.float32) if label_col else None,
            }

    print(f"  Built {len(panels)} panels:")
    for name, p in panels.items():
        print(f"    {name}: n={len(p['seqs'])}  has_labels={p['labels'] is not None}")

    # ── Run ensemble inference ─────────────────────────────────────────────
    fold_preds: dict[str, list[np.ndarray]] = {name: [] for name in panels}
    for fold in range(args.folds):
        t0 = time.time()
        predict_step, params, state = _build_predict_step(fold)
        for name, p in panels.items():
            preds = _predict_seqs(predict_step, params, state, p["seqs"])
            fold_preds[name].append(preds)
        print(f"  fold {fold} done in {time.time() - t0:.0f}s")
        # Free params before next fold
        del params, state, predict_step
        jax.clear_caches()

    # ── Aggregate to ensemble means + write summary ────────────────────────
    summary = {}
    fold_stacks = {}  # keep for cross-panel analyses below
    for name, p in panels.items():
        stack = np.stack(fold_preds[name], axis=0)
        fold_stacks[name] = stack
        ens = stack.mean(axis=0)
        s = _stats(ens, p["labels"])
        # Per-seq ensemble std (oracle uncertainty)
        per_seq_std = stack.std(axis=0)
        s["ensemble_std_mean"] = float(per_seq_std.mean())
        s["ensemble_std_p90"] = float(np.quantile(per_seq_std, 0.9))
        summary[name] = s

    # ── Activity-stratified residuals (test_real panel) ───────────────────
    # Bins by ground-truth activity decile; reports mean residual + count
    # per bin. Exposes whether overprediction is concentrated at the
    # high-activity tail (the user's specific concern).
    if "test_real" in panels and panels["test_real"]["labels"] is not None:
        ens = fold_stacks["test_real"].mean(axis=0)
        labels = panels["test_real"]["labels"]
        deciles = np.quantile(labels, np.linspace(0, 1, 11))
        bin_stats = []
        for i in range(10):
            lo, hi = deciles[i], deciles[i + 1]
            mask = (labels >= lo) & (labels <= hi if i == 9 else labels < hi)
            if mask.sum() == 0:
                continue
            resid = ens[mask] - labels[mask]
            bin_stats.append(
                {
                    "decile": i + 1,
                    "label_lo": float(lo),
                    "label_hi": float(hi),
                    "n": int(mask.sum()),
                    "mean_residual": float(resid.mean()),
                    "abs_mean_residual": float(np.abs(resid).mean()),
                    "rmse": float(np.sqrt((resid**2).mean())),
                }
            )
        summary["__activity_stratified__"] = bin_stats

    # ── SNV delta calibration ─────────────────────────────────────────────
    # Predicted alt - ref vs measured alt - ref. Pearson R + mean error.
    # Diagonal = perfect; mean residual signals systematic bias.
    if "snv_ref" in panels and "snv_alt" in panels:
        ref_ens = fold_stacks["snv_ref"].mean(axis=0)
        alt_ens = fold_stacks["snv_alt"].mean(axis=0)
        delta_pred = alt_ens - ref_ens
        ref_lab = panels["snv_ref"]["labels"]
        alt_lab = panels["snv_alt"]["labels"]
        if ref_lab is not None and alt_lab is not None:
            delta_true = alt_lab - ref_lab
            d_stats = {
                "n": int(len(delta_pred)),
                "pearson_r": float(pearsonr(delta_pred, delta_true)[0])
                if delta_true.std() > 0
                else 0.0,
                "spearman_r": float(spearmanr(delta_pred, delta_true)[0]),
                "mse": float(np.mean((delta_pred - delta_true) ** 2)),
                "mean_residual": float(np.mean(delta_pred - delta_true)),
                "delta_pred_std": float(delta_pred.std()),
                "delta_true_std": float(delta_true.std()),
            }
            summary["__snv_delta_calibration__"] = d_stats

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "bias_eval.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")

    # ── Print key bias indicators ──────────────────────────────────────────
    print(
        "\n=== Key bias indicators (ensemble mean prediction; 0 expected for negative controls) ==="
    )
    for name in sorted(summary):
        if name.startswith("random_gc_") or name.startswith("test_dinuc"):
            s = summary[name]
            print(f"  {name:30s}  mean={s['mean']:>+.3f}  pct_positive={s['pct_positive']:>5.1f}%")


if __name__ == "__main__":
    main()
