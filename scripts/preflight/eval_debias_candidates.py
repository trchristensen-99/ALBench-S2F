"""Score each fold-0 debias-sweep candidate on the panels we actually
have ground-truth (or a reliable expectation) for, and rank them.

Per project direction (May 2026):
  - We avoid post-hoc correction entirely (any output transform pulls us
    off the oracle's S2F landscape and could mis-extrapolate on the
    intermediate-distribution panels we care about).
  - We retrain SINGLE folds first to probe the design space, only
    extending the winner to a 10-fold ensemble.
  - For most v2 eval panels (PRM, indels, structural muts, intermediate-
    GC random) we have NO ground truth, so they're not useful as a
    headline bias signal. They become useful later for landscape-coherence
    QA and for cross-validation with real episomal MPRA data when we
    have it.

This script therefore scores each candidate on the panels where we DO
have a directly testable signal:

  Real-label (use Pearson / MSE):
    test_real        — in-dist test, K562_log2FC labels
    test_real_HepG2  — same seqs, HepG2 labels (cross-cell sanity)
    test_real_SKNSH  — same seqs, SKNSH labels (cross-cell sanity)
    snv_ref / snv_alt — variant-effect calibration

  Negative-control (expected ≈ 0; use mean-absolute):
    random_gc_25/35/45/55/65/75pct — CpG-shortcut probe at 6 GC levels
    test_dinuc_shuffled            — keeps dinuc stats, breaks motifs

Composite score (higher is better):
    in_dist_pearson + 0.5 * snv_delta_pearson - alpha * neg_bias_score
  where neg_bias_score = mean over GC levels of |ensemble_mean|.

Outputs:
    outputs/oracle_neg_sweep/<sweep>/eval_summary.json
    outputs/oracle_neg_sweep/<sweep>/eval_summary.csv
    outputs/oracle_neg_sweep/<sweep>/eval_summary_plot.png  (if matplotlib)

Usage:
    uv run --no-sync python scripts/preflight/eval_debias_candidates.py \\
        --base outputs/oracle_neg_sweep/debias_sweep_v1
"""

from __future__ import annotations

import argparse
import json
import time
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
POOL_DIR = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt" / "pool"

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
    rng = np.random.default_rng(seed)
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    bases = np.array(list("ACGT"))
    arr = rng.choice(4, size=(n, 200), p=p)
    return ["".join(bases[row]) for row in arr]


def _build_predict_step(ckpt_dir: Path, batch_size: int = 256):
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

    loaded_params, _ = ocp.StandardCheckpointer().restore(ckpt_dir)
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
    x_rev = x[:, ::-1, ::-1]
    pf_all, pr_all = [], []
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
        pf_all.append(pf)
        pr_all.append(pr)
    return (np.concatenate(pf_all) + np.concatenate(pr_all)) / 2.0


def _build_panels(n_random: int = 500, n_test: int = 4000, n_snv: int = 2000):
    """Build the evaluation panels once. Same panels for every candidate."""
    panels: dict[str, dict] = {}
    for gc in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
        seqs = _gen_random_at_gc(n_random, gc, seed=42 + int(gc * 100))
        panels[f"random_gc_{int(gc * 100):02d}pct"] = {
            "seqs": seqs,
            "labels": None,
            "kind": "neg_control",
            "gc": gc,
        }

    test_pool = POOL_DIR / "test.parquet"
    if test_pool.exists():
        td = pd.read_parquet(test_pool).head(n_test)
        seqs = td["sequence"].astype(str).tolist()
        panels["test_real"] = {
            "seqs": seqs,
            "labels": td["K562_log2FC"].to_numpy(np.float32),
            "kind": "labeled_in_dist",
        }
        for cell in ("HepG2", "SKNSH"):
            col = f"{cell}_log2FC"
            if col in td.columns:
                panels[f"test_real_{cell}_label"] = {
                    "seqs": seqs,
                    "labels": td[col].to_numpy(np.float32),
                    "kind": "labeled_cross_cell",
                }

    snv_pool = POOL_DIR / "snv_pairs.parquet"
    if snv_pool.exists():
        sd = pd.read_parquet(snv_pool).head(n_snv)
        ref_col = "ref_sequence" if "ref_sequence" in sd.columns else None
        alt_col = "alt_sequence" if "alt_sequence" in sd.columns else None
        if ref_col and alt_col:
            panels["snv_ref"] = {
                "seqs": sd[ref_col].astype(str).tolist(),
                "labels": sd["ref_log2FC"].to_numpy(np.float32)
                if "ref_log2FC" in sd.columns
                else None,
                "kind": "labeled_snv_ref",
            }
            panels["snv_alt"] = {
                "seqs": sd[alt_col].astype(str).tolist(),
                "labels": sd["alt_log2FC"].to_numpy(np.float32)
                if "alt_log2FC" in sd.columns
                else None,
                "kind": "labeled_snv_alt",
            }

    ood_path = REPO / "data" / "k562" / "test_sets" / "test_ood_designed_k562.tsv"
    if ood_path.exists():
        od = pd.read_csv(ood_path, sep="\t")
        if "K562_log2FC" in od.columns:
            panels["test_ood"] = {
                "seqs": od["sequence"].astype(str).tolist(),
                "labels": od["K562_log2FC"].to_numpy(np.float32),
                "kind": "labeled_ood",
            }
    return panels


def _stats(preds, labels):
    out = {
        "n": int(len(preds)),
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "abs_mean": float(np.mean(np.abs(preds))),
    }
    if labels is not None and len(labels) == len(preds) and labels.std() > 0 and preds.std() > 0:
        out["pearson_r"] = float(pearsonr(preds, labels)[0])
        out["spearman_r"] = float(spearmanr(preds, labels)[0])
        out["mse"] = float(np.mean((preds - labels) ** 2))
        out["mean_residual"] = float(np.mean(preds - labels))
    return out


def _composite(per_panel: dict, snv_delta_r: float | None, alpha: float) -> dict:
    in_dist_r = per_panel.get("test_real", {}).get("pearson_r", 0.0)
    ood_r = per_panel.get("test_ood", {}).get("pearson_r", 0.0)
    ood_mse = per_panel.get("test_ood", {}).get("mse", 0.0)
    snv_r = snv_delta_r if snv_delta_r is not None else 0.0
    gc_means = [abs(per_panel[k]["mean"]) for k in per_panel if k.startswith("random_gc_")]
    neg_bias = float(np.mean(gc_means)) if gc_means else 0.0
    # v3 lesson: OOD Pearson alone can stay deceptively high while OOD
    # MSE explodes 4-13× due to scale collapse. c31 (10% dinuc + grad
    # penalty) ranked top with score=1.37 but OOD MSE was 4.96 vs baseline
    # 1.15 — the model's predictions were 4× more wrong on average,
    # despite preserving rank-ordering.
    # Add a normalized OOD-MSE penalty anchored at baseline (~1.15). Only
    # penalize when MSE rises above baseline; cap to avoid runaway penalty.
    ood_mse_baseline = 1.15  # K562 OOD MSE for the original 10-fold oracle
    ood_mse_penalty = max(0.0, (ood_mse - ood_mse_baseline) / ood_mse_baseline)
    ood_mse_penalty = min(ood_mse_penalty, 5.0)  # cap so collapse doesn't dominate
    return {
        "in_dist_pearson": in_dist_r,
        "ood_pearson": ood_r,
        "ood_mse": ood_mse,
        "ood_mse_penalty": ood_mse_penalty,
        "snv_delta_pearson": snv_r,
        "neg_bias": neg_bias,
        "score": in_dist_r + ood_r + 0.5 * snv_r - alpha * neg_bias - 0.5 * ood_mse_penalty,
    }


def _eval_candidate(name: str, ckpt_dir: Path, panels: dict) -> dict:
    print(f"\n=== {name} ===  ckpt={ckpt_dir}")
    t0 = time.time()
    predict_step, params, state = _build_predict_step(ckpt_dir)
    per_panel = {}
    for pname, p in panels.items():
        preds = _predict_seqs(predict_step, params, state, p["seqs"])
        per_panel[pname] = _stats(preds, p["labels"])
        per_panel[pname]["__preds_for_delta__"] = preds  # ephemeral
    snv_delta_r = None
    if "snv_ref" in per_panel and "snv_alt" in per_panel:
        ref = per_panel["snv_ref"].pop("__preds_for_delta__")
        alt = per_panel["snv_alt"].pop("__preds_for_delta__")
        if panels["snv_ref"]["labels"] is not None and panels["snv_alt"]["labels"] is not None:
            dt = panels["snv_alt"]["labels"] - panels["snv_ref"]["labels"]
            dp = alt - ref
            if dt.std() > 0 and dp.std() > 0:
                snv_delta_r = float(pearsonr(dp, dt)[0])
    for v in per_panel.values():
        v.pop("__preds_for_delta__", None)
    composite = _composite(per_panel, snv_delta_r, alpha=1.0)
    print(
        f"  in_dist_R={composite['in_dist_pearson']:.4f}  "
        f"snv_dR={composite['snv_delta_pearson']:.4f}  "
        f"neg_bias={composite['neg_bias']:.3f}  "
        f"score={composite['score']:.4f}  "
        f"({time.time() - t0:.0f}s)"
    )
    del params, state, predict_step
    jax.clear_caches()
    return {"name": name, "ckpt_dir": str(ckpt_dir), "composite": composite, "per_panel": per_panel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=str,
        required=True,
        help="Path to outputs/oracle_neg_sweep/<sweep_tag>/. Each child <candidate>/fold_0/best_model/checkpoint is evaluated.",
    )
    ap.add_argument("--n_random", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=4000)
    ap.add_argument("--n_snv", type=int, default=2000)
    ap.add_argument(
        "--include_baseline",
        type=str,
        default=None,
        help="Optional path to current oracle's fold_0/best_model/checkpoint to include as a reference row.",
    )
    args = ap.parse_args()

    # orbax requires absolute paths for checkpoint reads — resolve early.
    base = Path(args.base).resolve()
    if not base.exists():
        raise SystemExit(f"missing {base}")

    panels = _build_panels(args.n_random, args.n_test, args.n_snv)
    print(f"Built {len(panels)} panels:")
    for pn, p in panels.items():
        print(
            f"  {pn:30s}  n={len(p['seqs']):,}  has_labels={p['labels'] is not None}  kind={p['kind']}"
        )

    candidates = sorted([d for d in base.iterdir() if d.is_dir()])
    rows = []
    if args.include_baseline:
        rows.append(
            _eval_candidate("__current_oracle__", Path(args.include_baseline).resolve(), panels)
        )
    for cand in candidates:
        ckpt = (cand / "fold_0" / "best_model" / "checkpoint").resolve()
        if not ckpt.exists():
            print(f"  SKIP {cand.name}: no ckpt at {ckpt}")
            continue
        rows.append(_eval_candidate(cand.name, ckpt, panels))

    rows.sort(key=lambda r: -r["composite"]["score"])

    out_json = base / "eval_summary.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out_json}")

    csv_rows = []
    for r in rows:
        c = r["composite"]
        csv_rows.append(
            {
                "name": r["name"],
                "score": c["score"],
                "in_dist_pearson": c["in_dist_pearson"],
                "ood_pearson": c["ood_pearson"],
                "ood_mse": c.get("ood_mse"),
                "ood_mse_penalty": c.get("ood_mse_penalty"),
                "snv_delta_pearson": c["snv_delta_pearson"],
                "neg_bias": c["neg_bias"],
                "test_ood_mse": r["per_panel"].get("test_ood", {}).get("mse"),
                **{
                    f"{k}_mean": v["mean"]
                    for k, v in r["per_panel"].items()
                    if k.startswith("random_gc_")
                },
                **{
                    f"{k}_R": v.get("pearson_r")
                    for k, v in r["per_panel"].items()
                    if k.startswith("test_real")
                },
            }
        )
    df = pd.DataFrame(csv_rows)
    out_csv = base / "eval_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    print("\n=== Ranking (higher score = better; promote top-K to 10-fold) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
