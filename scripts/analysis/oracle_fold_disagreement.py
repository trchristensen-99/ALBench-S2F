"""Oracle ensemble-disagreement "trust map" for the 10-fold oracle_v3 AG-S2 oracle.

Where does the 10-fold chromosome-CV oracle AGREE (low cross-fold std) vs DIVERGE
(high cross-fold std) on random DNA vs genomic, as a function of GC content?  The
production loader (``_load_k562_ag_s2_oracle`` / ``_live_predict``) returns the
MEAN across folds; here we keep the PER-FOLD stack so cross-fold std can be used
as an epistemic-uncertainty / extrapolation flag.

Model-loading mirrors the single-checkpoint build in
``scripts/analysis/gc_bias_ag_hepg2.py`` and the fold loop in
``_load_k562_ag_s2_oracle``: the full S2 checkpoints (fine-tuned encoder + head)
are restored directly and run live with RC averaging, once per fold.

Hypothesis: folds agree in the genomic/in-support GC range and diverge at extreme
GC on random DNA — disagreement flags where the oracle is extrapolating, and that
divergence co-occurs with the high-activity runaway at extreme GC.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/experiments")

from albench.composition import cpg_obs_exp, gc_content_batch  # noqa: E402

ORACLE_DIR = Path(REPO) / "outputs" / "oracle_v3" / "s2"
ORACLE_CHUNK = int(os.environ.get("AG_ORACLE_CHUNK", "128"))
N_GENOMIC = int(os.environ.get("N_GENOMIC", "8000"))  # subsample genomic for speed
GC_EVAL_POINTS = [0.1, 0.3, 0.5, 0.7, 0.9]


def build_perfold_scorer():
    """Load all 10 oracle_v3 S2 folds; return predict(list[str]) -> (n_folds, n)."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from experiments.exp1_1_scaling import _ag_weights_path

    # models/__init__.py -> dream_rnn.py still does `from .base import SequenceModel`,
    # but models/base.py was removed in the tree-consolidation commit (SequenceModel
    # now lives in albench.model). Importing any models submodule therefore crashes
    # the package init. Shim the missing module in before the import; no file edits.
    import types

    import albench.model as _am

    if "models.base" not in sys.modules:
        _base = types.ModuleType("models.base")
        _base.SequenceModel = _am.SequenceModel
        sys.modules["models.base"] = _base

    from models.alphagenome_heads import register_s2f_head

    ckpt_paths = sorted(
        p / "best_model" / "checkpoint"
        for p in sorted(ORACLE_DIR.glob("fold_*"))
        if (p / "best_model" / "checkpoint").exists()
    )
    if len(ckpt_paths) != 10:
        raise FileNotFoundError(f"Expected 10 oracle_v3 folds, found {len(ckpt_paths)} in {ORACLE_DIR}")

    checkpointer = ocp.StandardCheckpointer()
    params_list = []
    for ckpt_path in ckpt_paths:
        loaded_params, _ = checkpointer.restore(ckpt_path)
        params_list.append(jax.device_put(loaded_params))
    print(f"[fold-disagree] restored {len(params_list)} folds", flush=True)

    head_names = sorted(
        {k.split("/")[1] for k in params_list[0] if k.startswith("head/") and "/" in k[5:]}
    )
    if len(head_names) != 1:
        raise ValueError(f"Expected one head, found {head_names}")
    head_name = head_names[0]
    print(f"[fold-disagree] head: {head_name}", flush=True)

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
        checkpoint_path=_ag_weights_path(),
        use_encoder_output=True,
        detach_backbone=False,
    )
    model_state = model._state

    @jax.jit
    def predict_step(params, seqs):
        out = model._predict(
            params,
            model_state,
            seqs,
            jnp.zeros(len(seqs), dtype=jnp.int32),
            requested_outputs=[head_name],
            negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
            strand_reindexing=None,
        )[head_name]
        return jnp.squeeze(out, axis=-1) if out.ndim > 1 else out

    flank_5 = MPRA_UPSTREAM[-200:]
    flank_3 = MPRA_DOWNSTREAM[:200]
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _encode(seq: str) -> np.ndarray:
        seq = seq.upper()
        if len(seq) < 200:
            pad = 200 - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > 200:
            start = (len(seq) - 200) // 2
            seq = seq[start : start + 200]
        full = flank_5 + seq + flank_3
        out = np.zeros((600, 4), dtype=np.float32)
        for i, c in enumerate(full):
            if c in mapping:
                out[i, mapping[c]] = 1.0
        return out

    def predict(sequences: list[str]) -> np.ndarray:
        """Return (n_folds, n_seqs) RC-averaged per-fold predictions."""
        n = len(sequences)
        x_fwd = np.stack([_encode(s) for s in sequences])
        x_rev = x_fwd[:, ::-1, ::-1]
        per_fold = []
        for fi, params in enumerate(params_list):
            pf, pr = [], []
            for i in range(0, n, ORACLE_CHUNK):
                cf = jnp.array(x_fwd[i : i + ORACLE_CHUNK])
                cr = jnp.array(x_rev[i : i + ORACLE_CHUNK])
                pf.append(np.array(predict_step(params, cf)).reshape(-1))
                pr.append(np.array(predict_step(params, cr)).reshape(-1))
            per_fold.append((np.concatenate(pf) + np.concatenate(pr)) / 2.0)
            print(f"  fold {fi} scored {n} seqs", flush=True)
        return np.stack(per_fold).astype(np.float32)  # (10, n)

    return predict


def main():
    predict = build_perfold_scorer()
    bdir = Path(REPO) / "data" / "k562" / "test_sets_ag_s2_chrsplit"

    # ── RANDOM across GC ────────────────────────────────────────────────
    zr = np.load(bdir / "gc_sweep_oracle.npz", allow_pickle=True)
    rseq = [str(x) for x in zr["sequences"]]
    rgc_bin = np.asarray(zr["gc_bin"], dtype=np.float64)
    print(f"[fold-disagree] RANDOM n={len(rseq)} bins={sorted(set(rgc_bin.tolist()))}", flush=True)
    rfold = predict(rseq)  # (10, n)
    rmean = rfold.mean(axis=0)
    rstd = rfold.std(axis=0)
    rgc = gc_content_batch(rseq)
    rcpg = np.array([cpg_obs_exp(s) for s in rseq], dtype=np.float64)

    # ── GENOMIC (subsample) ─────────────────────────────────────────────
    zg = np.load(bdir / "genomic_oracle.npz", allow_pickle=True)
    gseq_all = [str(x) for x in zg["sequences"]]
    sub = np.random.default_rng(0).choice(
        len(gseq_all), size=min(N_GENOMIC, len(gseq_all)), replace=False
    )
    gseq = [gseq_all[i] for i in sub]
    print(f"[fold-disagree] GENOMIC n={len(gseq)} (of {len(gseq_all)})", flush=True)
    gfold = predict(gseq)
    gmean = gfold.mean(axis=0)
    gstd = gfold.std(axis=0)
    ggc = gc_content_batch(gseq)
    gcpg = np.array([cpg_obs_exp(s) for s in gseq], dtype=np.float64)

    # ── Per-GC-bin aggregation of RANDOM disagreement ───────────────────
    bins = np.array(sorted(set(rgc_bin.tolist())))
    bin_std_mean, bin_std_med, bin_pred_mean, bin_n = [], [], [], []
    for b in bins:
        m = rgc_bin == b
        bin_std_mean.append(float(rstd[m].mean()))
        bin_std_med.append(float(np.median(rstd[m])))
        bin_pred_mean.append(float(rmean[m].mean()))
        bin_n.append(int(m.sum()))
    bin_std_mean = np.array(bin_std_mean)
    bin_std_med = np.array(bin_std_med)
    bin_pred_mean = np.array(bin_pred_mean)

    g_std_mean = float(gstd.mean())
    g_std_med = float(np.median(gstd))
    g_pred_mean = float(gmean.mean())
    g_gc_mean = float(ggc.mean())

    # High-disagreement = above the genomic 95th percentile of cross-fold std.
    hi_thresh = float(np.percentile(gstd, 95))
    frac_hi_by_bin = []
    for b in bins:
        m = rgc_bin == b
        frac_hi_by_bin.append(float((rstd[m] > hi_thresh).mean()))
    frac_hi_by_bin = np.array(frac_hi_by_bin)
    extreme_mask = (rgc_bin <= 0.1) | (rgc_bin >= 0.9)
    frac_hi_extreme = float((rstd[extreme_mask] > hi_thresh).mean())

    # correlations (random): disagreement vs |mean - genomic baseline|, vs GC extremity
    from scipy.stats import pearsonr, spearmanr

    gc_extremity = np.abs(rgc - g_gc_mean)
    r_std_meanpred = pearsonr(rstd, rmean)[0]
    r_std_gcex = spearmanr(rstd, gc_extremity)[0]
    r_std_cpg = spearmanr(rstd, rcpg)[0]

    # ── Report numbers ──────────────────────────────────────────────────
    print("\n==== ORACLE FOLD-DISAGREEMENT TRUST MAP ====", flush=True)
    print(f"genomic: GC={g_gc_mean:.3f}  cross-fold std mean={g_std_mean:.4f} "
          f"med={g_std_med:.4f}  pred mean={g_pred_mean:.3f}", flush=True)
    print(f"hi-disagreement threshold (genomic 95th pct std) = {hi_thresh:.4f}", flush=True)
    print("RANDOM per GC bin:", flush=True)
    print(f"{'gc':>5} {'n':>5} {'std_mean':>9} {'std_med':>8} {'pred_mean':>9} {'frac_hi':>7}", flush=True)
    for i, b in enumerate(bins):
        print(f"{b:>5.2f} {bin_n[i]:>5} {bin_std_mean[i]:>9.4f} {bin_std_med[i]:>8.4f} "
              f"{bin_pred_mean[i]:>9.3f} {frac_hi_by_bin[i]:>7.3f}", flush=True)
    print(f"\nrandom std at GC={GC_EVAL_POINTS}:", flush=True)
    for gp in GC_EVAL_POINTS:
        idx = int(np.argmin(np.abs(bins - gp)))
        print(f"  GC={bins[idx]:.2f}: std_mean={bin_std_mean[idx]:.4f} "
              f"(genomic {g_std_mean:.4f}, ratio {bin_std_mean[idx]/g_std_mean:.2f}x)", flush=True)
    print(f"\nfrac high-disagreement at extreme GC (<=0.1 or >=0.9) = {frac_hi_extreme:.3f}", flush=True)
    print(f"corr(std, mean_pred) pearson = {r_std_meanpred:.3f}", flush=True)
    print(f"corr(std, |GC-genomic|) spearman = {r_std_gcex:.3f}", flush=True)
    print(f"corr(std, CpG O/E) spearman = {r_std_cpg:.3f}", flush=True)

    # ── Save npz ────────────────────────────────────────────────────────
    outdir = Path(REPO) / "outputs" / "gc_bias"
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "oracle_fold_disagreement.npz"
    np.savez_compressed(
        npz_path,
        bins=bins,
        bin_std_mean=bin_std_mean,
        bin_std_med=bin_std_med,
        bin_pred_mean=bin_pred_mean,
        bin_n=np.array(bin_n),
        frac_hi_by_bin=frac_hi_by_bin,
        frac_hi_extreme=np.array(frac_hi_extreme),
        hi_thresh=np.array(hi_thresh),
        random_gc=rgc,
        random_gc_bin=rgc_bin,
        random_mean=rmean,
        random_std=rstd,
        random_cpg=rcpg,
        random_fold_preds=rfold,
        genomic_gc=ggc,
        genomic_mean=gmean,
        genomic_std=gstd,
        genomic_cpg=gcpg,
        genomic_fold_preds=gfold,
        genomic_std_mean=np.array(g_std_mean),
        genomic_std_med=np.array(g_std_med),
        genomic_pred_mean=np.array(g_pred_mean),
        genomic_gc_mean=np.array(g_gc_mean),
        corr_std_meanpred=np.array(r_std_meanpred),
        corr_std_gcex=np.array(r_std_gcex),
        corr_std_cpg=np.array(r_std_cpg),
    )
    print(f"wrote {npz_path}", flush=True)

    # ── Figure ──────────────────────────────────────────────────────────
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: disagreement (cross-fold std) vs GC, random vs genomic.
    ax1.plot(bins, bin_std_mean, "o-", color="#b2182b", lw=2, label="random DNA (mean cross-fold std)")
    ax1.plot(bins, bin_std_med, "s--", color="#ef8a62", lw=1.3, alpha=0.8, label="random DNA (median)")
    ax1.axhline(g_std_mean, color="#2166ac", lw=1.8, ls=":", label=f"genomic mean std = {g_std_mean:.3f}")
    ax1.scatter([g_gc_mean], [g_std_mean], color="#2166ac", s=90, zorder=5, marker="D",
                label=f"genomic (GC={g_gc_mean:.2f})")
    ax1.axvspan(0.25, 0.72, color="grey", alpha=0.12, label="genomic in-support GC [0.25,0.72]")
    ax1.set_xlabel("GC content of random 200bp payload")
    ax1.set_ylabel("Oracle ensemble disagreement (cross-fold std)")
    ax1.set_title("Oracle 10-fold disagreement vs GC\nrandom DNA vs genomic")
    ax1.legend(fontsize=7.5, loc="upper center")
    ax1.grid(alpha=0.3)

    # Right: disagreement vs mean prediction (runaway co-occurrence), colored by GC.
    sc = ax2.scatter(rmean, rstd, c=rgc, cmap="viridis", s=5, alpha=0.3)
    ax2.scatter(gmean, gstd, c="red", s=4, alpha=0.15, label="genomic")
    cb = fig.colorbar(sc, ax=ax2)
    cb.set_label("random GC")
    ax2.axhline(hi_thresh, color="k", ls=":", lw=1, label=f"hi-disagree thr={hi_thresh:.3f}")
    ax2.set_xlabel("Oracle mean prediction (log2FC)")
    ax2.set_ylabel("Oracle ensemble disagreement (cross-fold std)")
    ax2.set_title("Disagreement vs predicted activity\n(does the runaway coincide with divergence?)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = outdir / "oracle_fold_disagreement.png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}", flush=True)


if __name__ == "__main__":
    main()
