"""GC-content sweep for the HepG2 chr-split Stage-2 single AG-MPRA model.

Completes the 3-model (AG / MPRA-LegNet / Malinois) x GC analysis: does the
HepG2 AlphaGenome-MPRA student show the same random-DNA GC bias (U-shape +
high-GC runaway) the K562 oracle/LegNet/Malinois show?

Mirrors scripts/analysis/gc_bias_oracle.py (same random-at-GC generation and the
same genomic reference pool) but swaps the labeller to the HepG2 chr_split_v2
s2 single model at outputs/chr_split_v2/hepg2/ag_s2_best_hp/.../best_model/checkpoint.
The model is the fine-tuned encoder + s2f head, loaded directly and run live with
RC averaging; sequences are encoded exactly as in training (200bp MPRA upstream
flank + 200bp payload + 200bp downstream flank = 600bp).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/experiments")

L = 200
N_PER_BIN = 10_000
GC_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_GENOMIC = 20_000  # subsample of the 100k genomic pool, relabelled by HepG2 model
BASES = np.array(list("ACGT"))

CKPT = (
    f"{REPO}/outputs/chr_split_v2/hepg2/ag_s2_best_hp/"
    "seed_42/genomic/n600000/hp0/seed42/best_model/checkpoint"
)
ORACLE_CHUNK = int(os.environ.get("AG_ORACLE_CHUNK", "128"))


def random_at_gc(gc: float, n: int, rng) -> list[str]:
    # per-base A,C,G,T probabilities; expected GC == gc in the limit.
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    idx = rng.choice(4, size=(n, L), p=p)
    return ["".join(BASES[row]) for row in idx]


def gc_of(seqs):
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])


def build_hepg2_scorer():
    """Load the HepG2 s2 single checkpoint and return predict(list[str])->np.ndarray.

    Replicates the exact build from _train_ag_s2_student (task='k562' code path:
    the head is named s2f_exp1_s2_k562_42 even though labels are HepG2) and the
    live RC-averaged inference from _load_k562_ag_s2_oracle's wrapper.
    """
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_ft import create_model_with_heads

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    from experiments.exp1_1_scaling import TASK_CONFIGS, _ag_weights_path
    from models.alphagenome_heads import register_s2f_head
    from models.embedding_cache import reinit_head_params

    ckpt_path = Path(CKPT).resolve()
    print(f"[hepg2-gc] restoring checkpoint {ckpt_path}", flush=True)
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path)
    if isinstance(restored, dict):
        params = restored
    elif isinstance(restored, (tuple, list)):
        params = restored[0]
    else:
        params = restored

    # Detect the s2f head module name from the checkpoint (keys like
    # "head/s2f_exp1_s2_k562_42/~predict/hidden_0").
    head_names = sorted(
        {k.split("/")[1] for k in params if isinstance(k, str) and k.startswith("head/s2f")}
    )
    if len(head_names) != 1:
        raise ValueError(f"Expected exactly one s2f head in checkpoint, found {head_names}")
    head_name = head_names[0]
    print(f"[hepg2-gc] s2f head detected: {head_name}", flush=True)

    task_mode = TASK_CONFIGS["k562"]["task_mode"]
    register_s2f_head(
        head_name=head_name,
        arch="boda-flatten-512-512",
        task_mode=task_mode,
        num_tracks=1,
        dropout_rate=0.1,
    )
    weights_path = _ag_weights_path()
    model = create_model_with_heads(
        "all_folds",
        heads=[head_name],
        checkpoint_path=weights_path,
        use_encoder_output=True,
        detach_backbone=False,
    )
    reinit_head_params(model, head_name, num_tokens=5, dim=1536, rng=42)
    # Overwrite with the restored fine-tuned params (trunk + trained head).
    model._params = jax.device_put(params)
    model_state = model._state

    @jax.jit
    def predict_step(p, seqs):
        out = model._predict(
            p,
            model_state,
            seqs,
            jnp.zeros(len(seqs), dtype=jnp.int32),
            requested_outputs=[head_name],
            negative_strand_mask=jnp.zeros(len(seqs), dtype=bool),
            strand_reindexing=None,
        )[head_name]
        return jnp.squeeze(out, axis=-1) if out.ndim > 1 else out

    # Sequence encoding: 200bp MPRA flank + 200bp payload + 200bp flank = 600bp.
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
        n = len(sequences)
        x_fwd = np.stack([_encode(s) for s in sequences])
        x_rev = x_fwd[:, ::-1, ::-1]  # reverse complement (A<->T, C<->G via ::-1 on 4-ch)
        pf, pr = [], []
        for i in range(0, n, ORACLE_CHUNK):
            cf = jnp.array(x_fwd[i : i + ORACLE_CHUNK])
            cr = jnp.array(x_rev[i : i + ORACLE_CHUNK])
            pf.append(np.array(predict_step(model._params, cf)).reshape(-1))
            pr.append(np.array(predict_step(model._params, cr)).reshape(-1))
        return ((np.concatenate(pf) + np.concatenate(pr)) / 2.0).astype(np.float32)

    return predict


def main():
    predict = build_hepg2_scorer()
    rng = np.random.default_rng(42)

    out = {"gc_levels": np.array(GC_LEVELS)}
    med, mean = [], []
    for gc in GC_LEVELS:
        seqs = random_at_gc(gc, N_PER_BIN, rng)
        lab = np.asarray(predict(seqs), dtype=np.float32).ravel()
        out[f"random_gc{int(round(gc * 100))}"] = lab
        med.append(float(np.median(lab)))
        mean.append(float(lab.mean()))
        print(
            f"  random GC={gc:.1f}: n={len(lab)} actual_gc={gc_of(seqs).mean():.3f} "
            f"pred mean={lab.mean():.3f} median={np.median(lab):.3f}",
            flush=True,
        )

    # Genomic reference: same pool sequences as the K562 figure, RELABELLED by HepG2.
    z = np.load(
        f"{REPO}/outputs/grid_pools/k562_cumulative/genomic_d100000.npz", allow_pickle=True
    )
    gseq_all = [str(x) for x in z["sequences"]]
    sub = np.random.default_rng(0).choice(len(gseq_all), size=min(N_GENOMIC, len(gseq_all)),
                                          replace=False)
    gseq = [gseq_all[i] for i in sub]
    gpred = np.asarray(predict(gseq), dtype=np.float32).ravel()
    out["genomic_gc"] = gc_of(gseq)
    out["genomic_pred"] = gpred
    print(
        f"  genomic (HepG2-relabelled): n={len(gseq)} mean_gc={out['genomic_gc'].mean():.3f} "
        f"pred mean={gpred.mean():.3f} median={np.median(gpred):.3f}",
        flush=True,
    )

    os.makedirs(f"{REPO}/outputs/gc_bias", exist_ok=True)
    npz_path = f"{REPO}/outputs/gc_bias/gc_bias_ag_hepg2.npz"
    np.savez_compressed(npz_path, **out)
    print(f"  wrote {npz_path}", flush=True)

    # ── Plot: median+mean predicted activity vs GC, genomic reference overlaid ──
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gcs = np.array(GC_LEVELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gcs, med, "o-", color="#2166ac", lw=2, label="random DNA (median)")
    ax.plot(gcs, mean, "s--", color="#67a9cf", lw=1.5, alpha=0.8, label="random DNA (mean)")
    g_gc_mean = float(out["genomic_gc"].mean())
    g_med = float(np.median(gpred))
    g_mean = float(gpred.mean())
    ax.axhline(g_med, color="#b2182b", lw=1.5, ls=":",
               label=f"genomic median = {g_med:.2f}")
    ax.scatter([g_gc_mean], [g_med], color="#b2182b", s=90, zorder=5, marker="D",
               label=f"genomic (GC={g_gc_mean:.2f})")
    ax.set_xlabel("GC content of random 200bp payload")
    ax.set_ylabel("Predicted HepG2 activity (log2FC)")
    ax.set_title("GC bias — HepG2 chr-split AG-MPRA single model\n"
                 "random DNA vs genomic reference")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = f"{REPO}/outputs/gc_bias/gc_bias_ag_hepg2.png"
    fig.savefig(png_path, dpi=150)
    print(f"  wrote {png_path}", flush=True)


if __name__ == "__main__":
    main()
