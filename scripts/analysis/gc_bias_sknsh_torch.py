"""GC-content sweep for the SK-N-SH MPRA-LegNet and Malinois models.

Completes the SK-N-SH column of the 3-cell x 3-model GC-bias matrix. For
--model {legnet,malinois} this mirrors the labeller-loading in
scripts/analysis/gc_mechanism_hepg2.py (LegNetStudent wrapper for the per-cell
legnet_sknsh.pt; load_pretrained_malinois + one_hot/pad for Malinois), but reads
the SK-N-SH output head and produces the same random-DNA-vs-genomic GC curve the
AG sknsh figure uses: random 200bp payloads at GC 0.0-1.0 (N=10k/bin) plus the
Gosai genomic test-window reference, relabelled by this model.

Malinois is a single 3-output model; the SK-N-SH head is output index 2
(cell order k562=0 / hepg2=1 / sknsh=2, per scripts/eval_pretrained_malinois.py).
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")

L = 200
N_PER_BIN = 10_000
GC_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_GENOMIC = 20_000
BASES = np.array(list("ACGT"))
MALINOIS_SKNSH_IDX = 2  # k562=0 / hepg2=1 / sknsh=2


def random_at_gc(gc: float, n: int, rng) -> list[str]:
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A,C,G,T
    idx = rng.choice(4, size=(n, L), p=p)
    return ["".join(BASES[row]) for row in idx]


def gc_of(seqs):
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])


def make_predict(model_name: str):
    """Return predict(list[str]) -> np.ndarray for the SK-N-SH head of `model_name`."""
    import torch

    if model_name == "legnet":
        from models.legnet_student import LegNetStudent, TrainConfig

        m = torch.load(
            f"{REPO}/outputs/gc_bias/legnet_percell/legnet_sknsh.pt",
            map_location="cuda",
            weights_only=False,
        ).eval()
        st = LegNetStudent(train_config=TrainConfig(num_workers=0), ensemble_size=1, device="cuda")
        st.models = [m]
        return lambda seqs: np.asarray(st.predict(list(seqs)), dtype=np.float64)

    if model_name == "malinois":
        from add_malinois_random_bias import one_hot_encode, pad_to_600bp
        from eval_pretrained_malinois import load_pretrained_malinois

        model = load_pretrained_malinois().cuda().eval()

        def pred(seqs, idx=MALINOIS_SKNSH_IDX):
            out = []
            for i in range(0, len(seqs), 2048):
                b = seqs[i : i + 2048]
                x = torch.tensor(
                    np.stack([pad_to_600bp(one_hot_encode(s)) for s in b])
                ).float().cuda()
                with torch.no_grad():
                    out.append(model(x)[:, idx].cpu().numpy())
            return np.concatenate(out).astype(np.float64)

        return pred

    raise ValueError(model_name)


def main():
    model_name = sys.argv[1]
    predict = make_predict(model_name)
    rng = np.random.default_rng(42)

    out = {"gc_levels": np.array(GC_LEVELS), "model": model_name}
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

    # Genomic reference: same pool the AG/K562 figures use, relabelled by this model.
    z = np.load(
        f"{REPO}/outputs/grid_pools/k562_cumulative/genomic_d100000.npz", allow_pickle=True
    )
    gseq_all = [str(x) for x in z["sequences"]]
    sub = np.random.default_rng(0).choice(
        len(gseq_all), size=min(N_GENOMIC, len(gseq_all)), replace=False
    )
    gseq = [gseq_all[i] for i in sub]
    gpred = np.asarray(predict(gseq), dtype=np.float32).ravel()
    out["genomic_gc"] = gc_of(gseq)
    out["genomic_pred"] = gpred
    print(
        f"  genomic (sknsh-relabelled): n={len(gseq)} mean_gc={out['genomic_gc'].mean():.3f} "
        f"pred mean={gpred.mean():.3f} median={np.median(gpred):.3f}",
        flush=True,
    )

    os.makedirs(f"{REPO}/outputs/gc_bias", exist_ok=True)
    npz_path = f"{REPO}/outputs/gc_bias/gc_bias_{model_name}_sknsh.npz"
    np.savez_compressed(npz_path, **out)
    print(f"  wrote {npz_path}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gcs = np.array(GC_LEVELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gcs, med, "o-", color="#2166ac", lw=2, label="random DNA (median)")
    ax.plot(gcs, mean, "s--", color="#67a9cf", lw=1.5, alpha=0.8, label="random DNA (mean)")
    g_gc_mean = float(out["genomic_gc"].mean())
    g_med = float(np.median(gpred))
    ax.axhline(g_med, color="#b2182b", lw=1.5, ls=":", label=f"genomic median = {g_med:.2f}")
    ax.scatter(
        [g_gc_mean], [g_med], color="#b2182b", s=90, zorder=5, marker="D",
        label=f"genomic (GC={g_gc_mean:.2f})",
    )
    ax.set_xlabel("GC content of random 200bp payload")
    ax.set_ylabel("Predicted SK-N-SH activity (log2FC)")
    ax.set_title(
        f"GC bias — SK-N-SH {model_name}\nrandom DNA vs genomic reference"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = f"{REPO}/outputs/gc_bias/gc_bias_{model_name}_sknsh.png"
    fig.savefig(png_path, dpi=150)
    print(f"  wrote {png_path}", flush=True)


if __name__ == "__main__":
    main()
