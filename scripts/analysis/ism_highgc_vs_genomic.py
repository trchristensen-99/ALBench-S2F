"""GC-bias attribution: is high-GC-random activity DIFFUSE composition or pseudo-MOTIFS?

For the MPRA-LegNet K562 student we ask where the model's activity signal sits along a
sequence. Pure base-composition reading should spread importance ~uniformly across every
GC base; a (pseudo-)motif reader should concentrate it in a few short windows. We quantify
concentration (Gini, entropy, top-10% mass) of the per-position importance profile and then
read out which k-mers the hot windows actually contain, comparing high-GC random, mid-GC
random, and real Gosai genomic test windows.

Importance per position via BOTH saliency (gradient x input) and ISM (mean |Delta pred| over
the 3 substitutions). ISM is the primary signal; saliency is reported for cross-check.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import torch

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)

from data.k562 import K562Dataset  # noqa: E402
from models.legnet_student import LegNetStudent, TrainConfig  # noqa: E402

import os  # noqa: E402

L = 200
N_PER_GROUP = int(os.environ.get("N_PER_GROUP", "500"))
BASES = np.array(list("ACGT"))
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
DEVICE = os.environ.get("ISM_DEVICE", "cuda")
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ model
def load_model():
    m = torch.load(
        f"{REPO}/outputs/gc_bias/legnet_percell/legnet_k562.pt",
        map_location=DEVICE,
        weights_only=False,
    ).eval()
    return m


def seqs_to_onehot(seqs: list[str]) -> np.ndarray:
    out = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, nuc in enumerate(s[:L]):
            k = NUC_TO_IDX.get(nuc)
            if k is not None:
                out[i, k, j] = 1.0
    return out


def predict_onehot(model, x: np.ndarray, bs: int = 2048) -> np.ndarray:
    """x: (N,4,L) float32 -> (N,) predictions."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            b = torch.from_numpy(x[i : i + bs]).to(DEVICE)
            preds.append(model(b).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds)


# ------------------------------------------------------------------ importance
def saliency_gradxinput(model, x: np.ndarray, bs: int = 1024) -> np.ndarray:
    """Return gradient x input summed over channels -> (N,L); abs taken by caller."""
    out = np.zeros((len(x), L), dtype=np.float32)
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i : i + bs]).to(DEVICE).requires_grad_(True)
        y = model(xb).reshape(-1).sum()
        (g,) = torch.autograd.grad(y, xb)
        gi = (g * xb).sum(dim=1)  # (b, L) : grad at the base that is present
        out[i : i + bs] = gi.detach().cpu().numpy()
    return out


def ism_profile(model, seqs: list[str], bs: int = 4096) -> np.ndarray:
    """Mean |Delta pred| over the 3 substitutions at each position. -> (N, L)."""
    oh = seqs_to_onehot(seqs)
    base = predict_onehot(model, oh)
    prof = np.zeros((len(seqs), L), dtype=np.float32)
    # Build all variants for a chunk of sequences, predict in one batched pass.
    buf_x, buf_idx = [], []  # idx = (seq_i, pos, present_base, alt_base)

    def flush():
        if not buf_x:
            return
        xb = np.stack(buf_x).astype(np.float32)
        pv = predict_onehot(model, xb)
        for (si, pos), p in zip(buf_idx, pv):
            prof[si, pos] += abs(p - base[si])
        buf_x.clear()
        buf_idx.clear()

    for si, s in enumerate(seqs):
        present = np.array([NUC_TO_IDX.get(c, -1) for c in s[:L]])
        for pos in range(L):
            pb = present[pos]
            if pb < 0:
                continue
            for alt in range(4):
                if alt == pb:
                    continue
                v = oh[si].copy()
                v[:, pos] = 0.0
                v[alt, pos] = 1.0
                buf_x.append(v)
                buf_idx.append((si, pos))
            if len(buf_x) >= bs:
                flush()
    flush()
    prof /= 3.0  # mean over the 3 substitutions
    return prof, base


# ------------------------------------------------------------------ concentration
def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x).astype(np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * x.sum()))


def norm_entropy(x: np.ndarray) -> float:
    x = np.abs(x).astype(np.float64)
    s = x.sum()
    if s == 0:
        return 1.0
    p = x / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(x)))


def top10_frac(x: np.ndarray) -> float:
    x = np.abs(x).astype(np.float64)
    if x.sum() == 0:
        return 0.0
    k = max(1, int(round(0.10 * len(x))))
    return float(np.sort(x)[-k:].sum() / x.sum())


def concentration(profiles: np.ndarray) -> dict:
    g = np.array([gini(p) for p in profiles])
    e = np.array([norm_entropy(p) for p in profiles])
    t = np.array([top10_frac(p) for p in profiles])
    return {"gini": g, "entropy": e, "top10": t}


# ------------------------------------------------------------------ motif read-out
def top_windows(seqs: list[str], prof: np.ndarray, w: int = 9, n_peaks: int = 2):
    """Extract up to n_peaks non-overlapping highest-ISM windows per sequence."""
    half = w // 2
    wins = []
    for s, p in zip(seqs, prof):
        taken = np.zeros(L, dtype=bool)
        pp = p.copy()
        for _ in range(n_peaks):
            c = int(np.argmax(np.where(taken, -np.inf, pp)))
            if not np.isfinite(pp[c]) or pp[c] <= 0:
                break
            a, b = max(0, c - half), min(L, c + half + 1)
            wins.append(s[a:b])
            taken[max(0, c - w) : min(L, c + w + 1)] = True
    return wins


def kmer_counts(windows: list[str], k: int = 6) -> Counter:
    c = Counter()
    for w in windows:
        for i in range(len(w) - k + 1):
            c[w[i : i + k]] += 1
    return c


def gc_frac(s: str) -> float:
    return (s.count("G") + s.count("C")) / max(1, len(s))


def cpg_density(windows: list[str]) -> float:
    tot_cg = sum(w.count("CG") for w in windows)
    tot_bp = sum(max(0, len(w) - 1) for w in windows)
    return tot_cg / max(1, tot_bp)


# GC-box / Sp1 cores and KLF/EGR-ish cores (checked as substring hits in hot windows)
SP1_CORES = ["GGGCGG", "CCGCCC", "GGGGCGGGG", "GGGCGGGG", "GCGGGGCGG"]
KLF_EGR_CORES = ["GCGTGGGCG", "TGGGGGTGG", "GTGGGTG", "GCGGGGG", "GGGGTGG"]


def core_hit_rate(windows: list[str], cores: list[str]) -> float:
    if not windows:
        return 0.0
    hits = sum(any(core in w for core in cores) for w in windows)
    return hits / len(windows)


# ------------------------------------------------------------------ sequence gen
def rand_at_gc(gc: float, n: int) -> list[str]:
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A C G T
    return ["".join(BASES[r]) for r in rng.choice(4, size=(n, L), p=p)]


def main():
    model = load_model()

    # ---- groups
    hi = rand_at_gc(0.80, N_PER_GROUP)
    mid = rand_at_gc(0.50, N_PER_GROUP)
    ds = K562Dataset(data_path=f"{REPO}/data/k562", split="test")  # chr7/chr13 holdout
    gi = rng.choice(len(ds.sequences), size=N_PER_GROUP, replace=False)
    gen = [ds.sequences[i] for i in gi]
    gen_lab = np.asarray([ds.labels[i] for i in gi], dtype=np.float64)

    groups = {"highgc": hi, "midgc": mid, "genomic": gen}
    out = {"gen_label": gen_lab}
    summary = {}

    for name, seqs in groups.items():
        print(f"== {name}: ISM on {len(seqs)} seqs ==", flush=True)
        prof, base = ism_profile(model, seqs)
        sal = saliency_gradxinput(model, seqs_to_onehot(seqs))
        conc = concentration(prof)
        conc_sal = concentration(np.abs(sal))
        # saliency vs ISM agreement (per-seq Pearson of the two profiles)
        agree = np.array(
            [np.corrcoef(prof[i], np.abs(sal[i]))[0, 1] for i in range(len(seqs))]
        )
        gc = np.array([gc_frac(s) for s in seqs])

        wins = top_windows(seqs, prof, w=9, n_peaks=2)
        kc = kmer_counts(wins, k=6)
        top_k = kc.most_common(25)
        sp1 = core_hit_rate(wins, SP1_CORES)
        klf = core_hit_rate(wins, KLF_EGR_CORES)
        cpg = cpg_density(wins)
        win_gc = float(np.mean([gc_frac(w) for w in wins])) if wins else 0.0

        out[f"{name}_prof"] = prof
        out[f"{name}_sal"] = sal
        out[f"{name}_pred"] = base
        out[f"{name}_gini"] = conc["gini"]
        out[f"{name}_entropy"] = conc["entropy"]
        out[f"{name}_top10"] = conc["top10"]
        out[f"{name}_gini_sal"] = conc_sal["gini"]
        out[f"{name}_seqgc"] = gc
        out[f"{name}_agree"] = agree

        summary[name] = {
            "n": len(seqs),
            "pred_med": float(np.median(base)),
            "seqgc_med": float(np.median(gc)),
            "gini_ISM": float(np.median(conc["gini"])),
            "gini_sal": float(np.median(conc_sal["gini"])),
            "entropy_ISM": float(np.median(conc["entropy"])),
            "top10_ISM": float(np.median(conc["top10"])),
            "sal_vs_ISM_r": float(np.nanmedian(agree)),
            "win_gc": win_gc,
            "cpg_density_win": cpg,
            "sp1_core_rate": sp1,
            "klf_egr_core_rate": klf,
            "top_kmers": top_k,
        }
        print(
            f"   pred_med={summary[name]['pred_med']:+.3f}  seqGC={summary[name]['seqgc_med']:.2f}  "
            f"Gini(ISM)={summary[name]['gini_ISM']:.3f}  entropy={summary[name]['entropy_ISM']:.3f}  "
            f"top10={summary[name]['top10_ISM']:.3f}  sal~ISM r={summary[name]['sal_vs_ISM_r']:.2f}",
            flush=True,
        )
        print(
            f"   hot-window GC={win_gc:.2f}  CpG/bp={cpg:.3f}  Sp1-core={sp1:.2f}  KLF/EGR-core={klf:.2f}",
            flush=True,
        )
        print(f"   top 6-mers: {[(w, c) for w, c in top_k[:12]]}", flush=True)

    np.savez_compressed(f"{REPO}/outputs/gc_bias/ism_highgc_vs_genomic.npz", **out)
    import json

    with open(f"{REPO}/outputs/gc_bias/ism_highgc_vs_genomic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote npz + summary.json", flush=True)

    # ---------------------------------------------------------------- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 11))
    colors = {"highgc": "#d62728", "midgc": "#1f77b4", "genomic": "#2ca02c"}
    labels = {
        "highgc": "high-GC random (GC~0.8)",
        "midgc": "mid-GC random (GC~0.5)",
        "genomic": "Gosai genomic test",
    }

    # Row 1: example per-position ISM profiles (3 examples each group)
    for col, name in enumerate(["highgc", "midgc", "genomic"]):
        prof = out[f"{name}_prof"]
        # pick 3 with highest prediction to show where "activity" sits
        order = np.argsort(out[f"{name}_pred"])[::-1][:3]
        ax = fig.add_subplot(3, 3, col + 1)
        for r, idx in enumerate(order):
            ax.plot(prof[idx] + r * prof.max() * 0.0, color=colors[name], alpha=0.5 + 0.2 * r, lw=0.8)
        ax.set_title(f"{labels[name]}\nISM profile, top-3 predicted", fontsize=9)
        ax.set_xlabel("position")
        ax.set_ylabel("mean |Δ pred|")

    # Row 2: concentration metric distributions
    metrics = [("gini", "Gini (ISM)"), ("top10", "top-10% mass (ISM)"), ("entropy", "norm. entropy (ISM)")]
    for j, (mk, mlabel) in enumerate(metrics):
        ax = fig.add_subplot(3, 3, 4 + j)
        data = [out[f"{n}_{mk}"] for n in ["highgc", "midgc", "genomic"]]
        bp = ax.boxplot(data, labels=["hiGC", "midGC", "gen"], patch_artist=True, showfliers=False)
        for patch, n in zip(bp["boxes"], ["highgc", "midgc", "genomic"]):
            patch.set_facecolor(colors[n])
            patch.set_alpha(0.6)
        ax.set_title(mlabel, fontsize=10)

    # Row 3: hot-window composition summary bars
    ax = fig.add_subplot(3, 3, 7)
    names = ["highgc", "midgc", "genomic"]
    sp1 = [summary[n]["sp1_core_rate"] for n in names]
    klf = [summary[n]["klf_egr_core_rate"] for n in names]
    xpos = np.arange(3)
    ax.bar(xpos - 0.2, sp1, 0.4, label="Sp1/GC-box core", color="#9467bd")
    ax.bar(xpos + 0.2, klf, 0.4, label="KLF/EGR core", color="#8c564b")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["hiGC", "midGC", "gen"])
    ax.set_ylabel("frac hot windows w/ core")
    ax.set_title("GC-box / KLF-EGR core hits", fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(3, 3, 8)
    cpg = [summary[n]["cpg_density_win"] for n in names]
    wgc = [summary[n]["win_gc"] for n in names]
    ax.bar(xpos - 0.2, cpg, 0.4, label="CpG / bp", color="#e377c2")
    ax.bar(xpos + 0.2, wgc, 0.4, label="hot-win GC", color="#7f7f7f")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["hiGC", "midGC", "gen"])
    ax.set_title("hot-window CpG & GC", fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(3, 3, 9)
    for n in names:
        ax.scatter(out[f"{n}_seqgc"], out[f"{n}_gini"], s=5, alpha=0.3, color=colors[n], label=labels[n])
    ax.set_xlabel("sequence GC")
    ax.set_ylabel("Gini (ISM)")
    ax.set_title("concentration vs GC", fontsize=10)
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(f"{REPO}/outputs/gc_bias/ism_highgc_vs_genomic.png", dpi=130)
    print("wrote png", flush=True)


if __name__ == "__main__":
    main()
