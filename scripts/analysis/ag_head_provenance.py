"""AG-HepG2 head provenance check: is `ag_s2_best_hp/hepg2` genuinely HepG2-trained?

The HepG2 checkpoint carries a head literally named `s2f_exp1_s2_k562_42` (the k562
code path was reused for the build), so the label/weights pairing is worth a direct
test. We insert a strong liver TF motif (HNF4A, not a K562 driver) and a strong
erythroid TF motif (GATA1, a K562 driver) into the centre of a fixed set of neutral
(composition-shuffled genomic) backgrounds, and measure the mean change in predicted
activity for (a) the AG-HepG2 model and (b) the canonical AG-K562 oracle_v3.

Genuine HepG2 weights => larger HNF4A activation under the HepG2 model than the K562
model; the K562 model should relatively favour GATA1. Run once per --model
{hepg2,k562}; the two runs regenerate identical backgrounds/instances (fixed seeds),
so the resulting Deltas are directly comparable. A combine pass reads both npz.

Motif consensus come from the Sahu HepG2 random-STARR table (same source and
consensus() extraction as scripts/analysis/sahu_hepg2_motif_compare.py).
"""

from __future__ import annotations

import gzip
import os
import re
import sys

import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")

L = 200
N_BG = 500
IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "S": "GC",
    "W": "AT", "K": "GT", "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT",
    "V": "ACG", "N": "ACGT",
}
SAHU = (
    f"{REPO}/outputs/sahu_gse180153/"
    "GSM5454442_random-STARR-seq_enhancer_HepG2.p1e-6matches.lfc.tsv.gz"
)
# Target TFs: HNF4A = liver/HepG2-specific; GATA1 = erythroid/K562-specific.
TARGET_TFS = ["HNF4A", "GATA1"]


def consensus(pfm: str):
    m = re.search(r"_([ACGTRYSWKMBDHVN]{4,})_m\d+_c\d+", pfm)
    return m.group(1) if m else None


def load_target_consensus() -> dict[str, tuple[str, float]]:
    """strongest-|effect| consensus per target TF from the Sahu HepG2 table."""
    best: dict[str, tuple[str, float]] = {}
    with gzip.open(SAHU, "rt") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            name = p[0].upper()
            if name not in TARGET_TFS:
                continue
            c = consensus(p[1])
            try:
                e = float(p[4])
            except ValueError:
                continue
            if c and (name not in best or abs(e) > abs(best[name][1])):
                best[name] = (c, e)
    return best


def make_backgrounds(rng) -> list[str]:
    """Neutral backgrounds: genomic windows with bases mono-shuffled (motifs
    destroyed, per-sequence composition preserved)."""
    z = np.load(
        f"{REPO}/outputs/grid_pools/k562_cumulative/genomic_d100000.npz", allow_pickle=True
    )
    seqs = [str(x) for x in z["sequences"]]
    sel = rng.choice(len(seqs), size=N_BG, replace=False)
    out = []
    for i in sel:
        s = seqs[i][:L]
        a = np.array(list(s))
        rng.shuffle(a)
        out.append("".join(a))
    return out


def plant(bg: list[str], inst: str) -> list[str]:
    half = len(inst) // 2
    c0 = L // 2
    return [(b[: c0 - half] + inst + b[c0 - half + len(inst):])[:L] for b in bg]


def build_predict(model_name: str):
    if model_name == "hepg2":
        from scripts.analysis.gc_bias_ag_hepg2 import build_hepg2_scorer

        return build_hepg2_scorer()
    if model_name == "k562":
        from experiments.exp1_1_scaling import _load_oracle
        from scripts.build_pool import _pin_ag_s2_oracle
        from scripts.generate_labeled_pools import _label_sequences

        _pin_ag_s2_oracle("oracle_v3")
        oracle = _load_oracle("k562", oracle_type="ag_s2")
        return lambda seqs: np.asarray(_label_sequences(oracle, seqs), dtype=np.float32).ravel()
    raise ValueError(model_name)


def main():
    model_name = sys.argv[1]  # hepg2 | k562
    cons = load_target_consensus()
    print(f"[prov:{model_name}] target consensus: {cons}", flush=True)

    # Deterministic backgrounds + motif instances (identical across both model runs).
    rng = np.random.default_rng(1234)
    bg = make_backgrounds(rng)
    inst_rng = np.random.default_rng(99)
    instances = {}
    for tf in TARGET_TFS:
        c = cons[tf][0]
        instances[tf] = "".join(inst_rng.choice(list(IUPAC[b])) for b in c)
    print(f"[prov:{model_name}] instances: {instances}", flush=True)

    predict = build_predict(model_name)

    bg_pred = np.asarray(predict(bg), dtype=np.float64).ravel()
    out = {"model": model_name, "bg_pred": bg_pred}
    print(f"[prov:{model_name}] bg mean={bg_pred.mean():+.4f} n={len(bg_pred)}", flush=True)

    for tf in TARGET_TFS:
        planted = plant(bg, instances[tf])
        pp = np.asarray(predict(planted), dtype=np.float64).ravel()
        paired = pp - bg_pred  # same background per entry
        out[f"{tf}_planted_pred"] = pp
        out[f"{tf}_consensus"] = cons[tf][0]
        out[f"{tf}_instance"] = instances[tf]
        out[f"{tf}_sahu_effect"] = cons[tf][1]
        print(
            f"[prov:{model_name}] {tf}: consensus={cons[tf][0]} inst={instances[tf]} "
            f"Delta(mean-diff)={pp.mean() - bg_pred.mean():+.4f} "
            f"paired mean={paired.mean():+.4f} sem={paired.std(ddof=1) / np.sqrt(len(paired)):.4f}",
            flush=True,
        )

    os.makedirs(f"{REPO}/outputs/gc_bias", exist_ok=True)
    path = f"{REPO}/outputs/gc_bias/provenance_{model_name}.npz"
    np.savez_compressed(path, **out)
    print(f"[prov:{model_name}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
