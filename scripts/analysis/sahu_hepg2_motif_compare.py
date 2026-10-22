"""Compare our HepG2 models' motif effects to Sahu 2022 experimental STARR-seq.

Sahu's HepG2 random-enhancer table gives, per TF motif, the log2 enrichment of that
motif in transcribed (active) random sequences vs input (the `effect`). We compute
the model analog by INSERTION: drop a consensus instance of each motif into the
centre of many random backgrounds (held at 50% GC) and measure the mean change in
predicted HepG2 activity. Two questions:
  concordance - does our model rank motifs like the experiment (HNF4A/HNF1A high)?
  GC over-amplification - is our model's motif-GC vs effect slope steeper than the
    experiment's (+0.375)? If so, the model assigns activity to GC beyond biology.
"""
from __future__ import annotations
import gzip, re, sys
import numpy as np
from scipy.stats import spearmanr

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
GC = {'A':0,'T':0,'G':1,'C':1,'R':.5,'Y':.5,'S':1,'W':0,'K':.5,'M':.5,
      'B':2/3,'D':1/3,'H':1/3,'V':2/3,'N':.5}
IUPAC = {'A':'A','C':'C','G':'G','T':'T','R':'AG','Y':'CT','S':'GC','W':'AT',
         'K':'GT','M':'AC','B':'CGT','D':'AGT','H':'ACT','V':'ACG','N':'ACGT'}
TABLE = f"{REPO}/outputs/sahu_gse180153/GSM5454442_random-STARR-seq_enhancer_HepG2.p1e-6matches.lfc.tsv.gz"
N_BG = 150
L = 200
rng = np.random.default_rng(0)


def consensus(pfm):
    m = re.search(r'_([ACGTRYSWKMBDHVN]{4,})_m\d+_c\d+', pfm)
    return m.group(1) if m else None


def load_sahu():
    rows = []
    with gzip.open(TABLE, "rt") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            c = consensus(p[1])
            try:
                e = float(p[4])
            except ValueError:
                continue
            if c:
                rows.append((p[0], c, e))
    # one entry per motif-consensus (dedupe exact repeats, keep strongest |effect|)
    best = {}
    for name, c, e in rows:
        k = (name, c)
        if k not in best or abs(e) > abs(best[k][1]):
            best[k] = (c, e)
    return [(k[0], v[0], v[1]) for k, v in best.items()]


def instance(cons):
    return "".join(rng.choice(list(IUPAC[b])) for b in cons)


def main():
    import torch
    from models.legnet_student import LegNetStudent, TrainConfig
    m = torch.load(f"{REPO}/outputs/gc_bias/legnet_percell/legnet_hepg2.pt",
                   map_location="cuda", weights_only=False).eval()
    st = LegNetStudent(train_config=TrainConfig(num_workers=0), ensemble_size=1, device="cuda")
    st.models = [m]

    motifs = load_sahu()
    print(f"  motifs: {len(motifs)}", flush=True)
    bg = ["".join(rng.choice(list("ACGT"), size=L, p=[.25,.25,.25,.25])) for _ in range(N_BG)]
    base = st.predict(bg)
    base_mean = float(np.mean(base))

    names, sahu_eff, our_eff, mgc = [], [], [], []
    for i, (name, cons, eff) in enumerate(motifs):
        inst = instance(cons); half = len(inst) // 2; c0 = L // 2
        planted = [b[:c0 - half] + inst + b[c0 - half + len(inst):] for b in bg]
        planted = [s[:L] for s in planted]
        d = float(np.mean(st.predict(planted)) - base_mean)
        names.append(name); sahu_eff.append(eff); our_eff.append(d)
        mgc.append(np.mean([GC[b] for b in cons]))
        if i % 200 == 0:
            print(f"    {i}/{len(motifs)}", flush=True)
    sahu_eff, our_eff, mgc = map(np.array, (sahu_eff, our_eff, mgc))
    rc, pc = spearmanr(sahu_eff, our_eff)
    rg_ours, _ = spearmanr(mgc, our_eff)
    rg_exp, _ = spearmanr(mgc, sahu_eff)
    print(f"\n  CONCORDANCE  Spearman(our Δ, Sahu effect) = {rc:+.3f} (p={pc:.1e})")
    print(f"  GC-dependence  motif-GC vs effect:  ours={rg_ours:+.3f}   experiment={rg_exp:+.3f}")
    np.savez(f"{REPO}/outputs/sahu_gse180153/hepg2_legnet_motif_compare.npz",
             names=np.array(names), sahu=sahu_eff, ours=our_eff, gc=mgc)
    # top experimental motifs: does our model agree?
    order = np.argsort(sahu_eff)[::-1][:8]
    print("\n  top experimental HepG2 motifs — Sahu vs our LegNet Δ:")
    for j in order:
        print(f"    {names[j]:<14} Sahu={sahu_eff[j]:+.2f}  ourΔ={our_eff[j]:+.3f}  GC={mgc[j]:.2f}")


if __name__ == "__main__":
    main()
