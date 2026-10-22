"""HepG2 GC-bias mechanism + Sahu concordance, per model.

Three analyses, --model {legnet,malinois}, HepG2:
  (1) GC curve: predicted activity of random DNA at GC 0-100% (does this model show
      the oracle's random>genomic/high-GC rise?).
  (2) Composition-shuffle control: predict each random sequence AND a mono-shuffle
      of it (same bases, GC preserved EXACTLY, all motifs/k-mer structure destroyed).
      If activity survives shuffling, the prediction is driven by base COMPOSITION
      (GC), not motifs/arrangement — i.e. the high-GC activity is compositional.
  (3) Sahu concordance: motif-insertion effect vs the experimental HepG2 effect.
"""
from __future__ import annotations
import gzip, re, sys
import numpy as np
from scipy.stats import spearmanr

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO)
GCW = {'A':0,'T':0,'G':1,'C':1,'R':.5,'Y':.5,'S':1,'W':0,'K':.5,'M':.5,
       'B':2/3,'D':1/3,'H':1/3,'V':2/3,'N':.5}
IUPAC = {'A':'A','C':'C','G':'G','T':'T','R':'AG','Y':'CT','S':'GC','W':'AT',
         'K':'GT','M':'AC','B':'CGT','D':'AGT','H':'ACT','V':'ACG','N':'ACGT'}
TABLE = f"{REPO}/outputs/sahu_gse180153/GSM5454442_random-STARR-seq_enhancer_HepG2.p1e-6matches.lfc.tsv.gz"
L, GC_LEVELS = 200, [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
rng = np.random.default_rng(0)
BASES = np.array(list("ACGT"))


def rand_at_gc(gc, n):
    p = [(1-gc)/2, gc/2, gc/2, (1-gc)/2]
    return ["".join(BASES[r]) for r in rng.choice(4, size=(n, L), p=p)]


def make_predict(model_name):
    import torch
    if model_name == "legnet":
        from models.legnet_student import LegNetStudent, TrainConfig
        m = torch.load(f"{REPO}/outputs/gc_bias/legnet_percell/legnet_hepg2.pt",
                       map_location="cuda", weights_only=False).eval()
        st = LegNetStudent(train_config=TrainConfig(num_workers=0), ensemble_size=1, device="cuda")
        st.models = [m]
        return lambda seqs: np.asarray(st.predict(list(seqs)), dtype=np.float64)
    if model_name == "malinois":
        sys.path.insert(0, f"{REPO}/scripts")
        from eval_pretrained_malinois import load_pretrained_malinois
        from add_malinois_random_bias import one_hot_encode, pad_to_600bp
        model = load_pretrained_malinois().cuda().eval()
        def pred(seqs, idx=1):   # HepG2 = output index 1
            out = []
            for i in range(0, len(seqs), 2048):
                b = seqs[i:i+2048]
                x = torch.tensor(np.stack([pad_to_600bp(one_hot_encode(s)) for s in b])).float().cuda()
                with torch.no_grad():
                    out.append(model(x)[:, idx].cpu().numpy())
            return np.concatenate(out).astype(np.float64)
        return pred
    raise ValueError(model_name)


def consensus(pfm):
    m = re.search(r'_([ACGTRYSWKMBDHVN]{4,})_m\d+_c\d+', pfm)
    return m.group(1) if m else None


def main():
    model_name = sys.argv[1]
    predict = make_predict(model_name)
    out = {"model": model_name, "gc_levels": np.array(GC_LEVELS)}

    # (1) GC curve + (2) composition-shuffle control
    print("  == GC curve + shuffle control ==", flush=True)
    for gc in GC_LEVELS:
        seqs = rand_at_gc(gc, 3000)
        a = predict(seqs)
        sh = ["".join(rng.permutation(list(s))) for s in seqs]   # mono-shuffle: GC exact
        ash = predict(sh)
        out[f"act_gc{int(gc*100)}"] = a
        out[f"actshuf_gc{int(gc*100)}"] = ash
        print(f"    GC={gc:.1f}: act med={np.median(a):+.3f}  shuffled med={np.median(ash):+.3f}"
              f"  Δ(orig-shuf)={np.median(a)-np.median(ash):+.3f}", flush=True)

    # (3) Sahu motif concordance
    print("  == Sahu motif concordance ==", flush=True)
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
    best = {}
    for name, c, e in rows:
        if (name, c) not in best or abs(e) > abs(best[(name, c)][1]):
            best[(name, c)] = (c, e)
    motifs = [(k[0], v[0], v[1]) for k, v in best.items()]
    bg = rand_at_gc(0.5, 150)
    base = float(np.mean(predict(bg)))
    sahu, ours, mgc = [], [], []
    for name, cons, eff in motifs:
        inst = "".join(rng.choice(list(IUPAC[b])) for b in cons)
        h = len(inst)//2; c0 = L//2
        planted = [(b[:c0-h] + inst + b[c0-h+len(inst):])[:L] for b in bg]
        ours.append(float(np.mean(predict(planted)) - base))
        sahu.append(eff); mgc.append(np.mean([GCW[b] for b in cons]))
    sahu, ours, mgc = map(np.array, (sahu, ours, mgc))
    rc, pc = spearmanr(sahu, ours)
    out.update(sahu=sahu, ours=ours, motif_gc=mgc)
    print(f"    concordance Spearman(our Δ, Sahu) = {rc:+.3f} (p={pc:.1e})", flush=True)
    print(f"    motif-GC vs effect: ours={spearmanr(mgc,ours)[0]:+.3f} exp={spearmanr(mgc,sahu)[0]:+.3f}", flush=True)
    np.savez(f"{REPO}/outputs/sahu_gse180153/hepg2_{model_name}_full.npz", **out)
    print(f"  wrote hepg2_{model_name}_full.npz", flush=True)


if __name__ == "__main__":
    main()
