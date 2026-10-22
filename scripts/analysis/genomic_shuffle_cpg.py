"""Why is genomic predicted BELOW GC-matched random (the over-correction)?
Hypothesis: CpG. Genomic DNA is CpG-depleted by arrangement; uniform random at the
same GC is CpG-rich. If the model reads CpG as activity, random > genomic at matched
GC. Test: predict real genomic vs its mono-shuffle (destroys CpG depletion -> CpG
rises to random level) vs dinuc-shuffle (preserves CpG depletion). If mono-shuffle
RAISES genomic toward random while dinuc-shuffle does not, the driver is CpG."""
from __future__ import annotations
import sys, numpy as np, pandas as pd
REPO="/grid/wsbs/home_norepl/christen/ALBench-S2F"; sys.path.insert(0,REPO)
rng=np.random.default_rng(0)
def cpg(s): return s.count("CG")/(len(s)-1)
def mono_shuffle(s): return "".join(rng.permutation(list(s)))
def dinuc_shuffle(s):
    # Altschul-Erikson-ish: shuffle within a dinucleotide-edge graph (simple walk)
    import random; random.seed(int(rng.integers(1e9)))
    # fallback: preserve dinuc counts via greedy Eulerian; use a known-good lib if present
    from collections import defaultdict
    edges=defaultdict(list)
    for a,b in zip(s[:-1],s[1:]): edges[a].append(b)
    for k in edges: rng.shuffle(edges[k])
    start=s[0]; out=[start]; cur=start
    for _ in range(len(s)-1):
        if edges[cur]: out.append(edges[cur].pop()); cur=out[-1]
        else: break
    r="".join(out)
    return r if len(r)==len(s) else s
def main():
    import torch
    from models.legnet_student import LegNetStudent, TrainConfig
    m=torch.load(f"{REPO}/outputs/gc_bias/legnet_percell/legnet_hepg2.pt",map_location="cuda",weights_only=False).eval()
    st=LegNetStudent(train_config=TrainConfig(num_workers=0),ensemble_size=1,device="cuda"); st.models=[m]
    pred=lambda s: np.asarray(st.predict(list(s)),dtype=np.float64)
    df=pd.read_csv(f"{REPO}/data/hepg2/DATA-Table_S2__MPRA_dataset.txt",sep="\t",usecols=["sequence"]).dropna()
    df=df[df["sequence"].str.len()==200]
    gc=np.array([(s.count("G")+s.count("C"))/200 for s in df["sequence"].values])
    BASES=np.array(list("ACGT"))
    for lo in (0.45,0.50,0.55):
        sel=df["sequence"].values[(gc>=lo)&(gc<lo+0.05)][:2000]
        if len(sel)<100: continue
        g=pred(sel); ms=pred([mono_shuffle(s) for s in sel]); ds=pred([dinuc_shuffle(s) for s in sel])
        # uniform random at same mean GC
        mg=np.mean([(s.count("G")+s.count("C"))/200 for s in sel])
        rr=["".join(BASES[r]) for r in rng.choice(4,size=(len(sel),200),p=[(1-mg)/2,mg/2,mg/2,(1-mg)/2])]
        rp=pred(rr)
        print(f"  GC[{lo:.2f}]: genomic={np.median(g):+.3f} (CpG={np.mean([cpg(s) for s in sel]):.3f}) | "
              f"mono-shuf={np.median(ms):+.3f} (CpG={np.mean([cpg(s) for s in [mono_shuffle(x) for x in sel[:200]]]):.3f}) | "
              f"dinuc-shuf={np.median(ds):+.3f} | uniform-random={np.median(rp):+.3f} (CpG={np.mean([cpg(s) for s in rr[:200]]):.3f})",flush=True)
if __name__=="__main__": main()
