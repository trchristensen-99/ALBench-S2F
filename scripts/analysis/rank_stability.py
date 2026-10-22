import json, glob, collections, itertools
import numpy as np
from scipy.stats import kendalltau
BAKE="outputs/hp_step1_bakeoff_e100"; D=30000
RES=["genomic","dinuc_shuffle","evoaug_heavy","gc_matched","motif_planted_v2",
     "phylogenetic_zoonomia","uncertainty_guided","diversity_guided"]
def band(v,edges):
    for i,e in enumerate(edges):
        if v<=e: return i
    return len(edges)
def sig(hp):
    return (hp.get("block_class","?"), hp.get("optimizer","?"),
            band(hp.get("n_layers",0),[6,9,12]),
            band(hp.get("width_base",0),[128,256,512]),
            band(hp.get("lr",0),[3e-4,1e-3,3e-3]))
# sig -> res -> (best_val, test_genomic_pearson)
tab=collections.defaultdict(dict)
for r in RES:
    for f in glob.glob(f"{BAKE}/k562_{r}_d{D}/seed42_0/*/*_meta.json"):
        try: d=json.load(open(f))
        except Exception: continue
        hp=d.get("hp"); vp=d.get("best_val_pearson") or d.get("val_pearson")
        ps=d.get("per_set_metrics") or {}
        g=(ps.get("genomic") or {}).get("pearson")
        if not hp or vp is None or g is None: continue
        s=sig(hp)
        if r not in tab[s] or vp>tab[s][r][0]: tab[s][r]=(vp,g)
full=[s for s,v in tab.items() if len(v)==8]
print(f"signatures with all 8 reservoirs AND test metrics: {len(full)}")
rankings={}
for s in full:
    sc=np.array([tab[s][r][1] for r in RES])
    rankings[s]=sc
taus=[]
for a,b in itertools.combinations(full,2):
    t=kendalltau(rankings[a],rankings[b]).correlation
    if not np.isnan(t): taus.append(t)
taus=np.array(taus)
print(f"\npairwise Kendall tau across {len(full)} signatures ({len(taus)} pairs):")
print(f"  mean {taus.mean():.3f}   median {np.median(taus):.3f}   min {taus.min():.3f}   max {taus.max():.3f}")
print(f"  frac >= 0.8 : {(taus>=0.8).mean():.2f}")
print(f"  frac >= 0.6 : {(taus>=0.6).mean():.2f}")
M=np.vstack([rankings[s] for s in full])
print("\nmean test-genomic pearson per reservoir (avg over the 25 common signatures):")
order=np.argsort(-M.mean(0))
for i in order: print(f"  {RES[i]:26s} {M[:,i].mean():.4f}  (sd across sigs {M[:,i].std():.4f})")
print("\nhow often each reservoir takes rank 1:")
w=collections.Counter(RES[i] for i in M.argmax(1))
for k,v in w.most_common(): print(f"  {k:26s} {v}/{len(full)}")
json.dump({"n_sigs":len(full),"tau_mean":float(taus.mean()),"tau_min":float(taus.min()),
           "res":RES,"mean_test":M.mean(0).tolist()}, open("/tmp/rank_stab.json","w"), indent=1)
