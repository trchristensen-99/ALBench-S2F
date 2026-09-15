import glob
import numpy as np

fs = sorted(glob.glob("outputs/screen/cache/*__labeled.npz"))
print(f"labelled files: {len(fs)}")
bad = []
ns, means, sds = [], [], []
for f in fs:
    z = np.load(f, allow_pickle=True)
    lab = np.asarray(z["oracle_labels"], dtype=float).ravel()
    nseq = len(z["sequences"])
    ok = (lab.size == nseq) and np.isfinite(lab).all() and len(np.unique(lab)) > 100
    if not ok:
        bad.append((f.split("/")[-1], nseq, lab.size, len(np.unique(lab))))
    ns.append(nseq)
    means.append(lab.mean())
    sds.append(lab.std())
print(f"  n_sequences: min {min(ns):,} max {max(ns):,}")
print(
    f"  label mean across files: {np.mean(means):+.3f} (range {min(means):+.3f}..{max(means):+.3f})"
)
print(f"  label sd   across files: {np.mean(sds):.3f} (range {min(sds):.3f}..{max(sds):.3f})")
print(f"  SUSPECT files: {len(bad)}")
for b in bad[:5]:
    print("   ", b)
for f in fs[:3]:
    z = np.load(f, allow_pickle=True)
    lab = np.asarray(z["oracle_labels"], dtype=float)
    print(
        f"  {f.split('/')[-1][:48]:<48} n={lab.size:>6} uniq={len(np.unique(lab)):>6} "
        f"mean={lab.mean():+.3f}"
    )
