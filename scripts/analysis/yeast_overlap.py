"""Do the yeast genomic/OOD eval sequences already appear in the oracle's training data?"""

import numpy as np

train = set()
with open("data/yeast/train.txt") as fh:
    for line in fh:
        s = line.split("\t", 1)[0].strip()
        if s:
            train.add(s)
print(f"train sequences: {len(train):,} distinct")

for name in ("genomic", "random"):
    z = np.load(f"data/yeast/test_sets/{name}_oracle.npz", allow_pickle=True)
    seqs = [str(s).strip() for s in z["sequences"]]
    exact = sum(1 for s in seqs if s in train)
    # train.txt is 110bp, test sets are 150bp -> compare the shared core too
    cores = {s[20:-20] for s in train} if len(next(iter(train))) != len(seqs[0]) else set()
    core_hit = sum(1 for s in seqs if s[20:-20] in cores) if cores else 0
    print(
        f"  {name:<8} n={len(seqs):>5} len={len(seqs[0])}  "
        f"exact matches in train: {exact}  core matches: {core_hit}"
    )

lab = np.load("data/yeast/test_sets/genomic_oracle.npz", allow_pickle=True)
t = np.asarray(lab["true_labels"], float)
o = np.asarray(lab["oracle_labels"], float)
ok = np.isfinite(t) & np.isfinite(o)
from scipy.stats import pearsonr

print(f"\ngenomic eval set: n={ok.sum()}  true mean={t[ok].mean():.2f} sd={t[ok].std():.2f}")
print(f"  existing oracle_labels vs true_labels r = {pearsonr(o[ok], t[ok])[0]:.4f}")
se = (1 - pearsonr(o[ok], t[ok])[0] ** 2) / np.sqrt(ok.sum() - 1)
print(f"  approx SE on r at this n: +-{se:.4f}  -> differences below ~{2 * se:.3f} are noise")
