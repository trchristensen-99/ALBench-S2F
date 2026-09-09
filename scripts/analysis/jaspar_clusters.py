"""What ARE the near-duplicate PFMs that clustering collapses?

Prints the actual clusters so the filtering decision can be judged on the real
groupings rather than on the abstract idea of 'near-duplicates'.
"""

import sys, itertools, collections
import numpy as np

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from albench.motifs import vocabulary as V

ms = V.build(cluster_at=None, trim_ic=0.0)  # human CORE, unclustered
print(f"human CORE, unclustered: {len(ms)} motifs\n")


def groups(motifs, thr):
    """Same greedy pass as cluster(), but record who got absorbed into whom."""
    order = sorted(motifs, key=lambda m: -m.info_content)
    reps, members = [], []
    for m in order:
        hit = None
        for i, r in enumerate(reps):
            if V._pwm_similarity(m.pwm, r.pwm) >= thr:
                hit = i
                break
        if hit is None:
            reps.append(m)
            members.append([m])
        else:
            members[hit].append(m)
    return reps, members


for thr in (0.90, 0.80):
    reps, members = groups(ms, thr)
    sizes = np.array([len(g) for g in members])
    print("=" * 78)
    print(
        f"cluster_at={thr}: {len(ms)} -> {len(reps)} clusters | "
        f"singletons {int((sizes == 1).sum())}, "
        f"size>=2 {int((sizes >= 2).sum())}, max size {sizes.max()}"
    )
    print("=" * 78)
    # Are absorbed members the same TF, or different TFs?
    same_tf = diff_tf = 0
    for g in members:
        base = g[0].name.upper().replace("(-)", "")
        for m in g[1:]:
            n = m.name.upper()
            # same gene symbol, or one contained in the other (paralogue families)
            if n == base or n in base or base in n:
                same_tf += 1
            else:
                diff_tf += 1
    print(f"absorbed members: {same_tf} same-TF-name, {diff_tf} different-TF-name\n")
    order = np.argsort(-sizes)
    print(f"10 largest clusters at thr={thr}:")
    for i in order[:10]:
        g = members[i]
        names = ", ".join(f"{m.name}({m.mid})" for m in g[:9])
        sims = [V._pwm_similarity(m.pwm, g[0].pwm) for m in g[1:6]]
        print(f"  [{len(g):>2}] rep={g[0].name:<14} IC={g[0].info_content:5.1f}  L={g[0].length}")
        print(f"       members: {names}{' ...' if len(g) > 9 else ''}")
        print(f"       consensus: " + " | ".join(m.consensus for m in g[:4]))
        print(f"       sim to rep: " + " ".join(f"{s:.3f}" for s in sims))
    print()

# How much IC / length diversity is thrown away inside clusters?
reps, members = groups(ms, 0.90)
multi = [g for g in members if len(g) >= 2]
print("=" * 78)
print("WHAT IS DISCARDED by keeping only the representative")
print("=" * 78)
print(f"{len(multi)} clusters have >=2 members, holding {sum(len(g) for g in multi)} motifs total")
ic_spread, len_spread, cons_distinct = [], [], []
for g in multi:
    ics = [m.info_content for m in g]
    L = [m.length for m in g]
    ic_spread.append(max(ics) - min(ics))
    len_spread.append(max(L) - min(L))
    cons_distinct.append(len({m.consensus for m in g}))
print(
    f"  within-cluster IC spread:     median {np.median(ic_spread):.2f} bits, max {max(ic_spread):.2f}"
)
print(
    f"  within-cluster length spread: median {np.median(len_spread):.0f} bp, max {max(len_spread)}"
)
print(
    f"  distinct consensus strings per cluster: median {np.median(cons_distinct):.0f}, max {max(cons_distinct)}"
)
n_same = sum(1 for g in multi if len({m.consensus for m in g}) == 1)
print(f"  clusters where ALL members share one consensus: {n_same} of {len(multi)}")
