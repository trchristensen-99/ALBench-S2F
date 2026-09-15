"""How are the paired eval sets connected? Determines whether 80/10/10 is even possible."""

import csv
from collections import defaultdict
from pathlib import Path

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
for fname in ("motif_perturbation.csv", "motif_tiling_seqs.csv", "all_SNVs_seqs.csv"):
    f = REPO / "data/yeast/test_subset_ids" / fname
    rows = list(csv.DictReader(open(f, newline="")))
    refs = defaultdict(int)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n_rec = 0
    for r in rows:
        a = (r.get("alt_sequence") or "").strip().upper()
        b = (r.get("ref_sequence") or "").strip().upper()
        if a and b:
            union(a, b)
            refs[b] += 1
            n_rec += 1
    comps = defaultdict(list)
    for s in list(parent):
        comps[find(s)].append(s)
    sizes = sorted((len(v) for v in comps.values()), reverse=True)
    print(f"{fname}")
    print(f"  records={n_rec:,}  distinct refs={len(refs):,}  components={len(comps):,}")
    print(f"  largest components: {sizes[:5]}")
    tot = sum(sizes)
    print(f"  largest component holds {100 * sizes[0] / max(tot, 1):.1f}% of the class' sequences")
    if len(comps) < 10:
        print(
            "  -> FEWER THAN 10 COMPONENTS: a 10-fold stratified split is impossible "
            "without breaking pairs"
        )
    print()
