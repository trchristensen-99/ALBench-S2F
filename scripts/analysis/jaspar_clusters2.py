"""Does the overlap requirement fix the spurious clustering?

Compares the old permissive similarity (4-position floor) against overlap-aware
settings, and reports whether absorbed cluster members are the same TF family or
unrelated factors -- which is the thing that tells us the clustering is real.
"""

import sys
import numpy as np

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from albench.motifs import vocabulary as V

ms = V.build(cluster_at=None, trim_ic=0.0)
print(f"human CORE, unclustered: {len(ms)} motifs\n")


def sim_factory(min_ov, frac):
    def f(x, y):
        return V._pwm_similarity(x, y, min_overlap=min_ov, min_overlap_frac=frac)

    return f


def groups(motifs, thr, sim):
    order = sorted(motifs, key=lambda m: -m.info_content)
    reps, members = [], []
    for m in order:
        hit = None
        for i, r in enumerate(reps):
            if sim(m.pwm, r.pwm) >= thr:
                hit = i
                break
        if hit is None:
            reps.append(m)
            members.append([m])
        else:
            members[hit].append(m)
    return members


def family(name):
    """Crude TF-family key: strip trailing digits and composite partners."""
    n = name.upper().replace("(-)", "").split("::")[0]
    return n.rstrip("0123456789").rstrip("-_.") or n


SETTINGS = [
    ("OLD (4-pos floor, no frac)", 4, 0.0),
    ("min_ov=6, frac=0.5", 6, 0.5),
    ("min_ov=6, frac=0.8 (new default)", 6, 0.8),
    ("min_ov=8, frac=1.0 (strict)", 8, 1.0),
]

for label, mo, fr in SETTINGS:
    sim = sim_factory(mo, fr)
    for thr in (0.90,):
        mem = groups(ms, thr, sim)
        sizes = np.array([len(g) for g in mem])
        same_fam = diff_fam = 0
        for g in mem:
            f0 = family(g[0].name)
            for m in g[1:]:
                if family(m.name) == f0:
                    same_fam += 1
                else:
                    diff_fam += 1
        tot = same_fam + diff_fam
        print(
            f"{label:<36} thr={thr}: {len(ms)} -> {len(mem):>3} clusters | "
            f"max size {sizes.max():>2} | absorbed {tot:>3} "
            f"({100 * same_fam / max(tot, 1):>4.0f}% same family)"
        )

print("\n" + "=" * 78)
print("Largest clusters under the NEW default (min_ov=6, frac=0.8)")
print("=" * 78)
sim = sim_factory(6, 0.8)
mem = groups(ms, 0.90, sim)
sizes = np.array([len(g) for g in mem])
for i in np.argsort(-sizes)[:8]:
    g = mem[i]
    print(f"  [{len(g):>2}] {g[0].name:<16} L={g[0].length}")
    print(f"       {', '.join(m.name for m in g[:10])}{' ...' if len(g) > 10 else ''}")
    print(f"       consensus: {' | '.join(m.consensus for m in g[:4])}")

print("\n" + "=" * 78)
print("Vocabulary size vs threshold, NEW similarity")
print("=" * 78)
for thr in (0.95, 0.90, 0.85, 0.80, 0.75):
    mem = groups(ms, thr, sim)
    sz = np.array([len(g) for g in mem])
    print(
        f"  cluster_at={thr}: {len(mem):>3} clusters, "
        f"{int((sz == 1).sum()):>3} singletons, max size {sz.max():>2}, "
        f"{int(sz.sum())} PFMs retained if members kept"
    )
