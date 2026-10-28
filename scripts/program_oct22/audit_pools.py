"""Integrity audit of the training pools used by the Oct-28 night program (sc31_, al31_): size, nesting, duplicates, labels, and exact-sequence overlap with
the validation set (V0c) and the three test sets. A pool that leaks test or validation sequences, or whose labels are not oracle_v3, would make the
results unusable without any error, so this is run before the results are trusted (cpuq).

    python scripts/program_oct22/audit_pools.py > outputs/audit_pools.txt
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get("ALBENCH_ROOT", "/grid/wsbs/home_norepl/christen/ALBench-S2F"))
sys.path.insert(0, str(R))
GP = R / "outputs/grid_pools"
TS = R / "data/k562/test_sets_ag_s2_chrsplit"
STRATS = [
    "genomic",
    "random",
    "motif_ct_enriched",
    "evoaug_paper2023",
    "zoonomia_orthologs",
    "motif_shared_core",
    "random_gc_sweep",
    "mutagenesis_mr10",
]


def seqs(f, key="sequences"):
    z = np.load(f, allow_pickle=True)
    k = key if key in z.files else z.files[0]
    return [str(s) for s in z[k]]


def gc(ss):
    return float(np.mean([(s.count("G") + s.count("C")) / len(s) for s in ss[:5000]]))


def main() -> None:
    held = {"val_V0c": set(seqs(R / "outputs/val_variants/val_V0.npz"))}
    for name in ("genomic", "ood", "random_32k"):
        f = TS / f"{name}_oracle.npz"
        if f.exists():
            held[name] = set(seqs(f))
    print({k: len(v) for k, v in held.items()})
    base = np.load(GP / "k562_reps/genomic_r1_d30000.npz", allow_pickle=True)
    bs = [str(s) for s in base["sequences"][:30000]]
    files = [
        (f"{st}@{D}", GP / f"k562_nested/{st}_d{D}.npz", D)
        for st in STRATS
        for D in (10_000, 30_000, 100_000)
    ]
    files += [
        (Path(f).stem, Path(f), 100_000)
        for f in sorted(
            glob.glob(str(GP / "k562_acq/res_*_d100000.npz"))
            + glob.glob(str(GP / "k562_acq/sel_*_d100000.npz"))
        )
    ]
    files.append(("genomic_r1_d100000", GP / "k562_reps/genomic_r1_d100000.npz", 100_000))
    bad = 0
    for name, f, D in files:
        if not f.exists():
            print(f"{name:34s} MISSING")
            continue
        z = np.load(f, allow_pickle=True)
        s = [str(x) for x in z["sequences"]]
        y = np.asarray(z["oracle_labels"], float)
        issues = []
        if len(s) != D:
            issues.append(f"size {len(s)} != {D}")
        if str(z["oracle_id"]) != "oracle_v3":
            issues.append(f"oracle {z['oracle_id']}")
        if not np.isfinite(y).all():
            issues.append("non-finite labels")
        uniq = len(set(s)) / len(s)
        if uniq < 0.995:
            issues.append(f"unique fraction {uniq:.4f}")
        if len({len(x) for x in s}) != 1 or len(s[0]) != 200:
            issues.append("sequence length")
        ov = {k: len(set(s) & v) for k, v in held.items()}
        if any(ov.values()):
            issues.append(f"overlap with held-out sets {ov}")
        if (name.startswith(("res_", "sel_")) or name == "genomic_r1_d100000") and s[:30000] != bs:
            issues.append("first 30k differ from the genomic r1 base")
        bad += bool(issues)
        print(
            f"{name:34s} n={len(s):6d} label mean {y.mean():.3f} sd {y.std():.3f} GC {gc(s):.3f} unique {uniq:.4f} {'OK' if not issues else 'ISSUES: ' + '; '.join(issues)}"
        )
    print("pools with issues:", bad)


if __name__ == "__main__":
    main()
