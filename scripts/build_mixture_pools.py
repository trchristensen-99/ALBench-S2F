"""Build mixed-reservoir pools by interleaving pools that are already labelled.

The reservoir ladder (rp47) ranks single strategies, but the library we actually have to specify is one corpus, and
the best corpus is not obviously the best single reservoir: at 300k `motif_shared_core` wins the designed set by
0.148 over genomic while `motif_ct_enriched` wins held-out genomic, and nothing says a mixture cannot take both.

Sources are interleaved round-robin rather than concatenated, so EVERY prefix of a mixture has the same composition
as the whole -- without that, a 100k prefix of a 300k half-and-half pool would be one reservoir only, and the size
axis would silently become a composition axis. No oracle call is needed: every source is already oracle_v3-labelled,
which is what makes this experiment cost training time and nothing else.

    python scripts/build_mixture_pools.py [--dry]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REPS = REPO / "outputs/grid_pools/k562_reps"
OUT = REPO / "outputs/grid_pools/k562_mixtures"
SIZE = 300_000
# the training script asserts the pool file is exactly D long, so each rung gets its own file; because the
# sources are interleaved, the 100k file is the first 100k rows of the 300k one and holds the same mixture
SIZES = (100_000, 300_000)
MIXES = {
    # name -> the reservoirs to draw from in equal parts
    "mix_ms_mc": ("motif_shared_core", "motif_ct_enriched"),
    "mix_ms_gen": ("motif_shared_core", "genomic"),
    "mix_ms_mc_gen": ("motif_shared_core", "motif_ct_enriched", "genomic"),
}


def interleave(parts: list[tuple[list[str], np.ndarray]], n: int):
    """Round-robin the sources so every prefix holds the same mixture."""
    seqs: list[str] = []
    labels: list[float] = []
    k = len(parts)
    for i in range(n):
        s, y = parts[i % k]
        j = i // k
        seqs.append(s[j])
        labels.append(float(y[j]))
    return seqs, np.asarray(labels, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    man = {}
    for name, sources in MIXES.items():
        for rep in (1, 2, 3):
            paths = [REPS / f"{s}_r{rep}_d{SIZE}.npz" for s in sources]
            if not all(p.exists() for p in paths):
                print(f"  {name} r{rep}: missing {[p.name for p in paths if not p.exists()]}")
                continue
            parts, oids = [], set()
            for p in paths:
                z = np.load(p, allow_pickle=True)
                parts.append(
                    ([str(s) for s in z["sequences"]], z["oracle_labels"].astype(np.float32))
                )
                oids.add(str(z["oracle_id"]))
            assert len(oids) == 1, f"mixing pools labelled by different oracles: {oids}"
            per_source = SIZE // len(sources)
            assert all(len(p[0]) >= per_source for p in parts), (
                "a source is too small for its share"
            )
            seqs, y = interleave(parts, SIZE)
            oid = oids.pop()
            for D in SIZES:
                s_d, y_d = seqs[:D], y[:D]
                n_uniq = len(set(s_d))
                dest = OUT / f"{name}_r{rep}_d{D}.npz"
                print(
                    f"  {dest.name:34s} {len(s_d):,} rows, {n_uniq:,} distinct, "
                    f"{D // len(sources):,} per source, label {y_d.mean():+.3f}+-{y_d.std():.3f}"
                    + ("  [DRY]" if args.dry else "")
                )
                man[dest.name] = {
                    "sources": list(sources),
                    "per_source": D // len(sources),
                    "n": len(s_d),
                    "n_distinct": n_uniq,
                    "label_mean": round(float(y_d.mean()), 4),
                    "label_sd": round(float(y_d.std()), 4),
                    "oracle_id": oid,
                    "composition": "round-robin, so every prefix has the same mixture",
                }
                if args.dry:
                    continue
                OUT.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmpnew.npz")
                np.savez(
                    tmp,
                    sequences=np.asarray(s_d, dtype=object),
                    oracle_labels=y_d,
                    oracle_id=oid,
                    strategy=name,
                    seed=np.int64(rep),
                )
                tmp.replace(dest)
    if not args.dry and man:
        (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
        print("wrote", OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
