"""Pure in-place RE-SCORE of reservoir caches with the canonical AG_S2 oracle.

User-chosen approach (2026-06-09): re-label the EXISTING cached sequences with the
canonical `full856k_clean` oracle and OVERRIDE the stale labels in place — sequences
are kept byte-identical (no regeneration), so even oracle-coupled strategies like
prm_uncertainty_1pct keep exactly the sequences the sweeps already consumed. This kills
the contamination from the old (UNSTAMPED / hashfrag-era) labels without introducing any
new sequence-provenance variable. The 5M regeneration/expansion is a separate follow-up.

For each `outputs/reservoir_cache/k562_*.npz`:
  - load `sequences` (unchanged) and `oracle_labels` (old, backed up in-file),
  - re-predict with the canonical oracle,
  - write back atomically (tmp + rename) with:
      oracle_labels       <- new canonical labels (float32)
      oracle_labels_prev  <- the old labels (never destroyed)
      prev_oracle_id      <- previous stamp or "UNSTAMPED"
      oracle_id           <- "full856k_clean"  (the guard token)
      rescore_provenance  <- json blob

Idempotent: a cache already stamped `full856k_clean` is skipped unless --force.
Sharded: pass --shard/--n-shards; files are LPT-balanced by D so each shard does a
similar number of sequences.

Run (single file, ad-hoc):
  uv run --no-sync python scripts/rescore_reservoir_cache.py --cache outputs/reservoir_cache/k562_random_d3000_seed42.npz
Run (one array shard):
  uv run --no-sync python scripts/rescore_reservoir_cache.py --shard $T --n-shards 16
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np

CANON_ID = "full856k_clean"
CACHE_DIR = "outputs/reservoir_cache"
_D_RE = re.compile(r"_d(\d+)_seed")


def list_caches() -> list[str]:
    files = [f for f in sorted(glob.glob(f"{CACHE_DIR}/k562_*.npz")) if "/test.npz" not in f]
    return files


def parse_d(path: str) -> int:
    m = _D_RE.search(path)
    return int(m.group(1)) if m else 0


def lpt_shard(files: list[str], shard: int, n_shards: int) -> list[str]:
    """Longest-processing-time bin-packing by D so shards carry ~equal seq counts."""
    loads = [0] * n_shards
    bins: list[list[str]] = [[] for _ in range(n_shards)]
    for f in sorted(files, key=parse_d, reverse=True):
        i = min(range(n_shards), key=lambda j: loads[j])
        bins[i].append(f)
        loads[i] += parse_d(f)
    return sorted(bins[shard])


def rescore_one(fp: str, oracle, force: bool) -> str:
    from scripts.generate_labeled_pools import _label_sequences

    z = np.load(fp, allow_pickle=True)
    files_in = set(z.files)
    prev_id = str(z["oracle_id"]) if "oracle_id" in files_in else "UNSTAMPED"
    if not force and prev_id == CANON_ID:
        return f"SKIP (already {CANON_ID}) {fp}"

    seqs = [str(s) for s in z["sequences"]]
    n = len(seqs)
    old_labels = np.asarray(z["oracle_labels"], dtype=np.float32)
    assert old_labels.shape == (n,), f"label/seq mismatch in {fp}: {old_labels.shape} vs n={n}"

    t0 = time.time()
    new_labels = _label_sequences(oracle, seqs)
    assert new_labels.shape == (n,), f"new labels wrong shape: {new_labels.shape} vs n={n}"
    assert np.isfinite(new_labels).all(), f"non-finite labels produced for {fp}"

    out = {k: z[k] for k in z.files}
    out["sequences"] = z["sequences"]  # explicit: unchanged
    out["oracle_labels"] = new_labels
    out["oracle_labels_prev"] = old_labels
    out["prev_oracle_id"] = np.asarray(prev_id)
    out["oracle_id"] = np.asarray(CANON_ID)
    out["rescore_provenance"] = np.asarray(
        json.dumps(
            {
                "oracle_id": CANON_ID,
                "oracle_dir": "outputs/oracle_full856k_clean/s2",
                "n": n,
                "new_mean": float(new_labels.mean()),
                "new_std": float(new_labels.std()),
                "prev_mean": float(old_labels.mean()),
                "prev_oracle_id": prev_id,
                "rescored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    )

    tmp = Path(fp).with_name(Path(fp).stem + ".rescore.tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.rename(fp)
    dt = time.time() - t0
    return (
        f"OK {fp}  n={n:,}  prev_mean={old_labels.mean():.3f}->new_mean={new_labels.mean():.3f}"
        f"  ({n / dt:.1f} seq/s)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--cache", nargs="*", help="explicit cache paths (overrides sharding)")
    ap.add_argument("--force", action="store_true", help="re-score even if already stamped")
    ap.add_argument("--list-only", action="store_true", help="print this shard's files and exit")
    args = ap.parse_args()

    files = args.cache if args.cache else lpt_shard(list_caches(), args.shard, args.n_shards)
    total = sum(parse_d(f) for f in files)
    print(
        f"=== rescore shard {args.shard}/{args.n_shards}: {len(files)} files, ~{total:,} seqs ==="
    )
    for f in files:
        print(f"  {f}  (D~{parse_d(f):,})")
    if args.list_only:
        return

    from experiments.exp1_1_scaling import _load_oracle

    oracle = _load_oracle("k562", "ag_s2")
    print(f"Loaded canonical oracle ({CANON_ID}).", flush=True)

    for f in files:
        print(rescore_one(f, oracle, args.force), flush=True)
    print("=== shard done ===", flush=True)


if __name__ == "__main__":
    main()
