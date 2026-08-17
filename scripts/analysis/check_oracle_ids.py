"""Verify reservoir caches carry the canonical oracle id (full856k_clean)."""

import sys

import numpy as np

for r in sys.argv[1:]:
    path = f"outputs/reservoir_cache/k562_{r}_d30000_seed42.npz"
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  {r}: LOAD FAIL {e}")
        continue
    oid = z["oracle_id"] if "oracle_id" in z.files else "NO oracle_id"
    print(f"  {r}: {oid}")
