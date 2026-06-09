"""Build the proper monoallelic snv_oracle.npz by subsetting the existing battery.

Why subset instead of re-scoring: the over-sized snv_oracle.npz (n=45,543) was scored
in the SAME Jun-2 run as genomic_oracle.npz / ood_oracle.npz, so its per-sequence AG_S2
predictions are already consistent with the rest of the chr-split battery. The only thing
wrong with it was the SET — it included context-expansion duplicates (pair_keys with >1
row). The strict-monoallelic build (chr 7+13, 1 row per pair_key, n=29,383) is exactly a
subset, and every mono (ref,alt) pair is present in the over-sized file (verified 100%).

Re-scoring via _load_oracle is currently broken (LayerNorm 'scale' missing at predict —
the stale AG_S2 loader), and the canonical chr-split natural ensemble no longer exists on
disk. Subsetting needs no GPU, avoids the broken loader, and is provably consistent with
the battery. The AG_S2 oracle is deterministic per sequence, so a sequence's prediction is
identical wherever it appears in the over-sized file.

Output: snv_oracle.npz stamped test_set_version='snv_mono_chrsplit_v1' (asserted at load
by experiments/test_set_guards.assert_mono_snv).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TEST_DIR = REPO / "data/k562/test_sets_ag_s2_chrsplit"
OVERSIZED = TEST_DIR / "snv_oracle.npz"
INTERMEDIATE = TEST_DIR / "snv_mono_chr7_13_intermediate.npz"
BACKUP_OVERSIZED = TEST_DIR / "snv_oracle_legacy_oversized_45543.npz"

EXPECTED_N_MIN = 25_000
EXPECTED_N_MAX = 35_000


def main():
    if not OVERSIZED.exists():
        raise FileNotFoundError(f"missing {OVERSIZED} (the battery-scored SNV file)")
    if not INTERMEDIATE.exists():
        raise FileNotFoundError(
            f"missing {INTERMEDIATE}; run scripts/build_chrsplit_snv_mono.py first"
        )

    big = np.load(OVERSIZED, allow_pickle=True)
    if "test_set_version" in big.files:
        raise RuntimeError(
            f"{OVERSIZED} already carries a version stamp — it appears to be a rebuilt "
            f"mono set, not the over-sized battery file. Aborting to avoid re-subsetting."
        )
    big_ref = [str(s) for s in big["ref_sequences"]]
    big_alt = [str(s) for s in big["alt_sequences"]]
    ref_mean_by_seq = dict(zip(big_ref, big["ref_mean"].astype(np.float32)))
    alt_mean_by_seq = dict(zip(big_alt, big["alt_mean"].astype(np.float32)))

    mono = np.load(INTERMEDIATE, allow_pickle=True)
    mr = [str(s) for s in mono["ref_sequences"]]
    ma = [str(s) for s in mono["alt_sequences"]]
    n = len(mr)
    if not (EXPECTED_N_MIN <= n <= EXPECTED_N_MAX):
        raise RuntimeError(f"mono count {n:,} outside expected range")

    missing = [i for i in range(n) if mr[i] not in ref_mean_by_seq or ma[i] not in alt_mean_by_seq]
    if missing:
        raise RuntimeError(
            f"{len(missing)} mono sequences absent from the battery file — cannot subset "
            f"consistently; the over-sized file is not a superset of the mono set."
        )

    ref_mean = np.array([ref_mean_by_seq[s] for s in mr], dtype=np.float32)
    alt_mean = np.array([alt_mean_by_seq[s] for s in ma], dtype=np.float32)
    delta_mean = (alt_mean - ref_mean).astype(np.float32)

    if not BACKUP_OVERSIZED.exists():
        # Copy (don't move) so the source remains for re-derivation/auditing.
        np.savez_compressed(BACKUP_OVERSIZED, **{k: big[k] for k in big.files})
        print(f"  backed up over-sized battery SNV → {BACKUP_OVERSIZED.name}")

    np.savez_compressed(
        OVERSIZED,
        pair_keys=mono["pair_keys"],
        ref_sequences=np.array(mr, dtype=object),
        alt_sequences=np.array(ma, dtype=object),
        ref_mean=ref_mean,
        alt_mean=alt_mean,
        delta_mean=delta_mean,
        true_ref_label=mono["true_ref_label"].astype(np.float32),
        true_alt_label=mono["true_alt_label"].astype(np.float32),
        true_delta=mono["true_delta"].astype(np.float32),
        test_set_version=np.str_("snv_mono_chrsplit_v1"),
        monoallelic=np.bool_(True),
        n_pairs=np.int64(n),
        oracle_dir=np.str_("subset_of_jun2_battery(snv_oracle.npz n=45543)"),
        source_tsv=np.str_("data/k562/test_sets/test_snv_pairs.tsv"),
        created_utc=np.str_(datetime.now(timezone.utc).isoformat()),
    )
    print(f"  saved {OVERSIZED}  (n={n:,}, version=snv_mono_chrsplit_v1)")
    print(f"  ref_mean μ={ref_mean.mean():+.3f} σ={ref_mean.std():.3f}")
    print(f"  delta_mean μ={delta_mean.mean():+.3f} σ={delta_mean.std():.3f}")


if __name__ == "__main__":
    main()
