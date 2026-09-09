import logging, sys, time
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from albench.reservoir.encode_accessibility import (
    EncodeAccessibilitySampler,
    TEST_CHROMS,
    VAL_CHROMS,
    PARTITIONS,
)

print("=" * 76)
print("ENCODE ACCESSIBILITY SAMPLER")
print("=" * 76)
for part in ("shared_open_both", "k562_only", "hepg2_only"):
    s = EncodeAccessibilitySampler(seed=3, partition=part)
    t = time.time()
    seqs, meta = s.generate(4000)
    el = time.time() - t
    held = set(TEST_CHROMS) | set(VAL_CHROMS)
    leak = meta[meta.chrom.isin(held)]
    print(
        f"  {part:>18}: {len(seqs)} seqs, unique {len(set(seqs))}, "
        f"GC {meta.gc_content.mean():.3f}, peak width med {int(meta.peak_width.median())}, "
        f"LEAK into test/val chroms: {len(leak)}  ({el:.1f}s)"
    )
    assert len(leak) == 0, f"LEAKAGE: {leak.chrom.unique()}"
    assert all(len(x) == 200 for x in seqs)

print("\n  depth/count matching (HepG2 has 2.1x K562's specific peaks):")
for part, n in (("k562_only", None), ("hepg2_only", None), ("hepg2_only", 54250)):
    s = EncodeAccessibilitySampler(seed=3, partition=part, match_count=n)
    s._load_peaks()
    print(f"    {part} match_count={n}: {len(s._peaks):,} usable peaks")

print("\n  guards:")
try:
    EncodeAccessibilitySampler(partition="bogus")
    print("    FAIL: bad partition accepted")
except ValueError:
    print("    OK: bad partition rejected")
s = EncodeAccessibilitySampler(seed=1, partition="shared_open_both")
seqs, meta = s.generate(200)
print(
    f"    GC vs random expectation: {meta.gc_content.mean():.3f} (peaks should be GC-rich, >0.42)"
)
print("\n  over-request (more sequences than peaks) uses replacement:")
s = EncodeAccessibilitySampler(seed=1, partition="shared_open_both")
n_pk = len(s._load_peaks())
seqs, meta = s.generate(n_pk + 500)
print(
    f"    asked {n_pk + 500:,} from {n_pk:,} peaks -> got {len(seqs):,}, "
    f"{meta.window_start.nunique():,} distinct windows"
)
print("\nDONE")
