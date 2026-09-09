"""Fast verification of the new sampler paths (skips O(n^2) clustering)."""

import logging, sys, time
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from albench.reservoir.motif_planted_v2 import MotifPlantedV2Sampler, PhylogeneticZoonomiaSampler

RATES = "/grid/wsbs/home_norepl/christen/ALBench-S2F/data/zoonomia/per_position_rates.npz"

print("=" * 76)
print("ZOONOMIA: does rate_mode change WHERE mutations land?")
print("=" * 76)
for mode, rate in [("flat", 0.02), ("per_position_matched", 0.02), ("per_position", None)]:
    s = PhylogeneticZoonomiaSampler(seed=11, rate_mode=mode, mut_rate=rate)
    if mode == "flat":
        z = np.load(RATES, allow_pickle=True)
        base = [str(x) for x in z["sequences"][:20000]]
        t = time.time()
        seqs, meta = s.generate(5000, base_sequences=base)
        el = time.time() - t
    else:
        t = time.time()
        seqs, meta = s.generate(5000)
        el = time.time() - t
    print(
        f"  {mode:>21}: {meta.n_mutations.mean():6.1f} mut/200bp "
        f"({meta.n_mutations.mean() / 200:.2%}), sd {meta.n_mutations.std():5.2f}  ({el:.1f}s/5k)"
    )
    if mode != "flat":
        c = np.corrcoef(meta.mean_phylop, meta.n_mutations)[0, 1]
        print(
            f"        corr(mean phyloP, n_mut) = {c:+.3f}  <-- negative: conserved regions mutated LESS"
        )

print("\n  guard checks:")
try:
    PhylogeneticZoonomiaSampler(seed=1, rate_mode="per_position").generate(
        10, base_sequences=["ACGT" * 50]
    )
    print("    FAIL: per_position accepted base_sequences")
except ValueError:
    print("    OK: per_position refuses base_sequences (rates must pair with their region)")
try:
    PhylogeneticZoonomiaSampler(seed=1, rate_mode="bogus")
    print("    FAIL: bad rate_mode accepted")
except ValueError:
    print("    OK: bad rate_mode rejected")

print("\n  ti/tv spectrum (does the knob work?):")
ref = np.load(RATES, allow_pickle=True)["sequences"]
TRANS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
for titv in (1.0, 2.0, 4.0):
    s = PhylogeneticZoonomiaSampler(seed=5, rate_mode="per_position", ti_tv=titv)
    seqs, meta = s.generate(1200)
    ti = tv = 0
    for i, ridx in enumerate(meta.region_idx.values):
        for x, y in zip(str(ref[ridx])[:200], seqs[i]):
            if x != y and x in "ACGT" and y in "ACGT":
                if (x, y) in TRANS:
                    ti += 1
                else:
                    tv += 1
    print(f"    ti_tv set to {titv}: observed {ti / max(tv, 1):.2f}  (ti={ti:,} tv={tv:,})")

print("\n" + "=" * 76)
print("JASPAR PLANTING: pwm_sample vs consensus (cluster_at=None for speed)")
print("=" * 76)
from collections import Counter

for mode in ("pwm_sample", "consensus"):
    s = MotifPlantedV2Sampler(
        seed=7,
        motif_set="jaspar",
        vocab_cluster_at=None,
        vocab_size=60,
        vocab_trim_ic=0.5,
        vocab_max_len=12,
        plant_mode=mode,
    )
    t = time.time()
    seqs, meta = s.generate(3000, task="k562")
    el = time.time() - t
    names = [n for row in meta.planted_motifs for n in row.split(",") if n != "none"]
    top = Counter(names).most_common(1)[0][0]
    print(
        f"  {mode:>12}: {len(seqs)} seqs, mean {meta.n_motifs_planted.mean():.1f} planted, "
        f"|V|={meta.vocab_size.iloc[0]}, method={meta.method.iloc[0]}"
    )
    print(
        f"                '{top}' planted as {len(set(s._inst[top]))} distinct strings, "
        f"{len(set(seqs))} unique sequences"
    )
print("\nDONE")
