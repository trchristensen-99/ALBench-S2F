"""CPU plumbing test for the oracle-guided evolution family: same call as build_val_variants.py gpu, with a stand-in predictor."""

import sys
import time

import numpy as np

sys.path.insert(0, "scripts/val_variants")
import build_val_variants as B  # noqa: E402

B.use_val_backgrounds()
from albench.core.registry import Context, get  # noqa: E402


class Fake:  # rewards CpG-rich sequences; deterministic, needs no GPU
    def predict(self, seqs):
        return np.array([s.count("CG") * 0.2 + s.count("GC") * 0.05 for s in seqs], np.float32)


val, _ = B.val_sequences()
params = [r for r in B.V2_FAMILIES if r[0] == "ise_oracle_maximize"][0][2]
t = time.time()
seqs, meta = get("ise_maximize").generate(
    200, Context(task="k562", pool_sequences=val, oracle=Fake()), seed=1, **params
)
print(
    "generated",
    len(seqs),
    "in %.1fs" % (time.time() - t),
    "len set",
    {len(s) for s in seqs},
    "cols",
    list(meta.columns),
)
par = np.asarray(meta["parent_idx"])
fit = Fake().predict(seqs)
base = Fake().predict([val[i] for i in par])
print(
    "mean fitness parents %.2f -> evolved %.2f ; distinct %d"
    % (base.mean(), fit.mean(), len(set(seqs)))
)
hd = [sum(a != b for a, b in zip(s, val[p])) for s, p in zip(seqs, par)]
print("mean Hamming distance to parent: %.1f of 200" % np.mean(hd))
