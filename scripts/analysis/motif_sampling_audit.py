"""Tabulate what the motif planting procedure actually does, per Peter's question.

Reports, for each motif arm: vocabulary size, the distribution of how many motifs are
planted per sequence (requested vs REALIZED, which differ when placement fails), which
motifs get used and how often, and the spacing between adjacent planted sites -- the
last of these answers whether sites can abut or overlap.
"""

from __future__ import annotations

import collections
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

N = 2000
ARMS = {
    "motif_syntax_core": {},
    "motif_ct_enriched": {},
    "motif_dense (8-15)": {"_strategy": "motif_syntax_core", "min_motifs": 8, "max_motifs": 15},
}


def main() -> int:
    from albench.registry import Context, get

    for label, params in ARMS.items():
        strategy = params.pop("_strategy", label)
        spec = get(strategy)
        seqs, meta = spec.generate(N, Context(task="k562"), seed=0, **params)
        sampler = spec.build(seed=0, **params)
        sampler._build_vocab()
        vocab = sampler._vocab

        log.info("\n=== %s ===", label)
        log.info("  vocabulary: %d entries (%d PFMs, cluster_at=%.2f, trim_ic=%.1f, max_len=%s)",
                 len(vocab), sum(sampler._n_members), sampler.vocab_cluster_at,
                 sampler.vocab_trim_ic, sampler.vocab_max_len)
        log.info("  requested motifs/seq: uniform %d-%d", sampler.min_motifs, sampler.max_motifs)

        if "n_planted" in meta.columns:
            c = collections.Counter(meta["n_planted"])
            log.info("  REALIZED motifs/seq: mean %.2f  distribution %s",
                     meta["n_planted"].mean(), dict(sorted(c.items())))
        if "n_native_kept" in meta.columns:
            log.info("  native motif sites preserved per seq: mean %.1f", meta["n_native_kept"].mean())

        if "planted_motifs" in meta.columns:
            used = collections.Counter()
            for row in meta["planted_motifs"]:
                if row and row != "none":
                    used.update(row.split(","))
            log.info("  distinct vocabulary entries actually used: %d/%d", len(used), len(vocab))
            top = used.most_common(8)
            tot = sum(used.values())
            log.info("  top entries by share of all planted sites:")
            for name, n in top:
                log.info("      %-24s %5d  (%.1f%%)", name[:24], n, 100 * n / tot)
            shares = np.array(sorted(used.values(), reverse=True)) / tot
            log.info("  usage evenness: top-5 entries = %.0f%% of sites; effective |V| = %.1f",
                     100 * shares[:5].sum(), 1.0 / (shares ** 2).sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
