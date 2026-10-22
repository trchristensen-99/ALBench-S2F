"""Does the 30-motif core set actually cover what drives the high-activity set?

A capped vocabulary is only a good bet if the motifs that matter are inside the cap.
This scans the high-activity (designed/OOD) eval set and the genomic eval set with the
FULL clustered PWM vocabulary and ranks motifs by their enrichment in high-activity
relative to genomic -- then reports how many of the top-ranked motifs the 30-entry
core set would have contained. If the enriched motifs sit outside the cap, a
motif-planted arm literally cannot represent what the high-activity set is made of.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
N_SUB = 3000
IDX = {c: i for i, c in enumerate("ACGT")}


def onehot(seqs, L=200):
    X = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(str(s)[:L]):
            k = IDX.get(c)
            if k is not None:
                X[i, j, k] = 1.0
    return X


def max_scores(X, pwm):
    """Best log-odds score per sequence for one PWM."""
    lo = np.log2(np.clip(pwm, 1e-4, None) / 0.25).astype(np.float32)
    L = lo.shape[0]
    n_win = X.shape[1] - L + 1
    if n_win <= 0:
        return np.full(X.shape[0], -np.inf, dtype=np.float32)
    win = np.lib.stride_tricks.sliding_window_view(X, (L, 4), axis=(1, 2))[:, :, 0]
    return np.einsum("nwlk,lk->nw", win, lo).max(axis=1)


def main() -> int:
    from albench.motifs import vocabulary as V

    meme = V._default_meme()
    log.info("MEME source: %s", meme)
    full = V.build(meme=meme, human_only=True, cluster_at=0.90, max_motifs=None,
                   trim_ic=0.5, max_len=12)
    core = V.build(meme=meme, human_only=True, cluster_at=0.90, max_motifs=30,
                   trim_ic=0.5, max_len=12)
    core_names = {m.name for m in core}
    log.info("FULL vocabulary: %d entries | CORE cap: %d entries", len(full), len(core))

    # How much did the old CIS-BP name mangling damage this vocabulary? The sampler
    # keys its per-entry instance pool by name, so entries that collided on a label
    # overwrote each other and planted from the wrong PFMs.
    old = [m.name.split(".")[-1] for m in full]
    n_collided = len(old) - len(set(old))
    log.info("collision audit: %d/%d entries shared a label under the old parser "
             "(%.0f%% of the vocabulary planted from a colliding instance pool)",
             n_collided, len(full), 100 * n_collided / max(len(full), 1))

    rng = np.random.default_rng(0)
    sets = {}
    for tag in ("ood", "genomic"):
        z = np.load(REPO / "outputs/eval_sets_v3/labelled" / f"{tag}.npz", allow_pickle=True)
        s = z["sequences"]
        sel = rng.choice(len(s), size=min(N_SUB, len(s)), replace=False)
        sets[tag] = onehot(s[sel])
        log.info("  %-8s n=%d", tag, len(sel))

    rows = []
    for m in full:
        a = max_scores(sets["ood"], m.pwm).mean()
        b = max_scores(sets["genomic"], m.pwm).mean()
        rows.append((float(a - b), m.name, m.name in core_names))
    rows.sort(reverse=True)

    log.info("\n=== Top 25 motifs ENRICHED in high-activity vs genomic ===")
    log.info("%-6s %-28s %10s  %s", "rank", "motif", "delta_bits", "in_core30")
    for i, (d, nm, inc) in enumerate(rows[:25], 1):
        log.info("%-6d %-28s %10.3f  %s", i, nm[:28], d, "YES" if inc else "--")

    for k in (10, 25, 50, 100):
        hit = sum(1 for d, nm, inc in rows[:k] if inc)
        log.info("top-%-4d enriched motifs: %2d/%d are in the 30-entry core (%.0f%%)",
                 k, hit, k, 100 * hit / k)

    log.info("\n=== weighting sanity (full vocab) ===")
    ic = np.array([m.info_content for m in full], dtype=float)
    cs = np.array([len(m.name) for m in full], dtype=float)  # placeholder scale check
    for temp in (0.0, 0.5, 1.0, 2.0):
        if temp == 0:
            p = np.ones(len(full)) / len(full)
        else:
            w = np.clip(ic, 1e-9, None) ** temp
            p = w / w.sum()
        log.info("  ic temp=%.1f -> effective |V| = %6.1f of %d", temp, 1.0 / (p ** 2).sum(), len(full))
    del cs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
